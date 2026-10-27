"""
Export the final YOLO classifier to ONNX and ship it to the Dino-Predictor app.

Steps:
  1. Export best.pt -> ONNX (static 1x3xSxS input, softmax output).
  2. Verify: run the ONNX model with a plain PIL/NumPy preprocessing (the same recipe
     the .NET API must implement) and compare it with the PyTorch model on the test split.
  3. Copy model.onnx + labels.json + model-info.json into <app>/ML.

Usage:
  python tools/export_onnx.py
  python tools/export_onnx.py --weights path/to/best.pt --target D:/projects/Dino-Predictor/ML
"""
import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "CNN/results_yolo/yolo_11_selected_9_classes-150-27/weights/best.pt"
DEFAULT_TARGET = ROOT.parent / "Dino-Predictor/ML"
DEFAULT_TEST_DIR = ROOT / "datasets/clever_split/test"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Допуски перевірки ONNX vs PyTorch (різниця дає лише реалізація resize)
MAX_PROB_DIFF = 0.02
MIN_TOP1_AGREEMENT = 0.98


def preprocess(image_path: Path, size: int) -> np.ndarray:
    """Еталонний препроцесинг, який треба повторити в .NET (= ultralytics classify_transforms)."""
    # cv2.imread у ultralytics відкидає альфа-канал без змішування, тому робимо так само
    img = Image.open(image_path).convert("RGB")

    # Коротка сторона -> size, довга -> int(size * long / short) (усічення, як у torchvision)
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Зсув центрального кропу: round((side - size) / 2)
    left, top = int(round((new_w - size) / 2)), int(round((new_h - size) / 2))
    img = img.crop((left, top, left + size, top + size))

    tensor = np.asarray(img, dtype=np.float32) / 255.0  # HWC, RGB, [0..1], без mean/std
    return tensor.transpose(2, 0, 1)[np.newaxis]        # NCHW


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_test_images(test_dir: Path, limit: int) -> list[Path]:
    # Аугментовані копії не несуть нової інформації для перевірки
    images = [p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug_" not in p.stem]
    random.Random(42).shuffle(images)
    return images[:limit]


def verify(pt_model: YOLO, onnx_path: Path, images: list[Path], size: int) -> dict:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    top1_matches, max_diff = 0, 0.0
    for image_path in images:
        pt_probs = pt_model.predict(str(image_path), imgsz=size, verbose=False)[0].probs.data.cpu().numpy()
        onnx_probs = session.run(None, {input_name: preprocess(image_path, size)})[0][0]

        top1_matches += int(pt_probs.argmax() == onnx_probs.argmax())
        max_diff = max(max_diff, float(np.abs(pt_probs - onnx_probs).max()))

    return {
        "images": len(images),
        "top1Agreement": round(top1_matches / len(images), 4),
        "maxProbabilityDiff": round(max_diff, 5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="ML folder of the Dino-Predictor app")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--samples", type=int, default=200, help="test images used for verification")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"[!] Weights not found: {args.weights}")
    if not args.target.parent.exists():
        raise SystemExit(f"[!] App folder not found: {args.target.parent}")

    print(f"--- Exporting {args.weights} ---")
    model = YOLO(str(args.weights))
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=False, simplify=True))

    images = collect_test_images(args.test_dir, args.samples)
    if not images:
        raise SystemExit(f"[!] No test images in {args.test_dir}")

    print(f"--- Verifying ONNX on {len(images)} test images ---")
    report = verify(model, onnx_path, images, args.imgsz)
    print(json.dumps(report, indent=2))
    if report["top1Agreement"] < MIN_TOP1_AGREEMENT or report["maxProbabilityDiff"] > MAX_PROB_DIFF:
        raise SystemExit("[!] ONNX output differs from PyTorch more than allowed. Nothing was copied.")

    labels = [model.names[i] for i in sorted(model.names)]
    info = {
        "model": "model.onnx",
        "sha256": sha256(onnx_path),
        "sourceWeights": args.weights.relative_to(ROOT).as_posix(),
        "architecture": "YOLO11n-cls",
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ultralytics": ULTRALYTICS_VERSION,
        "opset": args.opset,
        "input": {"name": "images", "shape": [1, 3, args.imgsz, args.imgsz], "dtype": "float32", "layout": "NCHW"},
        "output": {"name": "output0", "shape": [1, len(labels)], "meaning": "softmax probabilities, index -> labels.json"},
        "preprocessing": [
            "decode image, drop alpha channel without blending, RGB",
            f"resize (bilinear, antialiased): shortest side -> {args.imgsz}, longest -> floor({args.imgsz} * long / short)",
            f"center crop {args.imgsz}x{args.imgsz}, offset = round((side - {args.imgsz}) / 2)",
            "scale pixels to [0, 1] (value / 255), no mean/std normalization",
            "HWC -> CHW, add batch dimension",
        ],
        "verification": report,
    }

    args.target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, args.target / "model.onnx")
    (args.target / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    (args.target / "model-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(f"\n[+] Copied model.onnx, labels.json, model-info.json -> {args.target}")


if __name__ == "__main__":
    main()
