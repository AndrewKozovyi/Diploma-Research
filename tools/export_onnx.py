"""
Export the final YOLO classifier to ONNX and ship it to the Dino-Predictor app.

Steps:
  1. Export best.pt -> ONNX (static 1x3xSxS input, softmax output) and add a second output with the
     feature maps before pooling, so that the .NET API can build Grad-CAM without gradients.
  2. Verify: run the ONNX model with a plain PIL/NumPy preprocessing (the same recipe the .NET API
     must implement) and compare probabilities with PyTorch and CAM with gradcam_yolo.py on the test split.
  3. Copy model.onnx + cam-weights.bin + labels.json + model-info.json into <app>/ML.

Usage:
  python tools/export_onnx.py
  python tools/export_onnx.py --weights path/to/best.pt --target D:/projects/Dino-Predictor/ML
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image, ImageOps
from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CNN"))
from gradcam_yolo import YoloGradCAM  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "CNN/results_yolo/yolo_11_selected_9_classes-150-27/weights/best.pt"
DEFAULT_TARGET = ROOT.parent / "Dino-Predictor/ML"
DEFAULT_TEST_DIR = ROOT / "datasets/clever_split/test"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROBABILITIES_OUTPUT = "output0"
FEATURES_OUTPUT = "features"

# Допуски перевірки ONNX vs PyTorch (різниця дає лише реалізація resize)
MAX_PROB_DIFF = 0.02
MIN_TOP1_AGREEMENT = 0.98
MAX_CAM_DIFF = 0.01


def preprocess(image_path: Path, size: int) -> np.ndarray:
    """Еталонний препроцесинг, який треба повторити в .NET (= ultralytics classify_transforms)."""
    # cv2.imread у ultralytics застосовує EXIF-орієнтацію і відкидає альфа-канал без змішування
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

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


def add_cam_output(onnx_path: Path) -> np.ndarray:
    """Додає вихід FEATURES_OUTPUT (карти ознак перед pooling) і повертає ваги Linear (класи x канали).

    Голова Classify: conv -> GlobalAveragePool -> Linear -> Softmax. За такої будови градієнт логіта
    за картами ознак сталий, тому Grad-CAM = max(0, sum_k weights[cls, k] * features[k]) без backward.
    """
    model = onnx.load(str(onnx_path))
    nodes = model.graph.node
    pool = next(n for n in nodes if n.op_type == "GlobalAveragePool")
    linear = next(n for n in nodes if n.op_type == "Gemm")
    initializers = {i.name: i for i in model.graph.initializer}

    # Identity дає стабільну назву виходу, незалежну від внутрішніх назв вузлів експорту
    inferred = onnx.shape_inference.infer_shapes(model)
    features_info = next(v for v in inferred.graph.value_info if v.name == pool.input[0])
    nodes.append(onnx.helper.make_node("Identity", [pool.input[0]], [FEATURES_OUTPUT]))
    output = model.graph.output.add()
    output.CopyFrom(features_info)
    output.name = FEATURES_OUTPUT
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))

    return numpy_helper.to_array(initializers[linear.input[1]])


def build_cam(features: np.ndarray, weights: np.ndarray, cls: int) -> np.ndarray:
    """Еталонний розрахунок CAM, який треба повторити в .NET. features: канали x h x w."""
    cam = np.maximum((weights[cls][:, None, None] * features).sum(0), 0)
    return cam / (cam.max() + 1e-8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_test_images(test_dir: Path, limit: int) -> list[Path]:
    # Аугментовані копії не несуть нової інформації для перевірки
    images = [p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug_" not in p.stem]
    random.Random(42).shuffle(images)
    return images[:limit]


def verify(
    pt_model: YOLO, weights_path: Path, onnx_path: Path, cam_weights: np.ndarray, images: list[Path], size: int
) -> dict:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    grad_cam = YoloGradCAM(str(weights_path), size)

    top1_matches, max_diff, max_cam_diff = 0, 0.0, 0.0
    features_shape = None
    for image_path in images:
        pt_probs = pt_model.predict(str(image_path), imgsz=size, verbose=False)[0].probs.data.cpu().numpy()
        onnx_probs, features = session.run(
            [PROBABILITIES_OUTPUT, FEATURES_OUTPUT], {input_name: preprocess(image_path, size)}
        )
        onnx_probs, features = onnx_probs[0], features[0]
        features_shape = features.shape

        top1_matches += int(pt_probs.argmax() == onnx_probs.argmax())
        max_diff = max(max_diff, float(np.abs(pt_probs - onnx_probs).max()))

        cam = build_cam(features, cam_weights, int(onnx_probs.argmax()))
        max_cam_diff = max(max_cam_diff, float(np.abs(cam - grad_cam(image_path)[1]).max()))

    return {
        "images": len(images),
        "top1Agreement": round(top1_matches / len(images), 4),
        "maxProbabilityDiff": round(max_diff, 5),
        "maxCamDiffVsGradCam": round(max_cam_diff, 5),
        "featuresShape": [1, *features_shape],
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
    cam_weights = add_cam_output(onnx_path)

    images = collect_test_images(args.test_dir, args.samples)
    if not images:
        raise SystemExit(f"[!] No test images in {args.test_dir}")

    print(f"--- Verifying ONNX on {len(images)} test images ---")
    report = verify(model, args.weights, onnx_path, cam_weights, images, args.imgsz)
    print(json.dumps(report, indent=2))
    if (
        report["top1Agreement"] < MIN_TOP1_AGREEMENT
        or report["maxProbabilityDiff"] > MAX_PROB_DIFF
        or report["maxCamDiffVsGradCam"] > MAX_CAM_DIFF
    ):
        raise SystemExit("[!] ONNX output differs from PyTorch more than allowed. Nothing was copied.")

    labels = [model.names[i] for i in sorted(model.names)]
    args.target.mkdir(parents=True, exist_ok=True)
    cam_weights_path = args.target / "cam-weights.bin"
    cam_weights.astype("<f4").tofile(cam_weights_path)

    info = {
        "model": "model.onnx",
        "sha256": sha256(onnx_path),
        "sourceWeights": args.weights.relative_to(ROOT).as_posix(),
        "architecture": "YOLO11n-cls",
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ultralytics": ULTRALYTICS_VERSION,
        "opset": args.opset,
        "input": {"name": "images", "shape": [1, 3, args.imgsz, args.imgsz], "dtype": "float32", "layout": "NCHW"},
        "outputs": {
            "probabilities": {
                "name": PROBABILITIES_OUTPUT,
                "shape": [1, len(labels)],
                "meaning": "softmax probabilities, index -> labels.json",
            },
            "features": {
                "name": FEATURES_OUTPUT,
                "shape": report["featuresShape"],
                "layout": "NCHW",
                "meaning": "feature maps before global average pooling, used for CAM",
            },
        },
        "camWeights": {
            "file": cam_weights_path.name,
            "sha256": sha256(cam_weights_path),
            "shape": list(cam_weights.shape),
            "dtype": "float32, little-endian",
            "layout": "row-major: weights[cls * channels + channel]",
        },
        "cam": [
            "cls = index of the highest probability",
            "cam[i, j] = max(0, sum over channels k of weights[cls, k] * features[k, i, j])",
            "divide by the maximum of cam (skip if the maximum is 0)",
        ],
        "preprocessing": [
            "decode image, apply EXIF orientation, drop alpha channel without blending, RGB",
            f"resize (bilinear, antialiased): shortest side -> {args.imgsz}, longest -> floor({args.imgsz} * long / short)",
            f"center crop {args.imgsz}x{args.imgsz}, offset = round((side - {args.imgsz}) / 2)",
            "scale pixels to [0, 1] (value / 255), no mean/std normalization",
            "HWC -> CHW, add batch dimension",
        ],
        "verification": report,
    }

    shutil.copy2(onnx_path, args.target / "model.onnx")
    (args.target / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    (args.target / "model-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(f"\n[+] Copied model.onnx, {cam_weights_path.name}, labels.json, model-info.json -> {args.target}")


if __name__ == "__main__":
    main()
