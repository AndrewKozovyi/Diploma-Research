"""
Ансамбль і TTA для YOLO на v2r (results_yolo/gbif_compare/v2r_adamw_s0..2), без донавчання.

TTA - середнє ймовірностей по 4 варіантах фото: оригінал, віддзеркалення по горизонталі, по вертикалі, обидва
(при навчанні були fliplr=0.5 і flipud=0.5, тож усі варіанти моделі знайомі).
Ансамбль - середнє ймовірностей 3 сідів.
Ймовірності зберігаються в results_yolo/model_compare/<метод>.npz для зведеного порівняння (exp10_model_compare.py).

Запуск:  python exp8_yolo_ensemble_tta.py
"""
import os
import glob

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

ROOT = r"D:\projects\Diploma"
DATA = os.path.join(ROOT, r"datasets\transfer_learning\transfer_learning_datasets_v2r_filtered_by_5_r512")
RUNS = [os.path.join(ROOT, rf"CNN\results_yolo\gbif_compare\v2r_adamw_s{s}") for s in (0, 1, 2)]
OUT = os.path.join(ROOT, r"CNN\results_yolo\model_compare")
CLASSES = sorted(os.listdir(os.path.join(DATA, "train")))
VIEWS = [lambda im: im, ImageOps.mirror, ImageOps.flip, lambda im: ImageOps.flip(ImageOps.mirror(im))]


def originals(split):
    return [(c, p) for c in CLASSES for p in sorted(glob.glob(os.path.join(DATA, split, c, "*.png"))) if "_aug_" not in p]


def probs(model, paths, view):
    order = [list(model.names.values()).index(c) for c in CLASSES]
    out = []
    for i in range(0, len(paths), 32):
        images = [view(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]
        out += [r.probs.data.cpu().numpy()[order] for r in model.predict(images, imgsz=320, verbose=False)]
    return np.stack(out)


def save(name, split_data):
    np.savez(os.path.join(OUT, f"{name}.npz"), classes=np.array(CLASSES),
             **{f"{s}_{k}": v for s, d in split_data.items() for k, v in d.items()})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    splits = {s: originals(s) for s in ("val", "test")}
    meta = {s: dict(y=np.array([CLASSES.index(c) for c, _ in items]),
                    files=np.array([f"{c}/{os.path.basename(p)}" for c, p in items])) for s, items in splits.items()}
    single, tta = {}, {}
    for seed, run in enumerate(RUNS):
        model = YOLO(os.path.join(run, "weights", "best.pt"))
        for s, items in splits.items():
            views = [probs(model, [p for _, p in items], v) for v in VIEWS]
            single.setdefault(seed, {})[s] = views[0]
            tta.setdefault(seed, {})[s] = np.mean(views, 0)
        for kind, store in (("yolo_single", single), ("yolo_tta", tta)):
            save(f"{kind}_s{seed}", {s: dict(p=store[seed][s], **meta[s]) for s in splits})
            print(kind, seed, {s: round(float((store[seed][s].argmax(1) == meta[s]["y"]).mean()), 3) for s in splits}, flush=True)
    for kind, store in (("yolo_ensemble", single), ("yolo_ensemble_tta", tta)):
        p = {s: np.mean([store[k][s] for k in store], 0) for s in splits}
        save(kind, {s: dict(p=p[s], **meta[s]) for s in splits})
        print(kind, {s: round(float((p[s].argmax(1) == meta[s]["y"]).mean()), 3) for s in splits}, flush=True)
