"""
YOLO26m-cls на v2r з тією самою конфігурацією, що й YOLO11m (AdamW lr 2e-4, cosine, 40 епох, freeze 7, imgsz 320).
Архітектура класифікатора YOLO26 така сама, як у YOLO11 (той самий backbone і голова), відрізняються
попередньо навчені ваги, тож freeze=7 заморожує ті самі шари. 3 сіди, потім ймовірності для exp10_model_compare.py.

Запуск:  python exp11_yolo26_v2r.py
Результати: results_yolo/gbif_compare/v2r_yolo26_adamw_s*, results_yolo/model_compare/yolo26_*.npz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import transfer_learning_yolo as tly
import exp8_yolo_ensemble_tta as e8

ROOT = e8.ROOT
RUNS = os.path.join(ROOT, r"CNN\results_yolo\gbif_compare")
ARGS = dict(epochs=40, imgsz=320, batch=16, freeze=7, patience=40, optimizer="AdamW", lr0=2e-4, cos_lr=True,
            dropout=0.8, weight_decay=0.005, degrees=15, fliplr=0.5, translate=0.1, scale=0.7, flipud=0.5,
            erasing=0.6, deterministic=True, device=0, workers=4)

if __name__ == "__main__":
    splits = {s: e8.originals(s) for s in ("val", "test")}
    meta = {s: dict(y=np.array([e8.CLASSES.index(c) for c, _ in it]),
                    files=np.array([f"{c}/{os.path.basename(p)}" for c, p in it])) for s, it in splits.items()}
    probs = []
    for seed in (0, 1, 2):
        run = f"v2r_yolo26_adamw_s{seed}"
        d = os.path.join(RUNS, run)
        if not os.path.exists(os.path.join(d, "DONE")):
            results = YOLO(os.path.join(ROOT, r"CNN\yolo26m-cls.pt")).train(data=e8.DATA, project=RUNS, name=run,
                                                                            exist_ok=True, seed=seed, **ARGS)
            tly.EXPERIMENT_NAME = run
            tly.MODEL_WEIGHTS, tly.IMG_SIZE, tly.FREEZE_LAYERS, tly.EPOCHS = "yolo26m-cls.pt", 320, 7, 40
            tly.OPTIMIZER, tly.LR_INIT, tly.WEIGHT_DECAY = "AdamW", 2e-4, 0.005
            tly.plot_custom_results(results.save_dir)
            open(os.path.join(d, "DONE"), "w").close()
        model = YOLO(os.path.join(d, "weights", "best.pt"))
        p = {s: e8.probs(model, [q for _, q in it], e8.VIEWS[0]) for s, it in splits.items()}
        e8.save(f"yolo26_single_s{seed}", {s: dict(p=p[s], **meta[s]) for s in splits})
        probs.append(p)
        print(run, {s: round(float((p[s].argmax(1) == meta[s]["y"]).mean()), 3) for s in splits}, flush=True)
    e8.save("yolo26_ensemble", {s: dict(p=np.mean([p[s] for p in probs], 0), **meta[s]) for s in splits})
