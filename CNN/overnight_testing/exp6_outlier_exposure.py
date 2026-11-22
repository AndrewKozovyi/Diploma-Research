"""
Outlier exposure: фінальна конфігурація + 11-й клас "Unknown" (кістки інших родів динозаврів і не-динозаврів).

OOD ділиться за РОДОМ (seed 0): половина родів near і far - в Unknown (train/val), решта - лише для оцінки,
тож оцінка показує, чи відсіюються роди, яких модель не бачила. far_raw (фото з фоном) у навчання не йде.
Порівняння з базовими моделями v2exp_f7_m_img320_sa (3 сіди, ті самі дані без Unknown) на:
  ID test (точність серед 10 класів, частка прийнятих) і OOD eval-родах (частка відсіяних).
Правила відсіювання: OE - argmax = Unknown або MSP по 10 класах < порогу; базова - MSP < порогу.
Поріг MSP підбирається так, щоб на ID val проходило 95% фото.

Результати: results_yolo/oe_v2/ : oe_summary.csv, oe_seeds.csv
Запуск:  python exp6_outlier_exposure.py
"""
import os
import csv
import random
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = r"D:\projects\Diploma"
BASE = os.path.join(ROOT, r"datasets\transfer_learning\transfer_learning_datasets_v2_filtered_by_5_r512")
DATA = os.path.join(ROOT, r"datasets\transfer_learning\transfer_learning_datasets_v2_oe_r512")
OOD = os.path.join(ROOT, r"datasets\ood")
RES = os.path.join(ROOT, r"CNN\results_yolo")
OUT = os.path.join(RES, "oe_v2")
BASELINES = ["v2exp_f7_m_img320_sa", "v2exp_f7_m_img320_sa-2", "v2exp_f7_m_img320_sa-3"]
UNKNOWN = "Unknown"
VAL_SHARE = 0.15
TARGET_TPR = 0.95
ARGS = dict(epochs=30, imgsz=320, batch=16, freeze=7, patience=15, optimizer="auto", lr0=1e-5, dropout=0.8,
            weight_decay=0.005, degrees=15, fliplr=0.5, translate=0.1, scale=0.7, flipud=0.5, erasing=0.6,
            deterministic=True, device=0, workers=4)


def genus_split():
    rng = random.Random(0)
    split = {}
    for kind in ("near", "far"):
        genera = sorted(os.listdir(os.path.join(OOD, kind)))
        rng.shuffle(genera)
        for i, g in enumerate(genera):
            split[(kind, g)] = "train" if i % 2 == 0 else "eval"
    return split


def files(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d))


def build(split):
    if os.path.exists(os.path.join(DATA, "DONE")):
        return
    if os.path.exists(DATA):
        shutil.rmtree(DATA)
    shutil.copytree(BASE, DATA)
    rng = random.Random(0)
    pool = [p for (kind, g), s in split.items() if s == "train" for p in files(os.path.join(OOD, kind, g))]
    rng.shuffle(pool)
    n_val = int(len(pool) * VAL_SHARE)
    for i, p in enumerate(pool):
        dst = os.path.join(DATA, "val" if i < n_val else "train", UNKNOWN)
        os.makedirs(dst, exist_ok=True)
        Image.open(p).convert("RGB").save(os.path.join(dst, f"{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}.png"))
    open(os.path.join(DATA, "DONE"), "w").close()
    print("Unknown train/val:", len(pool) - n_val, n_val)


@torch.no_grad()
def probs(model, paths):
    out = []
    for i in range(0, len(paths), 32):
        out += [r.probs.data.cpu().numpy() for r in model.predict(paths[i:i + 32], imgsz=ARGS["imgsz"], verbose=False, device=0)]
    return np.stack(out)


def evaluate(weights, classes, split):
    model = YOLO(weights)
    names = [model.names[i] for i in range(len(model.names))]
    id_idx = [names.index(c) for c in classes]
    unk = names.index(UNKNOWN) if UNKNOWN in names else None

    def score(paths):
        p = probs(model, paths)
        p_id = p[:, id_idx]
        is_unk = p.argmax(1) == unk if unk is not None else np.zeros(len(p), bool)
        return p_id.argmax(1), p_id.max(1) / p_id.sum(1), is_unk

    def id_set(s):
        paths, y = [], []
        for i, c in enumerate(classes):
            for p in files(os.path.join(BASE, s, c)):
                if "_aug_" not in p:
                    paths.append(p)
                    y.append(i)
        return paths, np.array(y)

    vp, _ = id_set("val")
    _, v_msp, v_unk = score(vp)
    thr = np.quantile(v_msp[~v_unk], 1 - TARGET_TPR) if (~v_unk).any() else 1.0
    tp, ty = id_set("test")
    t_pred, t_msp, t_unk = score(tp)
    t_acc = ~t_unk & (t_msp >= thr)
    row = {"weights": os.path.relpath(weights, ROOT), "msp_thr": thr,
           "ID_test_top1": float(((t_pred == ty) & ~t_unk).mean()),
           "ID_test_predicted_unknown": float(t_unk.mean()), "ID_test_accepted": float(t_acc.mean()),
           "ID_acc_of_accepted": float((t_pred == ty)[t_acc].mean())}
    for kind in ("near", "far", "far_raw"):
        if kind == "far_raw":
            paths = [p for (k, g), s in split.items() if k == "far" and s == "eval"
                     for p in files(os.path.join(OOD, "far_raw", g)) if os.path.isdir(os.path.join(OOD, "far_raw", g))]
        else:
            paths = [p for (k, g), s in split.items() if k == kind and s == "eval" for p in files(os.path.join(OOD, kind, g))]
        _, o_msp, o_unk = score(paths)
        row[f"{kind}_n"] = len(paths)
        row[f"{kind}_pred_unknown"] = float(o_unk.mean())
        row[f"{kind}_rejected"] = float((o_unk | (o_msp < thr)).mean())
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    split = genus_split()
    with open(os.path.join(OUT, "genus_split.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "genus", "split"])
        w.writerows([[k, g, s] for (k, g), s in sorted(split.items())])
    build(split)
    classes = sorted(os.listdir(os.path.join(BASE, "train")))
    runs = []
    for seed in (0, 1, 2):
        name = f"oe_seed{seed}"
        if not os.path.exists(os.path.join(OUT, name, "weights", "best.pt")) or not os.path.exists(os.path.join(OUT, name, "DONE")):
            YOLO(os.path.join(ROOT, r"CNN\yolo11m-cls.pt")).train(data=DATA, project=OUT, name=name, exist_ok=True,
                                                                  plots=False, seed=seed, **ARGS)
            open(os.path.join(OUT, name, "DONE"), "w").close()
        runs.append(("outlier exposure", os.path.join(OUT, name, "weights", "best.pt")))
    runs += [("baseline", os.path.join(RES, b, "weights", "best.pt")) for b in BASELINES]
    rows = [{"model": m, **evaluate(w, classes, split)} for m, w in runs]
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(OUT, "oe_seeds.csv"), index=False)
    summary = df.drop(columns=["weights"]).groupby("model").mean().round(3)
    summary.to_csv(os.path.join(OUT, "oe_summary.csv"))
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
