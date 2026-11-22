"""
Порівняння методів відсіювання OOD-фото (не один з 10 класів) для фінальної моделі.

Оцінки (більше = схоже на навчальні класи):
  MSP           - максимальна softmax-ймовірність (те, що зараз віддає ONNX)
  MaxLogit      - максимальний логіт
  Energy        - logsumexp(логіти)
  kNN           - мінус косинусна відстань до k-го найближчого навчального фото в просторі ознак голови (1280-d)
  Mahalanobis   - мінус відстань Махаланобіса до центру найближчого класу (спільна коваріація зі стисненням)

Дані:
  ID     - val і test фінального поділу v2 (оригінали)
  OOD    - datasets/ood: near (інші роди динозаврів), far (не-динозаври, той самий конвеєр), far_raw (фото з фоном)
           + accepted_results/test_unknown. OOD ділиться 50/50 за записом на tune/eval.
Поріг кожного методу: TPR95 - проходить 95% ID val; Youden - максимум TPR-FPR на ID val + OOD tune (near+far).
Звіт - на ID test + OOD eval.

Запуск:  python ood_eval.py [weights]
Результати: results_yolo/ood_eval/
"""
import os
import re
import sys
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

ROOT = r"D:\projects\Diploma"
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, r"CNN\results_yolo\yolo_11m_v2_10_classes_best-150\weights\best.pt")
DATA = os.path.join(ROOT, r"datasets\transfer_learning\transfer_learning_datasets_v2_filtered_by_5")
OOD = os.path.join(ROOT, r"datasets\ood")
UNKNOWN = os.path.join(ROOT, r"accepted_results\test_unknown")
OUT = os.path.join(ROOT, r"CNN\results_yolo\ood_eval")
IMGSZ = 320
KNN_K = 5
TARGET_TPR = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXT = (".png", ".jpg", ".jpeg", ".webp")


class Extractor:
    def __init__(self, weights):
        self.yolo = YOLO(weights)
        self.net = self.yolo.model.float().eval().to(DEVICE)
        self.names = [self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.tf = classify_transforms(IMGSZ)
        self._feat = None
        self.net.model[-1].pool.register_forward_hook(lambda m, i, o: setattr(self, "_feat", o.flatten(1)))

    @torch.no_grad()
    def __call__(self, paths, batch=32):
        logits, feats = [], []
        for i in range(0, len(paths), batch):
            x = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]).to(DEVICE)
            out = self.net(x)
            logits.append(out[1].float().cpu())
            feats.append(self._feat.float().cpu())
        return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def files_in(d):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(EXT))


def record(path):
    m = re.match(r"(?:parsed_|yale_)?([0-9a-f]{8}-[0-9a-f\-]{27}|\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def id_split(split, names, originals_only=True):
    paths, labels = [], []
    for i, c in enumerate(names):
        for p in files_in(os.path.join(DATA, split, c)):
            if not (originals_only and "_aug_" in p):
                paths.append(p)
                labels.append(i)
    return paths, np.array(labels)


def scores(logits, feats, bank, means, prec):
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    knn = -np.sort(1 - f @ bank.T, axis=1)[:, KNN_K - 1]
    d = feats[:, None, :] - means[None]
    maha = -np.einsum("nkd,de,nke->nk", d, prec, d).min(1)
    return {"MSP": probs.max(1), "MaxLogit": logits.max(1),
            "Energy": logits.max(1) + np.log(np.exp(z).sum(1)), "kNN": knn, "Mahalanobis": maha}


def roc_curve(y, s):
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]], np.r_[np.inf, s[last]]


def roc_auc_score(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.trapezoid(tpr, fpr))


def fpr_at_tpr(y, s, tpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, tpr_target)])


def main():
    os.makedirs(OUT, exist_ok=True)
    ex = Extractor(WEIGHTS)
    names = ex.names

    tr_p, tr_y = id_split("train", names)
    tr_logits, tr_f = ex(tr_p)
    bank = tr_f / np.linalg.norm(tr_f, axis=1, keepdims=True)
    means = np.stack([tr_f[tr_y == k].mean(0) for k in range(len(names))])
    centered = tr_f - means[tr_y]
    cov = centered.T @ centered / len(tr_f)
    cov = 0.5 * cov + 0.5 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    prec = np.linalg.inv(cov)

    sets = {}
    for split in ("val", "test"):
        p, y = id_split(split, names)
        lg, ft = ex(p)
        sets["ID_" + split] = dict(paths=p, y=y, logits=lg, s=scores(lg, ft, bank, means, prec))
    rng = random.Random(0)
    ood = {kind: files_in(os.path.join(OOD, kind)) for kind in ("near", "far", "far_raw")}
    ood["far"] += files_in(UNKNOWN)
    recs = sorted({record(x) for paths in ood.values() for x in paths})
    rng.shuffle(recs)
    tune = set(recs[:len(recs) // 2])
    for kind, p in ood.items():
        for part, sel in (("tune", [x for x in p if record(x) in tune]), ("eval", [x for x in p if record(x) not in tune])):
            lg, ft = ex(sel)
            sets[f"{kind}_{part}"] = dict(paths=sel, logits=lg, s=scores(lg, ft, bank, means, prec))

    methods = list(sets["ID_val"]["s"])
    rows, thresholds = [], {}
    id_test = sets["ID_test"]
    id_ok = id_test["logits"].argmax(1) == id_test["y"]
    for m in methods:
        tune_ood = np.concatenate([sets[f"{k}_tune"]["s"][m] for k in ("near", "far")])
        y_tune = np.r_[np.ones(len(sets["ID_val"]["s"][m])), np.zeros(len(tune_ood))]
        fpr, tpr, thr = roc_curve(y_tune, np.r_[sets["ID_val"]["s"][m], tune_ood])
        rules = {"TPR95": float(np.quantile(sets["ID_val"]["s"][m], 1 - TARGET_TPR)),
                 "Youden": float(thr[np.argmax(tpr - fpr)])}
        thresholds[m] = rules
        for kind in ("near", "far", "far_raw", "near+far"):
            ood_s = np.concatenate([sets[f"{k}_eval"]["s"][m] for k in (["near", "far"] if kind == "near+far" else [kind])])
            ys = np.r_[np.ones(len(id_test["y"])), np.zeros(len(ood_s))]
            ss = np.r_[id_test["s"][m], ood_s]
            row = {"method": m, "ood": kind, "n_ood": len(ood_s), "AUROC": roc_auc_score(ys, ss),
                   "FPR@95TPR": fpr_at_tpr(ys, ss, 0.95)}
            for rule, t in rules.items():
                acc_mask = id_test["s"][m] >= t
                row.update({f"{rule}_thr": t, f"{rule}_ID_accepted": acc_mask.mean(),
                            f"{rule}_OOD_rejected": (ood_s < t).mean(), f"{rule}_acc_accepted": id_ok[acc_mask].mean()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(OUT, "ood_methods.csv"), index=False)

    msp_rows = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        r = {"MSP_threshold": thr, "ID_test_accepted": (id_test["s"]["MSP"] >= thr).mean(),
             "acc_of_accepted": id_ok[id_test["s"]["MSP"] >= thr].mean()}
        for kind in ("near", "far", "far_raw"):
            s = np.concatenate([sets[f"{kind}_{p}"]["s"]["MSP"] for p in ("tune", "eval")])
            r[f"{kind}_rejected"] = (s < thr).mean()
        msp_rows.append(r)
    pd.DataFrame(msp_rows).round(3).to_csv(os.path.join(OUT, "msp_thresholds.csv"), index=False)

    per_image = []
    for name, d in sets.items():
        for i, p in enumerate(d["paths"]):
            per_image.append({"set": name, "path": os.path.relpath(p, ROOT), "pred": names[int(d["logits"][i].argmax())],
                              **{m: float(d["s"][m][i]) for m in methods}})
    pd.DataFrame(per_image).to_csv(os.path.join(OUT, "per_image_scores.csv"), index=False)
    np.savez(os.path.join(OUT, "scores.npz"), **{f"{k}__{m}": d["s"][m] for k, d in sets.items() for m in methods})
    with open(os.path.join(OUT, "thresholds.json"), "w") as fh:
        json.dump({"weights": os.path.relpath(WEIGHTS, ROOT), "target_tpr_on_id_val": TARGET_TPR, "thresholds": thresholds}, fh, indent=2)
    print(df.round(3).to_string())
    print(pd.DataFrame(msp_rows).round(3).to_string())


if __name__ == "__main__":
    main()
