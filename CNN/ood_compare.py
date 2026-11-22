"""
OOD для моделей-кандидатів на v2r: YOLO11m, YOLO26m, DINOv2 ViT-B донавчений (4 блоки - exp9, 8 блоків - exp12) -
по 3 сіди і ансамблі.
Методи, OOD-набори, поділ OOD на tune/eval і пороги - як в ood_eval.py.
ID - val і test v2r (оригінали), банк для kNN / Mahalanobis - train v2r (з аугментованими копіями).
Ансамбль: MSP з середніх ймовірностей, решта оцінок - середнє оцінок членів.

Запуск:  python ood_compare.py
Результати: results_yolo/ood_compare/ (ood_compare_runs.csv, ood_compare_summary.csv, ood_compare_auroc.png)
"""
import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ood_eval as oe
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "overnight_testing"))
import exp9_dinov2_v2r as e9
import exp12_dino_sweep as e12

ROOT = oe.ROOT
DATA = e9.DATA
OUT = os.path.join(ROOT, r"CNN\results_yolo\ood_compare")
CLASSES = e9.CLASSES
MODELS = {
    "YOLO11m": [os.path.join(ROOT, rf"CNN\results_yolo\gbif_compare\v2r_adamw_s{s}\weights\best.pt") for s in (0, 1, 2)],
    "YOLO26m": [os.path.join(ROOT, rf"CNN\results_yolo\gbif_compare\v2r_yolo26_adamw_s{s}\weights\best.pt") for s in (0, 1, 2)],
    "DINOv2 ViT-B ft": [os.path.join(e9.OUT, rf"dinov2_vitb14_ft_s{s}\weights\best.pt") for s in (0, 1, 2)],
    "DINOv2 ViT-B ft, 8 blocks": [os.path.join(e12.OUT, rf"v2b_u8_s{s}\weights\best.pt") for s in (0, 1, 2)],
}


class YoloModel(oe.Extractor):
    def __init__(self, weights):
        super().__init__(weights)
        self.order = [self.names.index(c) for c in CLASSES]

    def __call__(self, paths, batch=32):
        logits, feats = super().__call__(paths, batch)
        return logits[:, self.order], feats


class DinoModel:
    def __init__(self, weights, sweep_config=None):
        if sweep_config:
            self.net = e12.Classifier(e12.cfg(sweep_config))
            self.tf, self.feat = e12.transforms(e12.cfg(sweep_config)["res"])[1], self.net.backbone.features
        else:
            self.net = e9.Classifier()
            self.tf, self.feat = e9.EVAL_TF, self.v2_features
        self.net.to(e9.DEVICE).eval().load_state_dict(torch.load(weights, map_location=e9.DEVICE))

    def v2_features(self, x):
        f = self.net.backbone.forward_features(x)
        return torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1)

    @torch.no_grad()
    def __call__(self, paths, batch=32):
        logits, feats = [], []
        for i in range(0, len(paths), batch):
            x = torch.stack([self.tf(oe.Image.open(p).convert("RGB")) for p in paths[i:i + batch]]).to(e9.DEVICE)
            f = self.feat(x)
            logits.append(self.net.head(f).cpu())
            feats.append(f.cpu())
        return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def id_split(split, with_aug=False):
    items = e9.items(split, with_aug)
    return [p for p, _ in items], np.array([y for _, y in items])


def softmax(z):
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def model_scores(model, sets):
    tr_p, tr_y = id_split("train", True)
    _, tr_f = model(tr_p)
    bank = tr_f / np.linalg.norm(tr_f, axis=1, keepdims=True)
    means = np.stack([tr_f[tr_y == k].mean(0) for k in range(len(CLASSES))])
    centered = tr_f - means[tr_y]
    cov = centered.T @ centered / len(tr_f)
    cov = 0.5 * cov + 0.5 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    prec = np.linalg.inv(cov)
    out = {}
    for name, paths in sets.items():
        lg, ft = model(paths)
        out[name] = dict(probs=softmax(lg), s=oe.scores(lg, ft, bank, means, prec))
    return out


def evaluate(res, y_test):
    rows = []
    id_ok = res["ID_test"]["probs"].argmax(1) == y_test
    for m in res["ID_val"]["s"]:
        tune_ood = np.concatenate([res[f"{k}_tune"]["s"][m] for k in ("near", "far")])
        y_tune = np.r_[np.ones(len(res["ID_val"]["s"][m])), np.zeros(len(tune_ood))]
        fpr, tpr, thr = oe.roc_curve(y_tune, np.r_[res["ID_val"]["s"][m], tune_ood])
        rules = {"TPR95": float(np.quantile(res["ID_val"]["s"][m], 1 - oe.TARGET_TPR)), "Youden": float(thr[np.argmax(tpr - fpr)])}
        for kind in ("near", "far", "far_raw", "near+far"):
            ood_s = np.concatenate([res[f"{k}_eval"]["s"][m] for k in (["near", "far"] if kind == "near+far" else [kind])])
            ys, ss = np.r_[np.ones(len(y_test)), np.zeros(len(ood_s))], np.r_[res["ID_test"]["s"][m], ood_s]
            row = {"method": m, "ood": kind, "n_ood": len(ood_s), "AUROC": oe.roc_auc_score(ys, ss),
                   "FPR@95TPR": oe.fpr_at_tpr(ys, ss, 0.95), "ID_test_acc": id_ok.mean()}
            for rule, t in rules.items():
                keep = res["ID_test"]["s"][m] >= t
                row.update({f"{rule}_ID_accepted": keep.mean(), f"{rule}_OOD_rejected": (ood_s < t).mean(),
                            f"{rule}_acc_accepted": id_ok[keep].mean() if keep.any() else np.nan})
            rows.append(row)
    return rows


def ensemble(members):
    out = {}
    for name in members[0]:
        probs = np.mean([m[name]["probs"] for m in members], 0)
        s = {k: np.mean([m[name]["s"][k] for m in members], 0) for k in members[0][name]["s"]}
        s["MSP"] = probs.max(1)
        out[name] = dict(probs=probs, s=s)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sets = {}
    for split in ("val", "test"):
        sets["ID_" + split], y = id_split(split)
        if split == "test":
            y_test = y
    ood = {kind: oe.files_in(os.path.join(oe.OOD, kind)) for kind in ("near", "far", "far_raw")}
    ood["far"] += oe.files_in(oe.UNKNOWN)
    recs = sorted({oe.record(x) for paths in ood.values() for x in paths})
    random.Random(0).shuffle(recs)
    tune = set(recs[:len(recs) // 2])
    for kind, p in ood.items():
        sets[f"{kind}_tune"] = [x for x in p if oe.record(x) in tune]
        sets[f"{kind}_eval"] = [x for x in p if oe.record(x) not in tune]
    print({k: len(v) for k, v in sets.items()}, flush=True)

    rows = []
    for family, weights in MODELS.items():
        members = []
        for seed, w in enumerate(weights):
            if not os.path.exists(w):
                print("skip", w)
                continue
            model = (DinoModel(w, "v2b_u8") if "8 blocks" in family else DinoModel(w) if "DINOv2" in family
                     else YoloModel(w))
            res = model_scores(model, sets)
            members.append(res)
            rows += [dict(model=family, variant="1 model", seed=seed, **r) for r in evaluate(res, y_test)]
            print(family, seed, "done", flush=True)
            del model
            torch.cuda.empty_cache()
        if len(members) > 1:
            rows += [dict(model=family, variant=f"ensemble of {len(members)}", seed=-1, **r)
                     for r in evaluate(ensemble(members), y_test)]
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT, "ood_compare_runs.csv"), index=False)
    s = (df.drop(columns=["seed", "n_ood"]).groupby(["model", "variant", "method", "ood"], sort=False).mean().round(3).reset_index())
    s.to_csv(os.path.join(OUT, "ood_compare_summary.csv"), index=False)
    view = s[s["ood"].isin(["near", "far", "far_raw"])].pivot_table(index=["model", "variant", "method"], columns="ood",
                                                                    values="AUROC", sort=False)
    print(view.to_string())
    yd = s[s["ood"] == "near+far"][["model", "variant", "method", "Youden_ID_accepted", "Youden_OOD_rejected", "Youden_acc_accepted"]]
    print(yd.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 0.45 * len(view) / 1 + 2), sharey=True)
    labels = [f"{a} | {b} | {c}" for a, b, c in view.index]
    for ax, kind in zip(axes, ("near", "far", "far_raw")):
        v = view[kind].to_numpy()[:, None]
        ax.imshow(v, cmap="Blues", vmin=0.5, vmax=1, aspect="auto")
        for i, x in enumerate(v[:, 0]):
            ax.text(0, i, f"{x:.3f}", ha="center", va="center", fontsize=8, color="white" if x > 0.8 else "#222")
        ax.set_xticks([0], [f"AUROC: {kind}"]); ax.set_yticks(range(len(labels)), labels, fontsize=8)
    fig.suptitle("OOD detection, AUROC (ID = v2r test)"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ood_compare_auroc.png"), dpi=120); plt.close(fig)
