"""
Зведене порівняння YOLO11, YOLO26 і DINOv2 на v2r за ймовірностями з results_yolo/model_compare/*.npz
(exp8_yolo_ensemble_tta.py, exp9_dinov2_v2r.py, exp11_yolo26_v2r.py).

Метрики: val / test accuracy, test top-3, balanced accuracy (середнє recall по класах), test log loss,
спільний test (фото, що в test і v2r, і v2c - однаковий набір для всіх порівнянь), точність по класах.
Для методів з 3 сідами - середнє ± std; ансамблі - середнє ймовірностей.
Комбінований ансамбль YOLO + DINOv2 - середнє ймовірностей обох ансамблів.

Запуск:  python exp10_model_compare.py
Результати: results_yolo/model_compare/ (compare_summary.csv, compare_per_class.csv, compare_*.png)
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\projects\Diploma"
OUT = os.path.join(ROOT, r"CNN\results_yolo\model_compare")
V2C_TEST = os.path.join(ROOT, r"datasets\transfer_learning\transfer_learning_datasets_v2c_filtered_by_5_r512\test")
METHODS = [  # (назва в таблиці, файл або шаблон для сідів, група)
    ("YOLO11m, 1 model", "yolo_single_s?", "YOLO"),
    ("YOLO11m, 1 model + TTA", "yolo_tta_s?", "YOLO"),
    ("YOLO11m, ensemble of 3", "yolo_ensemble", "YOLO"),
    ("YOLO11m, ensemble of 3 + TTA", "yolo_ensemble_tta", "YOLO"),
    ("YOLO26m, 1 model", "yolo26_single_s?", "YOLO26"),
    ("YOLO26m, ensemble of 3", "yolo26_ensemble", "YOLO26"),
    ("DINOv2 ViT-S, frozen + linear", "dinov2_vits14_linear", "DINOv2"),
    ("DINOv2 ViT-B, frozen + linear", "dinov2_vitb14_linear", "DINOv2"),
    ("DINOv2 ViT-B, fine-tuned, 1 model", "dinov2_ft_s?", "DINOv2"),
    ("DINOv2 ViT-B, fine-tuned, 1 model + TTA", "dinov2_ft_tta_s?", "DINOv2"),
    ("DINOv2 ViT-B, fine-tuned, ensemble of 3", "dinov2_ft_ensemble", "DINOv2"),
    ("DINOv2 ViT-B, fine-tuned, ensemble of 3 + TTA", "dinov2_ft_ensemble_tta", "DINOv2"),
    ("YOLO ensemble + DINOv2 ensemble", ("yolo_ensemble", "dinov2_ft_ensemble"), "YOLO + DINOv2"),
]
GROUP_COLORS = {"YOLO": "#2a78d6", "YOLO26": "#1baa6b", "DINOv2": "#e0702a", "YOLO + DINOv2": "#7a4fc9"}
CURVES = {"YOLO11m (v2r, AdamW)": (os.path.join(ROOT, r"CNN\results_yolo\gbif_compare"), "v2r_adamw_s?", "YOLO"),
          "YOLO26m (v2r, AdamW)": (os.path.join(ROOT, r"CNN\results_yolo\gbif_compare"), "v2r_yolo26_adamw_s?", "YOLO26"),
          "DINOv2 ViT-B fine-tuned": (OUT, "dinov2_vitb14_ft_s?", "DINOv2")}


def load(name):
    z = np.load(os.path.join(OUT, f"{name}.npz"))
    return {k: z[k] for k in z.files}


def metrics(z, common, classes):
    out = {}
    for s in ("val", "test"):
        pred = z[f"{s}_p"].argmax(1)
        out[f"{s}_acc"] = float((pred == z[f"{s}_y"]).mean())
    p, y, files = z["test_p"], z["test_y"], z["test_files"]
    pred = p.argmax(1)
    out["test_top3"] = float((np.argsort(-p, 1)[:, :3] == y[:, None]).any(1).mean())
    recall = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(classes)}
    out["test_balanced_acc"] = float(np.mean(list(recall.values())))
    out["test_log_loss"] = float(-np.log(p[np.arange(len(y)), y] + 1e-8).mean())
    mask = np.isin(files, common)
    out["common_test_acc"] = float((pred[mask] == y[mask]).mean())
    return out, recall


if __name__ == "__main__":
    common = np.array([f"{c}/{os.path.basename(p)}" for c in os.listdir(V2C_TEST)
                       for p in glob.glob(os.path.join(V2C_TEST, c, "*.png")) if "_aug_" not in p])
    rows, per_class = [], {}
    for label, src, group in METHODS:
        if isinstance(src, tuple):
            parts = [load(n) for n in src]
            runs = [{**parts[0], "val_p": np.mean([q["val_p"] for q in parts], 0), "test_p": np.mean([q["test_p"] for q in parts], 0)}]
        else:
            runs = [load(os.path.basename(f)[:-4]) for f in sorted(glob.glob(os.path.join(OUT, f"{src}.npz")))]
        if not runs:
            print("skip", label)
            continue
        classes = list(runs[0]["classes"])
        ms = [metrics(r, common, classes) for r in runs]
        row = dict(method=label, group=group, n_models=len(runs))
        for k in ms[0][0]:
            vals = [m[0][k] for m in ms]
            row[k] = np.mean(vals)
            row[f"{k}_std"] = np.std(vals) if len(vals) > 1 else np.nan
        rows.append(row)
        per_class[label] = {c: np.mean([m[1][c] for m in ms]) for c in classes}
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT, "compare_summary.csv"), index=False)
    pc = pd.DataFrame(per_class).round(3)
    pc.to_csv(os.path.join(OUT, "compare_per_class.csv"))
    print(f"common test: {np.isin(runs[0]['test_files'], common).sum()} photos")
    cols = ["method", "n_models", "val_acc", "test_acc", "test_acc_std", "common_test_acc", "test_balanced_acc", "test_top3", "test_log_loss"]
    print(df[cols].round(3).to_string(index=False))
    print(pc.to_string())

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(df) + 1.5))
    yy = np.arange(len(df))[::-1]
    for metric, off, alpha in (("test_acc", 0.2, 1.0), ("common_test_acc", -0.2, 0.45)):
        ax.barh(yy + off, df[metric], 0.38, color=[GROUP_COLORS[g] for g in df["group"]], alpha=alpha,
                xerr=df[f"{metric}_std"].fillna(0), error_kw=dict(lw=1, ecolor="#444"))
        for y_, v in zip(yy + off, df[metric]):
            ax.text(v + 0.006, y_, f"{v:.3f}", va="center", fontsize=8, color="#333")
    ax.set_yticks(yy, df["method"]); ax.set_xlim(0.5, 0.92)
    ax.axvspan(0.80, 0.82, color="#999", alpha=0.15, lw=0)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("accuracy (solid = test v2r, faded = common test; whiskers = std over 3 seeds)")
    ax.set_title("YOLO vs DINOv2 on v2r")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_accuracy.png"), dpi=130); plt.close(fig)

    keys = [m for m in ("YOLO11m, 1 model", "YOLO11m, ensemble of 3", "YOLO26m, ensemble of 3", "DINOv2 ViT-B, frozen + linear",
                        "DINOv2 ViT-B, fine-tuned, 1 model", "DINOv2 ViT-B, fine-tuned, ensemble of 3",
                        "YOLO ensemble + DINOv2 ensemble") if m in pc]
    fig, ax = plt.subplots(figsize=(1.6 * len(keys) + 3, 5.5))
    im = ax.imshow(pc[keys].to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(pc)):
        for j, k in enumerate(keys):
            v = pc[k].iloc[i]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else "#222")
    ax.set_xticks(range(len(keys)), [k.replace(", ", "\n", 1) for k in keys], fontsize=8)
    ax.set_yticks(range(len(pc)), pc.index)
    ax.set_title("Per-class test recall (v2r)"); fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_per_class.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for label, (base, pattern, group) in CURVES.items():
        rs = []
        for f in sorted(glob.glob(os.path.join(base, pattern, "results.csv"))):
            r = pd.read_csv(f); r.columns = r.columns.str.strip(); rs.append(r)
        if not rs:
            continue
        for ax, col in zip(axes, ("train/loss", "val/loss", "metrics/accuracy_top1")):
            n = min(len(r) for r in rs); ep = np.arange(1, n + 1)
            mm = np.array([r[col].to_numpy()[:n] for r in rs])
            ax.fill_between(ep, mm.min(0), mm.max(0), color=GROUP_COLORS[group], alpha=0.15, lw=0)
            ax.plot(ep, mm.mean(0), color=GROUP_COLORS[group], lw=2, label=f"{label} ({len(rs)} seeds)")
    for ax, t in zip(axes, ("Train loss", "Val loss", "Val accuracy top-1")):
        ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    axes[2].axhspan(0.80, 0.82, color="#999", alpha=0.15, lw=0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_curves.png"), dpi=130); plt.close(fig)
