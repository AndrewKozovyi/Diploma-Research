"""
Підбір параметрів на очищеному v2 (transfer_learning_datasets_v2c_filtered_by_5_r512).

Етап 1 (screen): кожна конфігурація по 1 сіду (seed 0), порівняння з базовою (фінальна конфігурація).
Етап 2 (confirm): найкращі конфігурації по val - ще сіди 1 і 2.
Вибір лише за val (test не дивимось до кінця) і за формою кривих:
  точність - рівень останніх 5 епох, падіння від піку, плавність (середній |Δ| між епохами);
  val loss - мінімум, рівень останніх 5 епох, нахил у другій половині (має йти вниз або стояти), стрибки.
Еталон - фінальна конфігурація на тих самих даних (v2exp_clean_f7_m_img320_sa*, 3 сіди).

Запуск:  python exp7_hparam_sweep_clean.py screen
         python exp7_hparam_sweep_clean.py confirm <назва> [<назва> ...]
         python exp7_hparam_sweep_clean.py summary
Результати: results_yolo/sweep_clean/ (прогони, sweep_runs.csv, sweep_summary.csv)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import transfer_learning_yolo as tly

ROOT = r"D:\projects\Diploma"
DATA = os.path.join(ROOT, r"datasets\transfer_learning\transfer_learning_datasets_v2c_filtered_by_5_r512")
OUT = os.path.join(ROOT, r"CNN\results_yolo\sweep_clean")
REFERENCE = "v2exp_clean_f7_m_img320_sa"
BASE = dict(model="yolo11m-cls.pt", epochs=30, imgsz=320, batch=16, freeze=7, patience=15, optimizer="auto", lr0=1e-5,
            dropout=0.8, weight_decay=0.005, degrees=15, fliplr=0.5, translate=0.1, scale=0.7, flipud=0.5,
            erasing=0.6, deterministic=True, device=0, workers=4)
CONFIGS = {
    "adamw_lr2e-4_cos": dict(optimizer="AdamW", lr0=2e-4, cos_lr=True, epochs=40),
    "adamw_lr2e-4_cos_wd0.05": dict(optimizer="AdamW", lr0=2e-4, cos_lr=True, epochs=40, weight_decay=0.05),
    "freeze9": dict(freeze=9),
    "img384": dict(imgsz=384),
    "yolo11s": dict(model="yolo11s-cls.pt"),
    "augmix": dict(auto_augment="augmix"),
    "strong_color": dict(hsv_h=0.05, hsv_s=0.9, hsv_v=0.5),
}


def train(name, seed):
    run = f"{name}_s{seed}"
    d = os.path.join(OUT, run)
    if os.path.exists(os.path.join(d, "DONE")):
        return d
    args = {**BASE, **CONFIGS[name]}
    model = args.pop("model")
    results = YOLO(os.path.join(ROOT, "CNN", model)).train(data=DATA, project=OUT, name=run, exist_ok=True, seed=seed, **args)
    tly.EXPERIMENT_NAME = run
    tly.MODEL_WEIGHTS, tly.IMG_SIZE, tly.FREEZE_LAYERS, tly.EPOCHS = model, args["imgsz"], args["freeze"], args["epochs"]
    tly.OPTIMIZER, tly.LR_INIT, tly.WEIGHT_DECAY = args["optimizer"], args["lr0"], args["weight_decay"]
    tly.plot_custom_results(results.save_dir)
    open(os.path.join(d, "DONE"), "w").close()
    return d


def curve_stats(r):
    acc, loss = r["metrics/accuracy_top1"], r["val/loss"]
    half = loss.iloc[len(loss) // 2:]
    return dict(best_val=acc.max(), best_val_epoch=int(acc.idxmax()) + 1,
                val_last5=acc.iloc[-5:].mean(),                         # рівень, на якому крива тримається в кінці
                val_drop=acc.max() - acc.iloc[-5:].mean(),               # наскільки впала від піку
                val_jumps=acc.diff().abs().iloc[4:].mean(),              # плавність кривої точності
                min_val_loss=loss.min(), val_loss_last5=loss.iloc[-5:].mean(),
                val_loss_trend=np.polyfit(np.arange(len(half)), half, 1)[0],  # нахил у другій половині (< 0 - іде вниз)
                val_loss_jumps=loss.diff().abs().iloc[4:].mean(),
                last_train_loss=r["train/loss"].iloc[-1])


def plot_curves(runs):
    import matplotlib.pyplot as plt
    groups = {}
    for run, d in runs:
        r = pd.read_csv(os.path.join(d, "results.csv")); r.columns = r.columns.str.strip()
        groups.setdefault(run.rsplit("_s", 1)[0], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for i, (name, rs) in enumerate(sorted(groups.items())):
        ref = name == "final_config"
        style = dict(color="black", lw=2.6) if ref else dict(color=plt.cm.tab10(i % 10), lw=1.5)
        for ax, col in zip(axes, ("metrics/accuracy_top1", "val/loss")):
            n = min(len(r) for r in rs)
            ax.plot(np.arange(1, n + 1), np.mean([r[col].to_numpy()[:n] for r in rs], 0),
                    label=f"{name} ({len(rs)} seed{'s' if len(rs) > 1 else ''})", **style)
    axes[0].axhspan(0.80, 0.82, color="#999", alpha=0.15, lw=0)
    axes[0].set_title("Val accuracy top-1"); axes[1].set_title("Val loss")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sweep_curves.png"), dpi=120); plt.close(fig)


def summarize():
    rows = []
    runs = [(run, os.path.join(OUT, run)) for run in sorted(os.listdir(OUT)) if os.path.exists(os.path.join(OUT, run, "DONE"))]
    runs += [(f"final_config_s{i}", os.path.join(ROOT, r"CNN\results_yolo", REFERENCE + suf)) for i, suf in enumerate(("", "-2", "-3"))]
    for run, d in runs:
        r = pd.read_csv(os.path.join(d, "results.csv")); r.columns = r.columns.str.strip()
        name, seed = run.rsplit("_s", 1)
        rows.append(dict(config=name, seed=int(seed), **curve_stats(r)))
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT, "sweep_runs.csv"), index=False)
    s = df.drop(columns="seed").groupby("config").agg(["mean", "std", "count"]).round(4)
    s = s.sort_values(("val_last5", "mean"), ascending=False)
    s.to_csv(os.path.join(OUT, "sweep_summary.csv"))
    plot_curves(runs)
    cols = ["best_val", "val_last5", "val_drop", "val_jumps", "min_val_loss", "val_loss_last5", "val_loss_trend", "val_loss_jumps"]
    print(s[[(c, "mean") for c in cols] + [("best_val", "count")]].round(3).to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "screen":
        for name in CONFIGS:
            train(name, 0)
            summarize()
    elif sys.argv[1] == "confirm":
        for name in sys.argv[2:]:
            for seed in (1, 2):
                train(name, seed)
                summarize()
    summarize()
