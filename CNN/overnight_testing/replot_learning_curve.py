"""
Перемальовує фірмовий графік навчання YOLO з РЕАЛЬНИМ learning rate,
який береться з results.csv (колонка lr/pg0), а не з аргументу lr0.
При optimizer='auto' Ultralytics ігнорує lr0 і сам обирає AdamW з lr = 0.002*5/(4+nc).

Запуск:  python replot_learning_curve.py <папка_прогону> [<папка_прогону> ...]
"""
import os
import sys
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def replot(run_dir, out_path=None):
    df = pd.read_csv(os.path.join(run_dir, "results.csv"))
    df.columns = df.columns.str.strip()
    with open(os.path.join(run_dir, "args.yaml"), encoding="utf-8") as f:
        a = yaml.safe_load(f)

    ep = df["epoch"]
    lr = df["lr/pg0"]
    peak_lr, final_lr = lr.max(), lr.iloc[-1]
    best_i = df["metrics/accuracy_top1"].idxmax()
    best_acc, best_ep = df.loc[best_i, "metrics/accuracy_top1"], int(df.loc[best_i, "epoch"])
    model = os.path.basename(str(a.get("model"))).replace(".pt", "")
    opt = a.get("optimizer")
    opt_txt = "auto -> AdamW (lr0 з аргументів ігнорується)" if opt == "auto" else opt

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    ax[0].plot(ep, df["metrics/accuracy_top1"], color="darkorange", label="Val Accuracy (Top-1)")
    ax[0].plot(ep, df["metrics/accuracy_top5"], color="tab:green", alpha=0.6, label="Val Accuracy (Top-5)")
    ax[0].scatter([best_ep], [best_acc], color="red", zorder=5, label=f"best = {best_acc:.3f} (epoch {best_ep})")
    ax[0].set_title("Validation Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()

    ax[1].plot(ep, df["train/loss"], color="tab:blue", label="Train Loss")
    ax[1].plot(ep, df["val/loss"], color="darkorange", label="Val Loss")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()

    ax[2].plot(ep, lr, color="tab:purple", label="фактичний LR (lr/pg0)")
    ax[2].set_title("Learning rate (реальний, з логу)"); ax[2].set_xlabel("epoch")
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); ax[2].legend()

    params = (
        f"MODEL: {model} | EXPERIMENT: {os.path.basename(os.path.normpath(run_dir))}\n"
        f"Img Size: {a.get('imgsz')} | Batch: {a.get('batch')} | Epochs: {a.get('epochs')} | Freeze: {a.get('freeze')} | "
        f"Optimizer: {opt_txt}\n"
        f"LR: warmup {a.get('warmup_epochs')} ep -> peak {peak_lr:.2e} -> linear decay to {final_lr:.2e} (lrf={a.get('lrf')}) | "
        f"Dropout: {a.get('dropout')} | Weight Decay: {a.get('weight_decay')} | Degrees: {a.get('degrees')}° | "
        f"Flip LR: {a.get('fliplr')} | Seed: {a.get('seed')}"
    )
    fig.text(0.5, 0.01, params, ha="center", fontsize=9,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"))
    plt.subplots_adjust(bottom=0.27, wspace=0.25)
    out_path = out_path or os.path.join(run_dir, "learning_curve_real_lr.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Збережено: {out_path}   (peak LR {peak_lr:.2e})")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        replot(d)
