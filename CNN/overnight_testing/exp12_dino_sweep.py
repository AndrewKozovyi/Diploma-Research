"""
Підбір параметрів донавчання DINOv2 / DINOv3 на v2r (продовження exp9_dinov2_v2r.py).

Фаза 1 (sweep) - кожна конфігурація з CONFIGS на сіді 0.
Фаза 2 (seeds) - 2 найкращі нові конфігурації ще на сідах 1 і 2.
Вибір тільки за val: середня val accuracy за останні 5 епох (стабільніша за одну найкращу епоху),
при рівності - мінімальна val loss. Test рахується і зберігається, але для вибору не використовується.
База - exp9 (dinov2_vitb14_ft_s0..2: DINOv2 ViT-B, 4 блоки, lr 2e-5, 224 px).

Запуск:  python exp12_dino_sweep.py bench          - швидкість і пам'ять кожної конфігурації
         python exp12_dino_sweep.py sweep
         python exp12_dino_sweep.py seeds
         python exp12_dino_sweep.py summary
         python exp12_dino_sweep.py confusion v2b_u8  - матриця помилок ансамблю 3 сідів (val, test)
Результати: results_yolo/model_compare/sweep/
"""
import os
import sys
import glob
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import exp9_dinov2_v2r as e9

OUT = os.path.join(e9.OUT, "sweep")
BASE = dict(src="v2", arch="dinov2_vitb14", unfrozen=4, lr_backbone=2e-5, llrd=1.0, res=224, epochs=20, batch=16,
            accum=1, lr_head=1e-3, weight_decay=0.05, label_smoothing=0.1, dropout=0.3)
V3 = "facebook/dinov3-{}-pretrain-lvd1689m"
CONFIGS = {
    "v2b_u8": dict(unfrozen=8),
    "v2b_u4_lr5e-5": dict(lr_backbone=5e-5),
    "v2b_u8_llrd": dict(unfrozen=8, lr_backbone=4e-5, llrd=0.8),
    "v2b_u4_r280": dict(res=280, batch=8, accum=2),
    "v3b_u4": dict(src="v3", arch=V3.format("vitb16")),
    "v3b_u8_llrd": dict(src="v3", arch=V3.format("vitb16"), unfrozen=8, lr_backbone=4e-5, llrd=0.8),
    "v3s_u8": dict(src="v3", arch=V3.format("vits16"), unfrozen=8, lr_backbone=4e-5, llrd=0.8),
    "v3cs_u9": dict(src="v3", arch=V3.format("convnext-small"), unfrozen=9, lr_backbone=5e-5, llrd=0.85),
}
BASELINE = [os.path.join(e9.OUT, f"dinov2_vitb14_ft_s{s}") for s in (0, 1, 2)]


def cfg(name):
    return {**BASE, **CONFIGS[name]}


class Backbone(nn.Module):
    """Спільний інтерфейс: features(x) -> CLS + середнє патчів; blocks - блоки знизу вгору; norm - фінальна нормалізація."""

    def __init__(self, c):
        super().__init__()
        self.src = c["src"]
        if self.src == "v2":
            self.net = e9.backbone(c["arch"])
            self.blocks, self.norm, self.dim, self.skip = list(self.net.blocks), self.net.norm, self.net.embed_dim, 0
        else:
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(c["arch"])
            if hasattr(self.net.model, "stages"):
                self.blocks = [b for s in self.net.model.stages for b in [s.downsample_layers, *s.layers]]
                self.norm, self.dim, self.skip = self.net.layer_norm, self.net.config.hidden_sizes[-1], 0
            else:
                self.blocks, self.norm = list(self.net.model.layer), self.net.norm
                self.dim, self.skip = self.net.config.hidden_size, self.net.config.num_register_tokens

    def features(self, x):
        if self.src == "v2":
            f = self.net.forward_features(x)
            return torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1)
        h = self.net(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1 + self.skip:].mean(1)], 1)


class Classifier(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.backbone = Backbone(c)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.trainable = self.backbone.blocks[-c["unfrozen"]:]
        for m in self.trainable + [self.backbone.norm]:
            for p in m.parameters():
                p.requires_grad = True
        self.head = nn.Sequential(nn.Dropout(c["dropout"]), nn.Linear(self.backbone.dim * 2, len(e9.CLASSES)))

    def forward(self, x):
        return self.head(self.backbone.features(x))

    def param_groups(self, c):
        groups = [{"params": self.head.parameters(), "lr": c["lr_head"]},
                  {"params": self.backbone.norm.parameters(), "lr": c["lr_backbone"]}]
        for depth, block in enumerate(reversed(self.trainable)):
            groups.append({"params": block.parameters(), "lr": c["lr_backbone"] * c["llrd"] ** depth})
        return groups


def transforms(res):
    train = T.Compose([T.RandomResizedCrop(res, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                       T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15, T.InterpolationMode.BILINEAR),
                       T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.ToTensor(), T.Normalize(e9.MEAN, e9.STD),
                       T.RandomErasing(p=0.4)])
    ev = T.Compose([T.Resize(res, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(res), T.ToTensor(),
                    T.Normalize(e9.MEAN, e9.STD)])
    return train, ev


def build(name):
    c = cfg(name)
    model = Classifier(c).to(e9.DEVICE)
    opt = torch.optim.AdamW(model.param_groups(c), weight_decay=c["weight_decay"])
    return c, model, opt


def bench():
    for name in CONFIGS:
        c, model, opt = build(name)
        train_tf, _ = transforms(c["res"])
        dl = e9.loader(e9.items("train", True), train_tf, shuffle=True, batch=c["batch"])
        torch.cuda.reset_peak_memory_stats()
        n, t0 = 0, None
        for i, (x, y) in enumerate(dl):
            if i == 3:
                torch.cuda.synchronize(); t0, n = time.time(), 0
            loss = F.cross_entropy(model(x.to(e9.DEVICE)), y.to(e9.DEVICE))
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            n += len(y)
            if i == 23:
                break
        torch.cuda.synchronize()
        speed = n / (time.time() - t0)
        print(f"{name}: {speed:.1f} img/s, ~{len(dl.dataset) / speed / 60:.1f} min/epoch, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        del model, opt, dl
        torch.cuda.empty_cache()


def run(name, seed):
    run_dir = os.path.join(OUT, f"{name}_s{seed}")
    if os.path.exists(os.path.join(run_dir, "DONE")):
        return
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    c, model, opt = build(name)
    train_tf, eval_tf = transforms(c["res"])
    splits = {"train": e9.items("train", True), "val": e9.items("val"), "test": e9.items("test")}
    train_dl = e9.loader(splits["train"], train_tf, shuffle=True, batch=c["batch"])
    eval_dl = {s: e9.loader(splits[s], eval_tf) for s in ("val", "test")}
    steps_per_epoch = len(train_dl) // c["accum"]
    steps, warmup = c["epochs"] * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warmup if i < warmup else 0.5 * (1 + math.cos(math.pi * (i - warmup) / (steps - warmup))))
    y_val = torch.tensor(e9.meta(splits["val"])["y"])
    rows, best = [], (-1, 0)
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(e9.DEVICE, non_blocking=True), y.to(e9.DEVICE, non_blocking=True)
            loss = F.cross_entropy(model(x), y, label_smoothing=c["label_smoothing"])
            (loss / c["accum"]).backward()
            if (i + 1) % c["accum"] == 0:
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            total += loss.item() * len(y); n += len(y)
        pv = e9.evaluate(model, eval_dl["val"])
        val_loss = float(F.nll_loss(torch.log(torch.tensor(pv) + 1e-8), y_val))
        acc = float((pv.argmax(1) == y_val.numpy()).mean())
        rows.append({"epoch": epoch, "train/loss": total / n, "val/loss": val_loss, "metrics/accuracy_top1": acc})
        print(f"{name}_s{seed} epoch {epoch}: train {total / n:.3f} | val loss {val_loss:.3f} | val acc {acc:.3f}", flush=True)
        if (acc, -val_loss) > best:
            best = (acc, -val_loss)
            torch.save(model.state_dict(), os.path.join(run_dir, "weights", "best.pt"))
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "results.csv"), index=False)
    e9.plot_curve(pd.DataFrame(rows), os.path.join(run_dir, "learning_curve.png"), f"{name}_s{seed}")
    model.load_state_dict(torch.load(os.path.join(run_dir, "weights", "best.pt")))
    np.savez(os.path.join(run_dir, "probs.npz"), classes=np.array(e9.CLASSES),
             **{f"{s}_{k}": v for s in ("val", "test")
                for k, v in dict(p=e9.evaluate(model, eval_dl[s]), **e9.meta(splits[s])).items()})
    open(os.path.join(run_dir, "DONE"), "w").close()
    del model, opt
    torch.cuda.empty_cache()


def curve_stats(run_dir):
    r = pd.read_csv(os.path.join(run_dir, "results.csv"))
    acc, loss = r["metrics/accuracy_top1"], r["val/loss"]
    return dict(val_acc_best=acc.max(), val_acc_last5=acc.tail(5).mean(), val_loss_min=loss.min(),
                val_loss_last=loss.iloc[-1], val_loss_rise=loss.iloc[-1] - loss.min())


def test_acc(run_dir):
    f = os.path.join(run_dir, "probs.npz")
    if not os.path.exists(f):
        z = np.load(os.path.join(e9.OUT, f"dinov2_ft_s{run_dir[-1]}.npz"))
    else:
        z = np.load(f)
    return float((z["test_p"].argmax(1) == z["test_y"]).mean())


def summary():
    rows = []
    for name, dirs in [("baseline_v2b_u4", BASELINE)] + [(n, sorted(glob.glob(os.path.join(OUT, f"{n}_s?")))) for n in CONFIGS]:
        dirs = [d for d in dirs if os.path.exists(os.path.join(d, "DONE"))]
        for d in dirs:
            rows.append(dict(config=name, seed=int(d[-1]), **curve_stats(d), test_acc=test_acc(d)))
    df = pd.DataFrame(rows)
    s0 = df[df.seed == 0].sort_values(["val_acc_last5", "val_loss_min"], ascending=[False, True])
    agg = df.groupby("config").agg(n_seeds=("seed", "count"), **{f"{k}": (k, "mean") for k in df.columns[2:]},
                                   test_acc_std=("test_acc", "std")).sort_values("val_acc_last5", ascending=False)
    s0.round(4).to_csv(os.path.join(OUT, "sweep_seed0.csv"), index=False)
    agg.round(4).to_csv(os.path.join(OUT, "sweep_summary.csv"))
    print("seed 0 (selection by val_acc_last5):\n", s0.round(3).to_string(index=False))
    print("\nall seeds:\n", agg.round(3).to_string())
    return s0


def confusion(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = [np.load(os.path.join(OUT, f"{name}_s{s}", "probs.npz")) for s in (0, 1, 2)]
    k = len(e9.CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    for ax, s in zip(axes, ("val", "test")):
        y, pred = zs[0][f"{s}_y"], np.mean([z[f"{s}_p"] for z in zs], 0).argmax(1)
        cm = np.zeros((k, k), int)
        np.add.at(cm, (y, pred), 1)
        rate = cm / cm.sum(1, keepdims=True)
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                if cm[i, j]:
                    ax.text(j, i, f"{cm[i, j]}\n{rate[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if rate[i, j] > 0.6 else "#222")
        ax.set_xticks(range(k), e9.CLASSES, rotation=45, ha="right"); ax.set_yticks(range(k), e9.CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{s}: accuracy {(pred == y).mean():.3f}, n = {len(y)}")
        pd.DataFrame(cm, index=e9.CLASSES, columns=e9.CLASSES).to_csv(os.path.join(OUT, f"{name}_confusion_{s}.csv"))
    fig.suptitle(f"{name}: ensemble of 3 seeds (count / share of true class)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}_confusion.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1]
    if mode == "bench":
        bench()
    elif mode == "sweep":
        for name in [a for a in sys.argv[2:]] or CONFIGS:
            run(name, 0)
        summary()
    elif mode == "seeds":
        top = [n for n in summary()["config"] if n in CONFIGS][:2]
        print("top-2:", top, flush=True)
        for name in top:
            for seed in (1, 2):
                run(name, seed)
        summary()
    elif mode == "confusion":
        confusion(sys.argv[2])
    else:
        summary()
