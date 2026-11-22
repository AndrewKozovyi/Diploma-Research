"""
DINOv2 на v2r (transfer_learning_datasets_v2r_filtered_by_5_r512) - порівняння з YOLO.

linear - заморожений backbone (ViT-S/14, ViT-B/14), ознаки CLS + середнє патчів, логістична регресія;
         сила регуляризації за val cross-entropy.
ft     - часткове донавчання ViT-B/14: розморожені останні UNFROZEN блоків + нова голова, AdamW, cosine, fp32
         (AMP на GTX 1650 для ViT у 4.5 раза повільніший);
         аугментації як у YOLO (повороти, віддзеркалення, масштаб, erasing). Найкраща епоха за val accuracy
         (як best.pt в ultralytics). 3 сіди, крива навчання в results.csv / learning_curve.png.
Ймовірності -> results_yolo/model_compare/<метод>.npz (exp10_model_compare.py).

Запуск:  python exp9_dinov2_v2r.py linear
         python exp9_dinov2_v2r.py ft [сід ...]
"""
import os
import sys
import glob
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

ROOT = r"D:\projects\Diploma"
DATA = os.path.join(ROOT, r"datasets\transfer_learning\transfer_learning_datasets_v2r_filtered_by_5_r512")
OUT = os.path.join(ROOT, r"CNN\results_yolo\model_compare")
CLASSES = sorted(os.listdir(os.path.join(DATA, "train")))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMGSZ = 224
FT = dict(arch="dinov2_vitb14", unfrozen=4, epochs=20, batch=16, lr_head=1e-3, lr_backbone=2e-5, weight_decay=0.05,
          label_smoothing=0.1, dropout=0.3)
DEVICE = "cuda"


def items(split, with_aug=False):
    return [(p, i) for i, c in enumerate(CLASSES) for p in sorted(glob.glob(os.path.join(DATA, split, c, "*.png")))
            if with_aug or "_aug_" not in p]


def meta(split_items):
    return dict(y=np.array([y for _, y in split_items]),
                files=np.array([f"{CLASSES[y]}/{os.path.basename(p)}" for p, y in split_items]))


def save(name, data):
    np.savez(os.path.join(OUT, f"{name}.npz"), classes=np.array(CLASSES),
             **{f"{s}_{k}": v for s, d in data.items() for k, v in d.items()})


def backbone(arch):
    return torch.hub.load("facebookresearch/dinov2", arch, verbose=False)


class Images(torch.utils.data.Dataset):
    def __init__(self, split_items, tf):
        self.items, self.tf = split_items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


EVAL_TF = T.Compose([T.Resize(IMGSZ, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(IMGSZ), T.ToTensor(),
                     T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([T.RandomResizedCrop(IMGSZ, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15, T.InterpolationMode.BILINEAR),
                      T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.ToTensor(), T.Normalize(MEAN, STD),
                      T.RandomErasing(p=0.4)])


def loader(split_items, tf, shuffle=False, batch=32):
    return torch.utils.data.DataLoader(Images(split_items, tf), batch_size=batch, shuffle=shuffle, num_workers=6,
                                       pin_memory=True, drop_last=shuffle, persistent_workers=True)


def features(model, dl):
    out = []
    with torch.no_grad():
        for x, _ in dl:
            f = model.forward_features(x.to(DEVICE))
            out.append(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1).float().cpu())
    return torch.cat(out)


def logreg(X, y, wd, steps=500):
    lin = nn.Linear(X.shape[1], len(CLASSES))
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(X), y) + wd * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lin


def linear_probe():
    splits = {"train": items("train", True), "val": items("val"), "test": items("test")}
    for arch in ("dinov2_vits14", "dinov2_vitb14"):
        model = backbone(arch).to(DEVICE).eval()
        X = {s: features(model, loader(it, EVAL_TF)) for s, it in splits.items()}
        Y = {s: torch.tensor([y for _, y in it]) for s, it in splits.items()}
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
        X = {s: (x - mu) / sd for s, x in X.items()}
        best = None
        for wd in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
            lin = logreg(X["train"], Y["train"], wd)
            with torch.no_grad():
                lv = lin(X["val"])
            ce, acc = float(F.cross_entropy(lv, Y["val"])), float((lv.argmax(1) == Y["val"]).float().mean())
            print(f"{arch} wd {wd:g}: val acc {acc:.3f}, val CE {ce:.3f}", flush=True)
            if best is None or ce < best[0]:
                best = (ce, lin)
        with torch.no_grad():
            p = {s: torch.softmax(best[1](X[s]), 1).numpy() for s in ("val", "test")}
        save(f"{arch}_linear", {s: dict(p=p[s], **meta(splits[s])) for s in ("val", "test")})
        print(arch, {s: round(float((p[s].argmax(1) == Y[s].numpy()).mean()), 3) for s in p}, flush=True)
        del model
        torch.cuda.empty_cache()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone(FT["arch"])
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.blocks[-FT["unfrozen"]:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        dim = self.backbone.embed_dim * 2
        self.head = nn.Sequential(nn.Dropout(FT["dropout"]), nn.Linear(dim, len(CLASSES)))

    def forward(self, x):
        f = self.backbone.forward_features(x)
        return self.head(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1))


def evaluate(model, dl, flips=False):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(DEVICE)
            views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if flips else [x]
            out.append(torch.stack([torch.softmax(model(v).float(), 1) for v in views]).mean(0).cpu())
    return torch.cat(out).numpy()


def plot_curve(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ep = res["epoch"]
    axes[0].plot(ep, res["train/loss"], label="train loss", color="#2a78d6", lw=2)
    axes[0].plot(ep, res["val/loss"], label="val loss", color="#e0702a", lw=2)
    axes[1].plot(ep, res["metrics/accuracy_top1"], label="val accuracy top-1", color="#1baa6b", lw=2)
    axes[1].axhspan(0.80, 0.82, color="#999", alpha=0.15, lw=0)
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25); ax.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fine_tune(seed):
    name = f"dinov2_vitb14_ft_s{seed}"
    run_dir = os.path.join(OUT, name)
    if os.path.exists(os.path.join(run_dir, "DONE")):
        return
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    splits = {"train": items("train", True), "val": items("val"), "test": items("test")}
    train_dl = loader(splits["train"], TRAIN_TF, shuffle=True, batch=FT["batch"])
    eval_dl = {s: loader(splits[s], EVAL_TF) for s in ("val", "test")}
    model = Classifier().to(DEVICE)
    params = [{"params": model.head.parameters(), "lr": FT["lr_head"]},
              {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": FT["lr_backbone"]}]
    opt = torch.optim.AdamW(params, weight_decay=FT["weight_decay"])
    steps, warmup = FT["epochs"] * len(train_dl), len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warmup if i < warmup else 0.5 * (1 + math.cos(math.pi * (i - warmup) / (steps - warmup))))
    y_val = torch.tensor(meta(splits["val"])["y"])
    rows, best = [], (-1, 0)
    for epoch in range(1, FT["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            loss = F.cross_entropy(model(x), y, label_smoothing=FT["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            total += loss.item() * len(y); n += len(y)
        pv = evaluate(model, eval_dl["val"])
        val_loss = float(F.nll_loss(torch.log(torch.tensor(pv) + 1e-8), y_val))
        acc = float((pv.argmax(1) == y_val.numpy()).mean())
        rows.append({"epoch": epoch, "train/loss": total / n, "val/loss": val_loss, "metrics/accuracy_top1": acc})
        print(f"{name} epoch {epoch}: train {total / n:.3f} | val loss {val_loss:.3f} | val acc {acc:.3f}", flush=True)
        if (acc, -val_loss) > best:
            best = (acc, -val_loss)
            torch.save(model.state_dict(), os.path.join(run_dir, "weights", "best.pt"))
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "results.csv"), index=False)
    res = pd.DataFrame(rows)
    plot_curve(res, os.path.join(run_dir, "learning_curve.png"), f"{name}: ViT-B/14, last {FT['unfrozen']} blocks unfrozen")
    model.load_state_dict(torch.load(os.path.join(run_dir, "weights", "best.pt")))
    for kind, flips in (("dinov2_ft", False), ("dinov2_ft_tta", True)):
        save(f"{kind}_s{seed}", {s: dict(p=evaluate(model, eval_dl[s], flips), **meta(splits[s])) for s in ("val", "test")})
    open(os.path.join(run_dir, "DONE"), "w").close()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "linear":
        linear_probe()
    else:
        for s in [int(a) for a in sys.argv[2:]] or [0, 1, 2]:
            fine_tune(s)
        for kind in ("dinov2_ft", "dinov2_ft_tta"):
            runs = [np.load(f) for f in sorted(glob.glob(os.path.join(OUT, f"{kind}_s?.npz")))]
            save(kind.replace("ft", "ft_ensemble"), {s: dict(p=np.mean([r[f"{s}_p"] for r in runs], 0), y=runs[0][f"{s}_y"],
                                                              files=runs[0][f"{s}_files"]) for s in ("val", "test")})
