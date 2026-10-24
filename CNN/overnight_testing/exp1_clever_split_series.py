"""
Серія 1 експериментів на clever_split (групове розбиття без витоку).
Кожен експеримент оцінюється на TEST і дописується в results_yolo/night_summary.csv.

Базова точка (yolo_11_clever_split-150-seed0, параметри -27): val 0.67 | TEST top1 0.62, top5 0.94.
Діагноз базового прогону:
  * сильне перенавчання: train loss 2.5 -> 0.10, val acc вийшла на плато ~0.64 вже з 5-ї епохи, val loss росте;
  * для класифікації Ultralytics НЕ використовує degrees/translate (лише RandomResizedCrop(scale),
    flips, hsv, RandAugment, erasing) -> поворотів в онлайн-аугментації фактично не було, flipud=0;
  * Grad-CAM: модель дивиться на текстуру поверхні, а не на форму кістки; 224 px - мало для текстури.

Експерименти:
  D1  DINOv2 ViT-S/14 (заморожений, self-supervised) + логістична регресія на ембедингах
      - інший підхід: сильні загальні ознаки без донавчання мережі, класично добре працює на малих даних.
  Y1  YOLO11n-cls як базовий, але flipud=0.5, cos_lr, 40 епох, patience 20 (кістка не має "верху").
  Y2  те саме, що Y1, але imgsz=320 (більше деталей текстури/форми).

Запуск:  python exp1_clever_split_series.py
"""
import os
import csv
import json
import time
import datetime
import traceback

import numpy as np
import torch

DATA_DIR = r"/datasets/clever_split"
PROJECT_DIR = r"/CNN/results_yolo"
SUMMARY = os.path.join(PROJECT_DIR, "night_summary.csv")

BASE_YOLO = dict(
    epochs=30, imgsz=224, batch=16, freeze=5, patience=15, optimizer="auto", lr0=1e-5,
    dropout=0.8, weight_decay=0.0005, degrees=15, fliplr=0.5, scale=0.5, translate=0.1,
    deterministic=True, seed=0,
)

EXPERIMENTS = [
    ("D1_dinov2_vits14_linear", "dinov2", dict(arch="dinov2_vits14", imgsz=224)),
    ("Y1_yolo11n_flipud_cos40", "yolo", dict(flipud=0.5, cos_lr=True, epochs=40, patience=20)),
    ("Y2_yolo11n_flipud_cos40_img320", "yolo", dict(flipud=0.5, cos_lr=True, epochs=40, patience=20, imgsz=320)),
]


def log_summary(row):
    new = not os.path.exists(SUMMARY)
    with open(SUMMARY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "name", "kind", "val_top1", "test_top1", "test_top5",
                                          "minutes", "params", "note"])
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\n>>> {row['name']}: val {row['val_top1']} | TEST top1 {row['test_top1']} top5 {row['test_top5']} "
          f"({row['minutes']} хв)\n")


# ------------------------------------------------------------------------------------------
# YOLO
# ------------------------------------------------------------------------------------------
def run_yolo(name, overrides):
    from ultralytics import YOLO
    from replot_learning_curve import replot
    from CNN.gradcam_yolo import run_gradcam

    args = {**BASE_YOLO, **overrides}
    model = YOLO("../yolo11n-cls.pt")
    res = model.train(data=DATA_DIR, project=PROJECT_DIR, name=name, exist_ok=True, **args)
    run_dir = str(res.save_dir)
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    v = best.val(data=DATA_DIR, split="val", imgsz=args["imgsz"], batch=16, project=run_dir, name="eval_val",
                 exist_ok=True)
    t = best.val(data=DATA_DIR, split="test", imgsz=args["imgsz"], batch=16, project=run_dir, name="eval_test",
                 exist_ok=True)
    m = {"val_top1": float(v.top1), "test_top1": float(t.top1), "test_top5": float(t.top5)}
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"run": name, **m, "args": args}, f, indent=2)
    try:
        replot(run_dir)
        run_gradcam(os.path.join(run_dir, "weights", "best.pt"), os.path.join(DATA_DIR, "test"),
                    os.path.join(run_dir, "gradcam_test"), imgsz=args["imgsz"])
    except Exception:
        traceback.print_exc()
    return m, json.dumps(overrides)


# ------------------------------------------------------------------------------------------
# DINOv2 + логістична регресія
# ------------------------------------------------------------------------------------------
def list_split(split):
    root = os.path.join(DATA_DIR, split)
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    items = []
    for i, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(root, c))):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(root, c, f), i))
    return items, classes


@torch.no_grad()
def extract(model, items, tf, bs=16):
    from PIL import Image
    feats, labels = [], []
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in chunk])
        f = model(x) + model(torch.flip(x, dims=[3]))  # TTA: оригінал + горизонтальний фліп
        feats.append(torch.nn.functional.normalize(f, dim=1))
        labels += [y for _, y in chunk]
        if (k // bs) % 10 == 0:
            print(f"   ознаки: {k + len(chunk)}/{len(items)}")
    return torch.cat(feats), torch.tensor(labels)


def train_logreg(Xtr, ytr, nc, wd, epochs=300):
    lin = torch.nn.Linear(Xtr.shape[1], nc)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xtr), ytr) + wd * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lin


def topk(logits, y, k):
    return float((logits.topk(k, dim=1).indices == y[:, None]).any(1).float().mean())


def run_dinov2(name, cfg):
    from torchvision import transforms as T
    out_dir = os.path.join(PROJECT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    model = torch.hub.load("facebookresearch/dinov2", cfg["arch"]).eval()
    s = cfg["imgsz"]
    tf = T.Compose([T.Resize(s, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(s), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    data = {}
    for split in ("train", "val", "test"):
        items, classes = list_split(split)
        print(f"  {split}: {len(items)} зображень")
        data[split] = extract(model, items, tf)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    nc = len(classes)

    # Вибір сили регуляризації за val cross-entropy (на 100 фото вона стабільніша за accuracy)
    best = None
    for wd in cfg.get("wd_grid", [1e-5, 1e-4, 1e-3, 1e-2]):
        lin = train_logreg(Xtr, ytr, nc, wd)
        with torch.no_grad():
            lv = lin(Xva)
            acc = topk(lv, yva, 1)
            ce = float(torch.nn.functional.cross_entropy(lv, yva))
        print(f"   weight decay {wd:g}: val top1 {acc:.3f}, val CE {ce:.3f}")
        if best is None or ce < best[0]:
            best = (ce, acc, wd, lin)
    _, val_acc, wd, lin = best
    with torch.no_grad():
        logits = lin(Xte)
        val_logits = lin(Xva)
    m = {"val_top1": val_acc, "test_top1": topk(logits, yte, 1), "test_top5": topk(logits, yte, 5)}
    # ймовірності для ансамблю (порядок файлів = list_split, класи за алфавітом)
    np.savez(os.path.join(out_dir, "probs.npz"),
             val=torch.softmax(val_logits, 1).numpy(), val_y=yva.numpy(),
             test=torch.softmax(logits, 1).numpy(), test_y=yte.numpy(), classes=np.array(classes))

    # матриця плутанини (рядки = справжній клас, стовпці = передбачений)
    cm = np.zeros((nc, nc), dtype=int)
    for t_, p_ in zip(yte.tolist(), logits.argmax(1).tolist()):
        cm[t_, p_] += 1
    with open(os.path.join(out_dir, "test_confusion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + classes)
        for i, c in enumerate(classes):
            w.writerow([c] + cm[i].tolist())
    torch.save({"linear": lin.state_dict(), "classes": classes, "arch": cfg["arch"], "wd": wd},
               os.path.join(out_dir, "linear_head.pt"))
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"run": name, **m, "weight_decay": wd, **cfg}, f, indent=2)
    return m, json.dumps({**cfg, "wd": wd})


def main(experiments=None):
    for name, kind, cfg in (experiments or EXPERIMENTS):
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        t0 = time.time()
        row = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "kind": kind,
               "val_top1": "", "test_top1": "", "test_top5": "", "params": "", "note": ""}
        try:
            m, params = run_yolo(name, cfg) if kind == "yolo" else run_dinov2(name, cfg)
            row.update({k: f"{v:.3f}" for k, v in m.items()}, params=params)
        except Exception as e:
            traceback.print_exc()
            row["note"] = f"ERROR: {type(e).__name__}: {e}"[:300]
        row["minutes"] = f"{(time.time() - t0) / 60:.1f}"
        log_summary(row)
    print(f"Готово. Підсумок: {SUMMARY}")


if __name__ == "__main__":
    main()
