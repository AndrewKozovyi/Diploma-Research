"""
5-кратна ГРУПОВА крос-валідація (GPU) - найнадійніша оцінка для диплома.

Навіщо: test на 100 фото дає похибку ~±10 п.п. (95% CI). При 5-fold кожне з 648 оригінальних фото
рівно один раз потрапляє в test, тож підсумкова точність рахується на всіх 648 фото (CI ~±4 п.п.),
плюс видно розкид між фолдами і точність по кожному класу.

Розбиття: ті самі групи, що й у clever_split (запис GBIF/Yale; для Monoclonius - окреме фото запису).
Групи кожного класу розкладаються по 5 фолдах, балансуючи кількість фото. Для фолду k:
  test  = фолд k
  val   = ~12% груп з решти (дрібні записи) - лише для вибору best.pt / регуляризації
  train = решта, YOLO: збалансовано офлайн-аугментацією до 150/клас (як у фінальній моделі)
Жоден запис ніколи не буває одночасно в train і test.

Моделі:
  CV_yolo11n   - фінальна конфігурація (параметри -27, 224 px)
  CV_dinov2s   - DINOv2 ViT-S + логістична регресія (контрольний, інший тип моделі)

Результати: results_yolo/gpu/cv/ : cv_folds.csv, cv_overall.csv, cv_per_class.csv, cv_confusion_<model>.csv
Запуск:  python exp4_group_cv.py
"""
import os
import sys
import csv
import json
import random
import shutil
import datetime
import statistics
import traceback
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, r"/data_preparation")
from clever_split_core import group_key, find_single_record_classes  # noqa: E402
from build_clever_split import augment_image  # noqa: E402

POOL = r"D:\projects\Diploma\datasets\clever_split_orig_r512"   # усі оригінали (512 px) з train/val/test
CV_ROOT = r"D:\projects\Diploma\datasets\cv5"
OUT = r"D:\projects\Diploma\CNN\results_yolo\gpu\cv"
K = 5
SEED = 42
VAL_FRAC = 0.12
TRAIN_PER_CLASS = 150

YOLO_ARGS = dict(epochs=30, imgsz=224, batch=16, freeze=5, patience=15, optimizer="auto", lr0=1e-5,
                 dropout=0.8, weight_decay=0.0005, fliplr=0.5, scale=0.5, deterministic=True,
                 device=0, workers=4, seed=0)


# ------------------------------------------------------------------------------------------
# Побудова фолдів
# ------------------------------------------------------------------------------------------
def collect_pool():
    files = {}
    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(POOL, split))):
            d = os.path.join(POOL, split, cls)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        files[(cls, f)] = os.path.join(d, f)
    return files


def assign_folds(files):
    by_cls = defaultdict(list)
    for cls, f in files:
        by_cls[cls].append(f)
    single = find_single_record_classes(by_cls)
    rng = random.Random(SEED)
    fold_of, group_of = {}, {}
    for cls in sorted(by_cls):
        groups = defaultdict(list)
        for f in by_cls[cls]:
            groups[group_key(cls, f, single)].append(f)
        keys = sorted(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda g: -len(groups[g]))
        load = [0] * K
        for g in keys:
            k = min(range(K), key=lambda i: (load[i], rng.random()))
            load[k] += len(groups[g])
            for f in groups[g]:
                fold_of[(cls, f)] = k
                group_of[(cls, f)] = g
        print(f"  {cls:14} фото по фолдах: {load}")
    return fold_of, group_of


def build_fold(k, files, fold_of, group_of):
    root = os.path.join(CV_ROOT, f"fold{k}")
    if os.path.exists(os.path.join(root, "DONE")):
        return root
    if os.path.exists(root):
        shutil.rmtree(root)
    rng = random.Random(SEED + k)
    random.seed(SEED + k)
    split_of = {}
    by_cls = defaultdict(lambda: defaultdict(list))
    for key, fk in fold_of.items():
        if fk == k:
            split_of[key] = "test"
        else:
            by_cls[key[0]][group_of[key]].append(key)
    for cls, groups in by_cls.items():
        n = sum(len(v) for v in groups.values())
        keys = sorted(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda g: len(groups[g]))  # дрібні записи - кандидати у val
        val_n = 0
        for g in keys:
            dst = "val" if val_n < VAL_FRAC * n else "train"
            if dst == "val":
                val_n += len(groups[g])
            for key in groups[g]:
                split_of[key] = dst
    for (cls, f), split in split_of.items():
        d = os.path.join(root, split, cls)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(files[(cls, f)], os.path.join(d, f))
    # балансування train до TRAIN_PER_CLASS
    for cls in sorted(os.listdir(os.path.join(root, "train"))):
        d = os.path.join(root, "train", cls)
        originals = sorted(os.listdir(d))
        made = 0
        while len(originals) + made < TRAIN_PER_CLASS:
            src = random.choice(originals)
            img = cv2.imread(os.path.join(d, src))
            base, ext = os.path.splitext(src)
            cv2.imwrite(os.path.join(d, f"{base}_aug_{made}{ext}"), augment_image(img))
            made += 1
    with open(os.path.join(root, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "file", "split", "group"])
        for (cls, f), s in sorted(split_of.items()):
            w.writerow([cls, f, s, group_of[(cls, f)]])
    open(os.path.join(root, "DONE"), "w").close()
    return root


# ------------------------------------------------------------------------------------------
# Моделі: повертають (class_names, y_true, y_pred, probs) на test фолду
# ------------------------------------------------------------------------------------------
def list_split(root, split, classes=None):
    classes = classes or sorted(os.listdir(os.path.join(root, split)))
    items = []
    for i, c in enumerate(classes):
        d = os.path.join(root, split, c)
        if os.path.isdir(d):
            items += [(os.path.join(d, f), i) for f in sorted(os.listdir(d))]
    return items, classes


def run_yolo(root, k):
    from ultralytics import YOLO
    model = YOLO("../yolo11n-cls.pt")
    res = model.train(data=root, project=OUT, name=f"yolo11n_fold{k}", exist_ok=True, plots=False, **YOLO_ARGS)
    best = YOLO(os.path.join(str(res.save_dir), "weights", "best.pt"))
    names = [best.names[i] for i in range(len(best.names))]
    classes = sorted(names)
    order = [names.index(c) for c in classes]
    items, _ = list_split(root, "test", classes)
    probs = []
    for p, _ in items:
        probs.append(best.predict(p, imgsz=YOLO_ARGS["imgsz"], verbose=False, device=0)[0].probs.data.cpu().numpy()[order])
    probs = np.stack(probs)
    return classes, np.array([y for _, y in items]), probs


_DINO = {}


def run_dinov2(root, k):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    if "m" not in _DINO:
        _DINO["m"] = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().cuda()
    m = _DINO["m"]
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @torch.no_grad()
    def feats(items):
        out = []
        for i in range(0, len(items), 32):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in items[i:i + 32]]).cuda()
            out.append(torch.nn.functional.normalize(m(x) + m(torch.flip(x, dims=[3])), dim=1).cpu())
        return torch.cat(out), torch.tensor([y for _, y in items])

    classes = sorted(os.listdir(os.path.join(root, "train")))
    Xtr, ytr = feats(list_split(root, "train", classes)[0])
    Xva, yva = feats(list_split(root, "val", classes)[0])
    Xte, yte = feats(list_split(root, "test", classes)[0])
    best = None
    for wd in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        lin = torch.nn.Linear(Xtr.shape[1], len(classes))
        opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lin(Xtr), ytr) + wd * lin.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            ce = float(torch.nn.functional.cross_entropy(lin(Xva), yva))
        if best is None or ce < best[0]:
            best = (ce, lin)
    with torch.no_grad():
        probs = torch.softmax(best[1](Xte), 1).numpy()
    return classes, yte.numpy(), probs


MODELS = {"CV_yolo11n": run_yolo, "CV_dinov2s": run_dinov2}


# ------------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def main():
    import torch
    assert torch.cuda.is_available(), "CUDA недоступна"
    os.makedirs(OUT, exist_ok=True)
    files = collect_pool()
    print(f"Оригіналів у пулі: {len(files)}")
    fold_of, group_of = assign_folds(files)
    roots = [build_fold(k, files, fold_of, group_of) for k in range(K)]

    pooled = {m: {"y": [], "p": []} for m in MODELS}
    folds_csv = os.path.join(OUT, "cv_folds.csv")
    with open(folds_csv, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["time", "model", "fold", "n_test", "top1", "top3", "top5", "note"])
    classes = None
    for k, root in enumerate(roots):
        for mname, fn in MODELS.items():
            print(f"\n{'=' * 70}\n{mname} fold {k}\n{'=' * 70}")
            row = [datetime.datetime.now().strftime("%H:%M"), mname, k]
            try:
                classes, y, p = fn(root, k)
                rank = np.argsort(-p, 1)
                accs = [float((rank[:, :t] == y[:, None]).any(1).mean()) for t in (1, 3, 5)]
                pooled[mname]["y"].append(y)
                pooled[mname]["p"].append(p)
                row += [len(y)] + [f"{a:.3f}" for a in accs] + [""]
                print(f">>> {mname} fold {k}: top1 {accs[0]:.3f} | top3 {accs[1]:.3f} | top5 {accs[2]:.3f}")
            except Exception as e:
                traceback.print_exc()
                row += ["", "", "", "", f"ERROR: {type(e).__name__}: {e}"[:300]]
            with open(folds_csv, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(row)

    # підсумки
    overall, per_class_rows = [], []
    for mname, d in pooled.items():
        if not d["y"]:
            continue
        y = np.concatenate(d["y"])
        p = np.concatenate(d["p"])
        rank = np.argsort(-p, 1)
        fold_top1 = [float((np.argmax(pp, 1) == yy).mean()) for yy, pp in zip(d["y"], d["p"])]
        hit1 = int((rank[:, 0] == y).sum())
        lo, hi = wilson(hit1, len(y))
        overall.append({
            "model": mname, "n_images": len(y), "folds": len(fold_top1),
            "pooled_top1": f"{hit1 / len(y):.3f}", "top1_95CI": f"[{lo:.3f}, {hi:.3f}]",
            "fold_top1_mean": f"{statistics.mean(fold_top1):.3f}",
            "fold_top1_std": f"{statistics.stdev(fold_top1):.3f}" if len(fold_top1) > 1 else "",
            "pooled_top3": f"{float((rank[:, :3] == y[:, None]).any(1).mean()):.3f}",
            "pooled_top5": f"{float((rank[:, :5] == y[:, None]).any(1).mean()):.3f}",
        })
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t_, p_ in zip(y, rank[:, 0]):
            cm[t_, p_] += 1
        with open(os.path.join(OUT, f"cv_confusion_{mname}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["true\\pred"] + classes)
            for i, c in enumerate(classes):
                w.writerow([c] + cm[i].tolist())
        for i, c in enumerate(classes):
            n = int(cm[i].sum())
            per_class_rows.append({"model": mname, "class": c, "n": n,
                                   "recall": f"{cm[i, i] / n:.3f}" if n else "",
                                   "precision": f"{cm[i, i] / cm[:, i].sum():.3f}" if cm[:, i].sum() else ""})
    for path, rows in ((os.path.join(OUT, "cv_overall.csv"), overall),
                       (os.path.join(OUT, "cv_per_class.csv"), per_class_rows)):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("\n===== ПІДСУМОК 5-FOLD GROUP CV =====")
    for r in overall:
        print(json.dumps(r, ensure_ascii=False))
    print(f"Файли: {OUT}")


if __name__ == "__main__":
    main()
