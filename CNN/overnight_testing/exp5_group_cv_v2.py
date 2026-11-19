"""
5-fold крос-валідація фінальної конфігурації (yolo11m-cls, 320 px, freeze 7) на датасеті v2 (10 класів, 883 фото).

Два розбиття одних і тих самих оригіналів:
  group  - усі фото одного запису GBIF/Yale в одному фолді (визначення роду для нових екземплярів);
           для Monoclonius (весь клас - один запис) група = окреме фото запису
  random - стратифіковане випадкове розбиття по фото (як у фінальній моделі)
Для фолду k: test = фолд k, val = ~12% груп/фото з решти (вибір best.pt), train = решта,
офлайн-аугментація train до 150/клас (як у build_v2_split.py).

Результати: results_yolo/cv_v2/ : cv_folds.csv, cv_overall.csv, cv_per_class.csv, cv_confusion_<split>.csv, preds_<split>.npz
Запуск:  python exp5_group_cv_v2.py [group|random ...]
"""
import os
import sys
import csv
import json
import random
import shutil
import datetime
import statistics
import importlib.util
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "..", "..", "data_preparation")
sys.path.insert(0, PREP)
from clever_split_core import group_key, find_single_record_classes  # noqa: E402

spec = importlib.util.spec_from_file_location("augment_train", os.path.join(PREP, "2_augment_train.py"))
augment_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augment_train)

POOL = r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_v2_filtered_by_5_r512"
CV_ROOT = r"D:\projects\Diploma\datasets\cv5_v2"
OUT = r"D:\projects\Diploma\CNN\results_yolo\cv_v2"
K = 5
SEED = 42
VAL_FRAC = 0.12
TRAIN_PER_CLASS = 150

YOLO_ARGS = dict(epochs=30, imgsz=320, batch=16, freeze=7, patience=15, optimizer="auto", lr0=1e-5,
                 dropout=0.8, weight_decay=0.005, degrees=15, fliplr=0.5, translate=0.1, scale=0.7,
                 flipud=0.5, erasing=0.6, deterministic=True, device=0, workers=4, seed=1)


def collect_pool():
    files = {}
    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(POOL, split))):
            for f in sorted(os.listdir(os.path.join(POOL, split, cls))):
                if "_aug_" not in f:
                    files[(cls, f)] = os.path.join(POOL, split, cls, f)
    return files


def make_groups(files, mode):
    by_cls = defaultdict(list)
    for cls, f in files:
        by_cls[cls].append(f)
    single = find_single_record_classes(by_cls)
    if mode == "group":
        return {(c, f): group_key(c, f, single) for c, f in files}
    return {(c, f): f"{c}/{f}" for c, f in files}


def assign_folds(group_of):
    rng = random.Random(SEED)
    by_cls = defaultdict(lambda: defaultdict(list))
    for key, g in group_of.items():
        by_cls[key[0]][g].append(key)
    fold_of = {}
    for cls in sorted(by_cls):
        groups = by_cls[cls]
        keys = sorted(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda g: -len(groups[g]))
        load = [0] * K
        for g in keys:
            k = min(range(K), key=lambda i: (load[i], rng.random()))
            load[k] += len(groups[g])
            for key in groups[g]:
                fold_of[key] = k
        print(f"  {cls:14} фото по фолдах: {load}")
    return fold_of


def build_fold(mode, k, files, fold_of, group_of):
    root = os.path.join(CV_ROOT, mode, f"fold{k}")
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
        keys.sort(key=lambda g: len(groups[g]))
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
    for cls in sorted(os.listdir(os.path.join(root, "train"))):
        d = os.path.join(root, "train", cls)
        originals = sorted(os.listdir(d))
        for made in range(TRAIN_PER_CLASS - len(originals)):
            src = random.choice(originals)
            base, ext = os.path.splitext(src)
            cv2.imwrite(os.path.join(d, f"{base}_aug_{made}{ext}"),
                        augment_train.augment_image(cv2.imread(os.path.join(d, src))))
    with open(os.path.join(root, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "file", "split", "group"])
        for (cls, f), s in sorted(split_of.items()):
            w.writerow([cls, f, s, group_of[(cls, f)]])
    open(os.path.join(root, "DONE"), "w").close()
    return root


def run_yolo(root, name):
    from ultralytics import YOLO
    weights = os.path.join(OUT, name, "weights", "best.pt")
    if not os.path.exists(os.path.join(OUT, name, "DONE")):
        res = YOLO(os.path.join(HERE, "..", "yolo11m-cls.pt")).train(
            data=root, project=OUT, name=name, exist_ok=True, plots=False, **YOLO_ARGS)
        weights = os.path.join(str(res.save_dir), "weights", "best.pt")
        open(os.path.join(OUT, name, "DONE"), "w").close()
    best = YOLO(weights)
    names = [best.names[i] for i in range(len(best.names))]
    classes = sorted(names)
    order = [names.index(c) for c in classes]
    items = [(os.path.join(root, "test", c, f), i) for i, c in enumerate(classes)
             for f in sorted(os.listdir(os.path.join(root, "test", c)))]
    probs = np.stack([r.probs.data.cpu().numpy()[order] for r in
                      best.predict([p for p, _ in items], imgsz=YOLO_ARGS["imgsz"], verbose=False, device=0, batch=16)])
    return classes, np.array([y for _, y in items]), probs, [p for p, _ in items]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def summarize(results, classes):
    overall, per_class = [], []
    for mode, folds in results.items():
        y = np.concatenate([f["y"] for f in folds])
        p = np.concatenate([f["p"] for f in folds])
        rank = np.argsort(-p, 1)
        fold_top1 = [float((np.argmax(f["p"], 1) == f["y"]).mean()) for f in folds]
        hit1 = int((rank[:, 0] == y).sum())
        lo, hi = wilson(hit1, len(y))
        overall.append({
            "split": mode, "n_images": len(y), "folds": len(folds),
            "pooled_top1": f"{hit1 / len(y):.3f}", "top1_95CI": f"[{lo:.3f}, {hi:.3f}]",
            "fold_top1_mean": f"{statistics.mean(fold_top1):.3f}",
            "fold_top1_std": f"{statistics.stdev(fold_top1):.3f}" if len(folds) > 1 else "",
            "pooled_top3": f"{float((rank[:, :3] == y[:, None]).any(1).mean()):.3f}",
            "pooled_top5": f"{float((rank[:, :5] == y[:, None]).any(1).mean()):.3f}",
        })
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t_, p_ in zip(y, rank[:, 0]):
            cm[t_, p_] += 1
        with open(os.path.join(OUT, f"cv_confusion_{mode}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["true\\pred"] + classes)
            for i, c in enumerate(classes):
                w.writerow([c] + cm[i].tolist())
        for i, c in enumerate(classes):
            n = int(cm[i].sum())
            per_class.append({"split": mode, "class": c, "n": n,
                              "recall": f"{cm[i, i] / n:.3f}" if n else "",
                              "precision": f"{cm[i, i] / cm[:, i].sum():.3f}" if cm[:, i].sum() else ""})
        np.savez(os.path.join(OUT, f"preds_{mode}.npz"), y=y, p=p, classes=classes,
                 paths=np.array(sum([f["paths"] for f in folds], [])))
    for path, rows in ((os.path.join(OUT, "cv_overall.csv"), overall),
                       (os.path.join(OUT, "cv_per_class.csv"), per_class)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for r in overall:
        print(json.dumps(r, ensure_ascii=False))


def main():
    modes = sys.argv[1:] or ["group", "random"]
    os.makedirs(OUT, exist_ok=True)
    files = collect_pool()
    print(f"Оригіналів у пулі: {len(files)}")
    folds_csv = os.path.join(OUT, "cv_folds.csv")
    if not os.path.exists(folds_csv):
        with open(folds_csv, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["time", "split", "fold", "n_test", "top1", "top3", "top5"])
    results, classes = {}, None
    for mode in modes:
        print(f"\n### {mode}")
        group_of = make_groups(files, mode)
        fold_of = assign_folds(group_of)
        results[mode] = []
        for k in range(K):
            root = build_fold(mode, k, files, fold_of, group_of)
            classes, y, p, paths = run_yolo(root, f"{mode}_fold{k}")
            rank = np.argsort(-p, 1)
            accs = [float((rank[:, :t] == y[:, None]).any(1).mean()) for t in (1, 3, 5)]
            results[mode].append({"y": y, "p": p, "paths": paths})
            print(f">>> {mode} fold {k}: top1 {accs[0]:.3f} | top3 {accs[1]:.3f} | top5 {accs[2]:.3f}")
            with open(folds_csv, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([datetime.datetime.now().strftime("%H:%M"), mode, k, len(y)]
                                        + [f"{a:.3f}" for a in accs])
    summarize(results, classes)


if __name__ == "__main__":
    main()
