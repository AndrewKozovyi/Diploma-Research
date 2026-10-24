"""
Серія 3 (GPU, GTX 1650) на груповому розбитті clever_split.
Мета - не "вибити" одне число, а дати стабільні висновки для диплома: кожна конфігурація
проганяється з кількома seed, звітуємо середнє ± std на TEST.

0. Підготовка: зменшені копії датасету (довша сторона 512 px) - щоб процесор не декодував
   PNG по 2-6 МБ щоепохи. Модель все одно бачить 224 px, на точність не впливає (перевіряє група S).
     clever_split_r512       - як clever_split (train збалансовано аугментацією до 150/клас)
     clever_split_orig_r512  - train лише з ОРИГІНАЛІВ, без офлайн-аугментації (val/test ті самі)

Групи експериментів (усі - YOLO-cls, параметри -27, якщо не вказано інше):
  S  yolo11n, 150/клас ................. seeds 0-4   -> стабільна оцінка бази і розкид
  O  yolo11n, лише оригінали ........... seeds 0-2   -> чи допомагає балансування до 150 на чесному спліті
  M  yolo11s / yolo11m ................. seeds 0-2   -> чи дає більша модель
  F  yolo11n, freeze=0 ................. seeds 0-2   -> повне донавчання
  P  yolo11n, dropout=0.5 .............. seeds 0-2

Результати: results_yolo/gpu_summary.csv (кожен прогін) і gpu_groups.csv (середнє ± std по групах).
Запуск:  python exp3_gpu_seeds.py
"""
import os
import csv
import json
import time
import shutil
import datetime
import statistics
import traceback

import cv2

SRC = r"/datasets/clever_split"
DS_FULL = r"D:\projects\Diploma\datasets\clever_split_r512"
DS_ORIG = r"D:\projects\Diploma\datasets\clever_split_orig_r512"
PROJECT_DIR = r"/CNN/results_yolo/gpu"
SUMMARY = os.path.join(PROJECT_DIR, "gpu_summary.csv")
GROUPS = os.path.join(PROJECT_DIR, "gpu_groups.csv")
MAX_SIDE = 512

BASE = dict(epochs=30, imgsz=224, batch=16, freeze=5, patience=15, optimizer="auto", lr0=1e-5,
            dropout=0.8, weight_decay=0.0005, fliplr=0.5, scale=0.5, deterministic=True,
            device=0, workers=4)

GROUP_DEFS = [
    ("S_n_150", "yolo11n-cls.pt", DS_FULL, {}, [0, 1, 2, 3, 4]),
    ("O_n_orig", "yolo11n-cls.pt", DS_ORIG, {}, [0, 1, 2]),
    ("M_s_150", "yolo11s-cls.pt", DS_FULL, {}, [0, 1, 2]),
    ("M_m_150", "yolo11m-cls.pt", DS_FULL, {}, [0, 1, 2]),
    ("F_n_freeze0", "yolo11n-cls.pt", DS_FULL, {"freeze": 0}, [0, 1, 2]),
    ("P_n_drop05", "yolo11n-cls.pt", DS_FULL, {"dropout": 0.5}, [0, 1, 2]),
]


def is_aug(name):
    return "_aug_" in name or name.startswith("aug_")


def make_resized(dst, originals_only_train):
    if os.path.exists(os.path.join(dst, "DONE")):
        print(f"  {dst} вже готовий")
        return
    n = 0
    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(SRC, split))):
            sdir = os.path.join(SRC, split, cls)
            if not os.path.isdir(sdir):
                continue
            ddir = os.path.join(dst, split, cls)
            os.makedirs(ddir, exist_ok=True)
            for f in sorted(os.listdir(sdir)):
                if split == "train" and originals_only_train and is_aug(f):
                    continue
                img = cv2.imread(os.path.join(sdir, f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                s = MAX_SIDE / max(h, w)
                if s < 1:
                    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(ddir, os.path.splitext(f)[0] + ".png"), img)
                n += 1
    open(os.path.join(dst, "DONE"), "w").close()
    print(f"  {dst}: {n} зображень")


def run(group, weights, data, overrides, seed):
    from ultralytics import YOLO
    from replot_learning_curve import replot
    name = f"{group}_seed{seed}"
    args = {**BASE, **overrides, "seed": seed}
    t0 = time.time()
    model = YOLO(weights)
    res = model.train(data=data, project=PROJECT_DIR, name=name, exist_ok=True, plots=(seed == 0), **args)
    run_dir = str(res.save_dir)
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    v = best.val(data=data, split="val", imgsz=args["imgsz"], batch=32, device=0, plots=False,
                 project=run_dir, name="eval_val", exist_ok=True)
    t = best.val(data=data, split="test", imgsz=args["imgsz"], batch=32, device=0, plots=(seed == 0),
                 project=run_dir, name="eval_test", exist_ok=True)
    m = {"val_top1": float(v.top1), "test_top1": float(t.top1), "test_top5": float(t.top5)}
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"run": name, **m, "weights": weights, "data": data, "args": args}, f, indent=2)
    try:
        replot(run_dir)
    except Exception:
        traceback.print_exc()
    return name, m, (time.time() - t0) / 60


def append(path, row, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    import torch
    assert torch.cuda.is_available(), "CUDA недоступна - спершу запусти install_cuda_torch.py"
    print("GPU:", torch.cuda.get_device_name(0))
    os.makedirs(PROJECT_DIR, exist_ok=True)

    print("\nПідготовка зменшених датасетів...")
    make_resized(DS_FULL, originals_only_train=False)
    make_resized(DS_ORIG, originals_only_train=True)

    fields = ["time", "group", "run", "seed", "val_top1", "test_top1", "test_top5", "minutes", "note"]
    for group, weights, data, overrides, seeds in GROUP_DEFS:
        tests = []
        for seed in seeds:
            print(f"\n{'=' * 70}\n{group} seed {seed}\n{'=' * 70}")
            row = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "group": group, "seed": seed,
                   "run": "", "val_top1": "", "test_top1": "", "test_top5": "", "minutes": "", "note": ""}
            try:
                name, m, mins = run(group, weights, data, overrides, seed)
                row.update(run=name, minutes=f"{mins:.1f}", **{k: f"{v:.3f}" for k, v in m.items()})
                tests.append(m)
            except Exception as e:
                traceback.print_exc()
                row["note"] = f"ERROR: {type(e).__name__}: {e}"[:300]
            append(SUMMARY, row, fields)
            print(f">>> {row}")
        if tests:
            t1 = [m["test_top1"] for m in tests]
            t5 = [m["test_top5"] for m in tests]
            g = {"group": group, "n": len(t1),
                 "test_top1_mean": f"{statistics.mean(t1):.3f}",
                 "test_top1_std": f"{statistics.stdev(t1):.3f}" if len(t1) > 1 else "",
                 "test_top5_mean": f"{statistics.mean(t5):.3f}",
                 "val_top1_mean": f"{statistics.mean(m['val_top1'] for m in tests):.3f}",
                 "params": json.dumps({"weights": weights, "data": os.path.basename(data), **overrides})}
            append(GROUPS, g, list(g.keys()))
            print(f"\n### {group}: TEST top1 {g['test_top1_mean']} ± {g['test_top1_std']} (n={len(t1)})\n")
    print(f"Готово. {SUMMARY}\n{GROUPS}")


if __name__ == "__main__":
    main()
