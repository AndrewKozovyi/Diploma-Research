"""
Чесна оцінка обраної моделі.
1) Оцінює best.pt на val і на test (test раніше ніде не використовувався).
2) Ділить кожен спліт на «знайомі» екземпляри (той самий ID запису GBIF/Yale є в train)
   та «нові» екземпляри (запис у train відсутній) і рахує точність окремо.
3) Дає 95% довірчий інтервал (Wilson) і macro-accuracy по класах.

Запуск:  python evaluate_honest.py
"""
import os
import re
import math
from collections import defaultdict
from ultralytics import YOLO

DATA_DIR = r"D:\projects\Diploma\transfer_learning_datasets_filtered_by_5"
WEIGHTS = r"D:\projects\Diploma\CNN\results_yolo\yolo_11_selected_9_classes-150-27\weights\best.pt"
IMG_SIZE = 224


def record_id(class_name, filename):
    """ID екземпляра: числовий ID запису GBIF або UUID Yale (без індексу фото/кістки)."""
    b = filename.replace("parsed_", "").replace("yale_", "")
    b = re.sub(r"^aug_\d+_", "", b)
    m = re.match(r"([0-9a-f\-]{36}|\d+)", b)
    return f"{class_name}/{m.group(1) if m else b}"


def list_split(split):
    items = []
    root = os.path.join(DATA_DIR, split)
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(cdir, f), cls, record_id(cls, f)))
    return items


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report(name, rows):
    n = len(rows)
    k = sum(r["ok"] for r in rows)
    lo, hi = wilson(k, n)
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["cls"]][0] += r["ok"]
        per[r["cls"]][1] += 1
    macro = sum(a / b for a, b in per.values()) / max(len(per), 1)
    acc = k / n if n else float("nan")
    print(f"  {name:<38} n={n:3d}  acc={acc:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]  macro={macro:.3f}")
    return per


def main():
    model = YOLO(WEIGHTS)
    names = model.names  # {idx: class}

    train_records = {rid for _, _, rid in list_split("train")}

    for split in ["val", "test"]:
        items = list_split(split)
        rows = []
        for path, cls, rid in items:
            res = model.predict(path, imgsz=IMG_SIZE, verbose=False)[0]
            pred = names[int(res.probs.top1)]
            rows.append({"cls": cls, "ok": int(pred == cls), "seen": rid in train_records,
                         "pred": pred, "path": path})

        print(f"\n===== {split.upper()} =====")
        per = report("ALL", rows)
        report("seen specimens (record in train)", [r for r in rows if r["seen"]])
        report("UNSEEN specimens (new record)", [r for r in rows if not r["seen"]])
        report("ALL without Monoclonius", [r for r in rows if r["cls"] != "Monoclonius"])
        print("  per class:", {c: f"{a}/{b}" for c, (a, b) in sorted(per.items())})
        errs = [r for r in rows if not r["ok"]]
        print(f"  errors ({len(errs)}):")
        for r in errs:
            print(f"    {r['cls']:>14} -> {r['pred']:<14} {'seen' if r['seen'] else 'NEW '}  {os.path.basename(r['path'])}")


if __name__ == "__main__":
    main()
