"""
Серія 2 на clever_split. Результати дописуються в results_yolo/night_summary.csv.

Підсумки першої серії (TEST top-1 / top-5):
  B0  YOLO11n, параметри -27 ..................... 0.62 / 0.94
  D1  DINOv2 ViT-S + логістична регресія ........ 0.63 / 0.92  (10 хв, без донавчання мережі)
  Y1  YOLO11n + flipud + cos_lr, 40 еп. ......... 0.59 / 0.87  (val-пік 0.70 був випадковим)
  Y2  YOLO11n як Y1, imgsz 320 .................. (див. night_summary.csv)

Експерименти:
  D1b DINOv2 ViT-S повтор: регуляризація за val cross-entropy, збереження ймовірностей для ансамблю
  D2  DINOv2 ViT-B/14 (більший backbone), ширша сітка регуляризації
  E1  Ансамбль: середнє ймовірностей YOLO (B0, Y2) і DINOv2 (D1b, D2) - YOLO і DINOv2 помиляються
      по-різному (CNN, донавчена на наших даних, проти self-supervised трансформера).
      Комбінацію обираємо за val, на test звітуємо всі - для прозорості.

Запуск:  python exp2_dinov2_ensemble.py
"""
import os
import json
import itertools

import numpy as np

import exp1_clever_split_series as ne

YOLO_RUNS = {
    "B0": ("yolo_11_clever_split-150-seed0", 224),
    "Y2": ("Y2_yolo11n_flipud_cos40_img320", 320),
}
DINO_RUNS = ["D1b_dinov2_vits14_linear", "D2_dinov2_vitb14_linear"]

EXPERIMENTS_2 = [
    ("D1b_dinov2_vits14_linear", "dinov2",
     dict(arch="dinov2_vits14", imgsz=224, wd_grid=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2])),
    ("D2_dinov2_vitb14_linear", "dinov2",
     dict(arch="dinov2_vitb14", imgsz=224, wd_grid=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2])),
]


def yolo_probs(run_dir, imgsz, split):
    from ultralytics import YOLO
    model = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    items, classes = ne.list_split(split)
    names = [model.names[i] for i in range(len(model.names))]
    order = [names.index(c) for c in classes]  # вирівнюємо порядок класів з DINOv2 (алфавітний)
    probs = []
    for path, _ in items:
        p = model.predict(path, imgsz=imgsz, verbose=False)[0].probs.data.cpu().numpy()
        probs.append(p[order])
    return np.stack(probs), np.array([y for _, y in items])


def acc_top(p, y, k=1):
    return float((np.argsort(-p, 1)[:, :k] == y[:, None]).any(1).mean())


def run_ensemble():
    members = {}
    for key, (run, imgsz) in YOLO_RUNS.items():
        run_dir = os.path.join(ne.PROJECT_DIR, run)
        if not os.path.exists(os.path.join(run_dir, "weights", "best.pt")):
            print(f"  пропуск {key}: немає {run_dir}")
            continue
        pv, yv = yolo_probs(run_dir, imgsz, "val")
        pt, yt = yolo_probs(run_dir, imgsz, "test")
        members[key] = (pv, pt)
        print(f"  {key}: val {acc_top(pv, yv):.3f} | test {acc_top(pt, yt):.3f}")
    for run in DINO_RUNS:
        f = os.path.join(ne.PROJECT_DIR, run, "probs.npz")
        if not os.path.exists(f):
            print(f"  пропуск {run}: немає probs.npz")
            continue
        z = np.load(f)
        yv, yt = z["val_y"], z["test_y"]
        members[run.split("_")[0]] = (z["val"], z["test"])

    keys = sorted(members)
    table = []
    for r in range(2, len(keys) + 1):
        for combo in itertools.combinations(keys, r):
            pv = np.mean([members[k][0] for k in combo], 0)
            pt = np.mean([members[k][1] for k in combo], 0)
            table.append({"combo": "+".join(combo), "val_top1": acc_top(pv, yv),
                          "test_top1": acc_top(pt, yt), "test_top5": acc_top(pt, yt, 5)})
    table.sort(key=lambda r: -r["val_top1"])
    out_dir = os.path.join(ne.PROJECT_DIR, "E1_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ensemble_table.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print("\n  Усі комбінації (відсортовано за val):")
    for r in table:
        print(f"   {r['combo']:<20} val {r['val_top1']:.3f} | TEST {r['test_top1']:.3f} / {r['test_top5']:.3f}")
    best = table[0]
    return ({"val_top1": best["val_top1"], "test_top1": best["test_top1"], "test_top5": best["test_top5"]},
            json.dumps({"chosen_by_val": best["combo"]}))


def main():
    ne.main(EXPERIMENTS_2)
    import time, datetime, traceback
    t0 = time.time()
    row = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "name": "E1_ensemble", "kind": "ensemble",
           "val_top1": "", "test_top1": "", "test_top5": "", "params": "", "note": ""}
    try:
        m, params = run_ensemble()
        row.update({k: f"{v:.3f}" for k, v in m.items()}, params=params)
    except Exception as e:
        traceback.print_exc()
        row["note"] = f"ERROR: {type(e).__name__}: {e}"[:300]
    row["minutes"] = f"{(time.time() - t0) / 60:.1f}"
    ne.log_summary(row)


if __name__ == "__main__":
    main()
