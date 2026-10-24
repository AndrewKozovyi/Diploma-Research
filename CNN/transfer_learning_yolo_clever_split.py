"""
Навчання YOLO11n-cls на датасеті clever_split (групове розбиття без витоку)
з ТИМИ САМИМИ параметрами, що й обрана модель yolo_11_selected_9_classes-150-27.

Після навчання:
  - best.pt оцінюється на TEST (на ньому нічого не підбиралось) -> test_metrics.json;
  - будується графік з реальним LR (learning_curve_real_lr.png);
  - будується Grad-CAM для test-зображень (папка gradcam_test/).

Запуск:  python transfer_learning_yolo_clever_split.py
Для оцінки стабільності можна вказати кілька seed у SEEDS, напр. [0, 1, 2].
"""
import os
import json
from ultralytics import YOLO

from CNN.overnight_testing.replot_learning_curve import replot
from CNN.gradcam_yolo import run_gradcam

DATA_DIR = r"D:\projects\Diploma\datasets\clever_split"
PROJECT_DIR = r"D:\projects\Diploma\CNN\results_yolo"
BASE_WEIGHTS = "yolo11n-cls.pt"
SEEDS = [0]

# --- Параметри як у yolo_11_selected_9_classes-150-27 (args.yaml) ---------------------------
TRAIN_ARGS = dict(
    epochs=30,
    imgsz=224,
    batch=16,
    freeze=5,
    patience=15,
    optimizer="auto",     # -> AdamW, lr підбирається автоматично: 0.002*5/(4+nc) ~ 7.7e-4
    lr0=1e-5,             # ігнорується при optimizer='auto', лишаємо для ідентичності args.yaml
    dropout=0.8,
    weight_decay=0.0005,
    degrees=15,
    fliplr=0.5,
    scale=0.5,
    translate=0.1,
    deterministic=True,
)


def main():
    summary = []
    for seed in SEEDS:
        name = f"yolo_11_clever_split-150-seed{seed}"
        print(f"\n===== {name} =====")
        model = YOLO(BASE_WEIGHTS)
        results = model.train(data=DATA_DIR, project=PROJECT_DIR, name=name, seed=seed, **TRAIN_ARGS)
        run_dir = str(results.save_dir)

        best = YOLO(os.path.join(run_dir, "weights", "best.pt"))
        val_m = best.val(data=DATA_DIR, split="val", imgsz=TRAIN_ARGS["imgsz"], batch=16,
                         project=run_dir, name="eval_val", plots=True)
        test_m = best.val(data=DATA_DIR, split="test", imgsz=TRAIN_ARGS["imgsz"], batch=16,
                          project=run_dir, name="eval_test", plots=True)
        metrics = {
            "run": name, "seed": seed,
            "val_top1": float(val_m.top1), "val_top5": float(val_m.top5),
            "test_top1": float(test_m.top1), "test_top5": float(test_m.top5),
        }
        with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(metrics)
        summary.append(metrics)

        replot(run_dir)
        run_gradcam(os.path.join(run_dir, "weights", "best.pt"),
                    os.path.join(DATA_DIR, "test"),
                    os.path.join(run_dir, "gradcam_test"),
                    imgsz=TRAIN_ARGS["imgsz"])

    print("\n===== ПІДСУМОК =====")
    for m in summary:
        print(f"{m['run']}: val top1={m['val_top1']:.3f} | TEST top1={m['test_top1']:.3f} top5={m['test_top5']:.3f}")
    if len(summary) > 1:
        import statistics as st
        t = [m["test_top1"] for m in summary]
        print(f"TEST top1: {st.mean(t):.3f} ± {st.stdev(t):.3f} (n={len(t)} seeds)")


if __name__ == "__main__":
    main()
