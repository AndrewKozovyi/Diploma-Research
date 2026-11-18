import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==============================================================================
# НАЛАШТУВАННЯ (Точна копія конфігурації з логу 88.2%)
# ==============================================================================
# DATA_DIR = r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_filtered_by_5"
DATA_DIR = r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_v2_filtered_by_5"
# EXPERIMENT_NAME = "yolo_8_baseline-250"
# EXPERIMENT_NAME = "yolo_11_baseline-250"
# EXPERIMENT_NAME = "yolo_8_9_classes_baseline-250"
# EXPERIMENT_NAME = "yolo_11_9_classes_baseline-250"
# EXPERIMENT_NAME = "yolo_8_9_classes_finetuned-250"
# EXPERIMENT_NAME = "yolo_11_9_classes_finetuned-250"
# EXPERIMENT_NAME = "yolo_8_finetuned-100"
# EXPERIMENT_NAME = "yolo_11_finetuned-250"
# EXPERIMENT_NAME = "yolo_11_selected_9_classes-150"
EXPERIMENT_NAME = "yolo_11m_v2_10_classes_best-150"
PROJECT_DIR = r"D:\projects\Diploma\CNN\results_yolo"
MODEL_WEIGHTS = 'yolo11m-cls.pt'

EPOCHS = 30
BATCH_SIZE = 16
# IMG_SIZE = 224
IMG_SIZE = 320
# FREEZE_LAYERS = 9
# FREEZE_LAYERS = 5
FREEZE_LAYERS = 7
# LR_INIT = 0.001
LR_INIT = 1e-5   # УВАГА: при OPTIMIZER='auto' ігнорується, реальний LR ~7.7e-4 (див. lr/pg0 у results.csv)
yolo_name = 'yolo11'

# ------------------------------------------------------------------------------
# АНТИ-ПЕРЕНАВЧАННЯ (РЕГУЛЯРИЗАЦІЯ ТА АУГМЕНТАЦІЯ)
# ------------------------------------------------------------------------------
OPTIMIZER = 'auto'          # В логах було optimizer=auto (дає AdamW з lr=0.000769)
DROPOUT = 0.8               # Саме 0.8 було у вдалому запуску
WEIGHT_DECAY = 0.005        # Штраф за великі ваги (раніше 0.0005)
LABEL_SMOOTHING = 0.0       # Згладжування міток
DEGREES = 15                # Повороти зображень з логу
FLIP_LR = 0.5               # Віддзеркалення по горизонталі з логу
SCALE = 0.7                 # Сильніший RandomResizedCrop (раніше 0.5)
TRANSLATE = 0.1             # Дефолтне значення YOLO, яке працювало у тому запуску
FLIP_UD = 0.5               # Віддзеркалення по вертикалі (раніше 0.0)
ERASING = 0.6               # Random erasing для класифікації (раніше 0.4)
COS_LR = False              # Косинусний спад LR (дефолт YOLO)
SEED = 1                    # Seed найкращого запуску (раніше 0)

# ------------------------------------------------------------------------------
# ШВИДКЕ НАВЧАННЯ
# ------------------------------------------------------------------------------
FAST = False                # True: навчання на копії DATA_DIR + "_r512" (фото зменшені до 512 px, модель так само бачить IMG_SIZE)
DEVICE = 0                  # '' - автовибір (GPU, якщо є), 0 - перша відеокарта, 'cpu'
WORKERS = 4                 # Кількість процесів завантаження даних

# ==============================================================================


def resolve_data_dir():
    if not FAST:
        return DATA_DIR
    fast_dir = DATA_DIR.rstrip("\\/") + "_r512"
    if not os.path.exists(fast_dir):
        import importlib.util
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_preparation", "make_r512.py")
        spec = importlib.util.spec_from_file_location("make_r512", script)
        make_r512 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(make_r512)
        make_r512.main(DATA_DIR)
    return fast_dir


def plot_custom_results(results_dir):
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        print("Файл results.csv не знайдено, графік не побудовано.")
        return

    # Зчитуємо дані та очищаємо назви колонок від зайвих пробілів
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    epochs = df['epoch']
    val_acc = df['metrics/accuracy_top1']
    train_loss = df['train/loss']
    val_loss = df['val/loss']

    plt.figure(figsize=(12, 5.5))

    # Графік точності (в YOLO точність рахується лише для валідації)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_acc, color='darkorange', label='Val Accuracy (Top-1)')
    plt.legend()
    plt.title(f"Validation Accuracy over Epochs ({yolo_name})")

    # Графік втрат
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, color='tab:blue', label='Train Loss')
    plt.plot(epochs, val_loss, color='darkorange', label='Val Loss')
    plt.legend()
    plt.title(f"Loss over Epochs ({yolo_name})")

    # Реальний LR береться з логу (lr/pg0). При optimizer='auto' Ultralytics ігнорує lr0
    # і сам обирає AdamW з lr = 0.002*5/(4+nc) (~7.7e-4 для 9 класів), тому LR_INIT тут не показуємо.
    if OPTIMIZER == 'auto':
        lr_text = (f"Optimizer: auto -> AdamW | LR (з логу): peak {df['lr/pg0'].max():.2e}, "
                   f"final {df['lr/pg0'].iloc[-1]:.2e}")
    else:
        lr_text = f"Optimizer: {OPTIMIZER} | Init LR: {LR_INIT} (peak з логу {df['lr/pg0'].max():.2e})"

    # Підпис (3 рядки)
    params_text = (
        f"MODEL: {os.path.splitext(MODEL_WEIGHTS)[0]} | EXPERIMENT: {EXPERIMENT_NAME}\n"
        f"Img Size: {IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | {lr_text} | Freeze: {FREEZE_LAYERS}\n"
        f"Dropout: {DROPOUT} | Weight Decay: {WEIGHT_DECAY} | Degrees: {DEGREES}° | Flip LR: {FLIP_LR}"
    )
    plt.figtext(0.5, 0.01, params_text, wrap=True, horizontalalignment='center',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.subplots_adjust(bottom=0.28)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(results_dir, f"custom_learning_curve_{timestamp}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Фірмовий графік збережено: {plot_path}")


def main():
    print(f"--- Initialization {yolo_name} Classification ---")
    # model = YOLO('yolov8n-cls.pt')
    model = YOLO(MODEL_WEIGHTS)

    # best_weights_path = r"D:\projects\Diploma\CNN\results_yolo\yolo_11_9_classes_baseline-250\weights\best.pt"
    # model = YOLO(best_weights_path)

    print(f"Starting training for {EXPERIMENT_NAME}...")
    results = model.train(
        data=resolve_data_dir(),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        freeze=FREEZE_LAYERS,
        patience=15,
        lr0=LR_INIT,
        optimizer=OPTIMIZER,        # 'auto' увімкне саме той режим, що був у логах
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
        degrees=DEGREES,
        fliplr=FLIP_LR,
        scale=SCALE,                # 0.5 (як було в логах)
        translate=TRANSLATE,        # 0.1 (як було в логах)
        flipud=FLIP_UD,
        erasing=ERASING,
        cos_lr=COS_LR,
        seed=SEED,
        device=DEVICE,
        workers=WORKERS
    )

    print("--- Training Completed ---")
    results_dir = results.save_dir

    plot_custom_results(results_dir)


if __name__ == '__main__':
    main()