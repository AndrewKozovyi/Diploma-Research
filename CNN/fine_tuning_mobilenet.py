import datetime
import os
import tensorflow as tf
from keras import callbacks
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# ==============================================================================
# 1. ГІПЕРПАРАМЕТРИ ДЛЯ ЧИСТОГО FINE-TUNING (Еталон 71.3%)
# ==============================================================================
BASE_DIR = r"D:\projects\Diploma\transfer_learning_datasets_filtered_by_5"
RESULTS_DIR = r"D:\projects\Diploma\CNN\results_mobilenet"
MODEL_PATH = "results_mobilenet/mobilenet_9_classes_frozen_epoches_100.keras"  # Твоя готова модель з Фази 1

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
FINE_TUNE_EPOCHS = 60

# Мікро-крок для безпечного донавчання
LEARNING_RATE = 1e-5
# Розморожуємо зі 100-го шару
FINE_TUNE_AT = 100

PATIENCE_EARLY_STOP = 10
PATIENCE_REDUCE_LR = 4
FACTOR_REDUCE_LR = 0.5
# ==============================================================================

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

print("Loading datasets for Fine-Tuning...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical'
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================================================================
# 2. ЗАВАНТАЖЕННЯ ТА РОЗМОРОЖУВАННЯ МОДЕЛІ
# ==============================================================================
print(f"Loading base model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Дістаємо внутрішню базу MobileNetV2
base_model = model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = True

# Заморожуємо все до 100-го шару
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Обов'язкова перекомпіляція після зміни trainable
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("Number of trainable variables:", len(model.trainable_variables))

# ==============================================================================
# 3. ДОНАВЧАННЯ
# ==============================================================================
checkpoint_cb = callbacks.ModelCheckpoint(
    "results_mobilenet/mobilenet_9_classes_finetuned_epoches_100.keras", save_best_only=True, monitor='val_accuracy', mode='max'
)
early_stopping_cb = callbacks.EarlyStopping(
    patience=PATIENCE_EARLY_STOP, monitor='val_accuracy', mode='max', restore_best_weights=True
)
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', mode='max', factor=FACTOR_REDUCE_LR, patience=PATIENCE_REDUCE_LR, min_lr=1e-7, verbose=1
)

print("Starting Fine-Tuning...")
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
)

# ==============================================================================
# 4. ПОБУДОВА ГРАФІКІВ
# ==============================================================================
plt.figure(figsize=(12, 5.5))
plt.subplot(1, 2, 1)
plt.plot(history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over Epochs (Fine-Tuning)')

plt.subplot(1, 2, 2)
plt.plot(history_fine.history['loss'], label='Train Loss')
plt.plot(history_fine.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs (Fine-Tuning)')

loaded_dropout_rate = model.get_layer("dropout").rate

params_text = (
    f"TRANSFER LEARNING: Fine-Tuning (From layer {FINE_TUNE_AT})\n"
    f"Img Size: {IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {FINE_TUNE_EPOCHS}\n"
    f"Opt: Adam(lr={LEARNING_RATE}) | Dropout: {loaded_dropout_rate}\n"
    f"Callbacks: ReduceLR(p={PATIENCE_REDUCE_LR}, f={FACTOR_REDUCE_LR}) | EarlyStop(p={PATIENCE_EARLY_STOP})"
)
plt.figtext(0.5, 0.01, params_text, wrap=True, horizontalalignment='center',
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
plt.subplots_adjust(bottom=0.25)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join(RESULTS_DIR, f"training_mobilenet_finetune_baseline_9_classes_{timestamp}.png")
plt.savefig(plot_path, bbox_inches='tight')
print(f"Plot saved as {plot_path}")