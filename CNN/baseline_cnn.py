import datetime
import os
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
import matplotlib.pyplot as plt
from keras.optimizers import Adam, AdamW

# ==============================================================================
# 1. ГІПЕРПАРАМЕТРИ (Змінюй значення ТІЛЬКИ в цьому блоці)
# ==============================================================================
BASE_DIR = r"D:\projects\Diploma\dataset_more_images_all"
RESULTS_DIR = r"/CNN/results_baseline"

# Параметри даних
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30

# Параметри регуляризації та архітектури
DROPOUT_RATE = 0.8    # Відсоток вимкнених нейронів (0.4 - 0.8)
L2_REG_RATE = 0.00      # Штраф за великі ваги (постав 0.0, щоб вимкнути L2)

# Параметри навчання (Optimizer & Learning Rate)
OPTIMIZER_NAME = "Adam"  # Текст для графіка (зміни на "AdamW", якщо міняєш нижче)
LEARNING_RATE = 0.0001   # Стартовий крок навчання

# Параметри коллбеків (Зупинка та зменшення кроку)
PATIENCE_EARLY_STOP = 7 # Скільки епох чекати до повної зупинки
PATIENCE_REDUCE_LR = 3   # Скільки епох чекати до зменшення кроку
FACTOR_REDUCE_LR = 0.5   # У скільки разів зменшувати крок (0.5 = вдвічі)
# ==============================================================================

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

print("Loading datasets...")

# 2. Завантаження даних
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(class_names)
print("Classes detected: {}".format(NUM_CLASSES))

# 3. Створення базової архітектури CNN
def build_baseline_model():
    l2_reg = regularizers.l2(L2_REG_RATE)

    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        # Блок 1
        layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        # Блок 2
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        # Блок 3
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        # Блок 4
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(64, kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),

        # Динамічний Dropout
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_baseline_model()
model.summary()

# 4. Компіляція моделі
# Якщо хочеш спробувати AdamW, закоментуй Adam і розкоментуй нижній рядок
custom_optimizer = Adam(learning_rate=LEARNING_RATE)
# custom_optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=0.001)

model.compile(
    optimizer=custom_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Коллбеки
checkpoint_cb = callbacks.ModelCheckpoint(
    "results_baseline/baseline_dino_model.keras",
    save_best_only=True,
    monitor='val_accuracy'
)

early_stopping_cb = callbacks.EarlyStopping(
    patience=PATIENCE_EARLY_STOP,
    monitor='val_loss',
    restore_best_weights=True
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=FACTOR_REDUCE_LR,
    patience=PATIENCE_REDUCE_LR,
    min_lr=0.00001,
    verbose=1
)

# 6. Навчання
print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
)
print("Training finished.")

# 7. Побудова графіків навчання
plt.figure(figsize=(12, 5.5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')

# Динамічне формування тексту з усіма змінними
params_text = (
    "HYPERPARAMETERS:\n"
    "Img Size: {} | Batch: {} | Epochs: {}\n"
    "Opt: {}(lr={}) | L2 Reg: {} | Dropout: {} | Blocks: 4 + BatchNorm\n"
    "Callbacks: ReduceLR(p={}, f={}) | EarlyStop(p={})"
).format(
    IMG_SIZE, BATCH_SIZE, EPOCHS,
    OPTIMIZER_NAME, LEARNING_RATE, L2_REG_RATE, DROPOUT_RATE,
    PATIENCE_REDUCE_LR, FACTOR_REDUCE_LR, PATIENCE_EARLY_STOP
)

plt.figtext(0.5, 0.01, params_text, wrap=True, horizontalalignment='center',
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

plt.subplots_adjust(bottom=0.25)

os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = "training_history_all_images_upd_{}.png".format(timestamp)
filepath = os.path.join(RESULTS_DIR, filename)

# Зберігаємо гарантовано з текстом (bbox_inches='tight')
plt.savefig(filepath, bbox_inches='tight')
print("Plot saved as {}".format(filepath))