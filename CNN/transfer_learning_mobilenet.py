import datetime
import os
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

# ==============================================================================
# 1. ГІПЕРПАРАМЕТРИ
# ==============================================================================
BASE_DIR = r"D:\projects\Diploma\transfer_learning_datasets_filtered_by_5"
RESULTS_DIR = r"D:\projects\Diploma\CNN\results_mobilenet"
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30

DROPOUT_RATE = 0.5  # Для Transfer Learning 0.5 є стандартом
LEARNING_RATE = 0.0001  # Малий крок, щоб акуратно підлаштувати верхівку

PATIENCE_EARLY_STOP = 15
PATIENCE_REDUCE_LR = 3
FACTOR_REDUCE_LR = 0.5
# ==============================================================================

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

print("Loading datasets...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical'
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(class_names)
print(f"Classes detected: {NUM_CLASSES}")


# ==============================================================================
# 2. АРХІТЕКТУРА TRANSFER LEARNING (MobileNetV2)
# ==============================================================================
def build_transfer_model():
    # Завантажуємо MobileNetV2 без верхнього шару (include_top=False)
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    # Заморожуємо ваги базової моделі (щоб не зруйнувати її знання)
    base_model.trainable = False
    # Перейменовуємо для легкого доступу при Grad-CAM
    base_model._name = "mobilenet_base"

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # MobileNetV2 вимагає специфічної обробки пікселів (від -1 до 1)
    x = preprocess_input(inputs)

    # Пропускаємо через базову модель (training=False обов'язково для збереження логіки BatchNorm)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="dense_output")(x)

    model = models.Model(inputs, outputs)
    return model


model = build_transfer_model()
model.summary()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================================================================
# 3. НАВЧАННЯ
# ==============================================================================
checkpoint_cb = callbacks.ModelCheckpoint(
    "results_mobilenet/mobilenet_9_classes_frozen.keras", save_best_only=True, monitor='val_accuracy'
)
early_stopping_cb = callbacks.EarlyStopping(
    patience=PATIENCE_EARLY_STOP, monitor='val_loss', restore_best_weights=True
)
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=FACTOR_REDUCE_LR, patience=PATIENCE_REDUCE_LR, min_lr=0.00001, verbose=1
)

print("Starting Transfer Learning...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
)

# ==============================================================================
# 4. ПОБУДОВА ГРАФІКІВ
# ==============================================================================
plt.figure(figsize=(12, 5.5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over Epochs (MobileNetV2)')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs (MobileNetV2)')

params_text = (
    f"TRANSFER LEARNING: MobileNetV2 (Frozen Base)\n"
    f"Img Size: {IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}\n"
    f"Opt: Adam(lr={LEARNING_RATE}) | Dropout: {DROPOUT_RATE}\n"
    f"Callbacks: ReduceLR(p={PATIENCE_REDUCE_LR}, f={FACTOR_REDUCE_LR}) | EarlyStop(p={PATIENCE_EARLY_STOP})"
)
plt.figtext(0.5, 0.01, params_text, wrap=True, horizontalalignment='center',
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
plt.subplots_adjust(bottom=0.25)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join(RESULTS_DIR, f"training_mobilenet_9_classes{timestamp}.png")
plt.savefig(plot_path, bbox_inches='tight')
print(f"Plot saved as {plot_path}")

# ==============================================================================
# 5. GRAD-CAM АНАЛІЗ (Візуалізація мислення мережі)
# ==============================================================================
print("\nGenerating Grad-CAM heatmaps for sample validation images...")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    base_model = model.get_layer("mobilenetv2_1.00_224")
    cam_base_model = tf.keras.Model(base_model.inputs, base_model.get_layer(last_conv_layer_name).output)

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)
    conv_outputs = cam_base_model(x)

    x_top = model.get_layer("global_avg_pool")(conv_outputs)
    x_top = model.get_layer("dropout")(x_top)
    predictions = model.get_layer("dense_output")(x_top)

    grad_model = tf.keras.Model(inputs, [conv_outputs, predictions])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy()


def save_gradcam(img_array, heatmap, filepath, alpha=0.5):
    img = img_array[0].astype("uint8")
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(filepath, superimposed_img)


# Беремо один батч із валідації і робимо теплові карти для перших 5 фото
for images, labels in val_dataset.take(1):
    for i in range(5):
        img_array = tf.expand_dims(images[i], axis=0)
        true_class_idx = tf.argmax(labels[i]).numpy()
        true_class_name = class_names[true_class_idx]

        heatmap, pred_class_idx = make_gradcam_heatmap(img_array, model)
        pred_class_name = class_names[pred_class_idx]

        status = "CORRECT" if true_class_idx == pred_class_idx else "WRONG"
        filename = f"cam_{i}_{status}_True-{true_class_name}_Pred-{pred_class_name}.jpg"

        save_gradcam(img_array.numpy(), heatmap, os.path.join(GRADCAM_DIR, filename))

print(f"Grad-CAM heatmaps saved to {GRADCAM_DIR}")