import os
import random
import shutil
import cv2

# ==============================================================================
# НАЛАШТУВАННЯ
# ==============================================================================
SOURCE_TRAIN_DIR = r"/transfer_learning_datasets/train_150"
DEST_TRAIN_DIR = r"/transfer_learning_datasets/train"
TARGET_COUNT = 250


# ==============================================================================

def augment_image(image_path, save_path):
    # Зчитуємо зображення
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 1. Випадкове віддзеркалення (горизонтальне або вертикальне)
    if random.choice([True, False]):
        img = cv2.flip(img, 1)
    if random.choice([True, False]):
        img = cv2.flip(img, 0)

        # 2. Випадковий поворот (від -45 до 45 градусів)
    angle = random.randint(-45, 45)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # BORDER_REFLECT заповнює пусті кути після повороту сусідніми пікселями
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Випадкова зміна яскравості та контрасту
    alpha = random.uniform(0.8, 1.2)  # контраст
    beta = random.randint(-20, 20)  # яскравість
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Зберігаємо нове зображення
    cv2.imwrite(save_path, img)
    return True


def make_oversampled_dataset(source_dir, dest_dir, target_count):
    if not os.path.exists(source_dir):
        print(f"❌ Папку {source_dir} не знайдено!")
        return

    os.makedirs(dest_dir, exist_ok=True)
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in class_names:
        src_cls_dir = os.path.join(source_dir, cls)
        dest_cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(dest_cls_dir, exist_ok=True)

        # 1. Копіюємо всі існуючі 150 файлів
        existing_files = os.listdir(src_cls_dir)
        for f in existing_files:
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(dest_cls_dir, f))

        current_count = len(existing_files)
        print(f"⏳ Клас {cls}: скопійовано {current_count}. Генеруємо ще {target_count - current_count}...")

        # 2. Генеруємо нові, поки не дійдемо до 250
        added = 0
        while current_count + added < target_count:
            # Обираємо випадкове зображення з оригіналів для мутації
            src_file = random.choice(existing_files)
            src_path = os.path.join(src_cls_dir, src_file)

            new_file_name = f"aug_{added}_{src_file}"
            dest_path = os.path.join(dest_cls_dir, new_file_name)

            if augment_image(src_path, dest_path):
                added += 1

        print(f"✅ Клас {cls}: готово (всього {current_count + added} файлів).")


if __name__ == "__main__":
    random.seed(42)
    print("Починаємо створення датасету на 250 фото...\n")
    make_oversampled_dataset(SOURCE_TRAIN_DIR, DEST_TRAIN_DIR, TARGET_COUNT)
    print("\n🎉 Генерацію успішно завершено!")