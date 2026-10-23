import os
import shutil

# ==============================================================================
# НАЛАШТУВАННЯ
# ==============================================================================
# Головна папка, де лежать оригінальні train, val, test
SOURCE_BASE_DIR = r"D:\projects\Diploma\transfer_learning_datasets"

# Нова папка для очищеного датасету
DEST_BASE_DIR = r"D:\projects\Diploma\transfer_learning_datasets_filtered_by_5"

# Мінімальна кількість фото у папці val, щоб клас пройшов фільтрацію
MIN_VAL_IMAGES = 5

# ==============================================================================

def filter_dataset(source_base, dest_base, min_val_images):
    source_val_dir = os.path.join(source_base, "val")

    if not os.path.exists(source_val_dir):
        print(f"❌ Папку {source_val_dir} не знайдено!")
        return

    print("🔍 Аналізуємо оригінальну папку val...")
    valid_classes = []

    # Отримуємо всі класи
    all_classes = [d for d in os.listdir(source_val_dir) if os.path.isdir(os.path.join(source_val_dir, d))]

    for cls in all_classes:
        cls_dir = os.path.join(source_val_dir, cls)
        num_files = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])

        if num_files >= min_val_images:
            valid_classes.append(cls)
            print(f"✅ {cls}: {num_files} фото (Залишаємо)")
        else:
            print(f"🚫 {cls}: {num_files} фото (Відкидаємо)")

    print(f"\n📊 Разом залишається класів: {len(valid_classes)} з {len(all_classes)}\n")

    # Копіюємо відфільтровані класи для train, val та test
    folders_to_copy = ["train", "val", "test", "train_50", "train_100", "train_250"]

    for folder_name in folders_to_copy:
        src_folder = os.path.join(source_base, folder_name)
        dest_folder = os.path.join(dest_base, folder_name)

        if not os.path.exists(src_folder):
            print(f"⚠️ Папка {src_folder} не існує, пропускаємо.")
            continue

        print(f"📂 Копіюємо {folder_name}...")
        os.makedirs(dest_folder, exist_ok=True)

        for cls in valid_classes:
            src_cls_dir = os.path.join(src_folder, cls)
            dest_cls_dir = os.path.join(dest_folder, cls)

            # ДОДАНО: Створюємо папку класу, якщо її ще немає
            os.makedirs(dest_cls_dir, exist_ok=True)

            if os.path.exists(src_cls_dir):
                for item in os.listdir(src_cls_dir):
                    s = os.path.join(src_cls_dir, item)
                    d = os.path.join(dest_cls_dir, item)
                    if not os.path.exists(d):  # Скіпає файл, якщо він вже є
                        shutil.copy2(s, d)

        print(f"   Готово для {folder_name}.")

if __name__ == "__main__":
    filter_dataset(SOURCE_BASE_DIR, DEST_BASE_DIR, MIN_VAL_IMAGES)
    print(f"\n🎉 Фільтрацію завершено! Новий датасет збережено у:\n{DEST_BASE_DIR}")