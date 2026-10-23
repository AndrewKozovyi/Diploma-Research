import os
import random
import shutil

# ==============================================================================
# НАЛАШТУВАННЯ
# ==============================================================================
# Вкажи шлях до твоєї ПАПКИ-ЕТАЛОНУ (тієї, де зараз лежить по 150 фото на клас)
SOURCE_TRAIN_DIR = r"/transfer_learning_datasets/train_150"

# Шлях до НОВОЇ папки, куди скопіюються по 100 фото
DEST_TRAIN_DIR = r"/transfer_learning_datasets/train"

# Скільки файлів залишаємо
TARGET_COUNT = 50


# ==============================================================================

def make_subset(source_dir, dest_dir, target_count):
    if not os.path.exists(source_dir):
        print(f"❌ Помилка: Папку {source_dir} не знайдено!")
        return

    # Створюємо головну папку призначення, якщо її немає
    os.makedirs(dest_dir, exist_ok=True)

    # Отримуємо список усіх класів (папок)
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    print(f"Знайдено класів: {len(class_names)}. Починаю копіювання...\n")

    total_copied = 0

    for cls in class_names:
        src_cls_dir = os.path.join(source_dir, cls)
        dest_cls_dir = os.path.join(dest_dir, cls)

        # Створюємо папку класу в новому місці
        os.makedirs(dest_cls_dir, exist_ok=True)

        # Беремо всі файли з папки класу
        all_files = [f for f in os.listdir(src_cls_dir) if os.path.isfile(os.path.join(src_cls_dir, f))]

        # Перевіряємо, чи вистачає файлів
        if len(all_files) < target_count:
            print(f"⚠️ У класі {cls} лише {len(all_files)} файлів. Копіюю всі.")
            files_to_copy = all_files
        else:
            # Обираємо випадкові 100 файлів
            files_to_copy = random.sample(all_files, target_count)

        # Копіюємо обрані файли
        for file_name in files_to_copy:
            src_file_path = os.path.join(src_cls_dir, file_name)
            dest_file_path = os.path.join(dest_cls_dir, file_name)
            shutil.copy2(src_file_path, dest_file_path)
            total_copied += 1

        print(f"✅ Клас {cls}: скопійовано {len(files_to_copy)} з {len(all_files)}.")

    print(f"\n🎉 Готово! Успішно створено датасет на {total_copied} фото.")
    print(f"Нова папка: {dest_dir}")


if __name__ == "__main__":
    # Задаємо фіксований seed для відтворюваності експерименту (щоб випадковість була однаковою, якщо запустиш ще раз)
    random.seed(42)
    make_subset(SOURCE_TRAIN_DIR, DEST_TRAIN_DIR, TARGET_COUNT)