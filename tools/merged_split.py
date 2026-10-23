import os
import shutil
import random
from tqdm import tqdm


def split_dataset(src_dir, dest_dir, min_files=11, train_ratio=0.70, val_ratio=0.15):
    print(f"🚀 Старт розбиття датасету з порогом {min_files} файлів...\n")

    # Створюємо головні папки для вибірок
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    kept_classes = 0
    skipped_classes = 0

    # Проходимо по всіх класах (папках)
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Збираємо всі файли класу
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        total_files = len(files)

        # Відсікаємо дрібні класи
        if total_files < min_files:
            skipped_classes += 1
            continue

        kept_classes += 1
        print(f"🔄 Обробка {class_name} ({total_files} файлів)...")

        # Випадково перемішуємо файли перед розбиттям
        random.shuffle(files)

        # Вираховуємо кількість файлів для кожної вибірки
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        # Усе, що залишилося (через округлення), йде в тест
        test_count = total_files - train_count - val_count

        # Розподіляємо файли по списках
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        # Функція для копіювання файлів у відповідну підпапку
        def copy_files(file_list, split_name):
            split_dir = os.path.join(dest_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for f in file_list:
                src_file = os.path.join(class_path, f)
                dest_file = os.path.join(split_dir, f)
                shutil.copy2(src_file, dest_file)

        # Копіюємо
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

    print("\n" + "=" * 50)
    print(f"✅ Розбиття успішно завершено!")
    print(f"📁 Новий датасет збережено у: {dest_dir}")
    print(f"🦖 Класів збережено: {kept_classes}")
    print(f"🗑 Класів відкинуто (менше {min_files} фото): {skipped_classes}")


if __name__ == "__main__":
    # Твоя очищена папка
    SOURCE_DATASET = r"D:\projects\Diploma\merged_dataset_filtered"

    # Нова папка, де будуть train, val, test
    DESTINATION_DATASET = r"D:\projects\Diploma\dataset_more_images_all"

    # Обов'язково фіксуємо random seed, щоб при повторному запуску розбиття було таким самим
    random.seed(42)

    split_dataset(SOURCE_DATASET, DESTINATION_DATASET, min_files=0)