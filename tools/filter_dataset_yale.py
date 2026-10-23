import os
import shutil


def sort_dataset(source_dir="parsed_dataset_yale", target_dir="yale_structured_dataset"):
    print(f"Сортування з {source_dir} у {target_dir}...")

    # Створюємо лише дві робочі папки
    train_dir = os.path.join(target_dir, "Train_Data")
    casts_dir = os.path.join(target_dir, "Casts_Data")

    for d in [train_dir, casts_dir]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(source_dir):
        print(f"[-] Папку {source_dir} не знайдено!")
        return

    stats = {"train": 0, "casts": 0}

    for folder_name in os.listdir(source_dir):
        source_folder = os.path.join(source_dir, folder_name)

        if not os.path.isdir(source_folder):
            continue

        lower_name = folder_name.lower()

        # 1. Сміття (undet, cf, aff) - просто пропускаємо цикл, нікуди не копіюємо
        if "undet" in lower_name or "cf." in lower_name or "aff." in lower_name:
            continue

        # 2. Зліпки (cast) - в окрему папку
        if "cast" in lower_name:
            dest_folder = os.path.join(casts_dir, folder_name)
            category = "casts"

        # 3. Чисті дані - ієрархія для навчання
        else:
            parts = folder_name.split('_')
            genus = parts[0]

            if len(parts) == 1 or "sp" in [p.lower() for p in parts]:
                species_folder = f"{genus}_sp"
            else:
                species_folder = folder_name

            dest_folder = os.path.join(train_dir, genus, species_folder)
            category = "train"

        os.makedirs(dest_folder, exist_ok=True)

        for file in os.listdir(source_folder):
            src_file = os.path.join(source_folder, file)
            dest_file = os.path.join(dest_folder, file)

            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
                stats[category] += 1

    print("\nСортування завершено!")
    print(f"Статистика:")
    print(f"  - Перенесено в Train_Data: {stats['train']}")
    print(f"  - Перенесено в Casts_Data: {stats['casts']}")
    print(f"  - Сміття проігноровано.")


if __name__ == "__main__":
    sort_dataset()