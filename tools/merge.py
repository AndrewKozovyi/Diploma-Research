import os
import shutil

# Головна папка з твоїм датасетом
DATASET_PATH = r"D:\projects\Diploma\merged_dataset"

# 1. СПЕЦІАЛЬНІ ПРАВИЛА (Виправляємо помилки таксономії GBIF)
# Формат: "Звідки (Відносний шлях)" : "Куди (Нова папка Роду)"
SPECIAL_MOVES = {
    r"Ceratopsidae\Triceratops_prorsus": r"Triceratops",
    r"Ceratopsidae\Triceratops_serratus": r"Triceratops",
    r"Ceratopsidae\Torosaurus_utahensis": r"Torosaurus",
    r"Ceratopsidae\Monoclonius_belli": r"Monoclonius",
    r"Tyrannosauridae\Tyrannosaurus_rex": r"Tyrannosaurus",
    r"Edmontosaurus\Claosaurus_annectens": r"Edmontosaurus",
    r"Paronychodon sp\Paronychodon sp_sp": r"Paronychodon",
    r"Dryosaurus\Laosaurus_altus": r"Dryosaurus"
}


def move_files(src_dir, dest_dir):
    """Переміщує всі файли з src_dir у dest_dir із захистом від перезапису"""
    if not os.path.exists(src_dir):
        return 0

    os.makedirs(dest_dir, exist_ok=True)
    count = 0

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        if os.path.isfile(src_path):
            # Захист від однакових імен файлів
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(item)
                dest_path = os.path.join(dest_dir, f"{base}_dup{ext}")

            shutil.move(src_path, dest_path)
            count += 1

    return count


def flatten_and_merge_dataset():
    print("🚀 Старт об'єднання та нормалізації датасету...\n")

    # КРОК 1: Виконуємо спеціальні переміщення
    print("🔄 1. Виправлення таксономічної плутанини...")
    for src_rel, dest_rel in SPECIAL_MOVES.items():
        src_full = os.path.join(DATASET_PATH, src_rel)
        dest_full = os.path.join(DATASET_PATH, dest_rel)

        moved = move_files(src_full, dest_full)
        if moved > 0:
            print(f"  [+] Перенесено {moved} файлів: {src_rel} -> {dest_rel}")
            # Видаляємо порожню папку звідки забрали файли
            if not os.listdir(src_full):
                os.rmdir(src_full)

    # КРОК 2: Схлопування підвидів у Родові папки
    print("\n🔄 2. Об'єднання всіх підвидів (sp, stenops, fragilis...) у загальні папки Роду...")
    for genus_folder in os.listdir(DATASET_PATH):
        genus_path = os.path.join(DATASET_PATH, genus_folder)

        if not os.path.isdir(genus_path):
            continue

        # Проходимо по всіх підпапках (наприклад, Stegosaurus_stenops всередині Stegosaurus)
        for species_folder in os.listdir(genus_path):
            species_path = os.path.join(genus_path, species_folder)

            if os.path.isdir(species_path):
                # Переміщуємо всі файли з підпапки на рівень вище (в папку Роду)
                moved = move_files(species_path, genus_path)
                if moved > 0:
                    print(f"  [+] Злито {moved} файлів: {species_folder} -> {genus_folder}")

                # Видаляємо підпапку, якщо вона стала порожньою
                if not os.listdir(species_path):
                    os.rmdir(species_path)

    # КРОК 3: Видалення абсолютно порожніх батьківських папок
    print("\n🧹 3. Очищення порожніх директорій...")
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path) and not os.listdir(folder_path):
            os.rmdir(folder_path)
            print(f"  [-] Видалено порожню папку: {folder}")

    print("\n✅ Готово! Усі підвиди об'єднані, структура плоска (один рівень папок).")


if __name__ == "__main__":
    flatten_and_merge_dataset()