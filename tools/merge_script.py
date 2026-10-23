import os
import shutil
from tqdm import tqdm


def merge_datasets(dir1, dir2, output_dir):
    # Створюємо фінальну папку, якщо її ще немає
    os.makedirs(output_dir, exist_ok=True)

    def copy_files(source_dir, prefix):
        # Збираємо всі файли з вихідної папки
        files = []
        for root, _, filenames in os.walk(source_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        if not files:
            print(f"Warning: No files found in {source_dir}")
            return

        # Копіюємо файли з прогрес-баром
        for filepath in tqdm(files, desc=f"Copying from {prefix}", unit="file"):
            try:
                # Отримуємо відносний шлях, щоб зберегти внутрішню структуру підпапок
                rel_path = os.path.relpath(filepath, source_dir)
                dir_name, file_name = os.path.split(rel_path)

                # Додаємо префікс до назви файлу
                new_file_name = f"{prefix}_{file_name}"

                # Формуємо кінцевий шлях
                dest_path = os.path.join(output_dir, dir_name, new_file_name)

                # Перевіряємо, чи існує потрібна підпапка в цільовій директорії
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Копіюємо файл
                shutil.copy2(filepath, dest_path)
            except Exception as e:
                print(f"Error copying {filepath}: {str(e)}")

    print(f"Starting dataset merge into: {output_dir}")

    # Запускаємо копіювання для обох директорій
    copy_files(dir1, "parsed")
    copy_files(dir2, "yale")

    print("\nMerge completed successfully! Your combined dataset is ready.")


if __name__ == "__main__":
    # Твої папки-джерела
    dataset_1 = r"D:\projects\Diploma\parsed_dataset_split"
    dataset_2 = r"D:\projects\Diploma\yale_dataset_split"

    # Папка, де будуть зібрані всі файли
    merged_output = r"D:\projects\Diploma\merged_dataset"

    merge_datasets(dataset_1, dataset_2, merged_output)