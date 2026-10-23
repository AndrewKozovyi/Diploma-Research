import os


def count_files_in_dataset(base_path):
    # Перевіряємо, чи існує папка
    if not os.path.exists(base_path):
        print(f"[-] Помилка: Шлях {base_path} не знайдено.")
        return

    print(f"📊 Статистика датасету: {base_path}\n" + "=" * 55)

    total_files = 0
    folder_stats = []

    # os.walk проходить по всіх папках і підпапках
    for root, dirs, files in os.walk(base_path):
        file_count = len(files)

        # Беремо відносний шлях для красивого виводу
        rel_path = os.path.relpath(root, base_path)
        if rel_path == '.':
            folder_name = "📁 Головна папка (корінь)"
        else:
            # Робимо відступи для підпапок
            level = rel_path.count(os.sep)
            indent = "  " * level
            folder_name = f"{indent}📁 {os.path.basename(root)}"

        # ЛОГІКА ВИВОДУ:
        if file_count > 0:
            # Якщо є файли — виводимо з підрахунком (вирівнюємо по 40 символу)
            folder_stats.append(f"{folder_name:<40} : {file_count} файлів")
            total_files += file_count
        elif len(dirs) > 0:
            # Якщо файлів немає, але є підпапки — виводимо просто як заголовок
            folder_stats.append(f"{folder_name}")

    # Виводимо результат
    for stat in folder_stats:
        print(stat)

    print("=" * 55)
    print(f"✅ Всього знайдено файлів: {total_files}")


if __name__ == "__main__":
    dataset_path = r"D:\projects\Diploma\transfer_learning_datasets_filtered_by_5"
    count_files_in_dataset(dataset_path)