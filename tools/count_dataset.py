import os


def count_dataset_images(target_dir):
    if not os.path.exists(target_dir):
        print(f"Помилка: Шлях не знайдено: {target_dir}")
        return

    print("Сканування директорій датасету...")
    total_images = 0
    class_counts = {}

    # Обходимо дерево папок
    for root, dirs, files in os.walk(target_dir):
        # Рахуємо лише PNG файли, які з'явилися після видалення фону
        png_files = [f for f in files if f.lower().endswith('.png')]

        if png_files:
            # Отримуємо відносний шлях, щоб бачити структуру класів
            rel_path = os.path.relpath(root, target_dir)
            class_counts[rel_path] = len(png_files)
            total_images += len(png_files)

    print("\nСтатистика розподілу даних за класами:")
    print("----------------------------------------")

    # Сортуємо за назвою класів для зручності читання
    for class_name, count in sorted(class_counts.items()):
        print(f"Клас: {class_name} -> {count} фото")

    print("----------------------------------------")
    print(f"Всього знайдено папок (класів): {len(class_counts)}")
    print(f"Загальна кількість PNG зображень: {total_images}")


if __name__ == "__main__":
    # Вказуємо шлях до очищених структурованих даних
    DATASET_PATH = r"D:\projects\Diploma\yale_dataset_nobg\Train_Data"
    count_dataset_images(DATASET_PATH)