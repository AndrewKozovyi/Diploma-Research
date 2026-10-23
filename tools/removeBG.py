import os
os.environ["ORT_LOGGING_LEVEL"] = "3"

print("⏳ Підключення штучного інтелекту...")
from rembg import remove
from PIL import Image
from tqdm import tqdm
import json
import shutil

def process_dataset(source_base="yale_structured_dataset", target_base="yale_dataset_nobg"):
    print(f"🚀 Запуск AI-видалення фону...\nДжерело: {source_base}\nРезультат: {target_base}")

    # Збираємо всі .jpg файли для обробки, щоб зробити гарний прогрес-бар
    files_to_process = []
    for root, _, files in os.walk(source_base):
        for file in files:
            if file.lower().endswith('.jpg'):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("[-] Зображень для обробки не знайдено.")
        return

    print(f"Всього знайдено фотографій: {len(files_to_process)}")

    # Використовуємо tqdm для візуалізації прогресу
    for src_img_path in tqdm(files_to_process, desc="Обробка фото", unit="шт"):
        try:
            # Формуємо шлях для збереження, зберігаючи структуру папок
            rel_path = os.path.relpath(src_img_path, source_base)
            dest_img_path = os.path.join(target_base, rel_path)

            # Змінюємо розширення з .jpg на .png для підтримки прозорості
            dest_img_path = os.path.splitext(dest_img_path)[0] + ".png"
            dest_dir = os.path.dirname(dest_img_path)

            os.makedirs(dest_dir, exist_ok=True)

            # Перевіряємо, чи ми вже обробили це фото (якщо скрипт перерветься)
            if os.path.exists(dest_img_path):
                continue

            # 1. Відкриваємо оригінал
            input_image = Image.open(src_img_path)

            # 2. Видаляємо фон (магія rembg)
            output_image = remove(input_image)

            # 3. Зберігаємо як PNG
            output_image.save(dest_img_path, format="PNG")

            # 4. Копіюємо відповідний JSON файл (метадані) поруч
            src_json_path = os.path.splitext(src_img_path)[0] + ".json"
            if os.path.exists(src_json_path):
                dest_json_path = os.path.join(dest_dir, os.path.basename(src_json_path))
                if not os.path.exists(dest_json_path):
                    shutil.copy2(src_json_path, dest_json_path)

        except Exception as e:
            print(f"\n[-] Помилка обробки файлу {src_img_path}: {str(e)}")

    print("\n✅ Усі фони успішно видалено! Датасет готовий.")


if __name__ == "__main__":
    process_dataset()