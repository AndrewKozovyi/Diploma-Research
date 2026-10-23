import cv2
import os
import numpy as np
from tqdm import tqdm


def split_bones_from_transparent_png(source_base=r"D:\projects\Diploma\yale_dataset_nobg\Train_Data",
                                     target_base=r"D:\projects\Diploma\yale_dataset_split"):
    print(f"Starting bone separation process...\nSource: {source_base}\nTarget: {target_base}")

    # Мінімальна площа в пікселях. Усе, що менше — вважається сміттям (артефакти rembg).
    MIN_AREA = 5000

    # Відступ у пікселях навколо вирізаної кістки, щоб вона не торкалася країв
    PADDING = 30

    files_to_process = []
    for root, _, files in os.walk(source_base):
        for file in files:
            if file.lower().endswith('.png'):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("No PNG images found for processing.")
        return

    print(f"Total photos to process: {len(files_to_process)}")

    for src_img_path in tqdm(files_to_process, desc="Splitting bones", unit="file"):
        try:
            # 1. Завантажуємо зображення, обов'язково зберігаючи Alpha-канал (IMREAD_UNCHANGED)
            img = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)

            # Перевіряємо, чи зображення дійсно має 4 канали (RGBA)
            if img is None or len(img.shape) < 3 or img.shape[2] != 4:
                print(f"\nSkipping {src_img_path}: Not a valid 4-channel PNG.")
                continue

            # 2. Витягуємо Alpha-канал як готову маску
            alpha_channel = img[:, :, 3]

            # 3. Знаходимо контури на Alpha-каналі
            # RETR_EXTERNAL шукає тільки зовнішні межі (ігнорує дірки всередині кістки)
            contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bone_index = 0
            img_height, img_width = img.shape[:2]

            # Створюємо папку для збереження, дублюючи структуру директорій
            rel_path = os.path.relpath(src_img_path, source_base)
            dest_dir = os.path.join(target_base, os.path.dirname(rel_path))
            os.makedirs(dest_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(src_img_path))[0]

            # 4. Перебираємо всі знайдені контури
            for contour in contours:
                area = cv2.contourArea(contour)

                # Відкидаємо дрібний шум
                if area > MIN_AREA:
                    # Отримуємо координати прямокутника (bounding box), який охоплює контур
                    x, y, w, h = cv2.boundingRect(contour)

                    # Додаємо відступи (PADDING), слідкуючи, щоб не вийти за межі оригінального фото
                    x1 = max(0, x - PADDING)
                    y1 = max(0, y - PADDING)
                    x2 = min(img_width, x + w + PADDING)
                    y2 = min(img_height, y + h + PADDING)

                    # 5. Вирізаємо частину оригінального зображення (усі 4 канали)
                    cropped_bone = img[y1:y2, x1:x2]

                    # Формуємо нове ім'я файлу: "original_name_bone_0.png", "original_name_bone_1.png", і т.д.
                    out_filename = f"{base_name}_bone_{bone_index}.png"
                    out_path = os.path.join(dest_dir, out_filename)

                    # 6. Зберігаємо вирізану кістку
                    cv2.imwrite(out_path, cropped_bone)
                    bone_index += 1

        except Exception as e:
            print(f"\nError processing file {src_img_path}: {str(e)}")

    print("\nBone separation completed successfully!")


if __name__ == "__main__":
    split_bones_from_transparent_png()