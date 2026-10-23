import os
import cv2
import random
import numpy as np


def augment_image(image):
    # Віддзеркалення (горизонтальне та вертикальне)
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)

    # Випадковий поворот від -45 до +45 градусів
    angle = random.randint(-45, 45)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Зміна яскравості та контрасту (допоможе мережі ігнорувати кольорові написи на кістках)
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image


def balance_train_dataset(train_dir, target_count=150):
    print("Starting augmentation for training dataset...")
    print("Target files per class: {}".format(target_count))
    print("-" * 50)

    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(files)

        if current_count == 0:
            continue

        print("Processing {}: current count = {}".format(class_name, current_count))

        if current_count >= target_count:
            print("  Skipping, already has enough images.")
            continue

        images_needed = target_count - current_count
        generated_count = 0

        while generated_count < images_needed:
            random_file = random.choice(files)
            img_path = os.path.join(class_path, random_file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            aug_img = augment_image(img)

            base, ext = os.path.splitext(random_file)
            new_filename = "{}_aug_{}{}".format(base, generated_count, ext)
            new_filepath = os.path.join(class_path, new_filename)

            cv2.imwrite(new_filepath, aug_img)
            generated_count += 1

        print("  Added {} augmented images. Total: {}".format(generated_count, target_count))

    print("-" * 50)
    print("Augmentation complete. Train dataset is balanced!")


if __name__ == "__main__":
    # Вказуємо шлях ТІЛЬКИ до папки train
    TRAIN_DATASET_PATH = r"D:\projects\Diploma\transfer_learning_datasets_100_train\train"
    TARGET_IMAGES_PER_CLASS = 100

    balance_train_dataset(TRAIN_DATASET_PATH, TARGET_IMAGES_PER_CLASS)