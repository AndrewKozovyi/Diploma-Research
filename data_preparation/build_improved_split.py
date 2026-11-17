import os
import sys
import random
import shutil
import importlib.util

import cv2

SPLIT_DIR = r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_filtered_by_5"
IMAGES_DIR = r"D:\projects\Diploma\datasets\merged_dataset_filtered_improved_bg"
DEST_DIR = r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_filtered_by_5_improved_bg"
TARGET_TRAIN_PER_CLASS = 150
SEED = 42

spec = importlib.util.spec_from_file_location(
    "augment_train", os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_augment_train.py"))
augment_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augment_train)


def main():
    if os.path.exists(DEST_DIR) and os.listdir(DEST_DIR):
        sys.exit(f"{DEST_DIR} already exists and is not empty")
    random.seed(SEED)

    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(SPLIT_DIR, split))):
            src_cls = os.path.join(SPLIT_DIR, split, cls)
            if not os.path.isdir(src_cls):
                continue
            dst_cls = os.path.join(DEST_DIR, split, cls)
            os.makedirs(dst_cls, exist_ok=True)
            for name in sorted(os.listdir(src_cls)):
                if "_aug_" in name:
                    continue
                improved = os.path.join(IMAGES_DIR, cls, name.replace("_dup", ""))
                if not os.path.exists(improved):
                    sys.exit(f"missing improved image for {split}/{cls}/{name}")
                shutil.copy2(improved, os.path.join(dst_cls, name))

    train_dir = os.path.join(DEST_DIR, "train")
    for cls in sorted(os.listdir(train_dir)):
        cls_dir = os.path.join(train_dir, cls)
        files = sorted(os.listdir(cls_dir))
        for made in range(TARGET_TRAIN_PER_CLASS - len(files)):
            src = random.choice(files)
            image = cv2.imread(os.path.join(cls_dir, src))
            base, ext = os.path.splitext(src)
            cv2.imwrite(os.path.join(cls_dir, f"{base}_aug_{made}{ext}"), augment_train.augment_image(image))

    for split in ("train", "val", "test"):
        counts = {c: len(os.listdir(os.path.join(DEST_DIR, split, c))) for c in sorted(os.listdir(os.path.join(DEST_DIR, split)))}
        print(split, counts)


if __name__ == "__main__":
    main()
