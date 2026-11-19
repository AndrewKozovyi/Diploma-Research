import os
import sys
import random
import shutil
import hashlib
import importlib.util

import cv2

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else r"D:\projects\Diploma\datasets\merged_dataset_filtered"
DEST_DIR = sys.argv[2] if len(sys.argv) > 2 else r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_v2_filtered_by_5"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
MIN_VAL_IMAGES = 5
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

    splits, seen = {}, set()
    for cls in sorted(os.listdir(SRC_DIR)):
        files = []
        for name in sorted(os.listdir(os.path.join(SRC_DIR, cls)), key=lambda n: ("_dup" in n, n)):
            with open(os.path.join(SRC_DIR, cls, name), "rb") as fh:
                digest = hashlib.md5(fh.read()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                files.append(name)
        files.sort()
        random.shuffle(files)
        train_count = int(len(files) * TRAIN_RATIO)
        val_count = int(len(files) * VAL_RATIO)
        splits[cls] = {
            "train": files[:train_count],
            "val": files[train_count:train_count + val_count],
            "test": files[train_count + val_count:],
        }

    kept = [cls for cls, s in splits.items() if len(s["val"]) >= MIN_VAL_IMAGES]
    print(f"kept {len(kept)} of {len(splits)}: {kept}")

    for cls in kept:
        for split, files in splits[cls].items():
            dst = os.path.join(DEST_DIR, split, cls)
            os.makedirs(dst, exist_ok=True)
            for name in files:
                shutil.copy2(os.path.join(SRC_DIR, cls, name), os.path.join(dst, name))

        train_dir = os.path.join(DEST_DIR, "train", cls)
        originals = sorted(os.listdir(train_dir))
        for made in range(TARGET_TRAIN_PER_CLASS - len(originals)):
            src = random.choice(originals)
            image = cv2.imread(os.path.join(train_dir, src))
            base, ext = os.path.splitext(src)
            cv2.imwrite(os.path.join(train_dir, f"{base}_aug_{made}{ext}"), augment_train.augment_image(image))

    print(f"{'class':15} {'train':>5} {'orig':>5} {'val':>4} {'test':>4}")
    for cls in kept:
        s = splits[cls]
        total = len(os.listdir(os.path.join(DEST_DIR, "train", cls)))
        print(f"{cls:15} {total:5} {len(s['train']):5} {len(s['val']):4} {len(s['test']):4}")


if __name__ == "__main__":
    main()
