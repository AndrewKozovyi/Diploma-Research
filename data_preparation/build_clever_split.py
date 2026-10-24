"""
Будує датасет D:\\projects\\Diploma\\datasets\\clever_split з груповим розбиттям БЕЗ витоку.

Що робить:
  1. Збирає всі ОРИГІНАЛИ (без *_aug_*) з train/val/test старого датасету filtered_by_5.
  2. Прибирає точні дублікати (однаковий MD5) - лишається одна копія.
  3. Ділить на train/val/test 70/15/15 по ГРУПАХ: усі фото одного запису GBIF/Yale потрапляють
     лише в один спліт (для Monoclonius, де весь клас - один запис, група = окреме фото запису).
     Великі записи завжди йдуть у train, щоб val/test складалися з багатьох різних екземплярів.
  4. Балансує train до 150 фото на клас тією ж аугментацією, що й 2_augment_train.py.
  5. Пише split_manifest.csv (файл, клас, спліт, група) і перевіряє, що жодна група
     не потрапила в два спліти.

Запуск:  python build_clever_split.py
Старий датасет не змінюється.
"""
import os
import csv
import shutil
import random
import hashlib
from collections import defaultdict, Counter

import cv2

from clever_split_core import is_aug, split_groups

SRC_DIR = r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_filtered_by_5"
DEST_DIR = r"D:\projects\Diploma\datasets\clever_split"
TARGET_TRAIN_PER_CLASS = 150
SEED = 42


# --- та сама аугментація, що в data_preparation/2_augment_train.py -------------------------
def augment_image(image):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
    angle = random.randint(-45, 45)
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    random.seed(SEED)
    if os.path.exists(DEST_DIR) and os.listdir(DEST_DIR):
        raise SystemExit(f"Папка {DEST_DIR} вже існує і не порожня. Видаліть її або змініть DEST_DIR.")

    # 1. Оригінали
    originals = {}  # (cls, filename) -> path
    for split in ("train", "val", "test"):
        root = os.path.join(SRC_DIR, split)
        for cls in sorted(os.listdir(root)):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")) and not is_aug(f):
                    originals.setdefault((cls, f), os.path.join(cdir, f))
    print(f"Оригіналів знайдено: {len(originals)}")

    # 2. Точні дублікати
    by_hash = defaultdict(list)
    for key, p in sorted(originals.items()):
        by_hash[md5_of(p)].append(key)
    removed = []
    for keys in by_hash.values():
        if len(keys) > 1:
            # лишаємо файл без суфікса _dup (або перший за алфавітом)
            keys = sorted(keys, key=lambda k: ("_dup" in k[1], k[1]))
            for k in keys[1:]:
                removed.append((k, keys[0]))
                originals.pop(k)
    print(f"Прибрано точних дублікатів: {len(removed)}")
    for (cls, f), (cls0, f0) in removed:
        note = "  (!) клас не збігається" if cls != cls0 else ""
        print(f"   - {cls}/{f}  ==  {cls0}/{f0}{note}")

    # 3. Групове розбиття
    files_by_class = defaultdict(list)
    for cls, f in originals:
        files_by_class[cls].append(f)
    assignment, single = split_groups(files_by_class, seed=SEED)
    if single:
        print(f"Класи з одним записом (групування по фото): {sorted(single)}")

    group_splits = defaultdict(set)
    for (cls, f), (split, g) in assignment.items():
        group_splits[g].add(split)
    leaks = [g for g, s in group_splits.items() if len(s) > 1]
    assert not leaks, f"Витік груп між сплітами: {leaks[:5]}"

    # 4. Копіювання
    rows = []
    for (cls, f), (split, g) in sorted(assignment.items()):
        dst_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(originals[(cls, f)], os.path.join(dst_dir, f))
        rows.append({"class": cls, "file": f, "split": split, "group": g})

    # 5. Балансування train аугментацією
    for cls in sorted(files_by_class):
        cdir = os.path.join(DEST_DIR, "train", cls)
        files = sorted(os.listdir(cdir))
        need = TARGET_TRAIN_PER_CLASS - len(files)
        made = 0
        while made < need:
            src = random.choice(files)
            img = cv2.imread(os.path.join(cdir, src))
            if img is None:
                continue
            base, ext = os.path.splitext(src)
            cv2.imwrite(os.path.join(cdir, f"{base}_aug_{made}{ext}"), augment_image(img))
            made += 1

    # 6. Маніфест і статистика
    with open(os.path.join(DEST_DIR, "split_manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "file", "split", "group"])
        w.writeheader()
        w.writerows(rows)

    cnt = defaultdict(Counter)
    grp = defaultdict(lambda: defaultdict(set))
    for r in rows:
        cnt[r["class"]][r["split"]] += 1
        grp[r["class"]][r["split"]].add(r["group"])
    print(f"\n{'клас':14} {'train':>6} {'val':>4} {'test':>4}   записів train/val/test")
    for cls in sorted(cnt):
        c, g = cnt[cls], grp[cls]
        print(f"{cls:14} {c['train']:6} {c['val']:4} {c['test']:4}   "
              f"{len(g['train'])}/{len(g['val'])}/{len(g['test'])}")
    print(f"\nГотово. Train збалансовано до {TARGET_TRAIN_PER_CLASS}/клас. Датасет: {DEST_DIR}")


if __name__ == "__main__":
    main()
