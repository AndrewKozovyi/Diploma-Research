"""
Пошук дублікатів між train / val / test (тільки оригінали, *_aug* пропускаються).
  1) точні дублікати  - однаковий MD5 файлу;
  2) майже-дублікати - однаковий перцептивний хеш (dHash 16x16): ресейв, інший формат, легке стиснення.
Запуск:  python find_duplicates.py
"""
import os, re, hashlib
from collections import defaultdict
from PIL import Image

DATA_DIR = r"D:\projects\Diploma\datasets\transfer_learning\transfer_learning_datasets_filtered_by_5"
SPLITS = ["train", "val", "test"]
HASH_MAX_DIST = 4   # бітів відмінності з 256, при яких вважаємо фото майже однаковими


def is_aug(name):
    return "_aug_" in name or re.match(r"^aug_\d+_", name)


def dhash(path, size=16):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


files = []
for split in SPLITS:
    root = os.path.join(DATA_DIR, split)
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and not is_aug(f):
                files.append((split, cls, f, os.path.join(cdir, f)))
print(f"Оригіналів: {len(files)}")

md5 = defaultdict(list)
hashes = []
for split, cls, f, p in files:
    with open(p, "rb") as fh:
        md5[hashlib.md5(fh.read()).hexdigest()].append(f"{split}/{cls}/{f}")
    hashes.append((dhash(p), f"{split}/{cls}/{f}"))

print("\n=== ТОЧНІ ДУБЛІКАТИ (MD5) ===")
exact = [g for g in md5.values() if len(g) > 1]
for g in exact:
    cross = len({x.split('/')[0] for x in g}) > 1
    print(("[МІЖ СПЛІТАМИ] " if cross else "[в одному спліті] ") + "  |  ".join(g))
print(f"Груп: {len(exact)}, з них між сплітами: {sum(len({x.split('/')[0] for x in g})>1 for g in exact)}")

print(f"\n=== МАЙЖЕ-ДУБЛІКАТИ (dHash, відстань <= {HASH_MAX_DIST}) ===")
exact_pairs = {frozenset(g) for g in exact}
n = 0
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        d = bin(hashes[i][0] ^ hashes[j][0]).count("1")
        if d <= HASH_MAX_DIST:
            a, b = hashes[i][1], hashes[j][1]
            if any({a, b} <= g for g in exact_pairs):
                continue
            n += 1
            tag = "[МІЖ СПЛІТАМИ]" if a.split('/')[0] != b.split('/')[0] else "[в одному спліті]"
            print(f"{tag} d={d}  {a}  |  {b}")
print(f"Пар: {n}  (переглянь очима - схожі ракурси однієї кістки теж можуть потрапити сюди)")
