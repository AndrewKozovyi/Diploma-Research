"""
Копія датасету, де кожне фото доповнене чорним до квадрата (кістка по центру).
Класифікатор при валідації/передбаченні робить центральний квадратний кроп, тому у видовжених кісток
відрізаються кінці; після доповнення кроп нічого не відрізає. Фон фото і так чорний.

Запуск:  python make_square.py <папка_датасету>   ->  <папка_датасету>_sq
"""
import os
import sys

import cv2


def main(src):
    dst = src.rstrip("\\/") + "_sq"
    if os.path.exists(dst) and os.listdir(dst):
        sys.exit(f"{dst} already exists and is not empty")
    count = 0
    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(src, split))):
            dst_dir = os.path.join(dst, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for name in sorted(os.listdir(os.path.join(src, split, cls))):
                image = cv2.imread(os.path.join(src, split, cls, name), cv2.IMREAD_COLOR)
                h, w = image.shape[:2]
                side = max(h, w)
                top, left = (side - h) // 2, (side - w) // 2
                image = cv2.copyMakeBorder(image, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)
                cv2.imwrite(os.path.join(dst_dir, os.path.splitext(name)[0] + ".png"), image)
                count += 1
    print(dst, count)


if __name__ == "__main__":
    main(sys.argv[1])
