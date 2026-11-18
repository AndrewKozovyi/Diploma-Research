import os
import sys

import cv2

MAX_SIDE = 512


def main(src):
    dst = src.rstrip("\\/") + "_r512"
    if os.path.exists(dst) and os.listdir(dst):
        sys.exit(f"{dst} already exists and is not empty")
    count = 0
    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(src, split))):
            src_dir = os.path.join(src, split, cls)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(dst, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for name in sorted(os.listdir(src_dir)):
                image = cv2.imread(os.path.join(src_dir, name), cv2.IMREAD_COLOR)
                h, w = image.shape[:2]
                scale = MAX_SIDE / max(h, w)
                if scale < 1:
                    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(dst_dir, os.path.splitext(name)[0] + ".png"), image)
                count += 1
    print(dst, count)


if __name__ == "__main__":
    main(sys.argv[1])
