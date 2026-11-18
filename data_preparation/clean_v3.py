import os
import sys
import importlib.util

import numpy as np
from PIL import Image

DATA_DIR = r"D:\projects\Diploma\datasets\merged_dataset_filtered_v3"
REMOVE_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3_removed.txt")

spec = importlib.util.spec_from_file_location(
    "improve_bg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "improve_bg.py"))
improve_bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(improve_bg)


def fix_background():
    fixed = 0
    for cls in sorted(os.listdir(DATA_DIR)):
        for name in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
            path = os.path.join(DATA_DIR, cls, name)
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            alpha = rgba[..., 3]
            hidden = rgba[..., :3][alpha == 0]
            if improve_bg.is_premultiplied(rgba) and (hidden.size == 0 or hidden.max() == 0):
                continue
            color = improve_bg.true_color(rgba, improve_bg.is_premultiplied(rgba))
            rgb = np.round(color * alpha[..., None] / 255.0).astype(np.uint8)
            Image.fromarray(np.dstack([rgb, alpha])).save(path)
            fixed += 1
    print("background fixed:", fixed)


def remove_listed():
    removed = 0
    with open(REMOVE_LIST, encoding="utf-8") as fh:
        for line in fh:
            rel = line.split("#")[0].strip()
            if rel and os.path.exists(os.path.join(DATA_DIR, rel)):
                os.remove(os.path.join(DATA_DIR, rel))
                removed += 1
    print("removed:", removed)


if __name__ == "__main__":
    {"background": fix_background, "remove": remove_listed}[sys.argv[1]]()
