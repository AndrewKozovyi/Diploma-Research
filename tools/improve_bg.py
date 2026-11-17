import os
import csv
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

SRC_DIR = r"D:\projects\Diploma\datasets\merged_dataset_filtered"
DEST_DIR = r"D:\projects\Diploma\datasets\merged_dataset_filtered_improved_bg"

CORE_ALPHA = 128
MIN_COMPONENT_SHARE = 0.03
EDGE_SIGMA = 0.7
PAD_SHARE = 0.05
TINY_SIDE = 224


def is_premultiplied(rgba):
    alpha = rgba[..., 3]
    semi = (alpha > 16) & (alpha < 128)
    if semi.sum() < 200:
        return True
    return (rgba[..., :3].max(axis=2)[semi] > alpha[semi] + 3).mean() < 0.02


def true_color(rgba, premultiplied):
    rgb = rgba[..., :3].astype(np.float32)
    if not premultiplied:
        return rgb
    alpha = rgba[..., 3:4].astype(np.float32)
    return np.clip(rgb * 255.0 / np.maximum(alpha, 1.0), 0, 255)


def clean_mask(alpha):
    core = alpha >= CORE_ALPHA
    labels, count = ndimage.label(core, structure=np.ones((3, 3)))
    if count == 0:
        return core, 0
    sizes = ndimage.sum(core, labels, range(1, count + 1))
    keep = np.flatnonzero(sizes >= sizes.max() * MIN_COMPONENT_SHARE) + 1
    return np.isin(labels, keep), count - len(keep)


def process(path):
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    premultiplied = is_premultiplied(rgba)
    color = true_color(rgba, premultiplied)

    mask, dropped = clean_mask(rgba[..., 3])
    if not mask.any():
        return None, {"premultiplied": premultiplied, "dropped_parts": dropped}

    soft = ndimage.gaussian_filter(mask.astype(np.float32), EDGE_SIGMA)
    soft[mask] = 1.0
    soft[soft < 0.02] = 0.0

    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    pad = max(4, int(round(max(np.ptp(ys), np.ptp(xs)) * PAD_SHARE)))
    top, bottom = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    left, right = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)

    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.round(color * soft[..., None]).astype(np.uint8)
    out[..., 3] = np.round(soft * 255).astype(np.uint8)
    out = out[top:bottom, left:right]

    old_fg = (rgba[..., 3] >= CORE_ALPHA).sum()
    info = {
        "premultiplied": premultiplied,
        "dropped_parts": dropped,
        "old_size": f"{w}x{h}",
        "new_size": f"{out.shape[1]}x{out.shape[0]}",
        "kept_fg": round(mask.sum() / max(old_fg, 1), 3),
        "filled_semi": round(((rgba[..., 3] < 224) & mask).sum() / max(mask.sum(), 1), 3),
        "tiny": max(out.shape[:2]) < TINY_SIDE,
    }
    return Image.fromarray(out, "RGBA"), info


def main():
    if os.path.exists(DEST_DIR) and os.listdir(DEST_DIR):
        sys.exit(f"{DEST_DIR} already exists and is not empty")
    rows = []
    for cls in sorted(os.listdir(SRC_DIR)):
        src_cls = os.path.join(SRC_DIR, cls)
        dst_cls = os.path.join(DEST_DIR, cls)
        os.makedirs(dst_cls, exist_ok=True)
        for name in sorted(os.listdir(src_cls)):
            if "_dup" in name:
                rows.append({"class": cls, "file": name, "status": "skipped_duplicate"})
                continue
            image, info = process(os.path.join(src_cls, name))
            if image is None:
                rows.append({"class": cls, "file": name, "status": "empty_mask", **info})
                continue
            image.save(os.path.join(dst_cls, name), optimize=True)
            rows.append({"class": cls, "file": name, "status": "ok", **info})
        print(f"{cls}: {sum(r['class'] == cls and r['status'] == 'ok' for r in rows)}")

    fields = ["class", "file", "status", "premultiplied", "dropped_parts", "old_size", "new_size",
              "kept_fg", "filled_semi", "tiny"]
    with open(os.path.join(DEST_DIR, "report.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"saved {len(ok)}, skipped duplicates {sum(r['status'] == 'skipped_duplicate' for r in rows)}, "
          f"empty {sum(r['status'] == 'empty_mask' for r in rows)}, tiny {sum(r['tiny'] for r in ok)}, "
          f"with dropped parts {sum(r['dropped_parts'] > 0 for r in ok)}")


if __name__ == "__main__":
    main()
