import os
import csv
import sys
import importlib.util

os.environ["ORT_LOGGING_LEVEL"] = "3"

import numpy as np
from PIL import Image
from scipy import ndimage
from rembg import remove, new_session

SRC_DIR = r"D:\projects\Diploma\not_prepared_images\parsed_dataset_new_institutions"
REMBG_DIR = r"D:\projects\Diploma\not_prepared_images\parsed_dataset_new_institutions_rembg"
DEST_DIR = r"D:\projects\Diploma\not_prepared_images\parsed_dataset_new_institutions_nobg"

CORE_ALPHA = 128
MIN_IMAGE_SHARE = 0.003
MIN_LARGEST_SHARE = 0.15
RECTANGULAR_FILL = 0.9
RULER_FILL = 0.85
RULER_ASPECT = 4.0
BORDER_SPAN = 0.5
BORDER_STRIP = 0.25
SCALE_COLOR_FILL = 0.7
SCALE_COLOR_SHARE = 0.6

spec = importlib.util.spec_from_file_location(
    "improve_bg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "improve_bg.py"))
improve_bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(improve_bg)


def scale_color_share(color, component):
    rgb = color[component]
    lum = rgb.mean(axis=1)
    blue = (rgb[:, 2] > rgb[:, 0] + 40) & (rgb[:, 2] > rgb[:, 1] + 20)
    return ((lum < 50) | (lum > 200) | blue).mean()


def is_scale_object(component, color, height, width):
    ys, xs = np.nonzero(component)
    box_h, box_w = np.ptp(ys) + 1, np.ptp(xs) + 1
    fill = component.sum() / (box_h * box_w)
    aspect = max(box_h, box_w) / min(box_h, box_w)
    touches_bottom_or_top = ys.max() >= height - 2 or ys.min() <= 1
    touches_side = xs.min() <= 1 or xs.max() >= width - 2
    edge_strip = ((touches_bottom_or_top and box_w >= width * BORDER_SPAN and box_h <= height * BORDER_STRIP)
                  or (touches_side and box_h >= height * BORDER_SPAN and box_w <= width * BORDER_STRIP))
    return (fill >= RECTANGULAR_FILL
            or (fill >= RULER_FILL and aspect >= RULER_ASPECT)
            or edge_strip
            or (fill >= SCALE_COLOR_FILL and scale_color_share(color, component) >= SCALE_COLOR_SHARE))


def bones(rgba, color):
    alpha = rgba[..., 3]
    core = alpha >= CORE_ALPHA
    labels, count = ndimage.label(core, structure=np.ones((3, 3)))
    if count == 0:
        return [], 0
    h, w = core.shape
    candidates, dropped_scale = [], 0
    areas = ndimage.sum(core, labels, range(1, count + 1))
    for index in np.flatnonzero(areas >= h * w * MIN_IMAGE_SHARE) + 1:
        component = labels == index
        area = areas[index - 1]
        if is_scale_object(component, color, h, w):
            dropped_scale += 1
            continue
        candidates.append((area, component))
    if not candidates:
        return [], dropped_scale
    largest = max(area for area, _ in candidates)
    kept = [c for area, c in candidates if area >= largest * MIN_LARGEST_SHARE]
    kept.sort(key=lambda c: (np.nonzero(c)[0].min() // max(h // 6, 1), np.nonzero(c)[1].min()))
    return kept, dropped_scale


def cut(rgba, color, component):
    soft = ndimage.gaussian_filter(component.astype(np.float32), improve_bg.EDGE_SIGMA)
    soft[component] = 1.0
    soft[soft < 0.02] = 0.0
    ys, xs = np.nonzero(component)
    h, w = component.shape
    pad = max(4, int(round(max(np.ptp(ys), np.ptp(xs)) * improve_bg.PAD_SHARE)))
    top, bottom = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    left, right = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.round(color * soft[..., None]).astype(np.uint8)
    out[..., 3] = np.round(soft * 255).astype(np.uint8)
    return Image.fromarray(out[top:bottom, left:right], "RGBA")


def main():
    if os.path.exists(DEST_DIR) and os.listdir(DEST_DIR):
        sys.exit(f"{DEST_DIR} already exists and is not empty")
    session = new_session("u2net")
    rows = []
    classes = [c for c in sorted(os.listdir(SRC_DIR)) if os.path.isdir(os.path.join(SRC_DIR, c)) and not c.startswith("_")]
    for cls in classes:
        os.makedirs(os.path.join(REMBG_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, cls), exist_ok=True)
        for name in sorted(os.listdir(os.path.join(SRC_DIR, cls))):
            base = os.path.splitext(name)[0]
            rembg_path = os.path.join(REMBG_DIR, cls, base + ".png")
            if os.path.exists(rembg_path):
                cutout = Image.open(rembg_path)
            else:
                cutout = remove(Image.open(os.path.join(SRC_DIR, cls, name)).convert("RGB"), session=session)
                cutout.save(rembg_path)
            rgba = np.asarray(cutout.convert("RGBA"))
            color = improve_bg.true_color(rgba, improve_bg.is_premultiplied(rgba))
            found, dropped_scale = bones(rgba, color)
            for k, component in enumerate(found):
                cut(rgba, color, component).save(os.path.join(DEST_DIR, cls, f"{base}_bone_{k}.png"), optimize=True)
            rows.append({"class": cls, "file": name, "bones": len(found), "dropped_scale_objects": dropped_scale})
        made = sum(r["bones"] for r in rows if r["class"] == cls)
        print(f"{cls}: {sum(r['class'] == cls for r in rows)} photos -> {made} bones", flush=True)

    with open(os.path.join(DEST_DIR, "report.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "file", "bones", "dropped_scale_objects"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"photos {len(rows)}, bones {sum(r['bones'] for r in rows)}, "
          f"no bones {sum(r['bones'] == 0 for r in rows)}, split {sum(r['bones'] > 1 for r in rows)}")


if __name__ == "__main__":
    main()
