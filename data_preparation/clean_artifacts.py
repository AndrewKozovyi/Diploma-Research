"""
Чистка артефактів вирізання фону в датасеті (структура <клас>/<фото>).

0. PNG з прозорістю накладаються на чорне тло (інакше під прозорими пікселями часто лежить білий колір,
   і завантажувач бачить білу рамку).

1. Білі заливки, що лишились після невдалого видалення фону (рамки, плями): компоненти пікселів >= 250,
   які щонайменше на 90% складаються з чистого (255, 255, 255). Білий гіпс чи світла кістка на фото такими
   не бувають (шум камери). Такі пікселі (з облямівкою 2 px) стають чорними.
2. Окремі дрібні об'єкти поза контуром кістки (лінійки, шматки інших кісток, залишки етикеток):
   компоненти маски, менші за KEEP_FRAC від найбільшої і розташовані поза замкненим контуром кістки,
   зафарбовуються чорним. Світлі плями всередині темної кістки не чіпаються.
3. Кадр обрізається до кістки з полем MARGIN.
Маски рахуються на копії з довшою стороною ANALYSIS_SIDE (як у навчальному r512) і переносяться на оригінал.

Запуск:  python clean_artifacts.py <src> <dst> <журнал.csv>
"""
import os
import sys
import csv

import cv2
import numpy as np

WHITE = 250
PURE_SHARE = 0.9
WHITE_MIN = 0.001
FG = 10
KEEP_FRAC = 0.10
MARGIN = 0.03
ANALYSIS_SIDE = 512


def components(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return lab, st[1:]


def analyse(image):
    h, w = image.shape[:2]
    info = {"white_share": 0.0, "removed_objects": 0, "removed_share": 0.0, "crop_share": 0.0}

    lab, st = components(image.min(2) >= WHITE)
    fill = np.zeros((h, w), bool)
    pure = image.min(2) == 255
    for k, area in enumerate(st[:, 4], 1):
        if area >= WHITE_MIN * h * w and pure[lab == k].mean() >= PURE_SHARE:
            fill |= lab == k
    if fill.any():
        fill = cv2.dilate(fill.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (image.min(2) >= 180) | fill
        image = image.copy()
        image[fill] = 0
        info["white_share"] = fill.mean()

    gone = np.zeros((h, w), bool)
    fg = cv2.morphologyEx((image.max(2) > FG).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lab, st = components(fg)
    if not len(st):
        return fill, gone, (0, 0, w, h), info
    main = st[:, 4].max()
    keep = [k for k, s in enumerate(st, 1) if s[4] >= KEEP_FRAC * main]
    body = cv2.morphologyEx(np.isin(lab, keep).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body = cv2.drawContours(np.zeros_like(body), contours, -1, 1, -1).astype(bool)
    drop = [k for k, s in enumerate(st, 1) if s[4] < KEEP_FRAC * main and s[4] > 30 and not body[lab == k].any()]
    if drop:
        gone = cv2.dilate(np.isin(lab, drop).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~body
        info["removed_objects"] = len(drop)
        info["removed_share"] = sum(st[k - 1, 4] for k in drop) / main

    ys, xs = np.nonzero(body)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = int(MARGIN * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    info["crop_share"] = 1 - (x1 - x0) * (y1 - y0) / (h * w)
    return fill, gone, (x0, y0, x1, y1), info


def clean(image):
    h, w = image.shape[:2]
    scale = min(1.0, ANALYSIS_SIDE / max(h, w))
    if scale == 1:
        fill, gone, (x0, y0, x1, y1), info = analyse(image)
    else:
        small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        fill_s, gone_s, box, info = analyse(small)

        def up(mask):
            return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        fill = up(cv2.dilate(fill_s.astype(np.uint8), np.ones((3, 3), np.uint8))) & (image.min(2) >= 180) | up(fill_s)
        gone = up(gone_s)
        x0, y0 = int(box[0] / scale), int(box[1] / scale)
        x1, y1 = min(w, int(np.ceil(box[2] / scale))), min(h, int(np.ceil(box[3] / scale)))
    image = image.copy()
    image[fill | gone] = 0
    return image[y0:y1, x0:x1], info


def main(src, dst, log):
    if os.path.exists(dst) and os.listdir(dst):
        sys.exit(f"{dst} already exists and is not empty")
    rows = []
    for cls in sorted(os.listdir(src)):
        os.makedirs(os.path.join(dst, cls), exist_ok=True)
        for name in sorted(os.listdir(os.path.join(src, cls))):
            image = cv2.imread(os.path.join(src, cls, name), cv2.IMREAD_UNCHANGED)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = (image[..., :3] * (image[..., 3:] / 255.0)).round().astype(np.uint8)
            out, info = clean(image)
            cv2.imwrite(os.path.join(dst, cls, name), out)
            rows.append({"cls": cls, "name": name, **{k: round(float(v), 4) for k, v in info.items()}})
    with open(log, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changed = [r for r in rows if r["white_share"] or r["removed_objects"]]
    print(f"{len(rows)} images, white fill removed in {sum(1 for r in rows if r['white_share'])}, "
          f"stray objects removed in {sum(1 for r in rows if r['removed_objects'])}, changed {len(changed)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
