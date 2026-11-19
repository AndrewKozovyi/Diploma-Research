"""
Додаткові фото з GBIF для слабких класів: записи з фото, яких ще немає в datasets/merged_dataset_filtered.
Записи YPM пропускаються: фото Єлю вже взято з їхнього порталу під іншими ідентифікаторами.

1. download - сирі фото в not_prepared_images/gbif_extra/raw/<рід>/<ключ>_<n>.jpg + metadata.csv
              (установа, номер, назва, ліцензія, позначка cast, якщо запис згадує зліпок/модель)
2. cut      - rembg u2net + tools/remove_bg_split_new (як для основного датасету)
              -> not_prepared_images/gbif_extra/cut/<рід>/<ключ>_<n>_bone_<k>.png

Запуск:  python download_gbif_extra.py download
         python download_gbif_extra.py cut
"""
import os
import re
import sys
import csv
import json
import importlib.util

import requests
import numpy as np
from PIL import Image

ROOT = r"D:\projects\Diploma"
DATASET = os.path.join(ROOT, r"datasets\merged_dataset_filtered")
OUT = os.path.join(ROOT, r"not_prepared_images\gbif_extra")
GENERA = ["Camarasaurus", "Camptosaurus", "Tyrannosaurus", "Edmontosaurus", "Allosaurus", "Monoclonius", "Deinonychus"]
SKIP_INSTITUTIONS = {"YPM"}
MAX_BONES_PER_PHOTO = 3
HEADERS = {"User-Agent": "dataset-research-script/1.0"}


def used_keys(genus):
    keys = set()
    for name in os.listdir(os.path.join(DATASET, genus)):
        keys |= set(re.findall(r"\d{6,}", name))
    return keys


def records(genus):
    key = requests.get("https://api.gbif.org/v1/species/match", headers=HEADERS, timeout=30,
                       params={"name": genus, "rank": "GENUS"}).json()["usageKey"]
    out, offset = [], 0
    while True:
        data = requests.get("https://api.gbif.org/v1/occurrence/search", headers=HEADERS, timeout=60,
                            params={"taxonKey": key, "mediaType": "StillImage", "limit": 300, "offset": offset}).json()
        out += data["results"]
        if data["endOfRecords"]:
            return out
        offset += 300


def download():
    rows = []
    for genus in GENERA:
        used = used_keys(genus)
        dst = os.path.join(OUT, "raw", genus)
        os.makedirs(dst, exist_ok=True)
        got = 0
        for item in records(genus):
            if str(item["key"]) in used or item.get("institutionCode") in SKIP_INSTITUTIONS:
                continue
            cast = bool(re.search(r"\b(cast|replica|model)\b", json.dumps(item).lower()))
            media = [m for m in item.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
            for n, m in enumerate(media):
                path = os.path.join(dst, f"{item['key']}_{n}.jpg")
                if not os.path.exists(path):
                    try:
                        resp = requests.get(m["identifier"], headers=HEADERS, timeout=60)
                        if resp.status_code != 200 or "image" not in resp.headers.get("Content-Type", ""):
                            continue
                        with open(path, "wb") as fh:
                            fh.write(resp.content)
                        Image.open(path).convert("RGB").save(path, "JPEG", quality=95)
                    except Exception as e:
                        print("  skip", m["identifier"], type(e).__name__)
                        if os.path.exists(path):
                            os.remove(path)
                        continue
                got += 1
                rows.append({"genus": genus, "file": os.path.basename(path), "key": item["key"],
                             "institution": item.get("institutionCode"), "catalog": item.get("catalogNumber"),
                             "name": item.get("scientificName"), "cast": cast,
                             "license": m.get("license") or item.get("license"), "url": m["identifier"]})
        print(f"{genus}: {got} photos", flush=True)
    with open(os.path.join(OUT, "metadata.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cut():
    from rembg import remove, new_session
    spec = importlib.util.spec_from_file_location("cut_bones", os.path.join(ROOT, "tools", "remove_bg_split_new.py"))
    cut_bones = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cut_bones)
    session = new_session("u2net")
    for genus in GENERA:
        src_dir = os.path.join(OUT, "raw", genus)
        dst = os.path.join(OUT, "cut", genus)
        os.makedirs(dst, exist_ok=True)
        made = 0
        for name in sorted(os.listdir(src_dir)):
            base = os.path.splitext(name)[0]
            img = Image.open(os.path.join(src_dir, name)).convert("RGB")
            img.thumbnail((2000, 2000))
            rgba = np.asarray(remove(img, session=session).convert("RGBA"))
            color = cut_bones.improve_bg.true_color(rgba, cut_bones.improve_bg.is_premultiplied(rgba))
            found, _ = cut_bones.bones(rgba, color)
            for k, component in enumerate(found[:MAX_BONES_PER_PHOTO]):
                cut_bones.cut(rgba, color, component).save(os.path.join(dst, f"{base}_bone_{k}.png"), optimize=True)
                made += 1
        print(f"{genus}: {made} bones", flush=True)


if __name__ == "__main__":
    {"download": download, "cut": cut}[sys.argv[1]]()
