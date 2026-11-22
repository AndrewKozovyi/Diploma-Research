"""
Набір для оцінки OOD (фото, яких модель не повинна впевнено відносити до жодного з 10 класів).

  near/<рід>      - кістки інших родів динозаврів (той самий конвеєр, що й датасет): 5 родів, відкинутих
                    через малу кількість фото, - з merged_dataset_filtered, решта - з not_prepared_images/merged_dataset
  far/<рід>       - кістки не-динозаврів (ссавці, морські рептилії, крокодили) з GBIF, оброблені тим самим конвеєром
  far_raw/<рід>   - ті самі фото GBIF без обробки (з фоном), як їх міг би завантажити користувач

Фото зі списку ood_removed.txt (етикетки, лінійки, порожні кадри, малюнки - перевірено вручну) видаляються.
Не беруться: таксони вищого рівня, що можуть містити роди з навчання (Theropoda, Hadrosauridae, ...),
та синоніми навчальних родів (Morosaurus, Trachodon, ...). MD5-дублікати навчальних фото відкидаються.

Запуск:  python build_ood_set.py near|far
"""
import os
import sys
import shutil
import hashlib
import importlib.util

import requests
import numpy as np
from PIL import Image

ROOT = r"D:\projects\Diploma"
ID_DIR = os.path.join(ROOT, r"datasets\merged_dataset_filtered")
NEAR_SRC = os.path.join(ROOT, r"not_prepared_images\merged_dataset")
OUT = os.path.join(ROOT, r"datasets\ood")
REMOVED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ood_removed.txt")
ID_CLASSES = {"Allosaurus", "Camarasaurus", "Camptosaurus", "Deinonychus", "Edmontosaurus",
              "Monoclonius", "Stegosaurus", "Torosaurus", "Triceratops", "Tyrannosaurus"}
EXCLUDED = {"Aublysodon", "Carnosauria", "Ceratopsia", "Ceratopsidae", "Coelurosauria", "Dromaeosauridae",
            "Hadrosauridae", "Megalosauridae", "Morosaurus", "Ornithischia", "Theropoda", "Thespesius",
            "Trachodon", "Tyrannosauridae", "Tyrannosauroidea", "Pteranodon"}
FAR_GENERA = ["Mammut", "Mammuthus", "Smilodon", "Bison", "Equus", "Megalonyx", "Canis",
              "Alligator", "Mosasaurus", "Ichthyosaurus", "Plesiosaurus"]
RECORDS_PER_GENUS = 12
MAX_BONES_PER_PHOTO = 2
HEADERS = {"User-Agent": "diploma-ood-builder/1.0"}

spec = importlib.util.spec_from_file_location("cut_bones", os.path.join(ROOT, "tools", "remove_bg_split_new.py"))
cut_bones = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cut_bones)


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def build_near():
    seen = {md5(os.path.join(ID_DIR, c, f)) for c in ID_CLASSES for f in os.listdir(os.path.join(ID_DIR, c))}
    total = 0
    for genus in sorted(os.listdir(NEAR_SRC)):
        if genus in ID_CLASSES or genus in EXCLUDED:
            continue
        src_dir = os.path.join(ID_DIR if os.path.isdir(os.path.join(ID_DIR, genus)) else NEAR_SRC, genus)
        dst = os.path.join(OUT, "near", genus)
        os.makedirs(dst, exist_ok=True)
        for name in sorted(os.listdir(src_dir)):
            src = os.path.join(src_dir, name)
            digest = md5(src)
            if digest not in seen:
                seen.add(digest)
                shutil.copy2(src, os.path.join(dst, name))
                total += 1
    print("near:", total)


def download_far():
    for genus in FAR_GENERA:
        dst = os.path.join(OUT, "far_raw", genus)
        os.makedirs(dst, exist_ok=True)
        got, offset = len(os.listdir(dst)), 0
        while got < RECORDS_PER_GENUS and offset < 500:
            data = requests.get("https://api.gbif.org/v1/occurrence/search", headers=HEADERS, timeout=30, params={
                "basisOfRecord": "FOSSIL_SPECIMEN", "mediaType": "StillImage", "scientificName": genus,
                "limit": 50, "offset": offset}).json()
            for item in data.get("results", []):
                if got >= RECORDS_PER_GENUS:
                    break
                path = os.path.join(dst, f"{item['key']}_0.jpg")
                url = next((m.get("identifier") for m in item.get("media", []) if m.get("type") == "StillImage"), None)
                if not url or os.path.exists(path):
                    continue
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=30)
                    if resp.status_code != 200 or "image" not in resp.headers.get("Content-Type", ""):
                        continue
                    with open(path, "wb") as fh:
                        fh.write(resp.content)
                    Image.open(path).convert("RGB").save(path, "JPEG", quality=95)
                    got += 1
                except Exception as e:
                    print("  skip", url, type(e).__name__)
                    if os.path.exists(path):
                        os.remove(path)
            if data.get("endOfRecords", True):
                break
            offset += 50
        print(f"{genus}: {got} photos", flush=True)


def build_far():
    from rembg import remove, new_session
    session = new_session("u2net")
    for genus in FAR_GENERA:
        src_dir = os.path.join(OUT, "far_raw", genus)
        dst = os.path.join(OUT, "far", genus)
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


def apply_removed():
    removed = 0
    with open(REMOVED, encoding="utf-8") as fh:
        for line in fh:
            path = os.path.join(OUT, line.split("#")[0].strip())
            if line.strip() and os.path.isfile(path):
                os.remove(path)
                removed += 1
    print("removed:", removed)


if __name__ == "__main__":
    if sys.argv[1] == "near":
        build_near()
    else:
        download_far()
        build_far()
    apply_removed()
