import os
import io
import re
import csv
import json
import time
import urllib.parse
import urllib.request

from PIL import Image

ROOT = r"D:\projects\Diploma"
OUTPUT_DIR = os.path.join(ROOT, r"not_prepared_images\parsed_dataset_new_institutions")
EXISTING_DIRS = [
    os.path.join(ROOT, r"not_prepared_images\parsed_dataset_gbif"),
    os.path.join(ROOT, r"not_prepared_images\parsed_dataset_yale"),
    os.path.join(ROOT, r"additional_Images\GBIF"),
    os.path.join(ROOT, r"datasets\merged_dataset_filtered"),
]
CLASSES = ["Allosaurus", "Ankylosauridae", "Camarasaurus", "Camptosaurus", "Coeluridae", "Deinonychus",
           "Dryosaurus", "Dryptosaurus", "Edmontonia", "Edmontosaurus", "Monoclonius", "Stegosaurus",
           "Torosaurus", "Triceratops", "Tyrannosaurus"]
INSTITUTIONS = {"USNM", "VMNH"}
HASH_DISTANCE = 6
HEADERS = {"User-Agent": "diploma-dataset-fetcher/1.0"}
GBIF = "https://api.gbif.org/v1"


def get_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def get_bytes(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def dhash(image, size=8):
    gray = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(gray.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def image_for_hash(path):
    image = Image.open(path)
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[3])
        return background
    return image.convert("RGB")


def existing_index():
    keys, hashes = set(), []
    for root in EXISTING_DIRS:
        for dirpath, _, files in os.walk(root):
            for name in files:
                match = re.match(r"(?:parsed_)?(\d{6,})_", name)
                if match:
                    keys.add(int(match.group(1)))
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        hashes.append(dhash(image_for_hash(os.path.join(dirpath, name))))
                    except Exception:
                        pass
    return keys, hashes


def is_duplicate(h, hashes):
    return any(bin(h ^ other).count("1") <= HASH_DISTANCE for other in hashes)


def resolve_url(identifier):
    if "flickr.com/photos/" in identifier:
        oembed = get_json("https://www.flickr.com/services/oembed/?format=json&maxwidth=2048&url="
                          + urllib.parse.quote(identifier))
        return oembed["url"]
    return identifier


def records(taxon_key):
    offset = 0
    while True:
        page = get_json(f"{GBIF}/occurrence/search?taxonKey={taxon_key}&mediaType=StillImage&limit=300&offset={offset}")
        for record in page["results"]:
            if record.get("institutionCode") in INSTITUTIONS:
                yield record
        offset += 300
        if page["endOfRecords"]:
            break


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Indexing existing images...")
    known_keys, known_hashes = existing_index()
    print(f"  keys: {len(known_keys)}, hashes: {len(known_hashes)}")

    meta_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    done = set()
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            done = {row["file"] for row in csv.DictReader(fh)}
        for row_file in done:
            path = os.path.join(OUTPUT_DIR, row_file)
            if os.path.exists(path):
                known_hashes.append(dhash(image_for_hash(path)))

    fields = ["file", "class", "gbif_key", "institution", "catalog_number", "scientific_name", "basis_of_record",
              "remarks", "license", "rights_holder", "image_url", "gbif_url"]
    stats = {"saved": 0, "known_key": 0, "duplicate": 0, "failed": 0}
    with open(meta_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            writer.writeheader()
        for cls in CLASSES:
            taxon_key = get_json(f"{GBIF}/species/match?name={cls}")["usageKey"]
            class_dir = os.path.join(OUTPUT_DIR, cls)
            os.makedirs(class_dir, exist_ok=True)
            saved_before = stats["saved"]
            for record in records(taxon_key):
                if record["key"] in known_keys:
                    stats["known_key"] += 1
                    continue
                images = [m for m in record.get("media", []) if m.get("identifier") and m.get("type") == "StillImage"]
                for index, media in enumerate(images):
                    name = f"{record['key']}_{index}.jpg"
                    rel = f"{cls}/{name}"
                    if rel in done:
                        continue
                    try:
                        url = resolve_url(media["identifier"])
                        image = Image.open(io.BytesIO(get_bytes(url))).convert("RGB")
                    except Exception as error:
                        stats["failed"] += 1
                        print(f"  [!] {rel}: {error}")
                        continue
                    h = dhash(image)
                    if is_duplicate(h, known_hashes):
                        stats["duplicate"] += 1
                        continue
                    known_hashes.append(h)
                    image.save(os.path.join(class_dir, name), quality=95)
                    writer.writerow({
                        "file": rel, "class": cls, "gbif_key": record["key"],
                        "institution": record.get("institutionCode"),
                        "catalog_number": record.get("catalogNumber"),
                        "scientific_name": record.get("scientificName"),
                        "basis_of_record": record.get("basisOfRecord"),
                        "remarks": record.get("occurrenceRemarks") or record.get("preparations") or "",
                        "license": media.get("license") or record.get("license"),
                        "rights_holder": media.get("rightsHolder") or record.get("rightsHolder") or "",
                        "image_url": url,
                        "gbif_url": f"https://www.gbif.org/occurrence/{record['key']}",
                    })
                    fh.flush()
                    stats["saved"] += 1
                    time.sleep(0.3)
            print(f"{cls}: +{stats['saved'] - saved_before}")
    print(stats)


if __name__ == "__main__":
    main()
