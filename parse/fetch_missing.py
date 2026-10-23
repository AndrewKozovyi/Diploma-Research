import os
import requests
import json
import urllib.parse
import time
import re

# ==========================================
# 1. РОЗШИРЕНИЙ СЛОВНИК ПОШУКУ (Клас : [Синоніми та Види])
# ==========================================
EXTENDED_SEARCH = {
    "Dryosaurus": ["Dryosaurus altus", "Dryosaurus lettowvorbecki"],
    "Dryptosaurus": ["Dryptosaurus aquilunguis", "Laelaps aquilunguis"],  # Laelaps - стара назва
    "Torosaurus": ["Torosaurus latus", "Torosaurus utahensis"],
    "Tyrannosaurus": ["Tyrannosaurus rex", "Tyrannosauridae", "Tarbosaurus"]
}

OUTPUT_DIR = "additional_Images"
GBIF_OUTPUT = os.path.join(OUTPUT_DIR, "GBIF")
YALE_OUTPUT = os.path.join(OUTPUT_DIR, "Yale")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


# ==========================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================
def clean_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)


def download_image(url, save_path):
    if os.path.exists(save_path):
        return False
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=10)
        if resp.status_code == 200 and 'image' in resp.headers.get('Content-Type', ''):
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return True
    except:
        pass
    return False


# ==========================================
# ПАРСЕР GBIF
# ==========================================
def fetch_gbif(search_term, main_class):
    target_dir = os.path.join(GBIF_OUTPUT, main_class)
    os.makedirs(target_dir, exist_ok=True)

    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "basisOfRecord": "FOSSIL_SPECIMEN",
        "mediaType": "StillImage",
        "q": search_term,  # Змінили scientificName на q (ширший пошук по тексту)
        "limit": 50,
        "offset": 0
    }

    downloaded = 0
    while True:
        try:
            res = requests.get(url, params=params, timeout=15).json()
            results = res.get("results", [])
            if not results: break

            for item in results:
                inst = str(item.get("institutionCode", "")).lower()
                if "yale" in inst or "ypm" in inst: continue

                for img_idx, media in enumerate(item.get("media", [])):
                    if media.get("type") == "StillImage" and media.get("identifier"):
                        file_name = f"{item.get('key')}_{img_idx}.jpg"
                        save_path = os.path.join(target_dir, file_name)

                        if download_image(media.get("identifier"), save_path):
                            print(f"  [+] GBIF: {file_name} (Знайдено за: {search_term})")
                            downloaded += 1

            if res.get("endOfRecords"): break
            params["offset"] += params["limit"]
        except Exception as e:
            print(f"  [!] Помилка GBIF: {e}")
            break

    return downloaded


# ==========================================
# ПАРСЕР YALE
# ==========================================
def fetch_yale(search_term, main_class):
    target_dir = os.path.join(YALE_OUTPUT, main_class)
    os.makedirs(target_dir, exist_ok=True)

    query_obj = {
        "AND": [
            {"text": search_term, "_lang": "en"},
            {"hasDigitalImage": 1},
            {"memberOf": {"name": "Vertebrate Paleontology"}}
        ]
    }
    encoded_query = urllib.parse.quote(json.dumps(query_obj))
    current_url = f"https://lux.collections.yale.edu/api/search/item?q={encoded_query}&page=1&pageLength=20"

    downloaded = 0
    while current_url:
        try:
            res = requests.get(current_url, timeout=15).json()
            items = res.get("orderedItems", [])
            if not items: break

            for item in items:
                record_id = item.get("id", "").split("/")[-1]

                try:
                    img_url = item["representation"][0]["digitally_shown_by"][0]["access_point"][0]["id"]
                    save_path = os.path.join(target_dir, f"{record_id}.jpg")

                    if download_image(img_url, save_path):
                        print(f"  [+] Yale: {record_id}.jpg (Знайдено за: {search_term})")
                        downloaded += 1
                except (KeyError, IndexError):
                    continue

                time.sleep(0.2)

            next_page = res.get("next")
            current_url = next_page.get("id") if isinstance(next_page, dict) else None

        except Exception as e:
            print(f"  [!] Помилка Yale: {e}")
            break

    return downloaded


# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print(f"Починаємо глибокий пошук за видами та синонімами...\n")

    for main_class, search_terms in EXTENDED_SEARCH.items():
        print(f"\n{'=' * 40}")
        print(f"КЛАС: {main_class}")
        print(f"{'=' * 40}")

        total_gbif = 0
        total_yale = 0

        for term in search_terms:
            total_gbif += fetch_gbif(term, main_class)
            total_yale += fetch_yale(term, main_class)

        print(f"\nПідсумок для {main_class}:")
        print(f"  GBIF: {total_gbif} нових фото")
        print(f"  Yale: {total_yale} нових фото")

    print("\n✅ Всі завантаження завершено! Шукай фото в папці additional_Images.")