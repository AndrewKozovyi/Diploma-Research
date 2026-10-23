import os
import requests

# Твої локальні шляхи
YALE_TRAIN_DIR = r"D:\projects\Diploma\yale_dataset_nobg\Train_Data"
GBIF_OUTPUT_DIR = r"D:\projects\Diploma\parsed_dataset_gbif"

API_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
YALE_KEYWORDS = ["yale", "ypm", "peabody"]

# Маскуємо скрипт під звичайний браузер
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_genera_from_local():
    """Проходить по папці Train_Data і збирає тільки назви родів (папки верхнього рівня)."""
    genera = []
    if not os.path.exists(YALE_TRAIN_DIR):
        print(f"[!] Directory {YALE_TRAIN_DIR} not found!")
        return genera

    for genus in os.listdir(YALE_TRAIN_DIR):
        genus_path = os.path.join(YALE_TRAIN_DIR, genus)
        if os.path.isdir(genus_path):
            genera.append(genus)
    return genera


def download_images_for_genus(search_genus):
    print(f"\n--- Searching GBIF for genus: {search_genus} ---")

    params = {
        "basisOfRecord": "FOSSIL_SPECIMEN",
        "mediaType": "StillImage",
        "scientificName": search_genus,
        "limit": 100,
        "offset": 0
    }

    while True:
        try:
            response = requests.get(API_OCCURRENCE_URL, params=params, timeout=15)
            if response.status_code != 200:
                print(f"[!] API returned status code {response.status_code}")
                break
        except Exception as e:
            print(f"[!] API request error: {e}")
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            print(f"No more results for {search_genus}.")
            break

        for item in results:
            item_key = item.get("key")

            # Ігноруємо Yale
            institution = str(item.get("institutionCode", "")).lower()
            publisher = str(item.get("publisherTitle", "")).lower()
            if any(kw in institution or kw in publisher for kw in YALE_KEYWORDS):
                continue

            # Визначаємо вид для створення підпапки
            item_species = item.get("species")
            if not item_species:
                # Якщо вид не вказаний, записуємо як "Genus_sp"
                item_species = f"{search_genus}_sp"

            folder_species = item_species.replace(" ", "_")

            # Цільова папка: parsed_dataset_gbif\Allosaurus\Allosaurus_fragilis
            target_dir = os.path.join(GBIF_OUTPUT_DIR, search_genus, folder_species)
            os.makedirs(target_dir, exist_ok=True)

            media_list = item.get("media", [])
            img_index = 0

            for media in media_list:
                if media.get("type") == "StillImage":
                    img_url = media.get("identifier")
                    if not img_url:
                        continue

                    file_name = f"{item_key}_{img_index}.jpg"
                    file_path = os.path.join(target_dir, file_name)

                    if os.path.exists(file_path):
                        print(f"[*] Skipped: {file_name} already exists in {search_genus}\\{folder_species}")
                        img_index += 1
                        continue

                    try:
                        img_resp = requests.get(img_url, headers=HEADERS, stream=True, timeout=15)

                        # Перевірка на HTML-сторінку замість картинки (без спаму в консоль)
                        content_type = img_resp.headers.get('Content-Type', '')
                        if 'image' not in content_type:
                            continue

                        if img_resp.status_code == 200:
                            with open(file_path, "wb") as f:
                                for chunk in img_resp.iter_content(1024):
                                    f.write(chunk)
                            print(f"[+] Downloaded {file_name} to {search_genus}\\{folder_species}")
                        else:
                            print(f"[!] Server returned {img_resp.status_code} for {img_url}")
                    except Exception as e:
                        print(f"[!] Download error for {img_url}: {e}")

                    img_index += 1

        if data.get("endOfRecords"):
            break

        params["offset"] += params["limit"]


def main():
    genera = get_genera_from_local()
    if not genera:
        print("No genus folders found in the local directory.")
        return

    print(f"Found {len(genera)} genera for synchronization with GBIF.")
    for genus in genera:
        download_images_for_genus(genus)


if __name__ == "__main__":
    main()