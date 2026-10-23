import os
import time
import requests
import json
import re


def get_existing_ids(dataset_dir="parsed_dataset_yale"):
    """Сканує існуючі файли для миттєвого пропуску без HTTP-запитів"""
    existing_ids = set()
    if not os.path.exists(dataset_dir):
        return existing_ids

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.json'):
                record_id = file.split('.')[0]
                existing_ids.add(record_id)
    return existing_ids


def clean_and_filter_name(name):
    """
    Правила фільтрації:
    1. Відкидає undet.
    2. Відкидає, якщо сумнівний рід (? перед першим словом).
    3. Якщо сумнівний вид (? у другому слові), замінює на sp.
    4. Очищає від заборонених символів Windows.
    """
    original_name = name.strip()
    lower_name = original_name.lower()

    # 1. Відкидаємо відверте сміття
    if "undet" in lower_name:
        return None, "Містить undet"

    parts = original_name.split()
    if not parts:
        return None, "Порожня назва"

    # 2. Якщо сумнівний рід
    if "?" in parts[0]:
        return None, f"Сумнівний рід: {parts[0]}"

    # 3. Якщо сумнівний вид
    for i in range(1, len(parts)):
        if "?" in parts[i]:
            parts[i] = "sp"

    clean_name = " ".join(parts)

    # 4. Рятуємо від крашів Windows (замінюємо | ? : * < > " \ / на підкреслення)
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', clean_name)

    return safe_name, "OK"


def extract_primary_name(data):
    identifiers = data.get("identified_by", [])
    for ident in identifiers:
        if ident.get("type") == "Name":
            for cls in ident.get("classified_as", []):
                if cls.get("_label") == "Primary Name":
                    return ident.get("content")
    return "Unknown_Dinosaur"


def extract_image_url(data):
    try:
        representations = data.get("representation", [])
        if not representations: return None
        digital_images = representations[0].get("digitally_shown_by", [])
        if not digital_images: return None
        access_points = digital_images[0].get("access_point", [])
        if not access_points: return None
        return access_points[0].get("id")
    except:
        return None


def download_fossil_data(item_url, dataset_dir="parsed_dataset_yale"):
    try:
        response = requests.get(item_url, timeout=10)
        if response.status_code != 200: return False

        data = response.json()
        raw_name = extract_primary_name(data)

        folder_name, status = clean_and_filter_name(raw_name)

        if folder_name is None:
            print(f"[~] Пропущено ({status}): {raw_name}")
            return True

        image_url = extract_image_url(data)
        if not image_url: return False

        class_dir = os.path.join(dataset_dir, folder_name)
        os.makedirs(class_dir, exist_ok=True)

        record_id = data.get("id", "").split("/")[-1]
        img_path = os.path.join(class_dir, f"{record_id}.jpg")
        json_path = os.path.join(class_dir, f"{record_id}.json")

        img_response = requests.get(image_url, timeout=10)
        if img_response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(img_response.content)
            with open(json_path, 'w', encoding='utf-8') as f_json:
                json.dump(data, f_json, ensure_ascii=False, indent=4)

            print(f"[+] Завантажено: {folder_name} -> {record_id}")
            return True

    except Exception as e:
        print(f"[-] Помилка файлу: {str(e)}")
    return False


def run_crawler(start_url):
    current_url = start_url
    page_num = 1
    total_downloaded = 0
    dataset_dir = "parsed_dataset_yale"

    print("Сканування дисків для швидкого пропуску")
    existing_ids = get_existing_ids(dataset_dir)
    print(f"Знайдено {len(existing_ids)} вже завантажених файлів. Запуск парсера\n")

    while current_url:
        print(f"\nОбробка сторінки {page_num}")
        try:
            response = requests.get(current_url, timeout=15)
            if response.status_code != 200:
                break

            data = response.json()
            items = data.get("orderedItems", [])

            if not items:
                break

            for item in items:
                item_url = item.get("id")
                if item_url:
                    record_id = item_url.split("/")[-1]

                    if record_id in existing_ids:
                        print(f"Вже є на диску: {record_id}")
                        continue

                    if download_fossil_data(item_url, dataset_dir):
                        total_downloaded += 1

                    time.sleep(0.5)

            next_page = data.get("next")
            if next_page and isinstance(next_page, dict):
                current_url = next_page.get("id")
                page_num += 1
            else:
                current_url = None

        except Exception as e:
            print(f"[-] Критична помилка пагінації: {str(e)}")
            break

    print(f"\nПарсинг завершено! Всього ДОЗАВАНТАЖЕНО нових записів: {total_downloaded}")


if __name__ == "__main__":
    SEARCH_API_URL = "https://lux.collections.yale.edu/api/search/item?q=%7B%22AND%22%3A%5B%7B%22text%22%3A%22Dinosaur%22%2C%22_lang%22%3A%22en%22%7D%2C%7B%22hasDigitalImage%22%3A1%7D%2C%7B%22memberOf%22%3A%7B%22name%22%3A%22Vertebrate%20Paleontology%22%7D%7D%5D%7D&page=1&pageLength=20"
    run_crawler(SEARCH_API_URL)