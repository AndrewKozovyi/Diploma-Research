import os
from ultralytics import YOLO

# ==============================================================================
# ШЛЯХ ДО ТВОЇХ НАЙКРАЩИХ ВАГ
# (заміни на актуальний шлях до папки твого найкращого запуску)
# ==============================================================================
WEIGHTS_PATH = r"D:\projects\Diploma\CNN\results_yolo\yolo_11_selected_9_classes-150-27\weights\best.pt"


def predict_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"[Помилка] Файл не знайдено: {image_path}")
        return

    # Запускаємо розпізнавання (verbose=False прибирає зайвий технічний спам YOLO з консолі)
    results = model.predict(source=image_path, imgsz=224, verbose=False)

    # Беремо результат для першого (і єдиного) переданого фото
    result = results[0]
    probs = result.probs  # Об'єкт із ймовірностями
    names = result.names  # Словник з іменами класів {0: 'Allosaurus', ...}

    # Найбільш ймовірний клас (Top-1)
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item() * 100
    top1_name = names[top1_idx]

    print("\n" + "=" * 50)
    print(f"Файл: {os.path.basename(image_path)}")
    print(f"Результат: {top1_name} (Впевненість: {top1_conf:.2f}%)")
    print("-" * 50)
    print("Топ-5 варіантів за версією моделі:")

    # Виводимо Топ-5 прогнозів з красивою текстовою шкалою
    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()

    for rank, (idx, conf) in enumerate(zip(top5_indices, top5_confs), start=1):
        class_name = names[idx]
        percentage = conf * 100
        bar_length = int(percentage / 5)  # 1 поділ = 5% (макс 20 символів)
        bar = "#" * bar_length + "." * (20 - bar_length)
        print(f"  {rank}. {class_name:<15} [{bar}] {percentage:.2f}%")
    print("=" * 50 + "\n")


def main():
    print("--- Завантаження моделі для класифікації динозаврів ---")
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[Критична помилка] Ваги не знайдено за шляхом:\n{WEIGHTS_PATH}")
        print("Перевір правильність шляху у змінній WEIGHTS_PATH.")
        return

    # Ініціалізуємо модель з твоїх ваг
    model = YOLO(WEIGHTS_PATH)
    print("Модель успішно завантажена та готова до роботи!")
    print(f"Доступні класи ({len(model.names)}): {', '.join(model.names.values())}\n")

    # Інтерактивний цикл
    while True:
        user_input = input("Введи шлях до фото (або 'q' для виходу): ").strip()

        # Дозволяє просто перетягувати файл мишкою у консоль PyCharm/CMD
        # (Windows при перетягуванні часто додає лапки навколо шляху, ми їх зрізаємо)
        user_input = user_input.strip('"').strip("'")

        if user_input.lower() in ['q', 'quit', 'exit', 'вихід']:
            print("Роботу програми завершено.")
            break

        if not user_input:
            continue

        predict_image(model, user_input)


if __name__ == '__main__':
    main()