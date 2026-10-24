# Експерименти: оцінка якості та стійкості моделі

Фінальна модель: **YOLO11n-cls**, параметри прогону `-27` (freeze=5, dropout=0.8, batch=16, imgsz=224,
optimizer=auto → AdamW, lr≈7.7e-4 з прогрівом 3 епохи та лінійним спаданням), train збалансовано до 150 фото/клас.
Ваги: `results_yolo/yolo_11_selected_9_classes-150-34/weights/best.pt` (перенавчено після видалення дублікатів).

## 1. Дві постановки оцінки

| Розбиття | Що вимірює | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|
| Випадкове по файлах (`transfer_learning_datasets_filtered_by_5`, val 92 фото) | впізнавання екземплярів, представлених у навчальній колекції (~75% val мають той самий запис GBIF/Yale у train) | **0.870** | — | 0.989 |
| Групове по записах, 5-fold CV (`exp4_group_cv.py`, 648 фото) | визначення роду для **нових** екземплярів | **0.620** (95% ДІ 0.582–0.657) | **0.847** | 0.921 |

Групове розбиття: усі фото одного запису GBIF/Yale потрапляють лише в один спліт
(`data_preparation/clever_split_core.py`; для Monoclonius, де весь клас — один запис, група = окреме фото).
Маніфест розбиття: `data_preparation/clever_split_manifest.csv`.

## 2. Що впливає на якість (групове розбиття, TEST, середнє ± std по seed)

| Конфігурація | Top-1 | Seeds | Скрипт |
|---|---|---|---|
| YOLO11n, 150/клас (фінальна) | 0.606 ± 0.024 | 5 | `exp3_gpu_seeds.py` |
| YOLO11n, лише оригінали (без балансування) | 0.590 ± 0.010 | 3 | `exp3_gpu_seeds.py` |
| YOLO11s | 0.600 ± 0.026 | 3 | `exp3_gpu_seeds.py` |
| YOLO11m | 0.617 ± 0.021 | 3 | `exp3_gpu_seeds.py` |
| YOLO11n, freeze=0 | 0.607 ± 0.070 | 3 | `exp3_gpu_seeds.py` |
| YOLO11n, dropout=0.5 | 0.587 ± 0.032 | 3 | `exp3_gpu_seeds.py` |
| YOLO11n + flipud + cos LR / imgsz 320 | 0.59 / 0.64 | 1 | `exp1_clever_split_series.py` |
| DINOv2 ViT-S / ViT-B + логістична регресія | 0.62–0.63 / 0.61 | 1 | `exp1`, `exp2` |
| Ансамбль YOLO + DINOv2 (обраний за val) | 0.63 | 1 | `exp2_dinov2_ensemble.py` |
| DINOv2 ViT-S, 5-fold CV | 0.569 (ДІ 0.531–0.607) | — | `exp4_group_cv.py` |

Висновки:
- Усі варіанти в межах 0.59–0.64, тобто в межах розкиду між seed (±2–3 п.п.): обмеження — кількість різних екземплярів, а не архітектура.
- YOLO11n стабільно краща за заморожений DINOv2 (+5 п.п. у 5-fold CV) — донавчання CNN на власних даних виправдане.
- freeze=5 стабільніше за повне донавчання (freeze=0: розкид 0.54–0.68).
- Балансування до 150/клас не шкодить, з тенденцією до покращення (+1.6 п.п., у межах шуму).

## 3. Точність по класах (5-fold CV, YOLO11n, recall)

Monoclonius 0.99 · Torosaurus 0.88 · Deinonychus 0.74 · Stegosaurus 0.64 · Triceratops 0.46 ·
Allosaurus 0.39 · Edmontosaurus 0.33 · Camptosaurus 0.31 · Camarasaurus 0.08

Найкраще розпізнаються класи, зібрані з одного джерела/серії фото (Monoclonius — одна колекція, Torosaurus — одна серія,
Deinonychus — лише Yale), що вказує на часткову залежність від стилю зйомки. Camarasaurus: половина класу — один запис (19 фото).

## 4. Обмеження
- Мала кількість унікальних записів на клас (17–54);
- Модель не має класу «не динозавр»: будь-яке зображення віднесе до одного з 9 родів.
- Test на ~100 фото має похибку ~±10 п.п.; для підсумкової оцінки використано 5-fold CV.

## Скрипти
| Файл | Призначення |
|---|---|
| `transfer_learning_yolo.py` | навчання фінальної конфігурації |
| `transfer_learning_yolo_clever_split.py` | та сама конфігурація на груповому розбитті + test + Grad-CAM |
| `exp1_clever_split_series.py` … `exp4_group_cv.py` | серії експериментів (результати: `results_yolo/night_summary.csv`, `results_yolo/gpu/*.csv`, `results_yolo/gpu/cv/*.csv`) |
| `gradcam_yolo.py` | Grad-CAM для YOLO-класифікатора |
| `replot_learning_curve.py` | графік навчання з фактичним LR |
| `evaluate_honest.py` | оцінка моделі з поділом на знайомі / нові екземпляри |
