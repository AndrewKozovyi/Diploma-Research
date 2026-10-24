"""Спільна логіка групового розбиття (використовується build_clever_split.py)."""
import re
import random
from collections import defaultdict

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def is_aug(name):
    return "_aug_" in name or re.match(r"^aug_\d+_", name) is not None


def group_key(cls, filename, single_record_classes):
    """
    Група = ID запису (GBIF occurrence або Yale UUID): усі фото одного запису йдуть в один спліт.
    Для класів, де ВЕСЬ клас - це один запис (Monoclonius), групуємо по номеру фото в записі
    (parsed_<record>_<photo>_bone_<k>): різні кістки одного фото не розносяться по сплітах.
    """
    b = re.sub(r"^(parsed|yale|idigbio|wiki|si)_", "", filename)
    m = re.match(r"([0-9a-f]{8}-[0-9a-f\-]{27}|\d+)(?:_(\d+))?", b)
    if not m:
        return f"{cls}/{b}"
    rec, photo = m.group(1), m.group(2)
    if cls in single_record_classes and photo is not None:
        return f"{cls}/{rec}_{photo}"
    return f"{cls}/{rec}"


def find_single_record_classes(files_by_class):
    out = set()
    for cls, files in files_by_class.items():
        recs = {group_key(cls, f, set()) for f in files}
        if len(recs) == 1:
            out.add(cls)
    return out


def split_groups(files_by_class, seed=42):
    """Стратифіковане групове розбиття: жадібно розкладаємо групи (від більших до менших)
    у той спліт, якому найбільше бракує фото до цільової частки."""
    single = find_single_record_classes(files_by_class)
    rng = random.Random(seed)
    assignment = {}
    for cls in sorted(files_by_class):
        groups = defaultdict(list)
        for f in sorted(files_by_class[cls]):
            groups[group_key(cls, f, single)].append(f)
        keys = sorted(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda k: -len(groups[k]))  # стабільне сортування: великі групи першими
        n = sum(len(v) for v in groups.values())
        target = {s: r * n for s, r in SPLIT_RATIOS.items()}
        cur = {s: 0 for s in SPLIT_RATIOS}
        # Великі записи (> 40% цільового розміру val) завжди йдуть у train:
        # інакше val/test класу складався б з 1-2 екземплярів і був би непредставницьким.
        big_limit = max(2, 0.4 * target["val"])
        for k in keys:
            if len(groups[k]) > big_limit:
                s = "train"
            else:
                s = max(SPLIT_RATIOS, key=lambda s: (target[s] - cur[s]) / target[s])
            cur[s] += len(groups[k])
            for f in groups[k]:
                assignment[(cls, f)] = (s, k)
    return assignment, single
