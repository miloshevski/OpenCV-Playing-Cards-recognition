import cv2
import numpy as np
from pathlib import Path

# Патеки
crops_dir = Path("crops")
ranks_dir = Path("templates/ranks")
suits_dir = Path("templates/suits")

def load_templates(template_path):
    templates = {}
    for file in template_path.glob("*.png"):
        name = file.stem.lower()
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        templates[name] = bin_img
    return templates

def match_template(img, templates):
    best_score = -1
    best_label = "unknown"
    for label, tpl in templates.items():
        tpl_resized = cv2.resize(tpl, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(img, tpl_resized, cv2.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_score:
            best_score = score
            best_label = label
    return best_label, best_score

# Вчитај темплејти
rank_templates = load_templates(ranks_dir)
suit_templates = load_templates(suits_dir)

# Прошетај низ сите rank слики и поврзи ги со соодветниот suit
for rank_file in crops_dir.glob("*_rank.png"):
    base_name = rank_file.stem.replace("_rank", "")
    suit_file = crops_dir / f"{base_name}_suit.png"
    if not suit_file.exists():
        print(f"Skipping {base_name}, missing suit.")
        continue

    # Вчитај и бинарај
    rank_img = cv2.imread(str(rank_file), cv2.IMREAD_GRAYSCALE)
    suit_img = cv2.imread(str(suit_file), cv2.IMREAD_GRAYSCALE)
    _, rank_bin = cv2.threshold(rank_img, 127, 255, cv2.THRESH_BINARY)
    _, suit_bin = cv2.threshold(suit_img, 127, 255, cv2.THRESH_BINARY)

    # Предвиди
    rank_label, rank_score = match_template(rank_bin, rank_templates)
    suit_label, suit_score = match_template(suit_bin, suit_templates)

    print(f"{base_name}: {rank_label.upper()} of {suit_label.lower()} (scores: {rank_score:.2f}, {suit_score:.2f})")
