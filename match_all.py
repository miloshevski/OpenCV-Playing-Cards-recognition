import cv2
import numpy as np
from pathlib import Path

# === Константи ===
RANK_TEMPLATES = Path("templates/ranks")
SUIT_TEMPLATES = Path("templates/suits")
INPUT_DIR = Path("crops")

# === Бинаризирање функција ===
def binarize(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# === Матч темплејт ===
def match_templates(input_img, templates_dir):
    input_bin = binarize(input_img)
    scores = []

    for tpl_path in sorted(templates_dir.glob("*.png")):
        tpl_img = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        tpl_bin = binarize(tpl_img)

        ih, iw = input_bin.shape
        th, tw = tpl_bin.shape

        if th > ih or tw > iw:
            scale = min(ih / th, iw / tw)
            tpl_bin = cv2.resize(tpl_bin, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_AREA)

        res = cv2.matchTemplate(input_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
        max_val = res.max() if res.size else -1.0
        scores.append((tpl_path.name, max_val))

    best_match = max(scores, key=lambda x: x[1])
    return best_match, scores

# === Главна логика ===
for img_path in sorted(INPUT_DIR.glob("*.png")):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"⚠ Не може да се прочита {img_path.name}")
        continue

    is_rank = "rank" in img_path.name
    templates_dir = RANK_TEMPLATES if is_rank else SUIT_TEMPLATES

    best_match, all_scores = match_templates(gray, templates_dir)

    print(f"\n🃏 {img_path.name}")
    for name, score in all_scores:
        print(f"  {name:<12} → {score:.3f}")
    print(f"✅ Најдобар match: {best_match[0]} (score: {best_match[1]:.3f})")
