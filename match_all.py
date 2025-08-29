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
    best_score = -1.0
    best_name = ""

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

        if max_val > best_score:
            best_score = max_val
            best_name = tpl_path.stem  # без .png

    return best_name

# === Главна логика ===
ranks = {}
suits = {}

for img_path in sorted(INPUT_DIR.glob("*.png")):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue

    card_id = "_".join(img_path.stem.split("_")[:2])  # card_01

    if "rank" in img_path.name:
        ranks[card_id] = match_templates(gray, RANK_TEMPLATES)
    elif "suit" in img_path.name:
        suits[card_id] = match_templates(gray, SUIT_TEMPLATES)

# === Комбинирање и прикажување на картите ===
cards = []
for cid in sorted(ranks.keys()):
    if cid in suits:
        rank = ranks[cid]
        suit = suits[cid][0].lower()  # Пр. 'club' → 'c'
        cards.append(f"{rank}{suit}")

print("\n🂠 Детектирани карти:")
print(" ".join(cards))
