import cv2
import numpy as np
from pathlib import Path

# Патеките
input_dir = Path("_debug_indices")
output_dir = Path("crops")
output_dir.mkdir(exist_ok=True)

# Препорачани параметри (suggested_params)
roi_w, roi_h = 0.18, 0.312
split_y_ratio = 0.568
left_band = 0.48
max_rel_area = 0.15
min_area = 19

# Рангирани темплејти (sintetički)
def synth_rank_templates(font_scale=1.0, thickness=2):
    chars = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    out = {}
    for ch in chars:
        canvas = np.zeros((56, 56), np.uint8)
        (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = max(1, (canvas.shape[1] - tw) // 2)
        y = max(th+2, int(0.6*canvas.shape[0]))
        cv2.putText(canvas, ch, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        out[ch] = canvas
    return out

def make_diamond(w=48, h=48):
    img = np.zeros((h, w), np.uint8)
    pts = np.array([[int(w*0.50), int(h*0.05)], [int(w*0.92), int(h*0.50)],
                    [int(w*0.50), int(h*0.95)], [int(w*0.08), int(h*0.50)]], np.int32)
    cv2.fillConvexPoly(img, pts, 255)
    return img

def tm_score(img_bin, tpl_bin):
    ih, iw = img_bin.shape
    th, tw = tpl_bin.shape
    if th > ih or tw > iw:
        scale = min(ih / th, iw / tw)
        tpl_bin = cv2.resize(tpl_bin, (max(1,int(tw*scale)), max(1,int(th*scale))), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(img_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else -1.0

def binarize(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshed

# Темплејти
rank_templates = synth_rank_templates()
diamond_template = make_diamond()

# Процесирање на сите слики
for file in sorted(input_dir.glob("*.jpg")):
    img = cv2.imread(str(file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[:2]
    roi_x1, roi_y1 = 0, 0
    roi_x2 = int(W * roi_w)
    roi_y2 = int(H * roi_h)
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    split_y = int((roi_y2 - roi_y1) * split_y_ratio)

    rank = roi[:split_y, :]
    suit = roi[split_y:, :]

    rank_bin = binarize(rank)
    suit_bin = binarize(suit)

    # Класификација на ранг
    best_rank, best_rank_score = "?", -1
    for name, tpl in rank_templates.items():
        score = tm_score(rank_bin, tpl)
        if score > best_rank_score:
            best_rank, best_rank_score = name, score

    # Класификација на боја
    suit_score = tm_score(suit_bin, diamond_template)
    suit_name = "diamonds" if suit_score > 0.5 else "unknown"

    # Зачувување
    name_base = file.stem
    cv2.imwrite(str(output_dir / f"{name_base}_rank.png"), rank_bin)
    cv2.imwrite(str(output_dir / f"{name_base}_suit.png"), suit_bin)

    print(f"{file.name}: Predicted -> {best_rank} {suit_name} (scores: {best_rank_score:.2f}, {suit_score:.2f})")
