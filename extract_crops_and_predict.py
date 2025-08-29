import cv2
import numpy as np
from pathlib import Path

# Патеките
input_dir = Path("_debug_indices")
output_dir = Path("crops")
output_dir.mkdir(exist_ok=True)

# Препорачани параметри
roi_w, roi_h = 0.18, 0.312
split_y_ratio = 0.568

# Рангирани темплејти (sintetički)
def synth_rank_templates(font_scale=1.0, thickness=2):
    chars = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    out = {}
    for ch in chars:
        canvas = np.zeros((120, 70), np.uint8)  # FINAL TEMPLATE SIZE
        (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = max(1, (canvas.shape[1] - tw) // 2)
        y = max(th + 2, int(0.6 * canvas.shape[0]))
        cv2.putText(canvas, ch, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        out[ch] = canvas
    return out

def make_diamond(w=70, h=70):
    img = np.zeros((h, w), np.uint8)
    pts = np.array([[int(w*0.50), int(h*0.05)], [int(w*0.92), int(h*0.50)],
                    [int(w*0.50), int(h*0.95)], [int(w*0.08), int(h*0.50)]], np.int32)
    cv2.fillConvexPoly(img, pts, 255)
    return img

def tm_score(img_bin, tpl_bin):
    res = cv2.matchTemplate(img_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else -1.0

def binarize(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshed

# Темплејти
rank_templates = synth_rank_templates()
diamond_template = make_diamond()

# Процесирање
for file in sorted(input_dir.glob("*.jpg")):
    img = cv2.imread(str(file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[:2]
    roi = gray[0:int(H * roi_h), 0:int(W * roi_w)]
    split_y = int(roi.shape[0] * split_y_ratio)

    rank = roi[:split_y, :]
    suit = roi[split_y:, :]

    rank_bin = binarize(rank)
    suit_bin = binarize(suit)

    # Resize на фиксна димензија
    rank_bin_resized = cv2.resize(rank_bin, (70, 120), interpolation=cv2.INTER_AREA)
    suit_bin_resized = cv2.resize(suit_bin, (70, 70), interpolation=cv2.INTER_AREA)

    # Класификација
    best_rank, best_rank_score = "?", -1
    for name, tpl in rank_templates.items():
        score = tm_score(rank_bin_resized, tpl)
        if score > best_rank_score:
            best_rank, best_rank_score = name, score

    suit_score = tm_score(suit_bin_resized, diamond_template)
    suit_name = "diamonds" if suit_score > 0.5 else "unknown"

    # Зачувување
    name_base = file.stem
    cv2.imwrite(str(output_dir / f"{name_base}_rank.png"), rank_bin_resized)
    cv2.imwrite(str(output_dir / f"{name_base}_suit.png"), suit_bin_resized)

    print(f"{file.name}: Predicted -> {best_rank} {suit_name} (scores: {best_rank_score:.2f}, {suit_score:.2f})")
