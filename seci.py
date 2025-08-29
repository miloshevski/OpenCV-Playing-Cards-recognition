import cv2
import numpy as np
from pathlib import Path

input_dir = Path("crops")

# Фиксни димензии
RANK_SIZE = (70, 120)  # w, h
SUIT_SIZE = (70, 70)

def get_target_size(filename):
    return SUIT_SIZE if "suit" in filename else RANK_SIZE

def crop_and_center(img, target_size):
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = img[y:y+h, x:x+w]

    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# Обработка со overwrite
for path in sorted(input_dir.glob("*.png")):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue

    target_size = get_target_size(path.name)
    processed = crop_and_center(gray, target_size)
    cv2.imwrite(str(path), processed)  # overwrite оригинал
    # print(f"✔ Overwritten {path.name} → {processed.shape}")
