from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import argparse

# ---------- helpers ----------
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)

def largest_white_component_mask(gray: np.ndarray, bgr: np.ndarray | None = None) -> np.ndarray:
    # 1) Otsu за бела карта на темна/сивкаста позадина
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # fallback ако Otsu „претера“ или „закаска“
    white_ratio = mask.mean() / 255.0
    if (white_ratio < 0.01 or white_ratio > 0.5) and bgr is not None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S_MAX, V_MIN = 60, 170
        mask = cv2.inRange(hsv, (0, 0, V_MIN), (180, S_MAX, 255))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # исечи 1px рамка за да не ги фаќа ивиците на самата слика
    mask[0, :] = 0; mask[-1, :] = 0; mask[:, 0] = 0; mask[:, -1] = 0
    return mask

def detect_card_quad(img: np.ndarray, min_area_ratio: float = 0.02) -> np.ndarray | None:
    h, w = img.shape[:2]
    img_area = w * h

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = largest_white_component_mask(gray, img)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return None

    candidates = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < min_area_ratio * img_area: 
            continue
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1: 
            continue
        candidates.append(c)
    if not candidates: 
        return None

    card_cnt = max(candidates, key=cv2.contourArea)

    # прво пробај ротирана правоаголна кутија
    rect = cv2.minAreaRect(card_cnt)
    box = cv2.boxPoints(rect)  # 4 точки
    box = order_corners(box)

    # ако аспектот изгледа сумнително, пробај approxPolyDP
    w_box = np.linalg.norm(box[1] - box[0])
    h_box = np.linalg.norm(box[3] - box[0])
    ratio = min(w_box, h_box) / max(w_box, h_box)
    if not (0.60 <= ratio <= 0.78):
        peri = cv2.arcLength(card_cnt, True)
        approx = cv2.approxPolyDP(card_cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            box = order_corners(approx.reshape(4, 2))
    return box.astype(np.float32)

def draw_overlay(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    out = img.copy()
    cv2.polylines(out, [quad.astype(np.int32)], True, (0, 255, 0), 3)
    for i, (x, y) in enumerate(quad.astype(int)):
        cv2.circle(out, (x, y), 6, (255, 0, 0), -1)
        cv2.putText(out, str(i), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return out

def compute_out_size(quad: np.ndarray, short=250, long=350) -> tuple[int, int]:
    # процени ориентација од должини на страните
    w1 = np.linalg.norm(quad[1] - quad[0])
    w2 = np.linalg.norm(quad[2] - quad[3])
    h1 = np.linalg.norm(quad[3] - quad[0])
    h2 = np.linalg.norm(quad[2] - quad[1])
    w = (w1 + w2) / 2.0
    h = (h1 + h2) / 2.0
    return (long, short) if w > h else (short, long)

def warp_topdown(img: np.ndarray, quad: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    w, h = out_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)   # пополни со бело како „скенирано“
    )
    return warped

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Detect playing-card edges and produce bird's-eye warp.")
    ap.add_argument("image", help="path to input image with ONE card")
    ap.add_argument("--edges-out", help="output overlay image (default: <name>_edges.jpg)")
    ap.add_argument("--warp-out", help="output warped image (default: <name>_warped.jpg)")
    ap.add_argument("--short", type=int, default=250, help="short edge of warped card (default 250)")
    ap.add_argument("--long", type=int, default=350, help="long edge of warped card (default 350)")
    ap.add_argument("--min-area", type=float, default=0.02, help="min contour area fraction (default 0.02)")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.exists():
        raise SystemExit(f"[ERR] Input image not found: {src}")

    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit("[ERR] Failed to read image.")

    quad = detect_card_quad(img, min_area_ratio=args.min_area)
    if quad is None:
        raise SystemExit("[ERR] No card found. Ensure white card on darker background and not touching image borders.")

    # 1) overlay со полигонот
    overlay = draw_overlay(img, quad)
    edges_path = Path(args.edges_out) if args.edges_out else src.with_name(src.stem + "_edges.jpg")
    cv2.imwrite(str(edges_path), overlay)

    # 2) bird's-eye warp (како скенирано)
    out_size = compute_out_size(quad, short=args.short, long=args.long)
    warped = warp_topdown(img, quad, out_size)
    warp_path = Path(args.warp_out) if args.warp_out else src.with_name(src.stem + "_warped.jpg")
    cv2.imwrite(str(warp_path), warped)

    print(f"[OK] Saved overlay: {edges_path.resolve()}")
    print(f"[OK] Saved warped : {warp_path.resolve()}  (size {out_size[0]}x{out_size[1]})")

if __name__ == "__main__":
    main()
