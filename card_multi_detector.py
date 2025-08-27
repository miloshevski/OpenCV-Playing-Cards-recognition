from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

# ---------------- helpers ----------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)

def white_card_mask(gray: np.ndarray, bgr: np.ndarray | None = None) -> np.ndarray:
    # Otsu за бело на темна/сивкаста позадина
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # fallback на HSV ако премалку/препремногу бело
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

def quad_from_contour(cnt: np.ndarray) -> np.ndarray | None:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = order_corners(box)

    w_box = np.linalg.norm(box[1] - box[0])
    h_box = np.linalg.norm(box[3] - box[0])
    ratio = min(w_box, h_box) / max(w_box, h_box)
    if 0.60 <= ratio <= 0.78:
        return box

    # ако аспектот е off, пробај approxPolyDP
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return order_corners(approx.reshape(4, 2))
    return box  # врати minAreaRect ако нема подобро

def compute_out_size(quad: np.ndarray, short=250, long=350) -> Tuple[int, int]:
    w1 = np.linalg.norm(quad[1] - quad[0])
    w2 = np.linalg.norm(quad[2] - quad[3])
    h1 = np.linalg.norm(quad[3] - quad[0])
    h2 = np.linalg.norm(quad[2] - quad[1])
    w = (w1 + w2) / 2.0
    h = (h1 + h2) / 2.0
    return (long, short) if w > h else (short, long)

def warp_topdown(img: np.ndarray, quad: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    w, h = out_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return warped

def draw_all_overlays(img: np.ndarray, quads: List[np.ndarray]) -> np.ndarray:
    out = img.copy()
    for i, q in enumerate(quads, start=1):
        cv2.polylines(out, [q.astype(np.int32)], True, (0, 255, 0), 3)
        cx, cy = q.mean(axis=0).astype(int)
        cv2.circle(out, (cx, cy), 7, (255, 0, 0), -1)
        cv2.putText(out, f"#{i}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return out

def sort_quads_readable(quads: List[np.ndarray]) -> List[np.ndarray]:
    # сортирај по „редови“ (од горе надолу), па внатре по x (лево->десно)
    centers = np.array([q.mean(axis=0) for q in quads])  # (N,2)
    ys = centers[:, 1]
    # процени висина на карта за кластеризација по редови
    heights = [np.linalg.norm(q[3] - q[0]) for q in quads]
    row_eps = (np.median(heights) if heights else 50) * 0.45

    # групирај по редови
    order = np.argsort(ys)
    rows = []
    current = []
    last_y = None
    for idx in order:
        y = ys[idx]
        if last_y is None or abs(y - last_y) <= row_eps:
            current.append(idx)
        else:
            rows.append(current); current = [idx]
        last_y = y
    if current: rows.append(current)

    # по редови сортирај по x
    sorted_indices = []
    for row in rows:
        row_sorted = sorted(row, key=lambda k: centers[k, 0])
        sorted_indices.extend(row_sorted)

    return [quads[i] for i in sorted_indices]

# ---------------- main pipeline ----------------
def process_multi(image_path: str, edges_out: str | None, warp_dir: str, prefix: str,
                  short: int, long: int, min_area: float, max_cards: int | None):
    src = Path(image_path)
    if not src.exists():
        raise SystemExit(f"[ERR] Input image not found: {src}")

    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit("[ERR] Failed to read image.")

    h, w = img.shape[:2]
    img_area = w * h

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = white_card_mask(gray, img)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # филтрирај премали и оние што допираат рамка
    cand = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < min_area * img_area: 
            continue
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1: 
            continue
        cand.append(c)

    if not cand:
        raise SystemExit("[ERR] No cards found. Make sure cards are brighter than background and not touching image borders.")

    # конвертирај во квадови
    quads = []
    for c in cand:
        q = quad_from_contour(c)
        if q is not None:
            quads.append(q)

    if not quads:
        raise SystemExit("[ERR] Contours found but failed to fit quadrilaterals.")

    # подреди за читливо именување
    quads = sort_quads_readable(quads)
    if max_cards is not None:
        quads = quads[:max_cards]

    # 1) снимaј overlay со сите
    edges_path = Path(edges_out) if edges_out else src.with_name(src.stem + "_edges.jpg")
    overlay = draw_all_overlays(img, quads)
    cv2.imwrite(str(edges_path), overlay)

    # 2) снимaј поединечно warp-нато
    outdir = Path(warp_dir) if warp_dir else src.with_name(src.stem + "_cards")
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, q in enumerate(quads, start=1):
        out_size = compute_out_size(q, short=short, long=long)
        warped = warp_topdown(img, q, out_size)
        out_path = outdir / f"{prefix}_{i:02d}.jpg"
        cv2.imwrite(str(out_path), warped)
        saved.append(str(out_path.resolve()))

    print(f"[OK] Saved overlay: {edges_path.resolve()}")
    print(f"[OK] Saved {len(saved)} warped cards in: {outdir.resolve()}")
    for p in saved:
        print(" -", p)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Detect multiple white playing cards, draw overlay, and warp each top-down.")
    ap.add_argument("image", help="path to input image with MULTIPLE cards")
    ap.add_argument("--edges-out", help="output overlay image with all boxes (default: <name>_edges.jpg)")
    ap.add_argument("--warp-dir", help="directory to save warped cards (default: <name>_cards)")
    ap.add_argument("--prefix", default="card", help="filename prefix for warped cards (default: card)")
    ap.add_argument("--short", type=int, default=250, help="short edge of warped card (default 250)")
    ap.add_argument("--long", type=int, default=350, help="long edge of warped card (default 350)")
    ap.add_argument("--min-area", type=float, default=0.01, help="min contour area fraction (default 0.01)")
    ap.add_argument("--max-cards", type=int, default=None, help="optional cap on number of cards to export")
    args = ap.parse_args()

    process_multi(args.image, args.edges_out, args.warp_dir, args.prefix,
                  args.short, args.long, args.min_area, args.max_cards)

if __name__ == "__main__":
    main()
