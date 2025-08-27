from __future__ import annotations
import cv2, numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

GREEN = (36, 255, 12)

# ---------- geometry ----------
def order_corners_ccw(pts: np.ndarray) -> np.ndarray:
    """Order 4 points CCW and start from top-left. Robust for any rotation."""
    pts = pts.astype(np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # [-pi, pi]
    pts = pts[np.argsort(ang)]  # CCW
    # rotate so index 0 is top-left (min y then min x)
    i0 = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    pts = np.roll(pts, -int(i0), axis=0)
    return pts.astype(np.float32)

def otsu_cards_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # осигурај: КАРТИ = бело (ако позадината излезе бела -> инвертирај)
    if m.mean() > 127:
        m = cv2.bitwise_not(m)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, 1)
    m[0,:]=0; m[-1,:]=0; m[:,0]=0; m[:,-1]=0
    return m

def approx_quad_from_contour(c: np.ndarray) -> np.ndarray:
    """
    1) Convex hull (картите се конвексни)
    2) Бинарна потрага по ε за да добиеме 4 точки со approxPolyDP
    3) Ако не успее -> minAreaRect
    """
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)

    # binary search for epsilon that yields 4 points
    lo, hi = 0.002, 0.06
    best = None
    for _ in range(18):  # доволно
        eps = (lo + hi) / 2.0
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) > 4:
            lo = eps
        elif len(ap) < 4:
            hi = eps
        else:
            if cv2.isContourConvex(ap):
                best = ap.reshape(4, 2)
                break
            else:
                lo = eps  # премногу детално, „пука“ конвексност

    if best is None:
        # пробај неколку фиксни eps како резерва
        for eps in (0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                best = ap.reshape(4, 2); break

    if best is None:
        box = cv2.boxPoints(cv2.minAreaRect(hull))
        best = box

    return order_corners_ccw(best)

def compute_out_size(q: np.ndarray, short=250, long=350) -> Tuple[int,int]:
    w = (np.linalg.norm(q[1]-q[0]) + np.linalg.norm(q[2]-q[3]))/2.0
    h = (np.linalg.norm(q[3]-q[0]) + np.linalg.norm(q[2]-q[1]))/2.0
    return (long,short) if w>h else (short,long)

def warp_topdown(img: np.ndarray, q: np.ndarray, out_size: Tuple[int,int]) -> np.ndarray:
    w,h = out_size
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

def draw_overlay(img: np.ndarray, quads: List[np.ndarray]) -> np.ndarray:
    out = img.copy()
    for i,q in enumerate(quads,1):
        cv2.polylines(out,[q.astype(np.int32)],True,GREEN,3)
        cx,cy = q.mean(axis=0).astype(int)
        cv2.circle(out,(cx,cy),6,(255,0,0),-1)
        cv2.putText(out,f"#{i}",(cx+8,cy-8),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    return out

def panel(mask: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    sep = np.zeros((mask.shape[0], 12, 3), dtype=np.uint8)
    return np.hstack([mask_bgr, sep, overlay])

# ---------- pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Cards via Otsu + contours (no Hough), robust ordering for rotated cards.")
    ap.add_argument("image", help="input image")
    ap.add_argument("--min-area", type=float, default=0.01, help="min contour bbox area / image area")
    ap.add_argument("--edges-out", help="overlay output (default: <name>_edges.jpg)")
    ap.add_argument("--mask-out", help="mask output (default: <name>_mask.png)")
    ap.add_argument("--panel-out", help="side-by-side mask|overlay (default: <name>_panel.jpg)")
    ap.add_argument("--warp-dir", help="optional directory to save warped cards")
    ap.add_argument("--prefix", default="card", help="filename prefix for warped crops")
    ap.add_argument("--short", type=int, default=250)
    ap.add_argument("--long", type=int, default=350)
    args = ap.parse_args()

    src = Path(args.image); img = cv2.imread(str(src))
    if img is None: raise SystemExit("[ERR] failed to read image")

    H,W = img.shape[:2]; img_area = H*W

    mask = otsu_cards_mask(img)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # филтрирај по област и допир со рамка
    good = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < args.min_area*img_area: continue
        if x<=1 or y<=1 or x+w>=W-1 or y+h>=H-1: continue
        good.append(c)
    if not good: raise SystemExit("[ERR] no contours above area threshold")

    quads = [approx_quad_from_contour(c) for c in good]

    # стабилна нумерација: редови горе→долу па лево→десно
    centers = np.array([q.mean(axis=0) for q in quads])
    row_eps = np.median([np.linalg.norm(q[3]-q[0]) for q in quads]) * 0.45
    order = np.argsort(centers[:,1])
    rows, cur, last = [], [], None
    for idx in order:
        y = centers[idx,1]
        if last is None or abs(y-last)<=row_eps:
            cur.append(idx)
        else:
            rows.append(cur); cur=[idx]
        last = y
    if cur: rows.append(cur)
    idxs = [j for row in rows for j in sorted(row, key=lambda k: centers[k,0])]
    quads = [quads[i] for i in idxs]

    # снимки
    edges_path = Path(args.edges_out) if args.edges_out else src.with_name(src.stem+"_edges.jpg")
    mask_path  = Path(args.mask_out)  if args.mask_out  else src.with_name(src.stem+"_mask.png")
    panel_path = Path(args.panel_out) if args.panel_out else src.with_name(src.stem+"_panel.jpg")

    overlay = draw_overlay(img, quads)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(edges_path), overlay)
    cv2.imwrite(str(panel_path), panel(mask, overlay))

    if args.warp_dir:
        outdir = Path(args.warp_dir); outdir.mkdir(parents=True, exist_ok=True)
        for i,q in enumerate(quads,1):
            wh = compute_out_size(q, args.short, args.long)
            cv2.imwrite(str(outdir / f"{args.prefix}_{i:02d}.jpg"),
                        warp_topdown(img, q, wh))

    print(f"[OK] mask   : {mask_path.resolve()}")
    print(f"[OK] overlay: {edges_path.resolve()}")
    print(f"[OK] panel  : {panel_path.resolve()}")
    if args.warp_dir: print(f"[OK] warped saved in: {Path(args.warp_dir).resolve()}")

if __name__ == "__main__":
    main()
