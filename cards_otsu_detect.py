# cards_otsu_detect.py
from __future__ import annotations
import cv2, numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

GREEN = (36, 255, 12)

# ---------- geometry ----------
def order_corners_ccw(pts: np.ndarray) -> np.ndarray:
    """Order 4 points CCW and start from top-left, robust for any rotation."""
    pts = pts.astype(np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # [-pi, pi]
    pts = pts[np.argsort(ang)]  # CCW
    # rotate so index 0 is top-left (min y then min x)
    i0 = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    pts = np.roll(pts, -int(i0), axis=0)
    return pts.astype(np.float32)

def otsu_cards_mask(bgr: np.ndarray) -> np.ndarray:
    """Otsu threshold → ensure cards are white; basic morphology cleanup."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure: CARDS = white (invert if background came out white)
    if m.mean() > 127:
        m = cv2.bitwise_not(m)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, 1)
    # cut 1px frame to ignore image borders
    m[0,:]=0; m[-1,:]=0; m[:,0]=0; m[:,-1]=0
    return m

def approx_quad_from_contour(c: np.ndarray) -> np.ndarray:
    """
    1) Convex hull (cards are convex)
    2) Binary-search epsilon for approxPolyDP to get 4 points
    3) Fallback to minAreaRect if needed
    """
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)

    lo, hi = 0.002, 0.06
    best = None
    for _ in range(18):
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
                lo = eps

    if best is None:
        for eps in (0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                best = ap.reshape(4, 2); break

    if best is None:
        box = cv2.boxPoints(cv2.minAreaRect(hull))
        best = box

    return order_corners_ccw(best)

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

# ---------- always-portrait warp ----------
def warp_topdown_portrait(img: np.ndarray, quad: np.ndarray, short: int, long: int) -> np.ndarray:
    """
    Produce bird's-eye warp **always portrait** (width=short, height=long).
    If the source quad is wider than tall, rotate the vertex order by 90°.
    """
    q = quad.astype(np.float32)
    w_src = (np.linalg.norm(q[1]-q[0]) + np.linalg.norm(q[2]-q[3]))/2.0
    h_src = (np.linalg.norm(q[3]-q[0]) + np.linalg.norm(q[2]-q[1]))/2.0
    if w_src > h_src:
        # rotate TL,TR,BR,BL -> TR,BR,BL,TL
        q = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

    out_w, out_h = short, long
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(
        img, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255,255,255)
    )

# ---------- pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Cards via Otsu + contours (no Hough), always-portrait warps.")
    ap.add_argument("image", help="input image")
    ap.add_argument("--min-area", type=float, default=0.01, help="min contour bbox area / image area")
    ap.add_argument("--edges-out", help="overlay output (default: <name>_edges.jpg)")
    ap.add_argument("--mask-out", help="mask output (default: <name>_mask.png)")
    ap.add_argument("--panel-out", help="side-by-side mask|overlay (default: <name>_panel.jpg)")
    ap.add_argument("--warp-dir", help="directory to save warped cards (portrait) (default: <name>_cards)")
    ap.add_argument("--prefix", default="card", help="filename prefix for warped crops")
    ap.add_argument("--short", type=int, default=250, help="portrait width")
    ap.add_argument("--long", type=int, default=350, help="portrait height")
    args = ap.parse_args()

    src = Path(args.image)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit("[ERR] failed to read image")

    H,W = img.shape[:2]; img_area = H*W

    # 1) Otsu mask
    mask = otsu_cards_mask(img)

    # 2) contours + area/border filter (StackOverflow logic)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < args.min_area*img_area: continue
        if x<=1 or y<=1 or x+w>=W-1 or y+h>=H-1: continue
        good.append(c)
    if not good:
        raise SystemExit("[ERR] no contours above area threshold")

    # 3) 4-pt polygons
    quads = [approx_quad_from_contour(c) for c in good]

    # 4) stable numbering: top->bottom then left->right
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

    # outputs
    edges_path = Path(args.edges_out) if args.edges_out else src.with_name(src.stem+"_edges.jpg")
    mask_path  = Path(args.mask_out)  if args.mask_out  else src.with_name(src.stem+"_mask.png")
    panel_path = Path(args.panel_out) if args.panel_out else src.with_name(src.stem+"_panel.jpg")

    overlay = draw_overlay(img, quads)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(edges_path), overlay)
    cv2.imwrite(str(panel_path), panel(mask, overlay))

    # 5) per-card **portrait** warps
    outdir = Path(args.warp_dir) if args.warp_dir else src.with_name(src.stem+"_cards")
    outdir.mkdir(parents=True, exist_ok=True)
    for i,q in enumerate(quads,1):
        warped = warp_topdown_portrait(img, q, args.short, args.long)
        cv2.imwrite(str(outdir / f"{args.prefix}_{i:02d}.jpg"), warped)

    print(f"[OK] mask   : {mask_path.resolve()}")
    print(f"[OK] overlay: {edges_path.resolve()}")
    print(f"[OK] panel  : {panel_path.resolve()}")
    print(f"[OK] warped : {outdir.resolve()} (portrait {args.short}x{args.long})")

if __name__ == "__main__":
    main()
