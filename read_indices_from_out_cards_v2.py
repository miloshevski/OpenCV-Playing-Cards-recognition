from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

"""
Fixed-layout index reader for already-cropped & rectified cards (from out_cards/).
Assumptions (per user):
  - Rank is ALWAYS at top-left.
  - Suit icon is ALWAYS directly below the rank (still in the top-left corner block).
  - Cards are upright (not rotated 180°).

Pipeline:
  1) Take a fixed top-left ROI (fraction of card width/height; tunable via --roi_w and --roi_h).
  2) Binarize (Otsu + small morph). Find external contours in ROI.
  3) Split contours into two bands by Y (upper band => rank; lower band => suit).
     - Rank band may contain multiple glyphs (e.g., "10"). We merge them into a single bounding box.
     - Suit band: take the largest component.
  4) Classify:
     - Suit: match against procedurally generated templates (hearts/diamonds/spades/clubs) via TM_CCOEFF_NORMED.
     - Rank: if pytesseract is available, OCR with whitelist "A234567890JQK"; otherwise, fallback to simple
       template matching on synthetic glyphs (Hershey font). For "10" we allow multi-character OCR result.

Outputs per image: {file, ok, rank, suit, scores, bboxes, card} and optional debug overlay image.
"""

# ---------------- Utils ----------------
def imread_bgr(p: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {p}")
    return img


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def otsu_inv(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def adaptive_inv(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    return th


def best_binary(gray: np.ndarray) -> np.ndarray:
    b1 = otsu_inv(gray)
    b2 = adaptive_inv(gray)
    return b1 if cv2.countNonZero(b1) >= cv2.countNonZero(b2) else b2


def morph_clean(bin_img: np.ndarray) -> np.ndarray:
    # small open/close to join gaps but remove noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    x = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, k, iterations=1)
    return x


def merge_boxes(boxes: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0]+b[2] for b in boxes)
    y2 = max(b[1]+b[3] for b in boxes)
    return x1, y1, x2-x1, y2-y1


# ---------------- Procedural suit templates ----------------
def make_hearts(w=48, h=48) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    # two circles + triangle
    cv2.circle(img, (int(w*0.35), int(h*0.30)), int(min(w,h)*0.18), 255, -1)
    cv2.circle(img, (int(w*0.65), int(h*0.30)), int(min(w,h)*0.18), 255, -1)
    pts = np.array([[int(w*0.10), int(h*0.35)], [int(w*0.90), int(h*0.35)], [int(w*0.50), int(h*0.90)]], np.int32)
    cv2.fillConvexPoly(img, pts, 255)
    return img


def make_diamond(w=48, h=48) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    pts = np.array([[int(w*0.50), int(h*0.05)], [int(w*0.92), int(h*0.50)],
                    [int(w*0.50), int(h*0.95)], [int(w*0.08), int(h*0.50)]], np.int32)
    cv2.fillConvexPoly(img, pts, 255)
    return img


def make_spade(w=48, h=48) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    # upside-down heart + stem
    heart = make_hearts(w, h)
    heart = cv2.rotate(heart, cv2.ROTATE_180)
    img = cv2.max(img, heart)
    cv2.rectangle(img, (int(w*0.45), int(h*0.70)), (int(w*0.55), int(h*0.95)), 255, -1)
    cv2.ellipse(img, (int(w*0.50), int(h*0.80)), (int(w*0.10), int(h*0.08)), 0, 0, 180, 255, -1)
    return img


def make_club(w=48, h=48) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    r = int(min(w,h)*0.16)
    centers = [(int(w*0.35), int(h*0.40)), (int(w*0.65), int(h*0.40)), (int(w*0.50), int(h*0.20))]
    for cx,cy in centers:
        cv2.circle(img, (cx,cy), r, 255, -1)
    # stem
    cv2.rectangle(img, (int(w*0.45), int(h*0.45)), (int(w*0.55), int(h*0.90)), 255, -1)
    return img


def suit_templates_bank() -> dict:
    return {
        'hearts': make_hearts(),
        'diamonds': make_diamond(),
        'spades': make_spade(),
        'clubs': make_club(),
    }


def tm_score(img_bin: np.ndarray, tpl_bin: np.ndarray) -> float:
    ih, iw = img_bin.shape
    th, tw = tpl_bin.shape
    if th > ih or tw > iw:
        scale = min(ih / th, iw / tw)
        if scale <= 0:
            return -1.0
        tpl_bin = cv2.resize(tpl_bin, (max(1,int(tw*scale)), max(1,int(th*scale))), interpolation=cv2.INTER_AREA)
        th, tw = tpl_bin.shape
        if th < 4 or tw < 4:
            return -1.0
    res = cv2.matchTemplate(img_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else -1.0


# ---------------- Rank OCR (optional) ----------------
try:
    import pytesseract  # type: ignore
    HAS_TESS = True
except Exception:
    HAS_TESS = False


def synth_rank_templates(font_scale=1.0, thickness=2) -> dict:
    """Generate simple glyph templates using Hershey fonts as fallback."""
    chars = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    out = {}
    for ch in chars:
        canvas = np.zeros((56, 56), np.uint8)
        text = ch
        # place text roughly centered
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = max(1, (canvas.shape[1] - tw) // 2)
        y = max(th+2, int(0.6*canvas.shape[0]))
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        out[ch] = canvas
    return out


# ---------------- Core reader (fixed layout) ----------------
class FixedLayoutReader:
    def __init__(self, roi_w: float = 0.24, roi_h: float = 0.28, split_y: float = 0.42,
                 min_area: int = 25, left_band: float = 0.60, max_rel_area: float = 0.18):
        """
        roi_w, roi_h  : top-left corner ROI in fractions of full card size.
        split_y       : fraction of ROI height that separates rank(upper) and suit(lower).
        left_band     : keep only components whose center-x <= left_band * ROI_W (hug the left edge).
        max_rel_area  : drop components whose area > max_rel_area * ROI_area (filters big central pips).
        """
        self.roi_w = roi_w
        self.roi_h = roi_h
        self.split_y = split_y
        self.min_area = min_area
        self.left_band = left_band
        self.max_rel_area = max_rel_area
        self.suit_bank = suit_templates_bank()
        self.rank_bank = synth_rank_templates()

    def _top_left_roi(self, gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        H, W = gray.shape
        x1, y1 = 0, 0
        x2, y2 = int(W * self.roi_w), int(H * self.roi_h)
        # guard rails
        x2 = max(x2, min(W, int(W*0.18)))
        y2 = max(y2, min(H, int(H*0.20)))
        return gray[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _split_rank_suit(self, roi_bin: np.ndarray) -> Tuple[Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]]]:
        H, W = roi_bin.shape
        roi_area = float(H * W)
        cnts, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comps = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w*h
            if area < self.min_area:
                continue
            # filter too big (likely central pip leak) and too far right
            cx = x + w/2.0
            if area > self.max_rel_area * roi_area:
                continue
            if cx > self.left_band * W:
                continue
            cy = y + h/2.0
            comps.append((x,y,w,h,cy,area))
        if not comps:
            return None, None
        # split by Y threshold
        thr = H * self.split_y
        rank_boxes = [(x,y,w,h) for (x,y,w,h,cy,_) in comps if cy <= thr]
        suit_boxes = [(x,y,w,h) for (x,y,w,h,cy,_) in comps if cy > thr]
        if not rank_boxes and not suit_boxes:
            return None, None
        if not rank_boxes:
            # pick the top-most component as rank
            top = min(comps, key=lambda t: t[1])
            rank_boxes = [(top[0], top[1], top[2], top[3])]
        if not suit_boxes:
            # pick the largest component that lies below (or closest below) the rank bottom
            ry = rank_boxes[0][1] + rank_boxes[0][3]
            below = [b for b in [(x,y,w,h) for (x,y,w,h,cy,_) in comps] if b[1] >= ry - int(0.04*H)]
            if not below:
                return merge_boxes(rank_boxes), None
            suit_boxes = [max(below, key=lambda b: b[2]*b[3])]
        # merge multiple rank glyphs (e.g., "10")
        rank_box = merge_boxes(rank_boxes)
        suit_box = max(suit_boxes, key=lambda b: b[2]*b[3])
        return rank_box, suit_box

    def _classify_suit(self, suit_crop_bin: np.ndarray) -> Tuple[str, float]:
        best_name, best_sc = "", -1.0
        for name, tpl in self.suit_bank.items():
            sc = tm_score(suit_crop_bin, tpl)
            if sc > best_sc:
                best_name, best_sc = name, sc
        return best_name, best_sc

    def _classify_rank(self, rank_crop_gray: np.ndarray, rank_crop_bin: np.ndarray) -> Tuple[str, float]:
        if HAS_TESS:
            cfg = "--psm 7 -c tessedit_char_whitelist=A234567890JQK"
            txt = pytesseract.image_to_string(255 - rank_crop_bin, config=cfg)
            txt = txt.strip().upper().replace("O","0").replace(" ","")
            if txt:
                if txt == "0": txt = "10"
                if len(txt) > 2: txt = txt[:2]
                if txt in {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}:
                    return txt, 0.99
        best_name, best_sc = "", -1.0
        for name, tpl in self.rank_bank.items():
            sc = tm_score(rank_crop_bin, tpl)
            if sc > best_sc:
                best_name, best_sc = name, sc
        return best_name, best_sc

    def read(self, bgr: np.ndarray) -> Optional[dict]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi_gray, (x1,y1,x2,y2) = self._top_left_roi(gray)
        if roi_gray.size == 0:
            return None
        roi_bin = morph_clean(best_binary(roi_gray))

        rank_box, suit_box = self._split_rank_suit(roi_bin)
        if rank_box is None or suit_box is None:
            return None

        rx,ry,rw,rh = rank_box
        sx,sy,sw,sh = suit_box
        rank_bin = roi_bin[ry:ry+rh, rx:rx+rw]
        suit_bin = roi_bin[sy:sy+sh, sx:sx+sw]
        rank_gray = roi_gray[ry:ry+rh, rx:rx+rw]

        suit_name, suit_sc = self._classify_suit(suit_bin)
        rank_name, rank_sc = self._classify_rank(rank_gray, rank_bin)
        if not suit_name or not rank_name:
            return None

        rank_xyxy = [x1+rx, y1+ry, x1+rx+rw, y1+ry+rh]
        suit_xyxy = [x1+sx, y1+sy, x1+sx+sw, y1+sy+sh]

        return {
            "corner": "top-left",
            "suit": suit_name,
            "suit_score": float(round(suit_sc, 3)),
            "suit_bbox_xyxy": list(map(int, suit_xyxy)),
            "rank": rank_name,
            "rank_score": float(round(rank_sc, 3)),
            "rank_bbox_xyxy": list(map(int, rank_xyxy)),
        }
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi_gray, (x1,y1,x2,y2) = self._top_left_roi(gray)
        if roi_gray.size == 0:
            return None
        roi_bin = morph_clean(best_binary(roi_gray))

        rank_box, suit_box = self._split_rank_suit(roi_bin)
        if rank_box is None or suit_box is None:
            return None

        rx,ry,rw,rh = rank_box
        sx,sy,sw,sh = suit_box
        rank_bin = roi_bin[ry:ry+rh, rx:rx+rw]
        suit_bin = roi_bin[sy:sy+sh, sx:sx+sw]
        rank_gray = roi_gray[ry:ry+rh, rx:rx+rw]

        suit_name, suit_sc = self._classify_suit(suit_bin)
        rank_name, rank_sc = self._classify_rank(rank_gray, rank_bin)

        if not suit_name or not rank_name:
            return None

        # translate boxes to full-image coords (xyxy)
        rank_xyxy = [x1+rx, y1+ry, x1+rx+rw, y1+ry+rh]
        suit_xyxy = [x1+sx, y1+sy, x1+sx+sw, y1+sy+sh]

        return {
            "corner": "top-left",
            "suit": suit_name,
            "suit_score": float(round(suit_sc, 3)),
            "suit_bbox_xyxy": list(map(int, suit_xyxy)),
            "rank": rank_name,
            "rank_score": float(round(rank_sc, 3)),
            "rank_bbox_xyxy": list(map(int, rank_xyxy)),
        }


# ---------------- Debug analyzer (returns geometry for drawing) ----------------
def analyze_image(reader, bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    roi_gray, (x1, y1, x2, y2) = reader._top_left_roi(gray)
    dbg = {
        "roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "split_y_abs": int(y1 + reader.split_y * (y2 - y1)),
        "left_band_abs": int(x1 + reader.left_band * (x2 - x1)),
        "rank_box_roi": None,
        "suit_box_roi": None,
    }
    rec = None
    if roi_gray.size:
        roi_bin = morph_clean(best_binary(roi_gray))
        rs = reader._split_rank_suit(roi_bin)
        if rs is not None:
            rank_box, suit_box = rs
            if rank_box is not None:
                dbg["rank_box_roi"] = [int(v) for v in rank_box]
            if suit_box is not None:
                dbg["suit_box_roi"] = [int(v) for v in suit_box]
            if rank_box is not None and suit_box is not None:
                rx, ry, rw, rh = rank_box
                sx, sy, sw, sh = suit_box
                rank_bin = roi_bin[ry:ry+rh, rx:rx+rw]
                suit_bin = roi_bin[sy:sy+sh, sx:sx+sw]
                rank_gray = roi_gray[ry:ry+rh, rx:rx+rw]
                suit_name, suit_sc = reader._classify_suit(suit_bin)
                rank_name, rank_sc = reader._classify_rank(rank_gray, rank_bin)
                if suit_name and rank_name:
                    rank_xyxy = [x1+rx, y1+ry, x1+rx+rw, y1+ry+rh]
                    suit_xyxy = [x1+sx, y1+sy, x1+sx+sw, y1+sy+sh]
                    rec = {
                        "corner": "top-left",
                        "suit": suit_name,
                        "suit_score": float(round(suit_sc, 3)),
                        "suit_bbox_xyxy": list(map(int, suit_xyxy)),
                        "rank": rank_name,
                        "rank_score": float(round(rank_sc, 3)),
                        "rank_bbox_xyxy": list(map(int, rank_xyxy)),
                    }
    return rec, dbg

# ---------------- Batch runner ----------------
def draw_box(img: np.ndarray, box: List[int], color=(0, 255, 0), txt: str = ""):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if txt:
        cv2.putText(img, txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def process_folder(in_dir: Path, out_json: Path, debug_dir: Optional[Path],
                   roi_w: float, roi_h: float, split_y: float,
                   left_band: float, max_rel_area: float, min_area: int):
    reader = FixedLayoutReader(roi_w=roi_w, roi_h=roi_h, split_y=split_y,
                             min_area=min_area, left_band=left_band, max_rel_area=max_rel_area)
    print(f"Params -> roi_w={roi_w}, roi_h={roi_h}, split_y={split_y}, left_band={left_band}, max_rel_area={max_rel_area}, min_area={min_area}")

    files = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    results = []
    if debug_dir:
        ensure_dir(debug_dir)

    for p in files:
        img = imread_bgr(p)
        rec, dbg = analyze_image(reader, img)
        if rec is None:
            results.append({"file": p.name, "ok": False})
            if debug_dir:
                vis = img.copy()
                # ROI rectangle
                rx1, ry1, rx2, ry2 = dbg["roi_xyxy"]
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0,255,255), 2)
                # split line
                sy_abs = dbg["split_y_abs"]
                cv2.line(vis, (rx1, sy_abs), (rx2, sy_abs), (255,255,0), 2)
                # left band line
                lx_abs = dbg["left_band_abs"]
                #cv2.line(vis, (lx_abs, ry1), (lx_abs, ry2), (255,0,255), 2)
                # candidate boxes if any
                if dbg.get("rank_box_roi"):
                    rbx, rby, rbw, rbh = dbg["rank_box_roi"]
                    cv2.rectangle(vis, (rx1+rbx, ry1+rby), (rx1+rbx+rbw, ry1+rby+rbh), (255,0,0), 3)
                if dbg.get("suit_box_roi"):
                    sbx, sby, sbw, sbh = dbg["suit_box_roi"]
                    cv2.rectangle(vis, (rx1+sbx, ry1+sby), (rx1+sbx+sbw, ry1+sby+sbh), (0,255,0), 3)
                cv2.imencode('.jpg', vis)[1].tofile(str(debug_dir / f"{p.stem}_layout.jpg"))
            continue

        card_symbol = {"spades": "♠", "hearts": "♥", "diamonds": "♦", "clubs": "♣"}[rec["suit"]]
        out = {
            "file": p.name,
            "ok": True,
            "rank": rec["rank"],
            "suit": rec["suit"],
            "rank_score": rec["rank_score"],
            "suit_score": rec["suit_score"],
            "corner": rec["corner"],
            "rank_bbox_xyxy": rec["rank_bbox_xyxy"],
            "suit_bbox_xyxy": rec["suit_bbox_xyxy"],
            "card": f"{rec['rank']}{card_symbol}"
        }
        results.append(out)

        if debug_dir:
            vis = img.copy()
            # ROI rectangle
            rx1, ry1, rx2, ry2 = dbg["roi_xyxy"]
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0,255,255), 2)
            # split line
            sy_abs = dbg["split_y_abs"]
            cv2.line(vis, (rx1, sy_abs), (rx2, sy_abs), (255,255,0), 2)
            # left band line
            lx_abs = dbg["left_band_abs"]
            #cv2.line(vis, (lx_abs, ry1), (lx_abs, ry2), (255,0,255), 2)
            # chosen boxes
            draw_box(vis, out["suit_bbox_xyxy"], (0, 200, 0), f"{out['suit']} {out['suit_score']:.2f}")
            draw_box(vis, out["rank_bbox_xyxy"], (255, 0, 0), f"{out['rank']} {out['rank_score']:.2f}")
            cv2.putText(vis, f"{out['card']} (TL)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imencode('.jpg', vis)[1].tofile(str(debug_dir / f"{p.stem}_layout.jpg"))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(files)} images. Wrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="out_cards", help="folder with cropped cards")
    ap.add_argument("--out_json", default="out_cards_indices.json")
    ap.add_argument("--debug", action="store_true", help="save debug overlays")
    ap.add_argument("--debug_dir", default="_debug_indices")
    ap.add_argument("--roi_w", type=float, default=0.18, help="fraction of width for top-left ROI")
    ap.add_argument("--roi_h", type=float, default=0.296, help="fraction of height for top-left ROI")
    ap.add_argument("--split_y", type=float, default=0.598, help="Y split (fraction of ROI height) between rank and suit")
    ap.add_argument("--left_band", type=float, default=0.513, help="keep blobs whose center-x <= left_band * ROI_W")
    ap.add_argument("--max_rel_area", type=float, default=0.15, help="reject blobs with area > max_rel_area * ROI_area")
    ap.add_argument("--min_area", type=int, default=18, help="min contour area in pixels between rank and suit")
    args = ap.parse_args()

    process_folder(Path(args.in_dir), Path(args.out_json),
                   Path(args.debug_dir) if args.debug else None,
                   roi_w=args.roi_w, roi_h=args.roi_h, split_y=args.split_y,
                   left_band=args.left_band, max_rel_area=args.max_rel_area, min_area=args.min_area)


if __name__ == "__main__":
    main()
