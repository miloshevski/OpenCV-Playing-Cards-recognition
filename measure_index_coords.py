from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np

"""
Interactive measurer for a single cropped card image.
Draw TWO rectangles in order: (1) RANK (top-left number/letter), (2) SUIT (small icon below).
Outputs W,H and the coordinates, plus suggested parameters for read_indices_from_out_cards_v2.py
"""

def pick_image(in_dir: Path | None, image: Path | None) -> Path:
    if image and image.exists():
        return image
    if in_dir is None:
        in_dir = Path("out_cards")
    imgs = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not imgs:
        raise SystemExit(f"No images found in {in_dir}")
    return imgs[0]


def suggest_params(W:int,H:int, rx1:int,ry1:int,rx2:int,ry2:int, sx1:int,sy1:int,sx2:int,sy2:int):
    # ROI size with small margin
    roi_w = (max(rx2, sx2) / W) * 1.08
    roi_h = (max(ry2, sy2) / H) * 1.08
    roi_w = float(max(0.18, min(0.6, roi_w)))
    roi_h = float(max(0.20, min(0.6, roi_h)))

    # split_y: halfway between rank bottom and suit top, normalized by ROI_H
    split_abs_y = (ry2 + sy1) / 2.0
    split_y = float(split_abs_y / (H * roi_h))
    split_y = float(max(0.30, min(0.80, split_y)))

    # left_band based on centers within ROI
    cx_rank = (rx1 + rx2)/2.0
    cx_suit = (sx1 + sx2)/2.0
    roi_w_px = W * roi_w
    left_band = 1.05 * max(cx_rank, cx_suit) / roi_w_px
    left_band = float(min(max(left_band, 0.48), 0.65))

    # area-based guards
    ROI_area = (W*roi_w) * (H*roi_h)
    area_rank = (rx2-rx1) * (ry2-ry1)
    area_suit = (sx2-sx1) * (sy2-sy1)
    max_rel_area = min( 2.5 * max(area_rank, area_suit) / ROI_area, 0.15 )
    min_area = max(int(0.004 * ROI_area), 15)

    return {
        "roi_w": round(roi_w,3),
        "roi_h": round(roi_h,3),
        "split_y": round(split_y,3),
        "left_band": round(left_band,3),
        "max_rel_area": round(float(max_rel_area),3),
        "min_area": int(min_area),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="", help="path to one cropped card image")
    ap.add_argument("--in_dir", type=str, default="out_cards", help="folder with cropped cards (used if --image not set)")
    ap.add_argument("--out", type=str, default="_measure", help="folder to save JSON/overlay")
    args = ap.parse_args()

    img_path = pick_image(Path(args.in_dir), Path(args.image) if args.image else None)
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read: {img_path}")

    H, W = img.shape[:2]
    view = img.copy()
    cv2.putText(view, "Select RANK box (top-left). ENTER to confirm.", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.namedWindow("measure", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("measure", min(1100,W), int(min(1100,W)*H/max(W,1)))

    # Use selectROIs to collect two boxes
    rois = cv2.selectROIs("measure", view, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("measure")

    if len(rois) < 2:
        print("Please draw TWO rectangles: first RANK, then SUIT.")
        return

    # interpret first as rank, second as suit
    (rx, ry, rw, rh) = rois[0]
    (sx, sy, sw, sh) = rois[1]

    rx1, ry1, rx2, ry2 = int(rx), int(ry), int(rx+rw), int(ry+rh)
    sx1, sy1, sx2, sy2 = int(sx), int(sy), int(sx+sw), int(sy+sh)

    params = suggest_params(W,H, rx1,ry1,rx2,ry2, sx1,sy1,sx2,sy2)

    # Save overlay and JSON
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay = img.copy()
    cv2.rectangle(overlay, (rx1,ry1), (rx2,ry2), (255,0,0), 2)
    cv2.rectangle(overlay, (sx1,sy1), (sx2,sy2), (0,255,0), 2)
    txt = f"W={W} H={H} | R=({rx1},{ry1},{rx2},{ry2}) S=({sx1},{sy1},{sx2},{sy2})"
    cv2.putText(overlay, txt, (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    out_img = out_dir / f"{img_path.stem}_measure.jpg"
    cv2.imencode('.jpg', overlay)[1].tofile(str(out_img))

    data = {
        "image": str(img_path),
        "W": W, "H": H,
        "rank_box_xyxy": [rx1,ry1,rx2,ry2],
        "suit_box_xyxy": [sx1,sy1,sx2,sy2],
        "suggested_params": params,
        "suggested_command": (
            f"python read_indices_from_out_cards_v2.py --roi_w {params['roi_w']} "
            f"--roi_h {params['roi_h']} --split_y {params['split_y']}"
        )
    }
    out_json = out_dir / f"{img_path.stem}_measure.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("=== MEASUREMENTS ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Saved overlay: {out_img}")
    print(f"Saved JSON:    {out_json}")


if __name__ == "__main__":
    main()
