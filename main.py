import subprocess
import sys
from pathlib import Path

image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path or not Path(image_path).exists():
    print("❌ Внеси постоечка .jpg слика: python main.py test.jpg")
    sys.exit(1)

# Користи истата Python патека (од .venv)
py = sys.executable

print("1️⃣ Детектирање на карти...")
subprocess.run([
    py, "cards_otsu_detect.py", image_path,
    "--warp-dir", "out_cards",
    "--mask-out", "mask.png",
    "--panel-out", "panel.jpg"
])

print("\n2️⃣ Читање на индекси (ранг и боја)...")
subprocess.run([
    py, "read_indices_from_out_cards_v2.py",
    "--roi_w", "0.18",
    "--roi_h", "0.312",
    "--split_y", "0.568",
    "--left_band", "0.48",
    "--max_rel_area", "0.15",
    "--min_area", "19",
    "--debug"
])

print("\n3️⃣ Извлекување binarized crops (rank/suit)...")
subprocess.run([py, "extract_crops_and_predict.py"])

print("\n4️⃣ Сечење и центрирање на crops...")
subprocess.run([py, "seci.py"])

print("\n5️⃣ Препознавање карти од crops...")
subprocess.run([py, "match_all.py"])
