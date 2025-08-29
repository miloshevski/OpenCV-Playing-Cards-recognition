import subprocess
import sys
import os
from pathlib import Path

# 1. Валидација на аргумент
image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path or not Path(image_path).exists():
    # print("❌ Внеси постоечка .jpg слика: python main.py test.jpg")
    sys.exit(1)

# 2. Користи истата Python патека (од .venv)
py = sys.executable

# 3. Бришење на .jpg/.png слики од дадена папка
def clear_images(folder: str, name: str):
    path = Path(folder)
    count = 0
    extensions = [".jpg", ".png", ".jpeg", ".webp"]
    if path.exists():
        for file in path.iterdir():
            if file.suffix.lower() in extensions:
                try:
                    file.unlink()
                    count += 1
                except Exception as e:
                    print(f"❌ Не може да се избрише {file.name}: {e}")
        # print(f"🧹 Избришани се {count} слики од {folder}/ ({name})")

# 4. Чистење на папки
clear_images("out_cards", "warped cards")
clear_images("_debug_indices", "debug индекси")
clear_images("crops", "binarized crops")  # <--- ново

# 5. Чекор 1: Детектирање на карти
# print("1️⃣ Детектирање на карти...")
subprocess.run([
    py, "cards_otsu_detect.py", image_path,
    "--warp-dir", "out_cards",
    "--mask-out", "mask.png",
    "--panel-out", "panel.jpg"
])

# 6. Чекор 2: Читање на индекси
# print("\n2️⃣ Читање на индекси (ранг и боја)...")
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

# 7. Чекор 3
# print("\n3️⃣ Извлекување binarized crops (rank/suit)...")
subprocess.run([py, "extract_crops_and_predict.py"])

# 8. Чекор 4
# print("\n4️⃣ Сечење и центрирање на crops...")
subprocess.run([py, "seci.py"])

# 9. Чекор 5
# print("\n5️⃣ Препознавање карти од crops...")
subprocess.run([py, "match_all.py"])
