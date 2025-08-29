import subprocess
import sys
import json
from pathlib import Path
import shutil

def run_pipeline(image_path: str) -> list[str]:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"❌ Не постои сликата: {image_path}")

    # Користи истата Python патека (.venv)
    py = sys.executable

    # Исчисти стари резултати
    for folder in ["out_cards", "crops"]:
        f = Path(folder)
        if f.exists() and f.is_dir():
            shutil.rmtree(f)
        f.mkdir(parents=True, exist_ok=True)

    print("1️⃣ Детектирање на карти...")
    subprocess.run([
        py, "cards_otsu_detect.py", image_path,
        "--warp-dir", "out_cards",
        "--mask-out", "mask.png",
        "--panel-out", "panel.jpg"
    ], check=True)

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
    ], check=True)

    print("\n3️⃣ Извлекување binarized crops (rank/suit)...")
    subprocess.run([py, "extract_crops_and_predict.py"], check=True)

    print("\n4️⃣ Сечење и центрирање на crops...")
    subprocess.run([py, "seci.py"], check=True)

    print("\n5️⃣ Препознавање карти од crops...")
    subprocess.run([py, "match_all.py"], check=True)

    # Прочитај JSON резултати (пр. од match_all.py)
    results_path = Path("final_cards.json")
    if not results_path.exists():
        raise FileNotFoundError("❌ Не се пронајдени резултати од match_all.py (final_cards.json)")

    with open(results_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    return cards  # пример: ["7c", "As", "10d"]

# Ако го повикуваш од терминал:
if __name__ == "__main__":
    detected = run_pipeline(sys.argv[1])
    print("\n🂠 Детектирани карти:", ", ".join(detected))
