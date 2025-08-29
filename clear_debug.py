import os
from pathlib import Path

debug_dir = Path("_debug_indices")
extensions = [".jpg", ".png", ".jpeg", ".webp"]

if not debug_dir.exists():
    print("📂 Папката _debug_indices не постои.")
    exit()

count = 0
for file in debug_dir.iterdir():
    if file.suffix.lower() in extensions:
        try:
            file.unlink()
            count += 1
        except Exception as e:
            print(f"❌ Не може да се избрише {file.name}: {e}")

print(f"✅ Избришани се {count} слики од _debug_indices/")
