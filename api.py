from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import subprocess
import json
import uuid
import time

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Зачувај ја сликата со уникатно име
    Path("uploads").mkdir(exist_ok=True)
    temp_image_path = Path("uploads") / f"{uuid.uuid4().hex}.jpg"
    with open(temp_image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. Повикај ја main.py со таа слика
        result = subprocess.run(
            ["python", "main.py", str(temp_image_path)],
            capture_output=True,
            text=True
        )

        # 3. Додај мала пауза во случај JSON да не се запише веднаш
        time.sleep(0.3)

        final_json_path = Path("final_cards.json")
        if not final_json_path.exists():
            return JSONResponse(
                content={"error": "❌ Не се пронајдени детектирани карти"},
                status_code=400
            )

        with open(final_json_path, "r", encoding="utf-8") as f:
            cards = json.load(f)

        return {"cards": cards}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # 4. Исчисти темп фајлови
        temp_image_path.unlink(missing_ok=True)
        Path("final_cards.json").unlink(missing_ok=True)
        Path("mask.png").unlink(missing_ok=True)
        Path("panel.jpg").unlink(missing_ok=True)
