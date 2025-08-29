# api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import subprocess
import json
import uuid

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Зачувај ја сликата со уникатно име
    temp_image_name = f"uploads/{uuid.uuid4().hex}.jpg"
    with open(temp_image_name, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. Повикај ја main.py логиката
        result = subprocess.run(
            ["python", "main.py", temp_image_name],
            capture_output=True,
            text=True
        )

        # 3. Прочитај го JSON фајлот што го создава match_all.py
        final_path = Path("final_cards.json")
        if not final_path.exists():
            return JSONResponse(content={"error": "No cards detected"}, status_code=400)

        with open(final_path, "r") as f:
            cards = json.load(f)

        return {"cards": cards}
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # 4. Исчисти привремени фајлови
        Path(temp_image_name).unlink(missing_ok=True)
