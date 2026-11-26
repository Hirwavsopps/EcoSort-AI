import os
import time
import json
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from src.predictor import EcoSortPredictor

ROOT = Path(os.getenv("ECOSORT_PROJECT_DIR", Path(__file__).resolve().parent.parent))
MODEL_DIR = ROOT / "models"
predictor = EcoSortPredictor(
    model_path=str(MODEL_DIR / json.loads((MODEL_DIR / "model_metadata.json").read_text())["model_file"]),
    metadata_path=str(MODEL_DIR / "model_metadata.json"),
    rules_path=str(ROOT / "disposal_rules.json"),
)
API_STARTED_AT = time.time()

app = FastAPI(title="EcoSort AI API", version="1.0.0",
              description="Waste material classification API")

async def run_prediction(upload: UploadFile):
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        from io import BytesIO
        return predictor.predict(BytesIO(content))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await upload.close()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor.model is not None,
        "model_version": predictor.metadata["model_version"],
        "uptime_seconds": round(time.time() - API_STARTED_AT, 2)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await run_prediction(file)
    return {"filename": file.filename, **result}

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    results = []
    for upload in files:
        try:
            result = await run_prediction(upload)
            results.append({"filename": upload.filename, **result})
        except HTTPException as exc:
            results.append({"filename": upload.filename, "error": exc.detail})
    return {"count": len(results), "results": results}
