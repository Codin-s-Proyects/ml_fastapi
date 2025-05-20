"""
FastAPI – endpoints REST completos.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from src.analysis.stats import detect_outliers
from src.config import DATA_DIR, MODEL_PATH, REPORT_DIR
from src.data_ingestion.load_file import load_txt
from src.preprocessing.clean_data import clean
from src.ml.train import train as train_model
from src.ml.predict import predict_one, ModelNotTrainedError
from src.reports.generate_excel import to_excel
from src.reports.generate_pdf import to_pdf

app = FastAPI(title="Accounting ML Backend", version="1.0.0")

# --- CORS (Next.js front local por defecto) ---------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- Schemas -----------------------------


class PredictIn(BaseModel):
    Ruc: int
    Debe_MN: float
    Haber_MN: float
    Debe_ME: float
    Haber_ME: float
    Saldo_MN: float
    Saldo_ME: float
    # … incluir cualquier feature adicional usada en el entrenamiento


class PredictOut(BaseModel):
    diario_predicho: int = Field(..., alias="diarioPredicho")


class MetricsOut(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


# ------------------------- Endpoints ----------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process_file(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Solo se permiten .txt")

    raw_path = DATA_DIR / file.filename
    with raw_path.open("wb") as f:
        f.write(await file.read())

    df_raw = load_txt(raw_path)
    df_clean = clean(df_raw)
    df_clean.to_pickle(DATA_DIR / "clean.pkl")
    logger.info("Archivo procesado y guardado clean.pkl")

    return {"detail": "Archivo procesado correctamente"}


@app.post("/train-model", response_model=MetricsOut)
def train_endpoint():
    clean_path = DATA_DIR / "clean.pkl"
    if not clean_path.exists():
        raise HTTPException(status_code=400, detail="No existe clean.pkl. Sube un archivo primero.")

    df = pd.read_pickle(clean_path)
    metrics = train_model(df)
    return metrics


@app.post("/predict", response_model=PredictOut)
def predict_endpoint(data: PredictIn):
    try:
        pred = predict_one(data.model_dump())
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictOut(diarioPredicho=pred)


@app.get("/download-excel")
def download_excel():
    clean_path = DATA_DIR / "clean.pkl"
    if not clean_path.exists():
        raise HTTPException(status_code=400, detail="No existe clean.pkl. Sube un archivo primero.")

    df = pd.read_pickle(clean_path)
    outliers = detect_outliers(
        df,
        ["Debe_MN", "Haber_MN", "Debe_ME", "Haber_ME"],
    )
    excel_path = REPORT_DIR / "resultado.xlsx"
    to_excel(outliers, excel_path)
    return FileResponse(excel_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/download-pdf")
def download_pdf():
    clean_path = DATA_DIR / "clean.pkl"
    if not clean_path.exists():
        raise HTTPException(status_code=400, detail="No existe clean.pkl. Sube un archivo primero.")

    df = pd.read_pickle(clean_path)
    outliers = detect_outliers(
        df,
        ["Debe_MN", "Haber_MN", "Debe_ME", "Haber_ME"],
    )
    pdf_path = REPORT_DIR / "resultado.pdf"
    to_pdf(outliers, pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf")
