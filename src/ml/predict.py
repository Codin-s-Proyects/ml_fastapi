"""
Cargador del modelo y función de predicción.
"""

from __future__ import annotations

import joblib
import pandas as pd
from loguru import logger
from src.config import MODEL_PATH


class ModelNotTrainedError(RuntimeError):
    ...


def _load_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError("Modelo no entrenado. Ejecuta /train-model primero.")
    data = joblib.load(MODEL_PATH)
    return data["model"], data["columns"]


def predict_one(payload: dict) -> int:
    model, expected_columns = _load_model()
    X = pd.DataFrame([payload])

    # Convertir columnas categóricas a dummies igual que en entrenamiento
    X = pd.get_dummies(X)

    # Asegurar que todas las columnas esperadas están presentes
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0

    # Reordenar columnas en el mismo orden que el entrenamiento
    X = X[expected_columns]

    pred = int(model.predict(X)[0])
    logger.debug(f"Predicción realizada: {pred}")
    return pred
