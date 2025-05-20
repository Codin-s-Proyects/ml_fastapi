"""
Entrenamiento y persistencia de RandomForestClassifier balanceado.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from loguru import logger

from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS, MAX_DEPTH


def train(df: pd.DataFrame) -> dict[str, float]:
    X = df.drop(columns=["Diario"])
    y = df["Diario"]

    # Detectar columnas categóricas
    non_numeric_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Columnas categóricas detectadas: {non_numeric_cols}")
        # Convertir columnas categóricas a variables dummy
        X = pd.get_dummies(X, columns=non_numeric_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro"),
    }

    logger.info("RandomForest metrics:\n" + classification_report(y_test, y_pred))
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_PATH)
    logger.info(f"Modelo y columnas guardados en {MODEL_PATH}")

    return metrics
