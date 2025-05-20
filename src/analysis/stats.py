"""
Cálculo de estadísticas y detección de outliers (μ ± 2σ).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


def detect_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Devuelve DataFrame con registros fuera de μ ± 2σ para cada col de *cols*,
    añadiendo columna 'observacion'.
    """
    out_frames = []
    for col in cols:
        mu, sigma = df[col].mean(), df[col].std(ddof=1)
        lower, upper = mu - 2 * sigma, mu + 2 * sigma
        mask = (df[col] < lower) | (df[col] > upper)
        tmp = df.loc[mask].copy()
        tmp["observacion"] = np.where(
            tmp[col] > upper,
            f"Mayor al intervalo {col}",
            f"Menor al intervalo {col}",
        )
        out_frames.append(tmp)

        logger.info(f"{col}: μ={mu:.2f}, σ={sigma:.2f}, outliers={mask.sum()}")

    resultado_final = (
        pd.concat(out_frames, axis=0).drop_duplicates().sort_values("ID").reset_index(drop=True)
    )
    return resultado_final
