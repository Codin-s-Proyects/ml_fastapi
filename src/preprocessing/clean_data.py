"""
Funciones de limpieza, casting y completado de 'Diario' / 'Sub_diario'.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from src.config import COLUMN_MAP, DIARIO_DICT


def rename_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas y castea tipos; agrega ID incremental."""
    df = df.rename(columns=COLUMN_MAP).copy()
    df["ID"] = range(1, len(df) + 1)

    numeric_cols = [
        "Ruc",
        "Diario",
        "Debe_MN",
        "Debe_ME",
        "Haber_MN",
        "Haber_ME",
        "Saldo_MN",
        "Saldo_ME",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Ruc"] = df["Ruc"].astype("Int64")
    df["Diario"] = df["Diario"].astype("Int64")

    logger.info("Columnas renombradas y casteadas")
    return df


def completar_diario(row: pd.Series) -> pd.Series:
    """
    Completa 'Diario' y 'Sub_diario' a partir de DIARIO_DICT.
    Reglas:
      – Si ambos faltan → misc.
      – Si falta uno → se infiere del otro.
    """
    diario = row.get("Diario")
    sub = row.get("Sub_diario")

    if pd.isna(diario) and pd.isna(sub):
        row["Diario"] = 709
        row["Sub_diario"] = "MISCELANEOS"
        return row

    if pd.isna(diario) and isinstance(sub, str):
        for k, v in DIARIO_DICT.items():
            if v == sub:
                row["Diario"] = int(k)
                return row
        row["Diario"] = 709

    if pd.isna(sub) and not pd.isna(diario):
        row["Sub_diario"] = DIARIO_DICT.get(str(int(diario))[:3], "MISCELANEOS")

    return row


def imputar_numericos(row: pd.Series) -> pd.Series:
    """Imputa MN/ME siguiendo las ecuaciones de balance."""
    # MN
    if pd.isna(row["Saldo_MN"]):
        row["Saldo_MN"] = row["Debe_MN"] - row["Haber_MN"]
    if pd.isna(row["Debe_MN"]):
        row["Debe_MN"] = row["Saldo_MN"] + row["Haber_MN"]
    if pd.isna(row["Haber_MN"]):
        row["Haber_MN"] = row["Debe_MN"] - row["Saldo_MN"]

    # ME
    if pd.isna(row["Saldo_ME"]):
        row["Saldo_ME"] = row["Debe_ME"] - row["Haber_ME"]
    if pd.isna(row["Debe_ME"]):
        row["Debe_ME"] = row["Saldo_ME"] + row["Haber_ME"]
    if pd.isna(row["Haber_ME"]):
        row["Haber_ME"] = row["Debe_ME"] - row["Saldo_ME"]

    return row


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpieza."""
    df = rename_cast(df)

    # Rellenar nulos de Diario/Sub_diario
    df["Diario"].fillna(709, inplace=True)
    df["Sub_diario"].fillna("MISCELANEOS", inplace=True)

    # Reglas por fila
    df = df.apply(completar_diario, axis=1)
    df = df.apply(imputar_numericos, axis=1)

    # Elimina duplicados globales
    before = df.shape[0]
    df = df.drop_duplicates()
    logger.info(f"Duplicados eliminados: {before - df.shape[0]}")

    return df
