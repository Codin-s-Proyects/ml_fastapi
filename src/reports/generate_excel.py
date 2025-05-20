import pandas as pd
from pathlib import Path
from loguru import logger


def to_excel(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df.to_excel(path, index=False)
    logger.info(f"Excel guardado en {path}")
    return path
