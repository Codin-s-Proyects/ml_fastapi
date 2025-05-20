"""
Carga archivos de texto contables, detecta encoding y devuelve un DataFrame.
"""

from pathlib import Path
import chardet
import pandas as pd
from loguru import logger


def detect_encoding(path: Path, sample_size: int = 10000) -> str:
    """Detecta el encoding leyendo los primeros *sample_size* bytes."""
    with path.open("rb") as f:
        raw = f.read(sample_size)
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    logger.info(f"Encoding detectado para {path.name}: {enc}")
    return enc


def load_txt(path: str | Path, delimiter: str = ";") -> pd.DataFrame:
    """Lee el .txt contable y retorna DataFrame crudo."""
    path = Path(path)
    enc = detect_encoding(path)
    df = pd.read_csv(
        path,
        encoding=enc,
        delimiter=delimiter,
        on_bad_lines="skip",
        dtype=str,  # leer todo como str para limpieza posterior
    )
    logger.info(f"Archivo {path.name} cargado con shape {df.shape}")
    return df
