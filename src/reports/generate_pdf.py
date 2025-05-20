from pathlib import Path

import matplotlib.pyplot as plt
from pandas.plotting import table as mpl_table
import pandas as pd
from loguru import logger


def to_pdf(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.25 + 2))
    ax.axis("off")
    mpl_table(ax, df, loc="center")
    plt.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)
    logger.info(f"PDF guardado en {path}")
    return path
