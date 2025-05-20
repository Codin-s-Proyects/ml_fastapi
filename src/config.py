from pathlib import Path
from loguru import logger

# --- Paths ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml"
REPORT_DIR = BASE_DIR / "reports"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"

# --- Column names mapping --------------------------------------
COLUMN_MAP = {
    "Mon.": "Moneda",
    "Debe1": "Debe_MN",
    "Haber1": "Haber_MN",
    "Saldo1": "Saldo_MN",
    "Debe": "Debe_ME",
    "Haber": "Haber_ME",
    "Saldo": "Saldo_ME",
}

# --- Business dictionaries -------------------------------------
DIARIO_DICT = {
    "700": "ASIENTO DE APERTURA",
    "701": "CAJA INGRESOS",
    "702": "CAJA EGRESOS",
    "703": "VENTAS",
    # … resto omitido por brevedad, incluye los 799
    "799": "DIFERENCIA DE CAMBIO",
}

# --- ML Settings -----------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_DEPTH = None

# --- Logging ----------------------------------------------------
logger.add("logs/backend.log", rotation="100 KB", retention="10 days")
