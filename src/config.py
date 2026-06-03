from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "semeval"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings_cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
TEXT_DIR = OUTPUTS_DIR / "text"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = RAW_DIR / "Train.txt"
VALIDATION_FILE = RAW_DIR / "Validation.txt"
TEST_FILE = RAW_DIR / "Test.txt"

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000
EVAL_TEST_SIZE = 0.20
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def ensure_project_dirs() -> None:
    for directory in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        EMBEDDINGS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        TEXT_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
