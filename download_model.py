from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "embedding_model"


def main() -> None:
    LOCAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(LOCAL_MODEL_DIR))
    print(f"Saved {MODEL_NAME} to {LOCAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
