from __future__ import annotations

import numpy as np
from pathlib import Path

from src.config import EMBEDDING_MODEL_NAME, MODELS_DIR


def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    local_model_dir = MODELS_DIR / "embedding_model"
    if local_model_dir.exists():
        return SentenceTransformer(str(local_model_dir))
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def encode_texts(
    texts: list[str],
    cache_path: Path | None = None,
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    model = load_sentence_transformer()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype("float32")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings
