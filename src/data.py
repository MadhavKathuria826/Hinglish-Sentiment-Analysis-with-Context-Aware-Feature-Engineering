from __future__ import annotations

import pandas as pd

from src.config import LABELS


LABEL_MAP = {
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "2": "neutral",
}


def clean_label(label: object) -> str | None:
    if label is None or pd.isna(label):
        return None
    value = str(label).strip().lower()
    return LABEL_MAP.get(value)


def parse_semeval_file(path) -> pd.DataFrame:
    """Parse SemEval SentiMix-style token files into one row per sentence."""
    rows: list[dict[str, object]] = []
    current_id: str | None = None
    current_label: str | None = None
    current_tokens: list[str] = []

    def flush_current() -> None:
        if current_id is None:
            return
        rows.append(
            {
                "id": current_id,
                "text": " ".join(current_tokens).strip(),
                "label": current_label,
            }
        )

    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if parts[0] == "meta":
                flush_current()
                current_id = parts[1].strip() if len(parts) > 1 else None
                current_label = clean_label(parts[2]) if len(parts) > 2 else None
                current_tokens = []
                continue
            current_tokens.append(parts[0].strip())

    flush_current()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].map(clean_label)
    return df


def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df[df["label"].isin(LABELS)].copy()
    return clean_df[["text", "label"]].reset_index(drop=True)


def dataset_profile(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    combined = pd.concat([labeled_only(train_df), labeled_only(validation_df)], ignore_index=True)
    return {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "official_test_rows": int(len(test_df)),
        "official_test_has_labels": bool(test_df["label"].notna().any()) if not test_df.empty else False,
        "combined_labeled_rows": int(len(combined)),
        "combined_class_distribution": combined["label"].value_counts().sort_index().to_dict(),
        "duplicate_texts_in_combined": int(combined.duplicated(subset=["text"]).sum()),
        "unique_texts_in_combined": int(combined["text"].nunique()),
        "average_words": float(combined["text"].str.split().str.len().mean()),
    }
