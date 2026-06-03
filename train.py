from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import (
    EVAL_TEST_SIZE,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL_NAME,
    FIGURES_DIR,
    LABELS,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TEST_FILE,
    TEXT_DIR,
    TRAIN_FILE,
    VALIDATION_FILE,
    ensure_project_dirs,
)
from src.data import dataset_profile, labeled_only, parse_semeval_file
from src.embeddings import encode_texts
from src.modeling import (
    MODEL_DISPLAY_NAMES,
    MODEL_FACTORIES,
    MODEL_TYPES,
    build_vectorizer,
    evaluate_model,
    prepare_embedding_texts,
    prepare_texts,
)
from src.preprocessing import analyze_error_type, preprocess_text
from src.reporting import generate_latex_report


TFIDF_KEYS = ["tfidf_logistic_regression", "tfidf_svm"]
EMBEDDING_KEYS = ["embedding_logistic_regression", "embedding_svm"]


def save_class_distribution(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    order = [label for label in LABELS if label in set(df["label"])]
    sns.countplot(data=df, x="label", order=order, palette="Set2", hue="label", legend=False)
    plt.title("SemEval Hinglish Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_model_comparison(results: pd.DataFrame, path: Path) -> None:
    plot_df = results.melt(
        id_vars=["model", "model_type"],
        value_vars=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10.5, 5.2))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", palette="Set2")
    plt.ylim(0, 1)
    plt.title("TF-IDF Baselines vs Sentence Embedding Models")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=18, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion_matrix(matrix, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=LABELS, yticklabels=LABELS)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_error_category_chart(errors: pd.DataFrame, path: Path) -> None:
    if errors.empty:
        return
    exploded = (
        errors.assign(error_type=errors["error_type"].str.split("; "))
        .explode("error_type")
        .groupby("error_type")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=exploded, x="count", y="error_type", palette="Set2", hue="error_type", legend=False)
    plt.title("Error Analysis Categories")
    plt.xlabel("Misclassified Examples")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def normalize_hf_label(value: object) -> str | None:
    value = str(value).lower().strip()
    if value in {"positive", "pos", "1"}:
        return "positive"
    if value in {"negative", "neg", "0"}:
        return "negative"
    if value in {"neutral", "neu", "2"}:
        return "neutral"
    return None


def try_evaluate_indic_sentiment(vectorizer, final_models: dict[str, object]) -> dict[str, object]:
    """Load ai4bharat/IndicSentiment through HuggingFace for external evaluation only."""
    try:
        from datasets import DatasetDict, load_dataset
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    except Exception as exc:
        return {"status": "skipped", "reason": f"datasets package unavailable: {exc}"}

    try:
        dataset = load_dataset("ai4bharat/IndicSentiment", "translation-hi", trust_remote_code=True)
    except Exception as exc:
        return {"status": "skipped", "reason": f"HuggingFace load failed: {exc}"}

    if isinstance(dataset, DatasetDict):
        split_name = "test" if "test" in dataset else list(dataset.keys())[0]
        hf_split = dataset[split_name]
    else:
        split_name = "default"
        hf_split = dataset

    df = hf_split.to_pandas()
    text_column = "ENGLISH REVIEW" if "ENGLISH REVIEW" in df.columns else "INDIC REVIEW"
    label_column = "LABEL" if "LABEL" in df.columns else "label"
    eval_df = pd.DataFrame(
        {
            "text": df[text_column].astype(str),
            "label": df[label_column].map(normalize_hf_label),
        }
    ).dropna()
    if eval_df.empty:
        return {"status": "skipped", "reason": "No positive/negative/neutral labels after normalization"}

    eval_df = eval_df.sample(n=min(5000, len(eval_df)), random_state=RANDOM_STATE)
    processed = prepare_texts(eval_df["text"])
    embedding_texts = prepare_embedding_texts(eval_df["text"])
    x_tfidf = vectorizer.transform(processed)
    x_embedding = encode_texts(
        embedding_texts,
        EMBEDDINGS_DIR / "combo_indic_sentiment_translation_hi_test_embeddings.npy",
        show_progress_bar=False,
    )

    rows = []
    for model_key, model in final_models.items():
        features = x_tfidf if model_key in TFIDF_KEYS else x_embedding
        predictions = model.predict(features)
        precision, recall, f1, _ = precision_recall_fscore_support(
            eval_df["label"], predictions, labels=LABELS, average="weighted", zero_division=0
        )
        rows.append(
            {
                "model_key": model_key,
                "model_type": MODEL_TYPES[model_key],
                "model": MODEL_DISPLAY_NAMES[model_key],
                "accuracy": accuracy_score(eval_df["label"], predictions),
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
                "samples": len(eval_df),
            }
        )

    pd.DataFrame(rows).to_csv(TABLES_DIR / "indic_sentiment_generalization.csv", index=False)
    return {
        "status": "completed",
        "dataset": "ai4bharat/IndicSentiment",
        "config": "translation-hi",
        "split": split_name,
        "text_column": text_column,
        "samples": int(len(eval_df)),
        "results": rows,
    }


def main() -> None:
    ensure_project_dirs()

    train_raw = parse_semeval_file(TRAIN_FILE)
    validation_raw = parse_semeval_file(VALIDATION_FILE)
    official_test_raw = parse_semeval_file(TEST_FILE)

    train_df = labeled_only(train_raw)
    validation_df = labeled_only(validation_raw)
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    combined_df["processed_text"] = combined_df["text"].map(preprocess_text)

    official_test_raw[["id", "text", "label"]].to_csv(PROCESSED_DIR / "official_test_unlabeled.csv", index=False)
    combined_df[["text", "label"]].to_csv(PROCESSED_DIR / "train_validation_combined.csv", index=False)
    combined_df[["text", "processed_text", "label"]].to_csv(PROCESSED_DIR / "train_validation_processed.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)

    profile = dataset_profile(train_raw, validation_raw, official_test_raw)
    with open(TEXT_DIR / "dataset_profile.json", "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    x_train_df, x_eval_df, y_train, y_eval = train_test_split(
        combined_df[["text", "processed_text"]],
        combined_df["label"],
        test_size=EVAL_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=combined_df["label"],
    )

    tfidf_vectorizer = build_vectorizer()
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train_df["processed_text"])
    x_eval_tfidf = tfidf_vectorizer.transform(x_eval_df["processed_text"])
    x_train_embeddings = encode_texts(
        (x_train_df["text"].astype(str) + " normalized " + x_train_df["processed_text"].astype(str)).tolist(),
        EMBEDDINGS_DIR / "combo_holdout_train_embeddings.npy",
    )
    x_eval_embeddings = encode_texts(
        (x_eval_df["text"].astype(str) + " normalized " + x_eval_df["processed_text"].astype(str)).tolist(),
        EMBEDDINGS_DIR / "combo_holdout_eval_embeddings.npy",
    )

    eval_rows: list[dict[str, object]] = []
    eval_details: dict[str, dict[str, object]] = {}
    for model_key in TFIDF_KEYS + EMBEDDING_KEYS:
        model = MODEL_FACTORIES[model_key]()
        x_train = x_train_tfidf if model_key in TFIDF_KEYS else x_train_embeddings
        x_eval = x_eval_tfidf if model_key in TFIDF_KEYS else x_eval_embeddings
        model.fit(x_train, y_train)
        result = evaluate_model(model_key, model, x_eval, y_eval)
        eval_details[model_key] = result
        eval_rows.append(
            {key: value for key, value in result.items() if key not in {"predictions", "classification_report", "confusion_matrix"}}
        )
        save_confusion_matrix(
            result["confusion_matrix"],
            f"{MODEL_DISPLAY_NAMES[model_key]} Confusion Matrix",
            FIGURES_DIR / f"confusion_matrix_{model_key}.png",
        )
        with open(TEXT_DIR / f"classification_report_{model_key}.txt", "w", encoding="utf-8") as file:
            file.write(result["classification_report"])

    results_df = pd.DataFrame(eval_rows).sort_values("f1_weighted", ascending=False)
    results_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    best_key = str(results_df.iloc[0]["model_key"])
    best_result = eval_details[best_key]
    error_df = pd.DataFrame(
        {
            "text": x_eval_df["text"].reset_index(drop=True),
            "processed_text": x_eval_df["processed_text"].reset_index(drop=True),
            "actual": y_eval.reset_index(drop=True),
            "predicted": pd.Series(best_result["predictions"]),
        }
    )
    error_df = error_df[error_df["actual"] != error_df["predicted"]].copy()
    error_df["error_type"] = error_df.apply(lambda row: analyze_error_type(row["text"], row["processed_text"]), axis=1)
    error_df.to_csv(TABLES_DIR / "error_analysis.csv", index=False)

    save_class_distribution(combined_df, FIGURES_DIR / "class_distribution.png")
    save_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")
    save_confusion_matrix(
        best_result["confusion_matrix"],
        f"Best Model Confusion Matrix: {MODEL_DISPLAY_NAMES[best_key]}",
        FIGURES_DIR / "confusion_matrix_best.png",
    )
    save_error_category_chart(error_df, FIGURES_DIR / "error_categories.png")

    final_vectorizer = build_vectorizer()
    x_full_tfidf = final_vectorizer.fit_transform(combined_df["processed_text"])
    x_full_embedding = encode_texts(
        (combined_df["text"].astype(str) + " normalized " + combined_df["processed_text"].astype(str)).tolist(),
        EMBEDDINGS_DIR / "combo_full_train_validation_embeddings.npy",
    )
    joblib.dump(final_vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    final_models: dict[str, object] = {}
    for model_key in TFIDF_KEYS + EMBEDDING_KEYS:
        final_model = MODEL_FACTORIES[model_key]()
        features = x_full_tfidf if model_key in TFIDF_KEYS else x_full_embedding
        final_model.fit(features, combined_df["label"])
        final_models[model_key] = final_model
        joblib.dump(final_model, MODELS_DIR / f"{model_key}.pkl")

    metadata = {
        "labels": LABELS,
        "model_display_names": MODEL_DISPLAY_NAMES,
        "model_types": MODEL_TYPES,
        "best_model_key": best_key,
        "best_model": MODEL_DISPLAY_NAMES[best_key],
        "best_model_type": MODEL_TYPES[best_key],
        "training_rows": int(len(combined_df)),
        "tfidf_max_features": final_vectorizer.max_features,
        "ngram_range": final_vectorizer.ngram_range,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "official_test_has_gold_labels": profile["official_test_has_labels"],
        "evaluation_note": "Official Test.txt has no gold labels in the provided files; metrics use a stratified holdout from the labeled train+validation pool.",
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    official_test_predictions = official_test_raw[["id", "text"]].copy()
    if not official_test_predictions.empty:
        official_processed = prepare_texts(official_test_predictions["text"])
        official_embedding_texts = prepare_embedding_texts(official_test_predictions["text"])
        x_official_tfidf = final_vectorizer.transform(official_processed)
        x_official_embedding = encode_texts(
            official_embedding_texts,
            EMBEDDINGS_DIR / "combo_official_test_embeddings.npy",
        )
        for model_key, final_model in final_models.items():
            features = x_official_tfidf if model_key in TFIDF_KEYS else x_official_embedding
            official_test_predictions[f"{model_key}_prediction"] = final_model.predict(features)
        official_test_predictions.to_csv(PROCESSED_DIR / "official_test_predictions.csv", index=False)

    hf_status = try_evaluate_indic_sentiment(final_vectorizer, final_models)
    with open(TEXT_DIR / "indic_sentiment_status.json", "w", encoding="utf-8") as file:
        json.dump(hf_status, file, indent=2)

    generate_latex_report(profile, results_df, error_df, metadata, hf_status)

    print("Training complete.")
    print(f"Best model: {metadata['best_model']} ({metadata['best_model_type']})")
    print(f"Artifacts saved under: {MODELS_DIR}")
    print("LaTeX report saved as reports/main.tex")


if __name__ == "__main__":
    main()
