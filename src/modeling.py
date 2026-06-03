from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.svm import LinearSVC, SVC

from src.config import LABELS, RANDOM_STATE, TFIDF_MAX_FEATURES
from src.preprocessing import preprocess_text


MODEL_FACTORIES = {
    "tfidf_logistic_regression": lambda: LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "tfidf_svm": lambda: LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    "embedding_logistic_regression": lambda: LogisticRegression(
        max_iter=2000,
        C=3.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "embedding_svm": lambda: SVC(
        kernel="rbf",
        C=64.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

MODEL_DISPLAY_NAMES = {
    "tfidf_logistic_regression": "Logistic Regression (TF-IDF)",
    "tfidf_svm": "SVM (TF-IDF)",
    "embedding_logistic_regression": "Logistic Regression (Embeddings)",
    "embedding_svm": "SVM (Embeddings)",
}

MODEL_TYPES = {
    "tfidf_logistic_regression": "Baseline",
    "tfidf_svm": "Baseline",
    "embedding_logistic_regression": "Embedding",
    "embedding_svm": "Embedding",
}


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        max_features=TFIDF_MAX_FEATURES,
        min_df=2,
        sublinear_tf=True,
    )


def prepare_texts(texts: pd.Series | list[str]) -> list[str]:
    return [preprocess_text(text) for text in texts]


def prepare_embedding_texts(texts: pd.Series | list[str]) -> list[str]:
    return [f"{text} normalized {preprocess_text(text)}" for text in texts]


def evaluate_model(name: str, model, x_test, y_test: pd.Series) -> dict[str, object]:
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    return {
        "model_key": name,
        "model": MODEL_DISPLAY_NAMES[name],
        "model_type": MODEL_TYPES[name],
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "predictions": predictions,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS),
    }


def prediction_confidence(model, x_vector) -> tuple[str, float]:
    prediction = str(model.predict(x_vector)[0])
    if hasattr(model, "predict_proba"):
        probability = float(np.max(model.predict_proba(x_vector)[0]))
        return prediction, probability

    scores = model.decision_function(x_vector)
    scores = np.atleast_2d(scores).astype(float)
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return prediction, float(np.max(probabilities[0]))
