from __future__ import annotations

import json

import joblib
import streamlit as st

from src.config import MODELS_DIR
from src.embeddings import load_sentence_transformer
from src.modeling import MODEL_DISPLAY_NAMES, MODEL_TYPES, prediction_confidence
from src.modeling import prepare_embedding_texts
from src.preprocessing import (
    has_abusive_language,
    has_negative_contrast_shift,
    has_negative_signal,
    has_strong_positive_signal,
    preprocess_text,
)


MODEL_OPTIONS = {
    "Logistic Regression (TF-IDF)": "tfidf_logistic_regression",
    "SVM (TF-IDF)": "tfidf_svm",
    "Logistic Regression (Embeddings)": "embedding_logistic_regression",
    "SVM (Embeddings)": "embedding_svm",
}


@st.cache_data
def load_artifacts():
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    metadata_path = MODELS_DIR / "metadata.json"
    if not vectorizer_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python train.py` once before starting the app.")

    vectorizer = joblib.load(vectorizer_path)
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    models = {
        key: joblib.load(MODELS_DIR / f"{key}.pkl")
        for key in MODEL_DISPLAY_NAMES
        if (MODELS_DIR / f"{key}.pkl").exists()
    }
    return vectorizer, models, metadata


@st.cache_resource
def get_embedder():
    return load_sentence_transformer()


st.set_page_config(page_title="Hinglish Sentiment Analyzer", page_icon=":bar_chart:", layout="centered")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 900px;}
    .metric-card {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        background: #f8fafc;
    }
    .small-label {font-size: 0.82rem; color: #5b6472; margin-bottom: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hinglish Sentiment Analyzer")

try:
    vectorizer, models, metadata = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

embedder = get_embedder()

left, right = st.columns([2, 1])
with left:
    text = st.text_area(
        "Enter Hinglish text",
        value="yeh movie mast hai but ending bakwaas thi",
        height=140,
        placeholder="Type a Hinglish sentence or tweet...",
    )
with right:
    available_options = [label for label, key in MODEL_OPTIONS.items() if key in models]
    selected_model = st.selectbox("Model", available_options, index=max(0, len(available_options) - 1))
    selected_key = MODEL_OPTIONS[selected_model]
    st.metric("Training rows", metadata.get("training_rows", "n/a"))
    st.caption(f"Best saved model: {metadata.get('best_model', 'n/a')}")

analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if analyze:
    processed_text = preprocess_text(text)
    model_key = MODEL_OPTIONS[selected_model]
    model = models[model_key]

    if MODEL_TYPES[model_key] == "Baseline":
        features = vectorizer.transform([processed_text])
    else:
        features = embedder.encode(
            prepare_embedding_texts([text]),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    prediction, confidence = prediction_confidence(model, features)
    if has_abusive_language(text):
        prediction = "negative"
        confidence = max(confidence, 0.90)
    elif has_negative_contrast_shift(processed_text):
        prediction = "negative"
        confidence = max(confidence, 0.75)
    elif has_strong_positive_signal(processed_text) and not has_negative_signal(processed_text):
        prediction = "positive"
        confidence = min(max(confidence, 0.65), 0.70)

    st.subheader(prediction.title())
    st.progress(confidence)
    st.write(f"Confidence score: **{confidence:.2%}**")
    st.write(f"Model type: **{MODEL_TYPES[model_key]}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="small-label">Original text</div>', unsafe_allow_html=True)
        st.write(text)
    with col2:
        st.markdown('<div class="small-label">Processed text</div>', unsafe_allow_html=True)
        st.write(processed_text if processed_text else "(empty after preprocessing)")

st.divider()
st.caption("The app loads saved .pkl classifiers and does not retrain models during inference.")
