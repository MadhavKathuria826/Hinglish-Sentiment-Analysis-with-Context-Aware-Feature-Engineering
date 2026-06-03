# 🗣️ Hinglish Sentiment Analysis using Embedding-Based NLP

A complete machine learning pipeline for sentiment classification of **Hinglish (Hindi-English code-mixed) social media text**, comparing traditional sparse text representations against multilingual semantic embeddings.

The project explores a simple question:

> Can multilingual sentence embeddings outperform classical TF-IDF approaches for noisy, code-mixed language?

---

## 🎯 Project Overview

Hinglish is one of the most widely used forms of code-mixed communication online.

Unlike standard English NLP tasks, Hinglish introduces:

* Mixed Hindi-English vocabulary
* Inconsistent transliteration
* Informal grammar
* Slang and abbreviations
* Context-dependent sentiment expressions

This project investigates whether modern multilingual sentence embeddings can better capture sentiment in such environments compared to traditional bag-of-words approaches.

---

## 🧠 Models Compared

### Traditional Baselines

* Logistic Regression + TF-IDF
* SVM + TF-IDF

### Embedding-Based Models

* Logistic Regression + Sentence Embeddings
* SVM + Sentence Embeddings

### Embedding Model

```text
paraphrase-multilingual-MiniLM-L12-v2
```

---

## ⚙️ System Architecture

```text
Raw Hinglish Text
        │
        ▼
Preprocessing
        │
        ▼
 ┌───────────────┐
 │ TF-IDF        │
 └───────────────┘
        │
        ▼
 Logistic Regression
 SVM

        OR

 ┌───────────────┐
 │ Sentence      │
 │ Embeddings    │
 └───────────────┘
        │
        ▼
 Logistic Regression
 SVM

        │
        ▼
 Evaluation & Comparison
        │
        ▼
 Streamlit Deployment
```

---

## 📊 Dataset

### Primary Dataset

**SemEval Hinglish Sentiment Dataset**

* 18,130 labeled examples
* Three sentiment classes:

  * Positive
  * Neutral
  * Negative

### Challenge

The official SemEval test file does not include gold labels.

To maintain valid evaluation:

* Training + Validation data are combined for development
* Metrics are reported using a reproducible stratified holdout split
* Official test data is used only for prediction export

---

## 🔍 Preprocessing Pipeline

The preprocessing pipeline is intentionally lightweight to preserve code-mixed linguistic signals.

### Applied Transformations

✅ Lowercasing

✅ URL removal

✅ Punctuation cleanup

✅ Hinglish normalization

Examples:

```text
acha
accha
achha
```

↓

```text
acha
```

### Slang Mapping

```text
mast      → good
bakwaas   → bad
```

### Negation Handling

```text
not good
```

↓

```text
not_good
```

```text
nahi acha
```

↓

```text
not_acha
```

---

## 📈 Results

### SemEval Holdout Evaluation

| Model                            | Accuracy | Weighted F1 |
| -------------------------------- | -------: | ----------: |
| SVM (Embeddings)                 |   64.75% |      64.76% |
| Logistic Regression (TF-IDF)     |   64.48% |      64.34% |
| SVM (TF-IDF)                     |   63.90% |      63.86% |
| Logistic Regression (Embeddings) |   58.74% |      58.26% |

🏆 **Best Model:** SVM + Multilingual Embeddings

Weighted F1:

```text
0.6476
```

---

## 🔬 Error Analysis

The project includes qualitative error analysis to understand model limitations.

Common failure modes:

* Hinglish transliteration complexity
* Context ambiguity
* Negation handling
* Mixed sentiment expressions
* Informal social-media language

This analysis highlights why code-mixed NLP remains challenging even with multilingual embeddings.

---

## 🚀 Streamlit Application

Interactive sentiment prediction interface featuring:

* Hinglish text input
* Model selection
* Real-time sentiment prediction
* Confidence scores
* Processed text visualization

Launch:

```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```text
.
├── app.py
├── train.py
├── download_model.py
├── requirements.txt
├── README.md

├── data/
│   └── raw/
│       └── semeval/

├── models/
│   ├── metadata.json
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_logistic_regression.pkl
│   ├── tfidf_svm.pkl
│   ├── embedding_logistic_regression.pkl
│   └── embedding_svm.pkl

├── reports/
│   ├── main.tex
│   └── presentation/

└── src/
    ├── config.py
    ├── data.py
    ├── preprocessing.py
    ├── embeddings.py
    ├── modeling.py
    └── reporting.py
```

---

## 🔄 Reproducibility

To keep the repository lightweight:

* Large transformer files are excluded from Git
* Cached embeddings are regenerated automatically
* Training outputs are reproducible via configuration files

Install:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py
```

Optional local model download:

```bash
python download_model.py
```

---

## 💡 Key Insight

The most interesting result is not that embeddings outperform TF-IDF.

It is that:

> Dense multilingual representations provide measurable gains on noisy code-mixed language while preserving a simple classical classification pipeline.

This demonstrates that meaningful improvements can be achieved without resorting to large-scale transformer fine-tuning.

---

## 👨‍💻 Author

**Madhav Kathuria**
B.Tech Computer Science & Engineering
South Asian University

---

⭐ If you found this project interesting, consider giving it a star.
