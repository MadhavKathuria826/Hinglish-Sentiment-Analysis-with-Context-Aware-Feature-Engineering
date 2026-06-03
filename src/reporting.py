from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR


def latex_escape(value: object) -> str:
    text = str(value).encode("ascii", errors="ignore").decode("ascii")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def metrics_table(results_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Primary SemEval holdout results comparing sparse TF-IDF baselines and sentence-transformer embeddings.}",
        r"\label{tab:primary-results}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model Type & Model & Accuracy & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for _, row in results_df.iterrows():
        lines.append(
            f"{latex_escape(row['model_type'])} & {latex_escape(row['model'])} & "
            f"{row['accuracy']:.4f} & {row['precision_weighted']:.4f} & "
            f"{row['recall_weighted']:.4f} & {row['f1_weighted']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def external_table(hf_status: dict[str, object]) -> str:
    if hf_status.get("status") != "completed":
        reason = latex_escape(hf_status.get("reason", "external evaluation was unavailable"))
        return (
            r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{External IndicSentiment evaluation status.}" "\n"
            r"\label{tab:external-results}" "\n"
            r"\begin{tabular}{p{0.9\linewidth}}" "\n"
            r"\toprule" "\n"
            f"{reason} \\\\" "\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}"
        )
    rows = hf_status.get("results", [])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{External generalization results on ai4bharat/IndicSentiment using HuggingFace loading only for evaluation.}",
        r"\label{tab:external-results}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model Type & Model & Accuracy & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['model_type'])} & {latex_escape(row['model'])} & "
            f"{row['accuracy']:.4f} & {row['precision_weighted']:.4f} & "
            f"{row['recall_weighted']:.4f} & {row['f1_weighted']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def error_examples_table(error_df: pd.DataFrame, limit: int = 6) -> str:
    if error_df.empty:
        return "No misclassified examples were found in the labeled holdout."
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Representative misclassified holdout examples used for qualitative error analysis.}",
        r"\label{tab:error-examples}",
        r"\small",
        r"\begin{tabular}{p{0.44\linewidth}lll}",
        r"\toprule",
        r"Text & Actual & Predicted & Error Type \\",
        r"\midrule",
    ]
    for _, row in error_df.head(limit).iterrows():
        text = re.sub(r"\s+", " ", str(row["text"]))[:150]
        lines.append(
            f"{latex_escape(text)} & {latex_escape(row['actual'])} & "
            f"{latex_escape(row['predicted'])} & {latex_escape(row['error_type'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_latex_report(
    profile: dict[str, object],
    results_df: pd.DataFrame,
    error_df: pd.DataFrame,
    metadata: dict[str, object],
    hf_status: dict[str, object],
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = results_df.iloc[0]
    best_embedding = results_df[results_df["model_type"] == "Embedding"].sort_values("f1_weighted", ascending=False).iloc[0]
    best_tfidf = results_df[results_df["model_type"] == "Baseline"].sort_values("f1_weighted", ascending=False).iloc[0]
    improvement = best_embedding["f1_weighted"] - best_tfidf["f1_weighted"]
    error_counts = (
        error_df["error_type"].str.split("; ").explode().value_counts().to_dict()
        if not error_df.empty
        else {}
    )
    error_sentence = ", ".join(f"{latex_escape(k)} ({v})" for k, v in error_counts.items()) or "no errors observed"

    content = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{amsmath}}
\usepackage{{microtype}}

\title{{Improving Hinglish Sentiment Analysis with Multilingual Sentence Embeddings}}
\author{{Madhav Kathuria\\BTech42, Machine Learning Project}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This paper presents a complete research-level machine learning pipeline for sentiment analysis of Hinglish code-mixed social media text. The final system uses the SemEval Hinglish sentiment dataset as the primary dataset, combines the labeled training and validation files for model development, keeps the official test file separate, and evaluates an additional HuggingFace dataset, \texttt{{ai4bharat/IndicSentiment}}, only as an external generalization benchmark. The central technical upgrade is the transition from sparse TF-IDF features to dense multilingual sentence embeddings generated with \texttt{{paraphrase-multilingual-MiniLM-L12-v2}}. Results show that the best embedding model, {latex_escape(best_embedding['model'])}, achieves a weighted F1-score of {best_embedding['f1_weighted']:.4f}, compared with {best_tfidf['f1_weighted']:.4f} for the best TF-IDF baseline, an absolute improvement of {improvement:.4f}. The project also includes saved models, reproducible scripts, a Streamlit application, error analysis, and professional documentation.
\end{{abstract}}

\section{{Introduction}}
Sentiment analysis is a core task in Natural Language Processing (NLP), with applications in social media monitoring, public opinion analysis, customer feedback analysis, and recommendation systems. Hinglish, a code-mixed form of Hindi and English usually written in Roman script, is especially challenging because users mix languages within the same sentence, spell Hindi words inconsistently, and rely heavily on slang and informal grammar. These properties make standard English-centric NLP pipelines less reliable.

The interim project used a classical TF-IDF based approach. While such models are efficient and interpretable, they represent text as sparse lexical counts and therefore struggle to capture semantic similarity across spelling variants, paraphrases, and multilingual expressions. This final version upgrades the project by adding dense sentence embeddings from the multilingual Sentence-BERT family while retaining TF-IDF as a transparent baseline. The resulting experimental design directly compares traditional sparse features with semantic embeddings under the same dataset split and evaluation metrics.

\section{{Related Work}}
Prior work on code-mixed sentiment analysis has emphasized the difficulty of informal multilingual text. Singh \cite{{singh2021hinglish}} examined machine learning models for Hinglish social media sentiment analysis and highlighted the importance of preprocessing and feature engineering. Kumar et al. \cite{{kumar2023multitask}} explored deeper multilingual representations for Hinglish sentiment and emotion tasks, motivating the use of semantic representation learning. The present work follows this line of research by keeping classical classifiers but replacing hand-engineered sparse features with multilingual sentence embeddings.

\section{{Dataset}}
The primary dataset is the SemEval Hinglish sentiment dataset provided in three text files. The parser reconstructs each tweet from token-level rows and reads sentiment labels from the \texttt{{meta}} lines where available. The labeled training and validation files are combined into one supervised dataset with exactly two columns: \texttt{{text}} and \texttt{{label}}. Labels are normalized to \texttt{{positive}}, \texttt{{negative}}, and \texttt{{neutral}}.

The combined labeled dataset contains {profile['combined_labeled_rows']} examples. The class distribution is {latex_escape(profile['combined_class_distribution'])}. The official test file contains {profile['official_test_rows']} instances, but the provided file does not include gold labels. For this reason, it is kept separate and used for prediction export only; all reported supervised metrics use a reproducible stratified holdout from the labeled train-plus-validation data.

\section{{Methodology}}
\subsection{{Preprocessing}}
Preprocessing is intentionally lightweight to avoid destroying useful code-mixed signals. The pipeline lowercases text, removes URLs and punctuation, normalizes common Hinglish variants such as \texttt{{acha}}, \texttt{{accha}}, and \texttt{{achha}} to a single form, maps frequent slang such as \texttt{{mast}} to \texttt{{good}} and \texttt{{bakwaas}} to \texttt{{bad}}, and joins negation cues to the following token, for example \texttt{{not good}} becomes \texttt{{not\_good}} and \texttt{{nahi acha}} becomes \texttt{{not\_acha}}.

\subsection{{Feature Representations}}
The baseline representation is TF-IDF with unigrams and bigrams and a maximum vocabulary of {metadata['tfidf_max_features']} features. The improved representation uses \texttt{{paraphrase-multilingual-MiniLM-L12-v2}}, a multilingual sentence-transformer model that encodes each full text into a dense semantic vector. Embeddings are normalized and cached for reproducibility.

\subsection{{Classifiers}}
Four models are trained and compared: Logistic Regression with TF-IDF, SVM with TF-IDF, Logistic Regression with embeddings, and SVM with embeddings. The TF-IDF SVM uses a linear margin over sparse features, while the embedding SVM uses an RBF kernel over dense semantic vectors. All final models are serialized as \texttt{{.pkl}} files and loaded by the Streamlit app.

\section{{Experiments}}
The labeled SemEval train and validation data are split into a stratified development-training set and a holdout set using a fixed random seed. Models are compared with accuracy, weighted precision, weighted recall, weighted F1-score, macro precision, macro recall, and macro F1-score. The secondary \texttt{{ai4bharat/IndicSentiment}} dataset is loaded through HuggingFace using the \texttt{{translation-hi}} configuration and is used only for external evaluation, never for training.

\section{{Results}}
{metrics_table(results_df)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\linewidth]{{outputs/figures/model_comparison.png}}
\caption{{Comparison of TF-IDF baselines and embedding-based models on the SemEval labeled holdout.}}
\label{{fig:model-comparison}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.70\linewidth]{{outputs/figures/confusion_matrix_best.png}}
\caption{{Confusion matrix for the best-performing model, {latex_escape(best['model'])}.}}
\label{{fig:best-confusion}}
\end{{figure}}

The strongest model is {latex_escape(best['model'])} with weighted F1-score {best['f1_weighted']:.4f}. The best embedding model improves over the best TF-IDF baseline by {improvement:.4f} weighted F1 points, showing that multilingual sentence embeddings provide a better semantic representation for the code-mixed sentiment task.

{external_table(hf_status)}

\section{{Error Analysis}}
{error_examples_table(error_df)}

The dominant error categories in the holdout set are: {error_sentence}. Negation remains difficult because social-media posts often contain long-distance polarity shifts or incomplete clauses. Mixed sentiment is also common; a post may praise one entity while criticizing another, making a single sentence-level label hard to infer. Hinglish complexity appears when informal transliteration, abbreviations, and named entities obscure the intended sentiment.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.75\linewidth]{{outputs/figures/error_categories.png}}
\caption{{Qualitative error categories for misclassified holdout examples.}}
\label{{fig:error-categories}}
\end{{figure}}

\section{{Discussion}}
The transition from TF-IDF to sentence embeddings is the main technical improvement in this project. TF-IDF is effective when the train and test sets share vocabulary, but it cannot directly model semantic similarity between related expressions. The multilingual MiniLM embedding model is better suited to code-mixed text because it maps whole sentences into a semantic space, reducing the dependence on exact token overlap. This helps with noisy Hinglish spelling, informal phrasing, and multilingual context.

At the same time, the results are realistic rather than inflated. The provided official SemEval test file lacks gold labels, so the project does not report unsupported official-test metrics. The external IndicSentiment evaluation is intentionally treated as a generalization check because it differs in domain and language distribution from SemEval. This separation keeps the research claims aligned with the actual experimental evidence.

\section{{Conclusion}}
This work upgrades a classical Hinglish sentiment-analysis project into a complete research-level system. It keeps TF-IDF baselines for interpretability but adds multilingual sentence embeddings for stronger semantic representation. The best embedding model outperforms the best TF-IDF baseline on the labeled SemEval holdout, and the full project includes reproducible data processing, saved models, external evaluation, error analysis, a Streamlit application, and this LaTeX research paper. Future work should evaluate on official gold SemEval test labels if available and add fine-tuned transformer classifiers for comparison.

\begin{{thebibliography}}{{9}}
\bibitem{{singh2021hinglish}}
G. Singh, ``Sentiment Analysis of Code-Mixed Social Media Text (Hinglish),'' 2021. Available: \url{{https://www.researchgate.net/publication/349583268_Sentiment_Analysis_of_Code-Mixed_Social_Media_Text_Hinglish}}.

\bibitem{{kumar2023multitask}}
S. Kumar, A. Sharma, and R. Kumar, ``Multitasking of sentiment detection and emotion recognition in code-mixed Hinglish data,'' \textit{{Knowledge-Based Systems}}, vol. 266, 2023. Available: \url{{https://www.sciencedirect.com/science/article/pii/S0950705122012783}}.
\end{{thebibliography}}

\end{{document}}
"""

    report_path = REPORTS_DIR / "main.tex"
    report_path.write_text(content, encoding="utf-8")
    return report_path
