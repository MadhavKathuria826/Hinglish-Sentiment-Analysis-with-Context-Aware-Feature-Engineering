from __future__ import annotations

import re
import string


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})

SPELLING_VARIANTS = {
    "acha": "acha",
    "accha": "acha",
    "achha": "acha",
    "achaa": "acha",
    "aacha": "acha",
    "bakwas": "bakwaas",
    "bekar": "bekaar",
    "bohot": "bahut",
    "bahot": "bahut",
    "bhot": "bahut",
    "nyc": "nice",
    "gud": "good",
    "gr8": "great",
}

SLANG_MAP = {
    "mast": "amazing",
    "badiya": "good",
    "badhiya": "excellent",
    "kamaal": "excellent",
    "zabardast": "excellent",
    "bakwaas": "bad",
    "bekaar": "bad",
    "ghatiya": "bad",
    "faltu": "bad",
    "badtameez": "bad",
    "bewakoof": "bad",
    "bevakoof": "bad",
    "nalayak": "bad",
    "nikamma": "bad",
    "kamina": "bad",
    "kameena": "bad",
    "harami": "bad",
    "haraami": "bad",
    "chutiya": "bad",
    "chutia": "bad",
    "chutiye": "bad",
    "chutiyo": "bad",
    "gandu": "bad",
    "gaand": "bad",
    "bhosdi": "bad",
    "bhosdike": "bad",
    "bsdk": "bad",
    "bhosdiwala": "bad",
    "bhosdiwale": "bad",
    "madarchod": "bad",
    "behenchod": "bad",
    "bhenchod": "bad",
    "bkl": "bad",
    "bc": "bad",
    "mc": "bad",
    "randi": "bad",
    "saala": "bad",
    "sala": "bad",
    "kutte": "bad",
    "kutta": "bad",
    "kutiya": "bad",
}

INTENSIFIERS = {"bohot", "bahut", "very", "extremely"}

INTENSIFIED_POSITIVE_MAP = {
    "acha": "very_good",
    "good": "very_good",
    "amazing": "very_amazing",
    "excellent": "very_amazing",
}

STRONG_POSITIVE_TOKENS = {
    "amazing",
    "excellent",
    "very_good",
    "very_amazing",
}

CONTRAST_TOKENS = {"but", "par", "lekin", "magar", "however", "though"}
CLEAR_NEGATIVE_TOKENS = {"bad", "worst", "hate", "angry", "sad", "poor", "negative"}

ABUSIVE_TERMS = {
    "badtameez",
    "bewakoof",
    "bevakoof",
    "nalayak",
    "nikamma",
    "kamina",
    "kameena",
    "harami",
    "haraami",
    "chutiya",
    "chutia",
    "chutiye",
    "chutiyo",
    "gandu",
    "gaand",
    "bhosdi",
    "bhosdike",
    "bsdk",
    "bhosdiwala",
    "bhosdiwale",
    "madarchod",
    "behenchod",
    "bhenchod",
    "bkl",
    "bc",
    "mc",
    "randi",
    "saala",
    "sala",
    "kutte",
    "kutta",
    "kutiya",
}

NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "nahi",
    "nahin",
    "nai",
    "mat",
    "dont",
    "don't",
    "didnt",
    "didn't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "cant",
    "can't",
}

POSITIVE_TERMS = {
    "good",
    "great",
    "amazing",
    "excellent",
    "very_good",
    "very_amazing",
    "nice",
    "love",
    "best",
    "happy",
    "acha",
    "positive",
}

NEGATIVE_TERMS = {
    "bad",
    "worst",
    "hate",
    "angry",
    "sad",
    "poor",
    "negative",
    "bekaar",
    "bakwaas",
    "badtameez",
    "bewakoof",
    "bevakoof",
    "nalayak",
    "nikamma",
    "kamina",
    "kameena",
    "harami",
    "haraami",
    "chutiya",
    "chutia",
    "chutiye",
    "gandu",
    "bhosdike",
    "bsdk",
    "madarchod",
    "behenchod",
    "bhenchod",
    "bkl",
    "bc",
    "mc",
    "randi",
    "saala",
    "sala",
    "kutte",
    "kutta",
    "kutiya",
}

HINGLISH_TERMS = {
    "acha",
    "nahi",
    "nahin",
    "hai",
    "ka",
    "ki",
    "ke",
    "mein",
    "mai",
    "mat",
    "bahut",
    "yaar",
    "bhai",
    "bakwaas",
    "mast",
    "kya",
    "kyu",
    "kyun",
    "badtameez",
    "bewakoof",
    "bevakoof",
    "nalayak",
    "nikamma",
    "kamina",
    "kameena",
    "harami",
    "haraami",
    "chutiya",
    "chutia",
    "chutiye",
    "gandu",
    "bhosdike",
    "bsdk",
    "madarchod",
    "behenchod",
    "bhenchod",
    "bkl",
    "bc",
    "mc",
    "randi",
    "saala",
    "sala",
    "kutte",
    "kutta",
    "kutiya",
}


def preprocess_text(text: str) -> str:
    """Normalize Roman-script Hinglish text for classical ML features."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = text.translate(PUNCT_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in text.split():
        token = SPELLING_VARIANTS.get(token, token)
        token = SLANG_MAP.get(token, token)
        tokens.append(token)

    intensity_tokens: list[str] = []
    intensify_next = False
    for token in tokens:
        if token in INTENSIFIERS:
            intensify_next = True
            continue
        if intensify_next:
            intensity_tokens.append(INTENSIFIED_POSITIVE_MAP.get(token, token))
            intensify_next = False
        else:
            intensity_tokens.append(token)
    tokens = intensity_tokens

    negation_aware_tokens: list[str] = []
    negate_next = False
    for token in tokens:
        if token in NEGATION_TOKENS:
            negate_next = True
            continue
        if negate_next:
            negation_aware_tokens.append(f"not_{token}")
            negate_next = False
        else:
            negation_aware_tokens.append(token)

    return " ".join(negation_aware_tokens)


def has_abusive_language(text: str) -> bool:
    """Detect explicit abusive Hinglish/Hindi terms before normalization."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = text.translate(PUNCT_TRANSLATION)
    tokens = {SPELLING_VARIANTS.get(token, token) for token in text.split()}
    return bool(tokens & ABUSIVE_TERMS)


def has_strong_positive_signal(processed_text: str) -> bool:
    """Detect strong positive tokens introduced by preprocessing."""
    return bool(set(str(processed_text).split()) & STRONG_POSITIVE_TOKENS)


def has_negative_signal(processed_text: str) -> bool:
    """Detect clear negative tokens after preprocessing."""
    tokens = set(str(processed_text).split())
    return bool(tokens & CLEAR_NEGATIVE_TOKENS)


def has_negative_contrast_shift(processed_text: str) -> bool:
    """Detect cases where sentiment turns negative after a contrast marker."""
    tokens = str(processed_text).split()
    for index, token in enumerate(tokens):
        if token in CONTRAST_TOKENS and any(
            later_token in CLEAR_NEGATIVE_TOKENS for later_token in tokens[index + 1 :]
        ):
            return True
    return False


def analyze_error_type(original_text: str, processed_text: str) -> str:
    raw_tokens = set(str(original_text).lower().split())
    processed_tokens = set(processed_text.split())
    categories: list[str] = []

    if any(token in NEGATION_TOKENS for token in raw_tokens) or any(
        token.startswith("not_") for token in processed_tokens
    ):
        categories.append("negation failure")

    has_positive = bool(processed_tokens & POSITIVE_TERMS)
    has_negative = bool(processed_tokens & NEGATIVE_TERMS) or any(
        token.startswith("not_") and token.replace("not_", "") in POSITIVE_TERMS
        for token in processed_tokens
    )
    if has_positive and has_negative:
        categories.append("mixed sentiment")

    if raw_tokens & HINGLISH_TERMS or processed_tokens & HINGLISH_TERMS:
        categories.append("Hinglish complexity")

    return "; ".join(categories) if categories else "lexical/context ambiguity"
