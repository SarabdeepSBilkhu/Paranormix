import os
import re
import spacy

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    nlp = None


def lemmatize_tokenizer(text):
    """
    Custom tokenizer for TF-IDF:
    - cleaning
    - lemmatization
    - controlled filtering
    """

    if not isinstance(text, str):
        return []

    # Basic cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    if not nlp:
        return text.split()

    doc = nlp(text)

    tokens = []
    for token in doc:
        lemma = token.lemma_.strip()

        # Skip short tokens
        if len(lemma) < 3:
            continue

        # Skip punctuation / spaces
        if token.is_punct or token.is_space:
            continue

        # Skip pure stopwords EXCEPT important negations
        if token.is_stop and lemma not in {"no", "not"}:
            continue

        # Skip useless lemmas
        if lemma in {"be", "have", "do", "say", "go", "get"}:
            continue

        tokens.append(lemma)

    return tokens


def clean_text_simple(text):
    """
    Basic cleaning without tokenization
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text