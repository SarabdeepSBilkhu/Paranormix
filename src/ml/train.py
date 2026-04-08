import json
import os
import sys
import joblib

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.ml.preprocessing import lemmatize_tokenizer

# ─── Configuration ────────────────────────────────────────────────────────────
CLASSES = ["internal", "material", "immaterial", "environmental", "rule_bound"]
TRAIN_DATA_PATH = os.path.join("data", "train.json")
FROZEN_CORPUS_PATH = os.path.join("data", "frozen_corpus.json")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    """
    Train the Signal Extractor model.
    
    Architecture Role:
        This model is a SIGNAL EXTRACTOR, not a final classifier.
        It learns to detect signal categories from unstructured narrative.
        Final classification is determined by the Rule Resolver (resolver.py).
    
    Design Decisions:
        - class_weight = None → No bias in signal detection.
        - log_loss → Enables probability output for signal confidence.
        - The model output is consumed as boolean flags, not as final labels.
    """
    print("=" * 60)
    print("Paranormix Signal Extractor — Training")
    print(f"Classes: {CLASSES}")
    print("=" * 60)

    # Load labeled data
    if not os.path.exists(TRAIN_DATA_PATH):
        print("ERROR: Training data not found.")
        return

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    X = [item["text"] for item in train_data]
    y = [item["label"] for item in train_data]

    print(f"Loaded {len(X)} labeled samples.")

    # Validate labels
    unique_labels = set(y)
    unexpected = unique_labels - set(CLASSES)
    if unexpected:
        print(f"WARNING: Unexpected labels found: {unexpected}")

    # Load frozen corpus (UNLABELED — for TF-IDF vocabulary expansion)
    if os.path.exists(FROZEN_CORPUS_PATH):
        with open(FROZEN_CORPUS_PATH, "r", encoding="utf-8") as f:
            frozen_data = json.load(f)
        frozen_texts = [item["text"] for item in frozen_data]
        print(f"Loaded {len(frozen_texts)} frozen corpus samples.")
    else:
        frozen_texts = []
        print("WARNING: Frozen corpus not found.")

    # Split labeled data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # TF-IDF Vectorizer (fit on frozen + train for vocabulary richness)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        tokenizer=lemmatize_tokenizer,
        token_pattern=None,
        sublinear_tf=True
    )

    print("Fitting TF-IDF on combined corpus...")
    vectorizer.fit(frozen_texts + X_train)

    # Transform
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Signal Extractor Classifier
    # class_weight=None: No artificial bias in signal detection
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=5e-4,
        random_state=42,
        max_iter=5000,
        class_weight=None
    )

    print("Training signal extractor...")
    clf.fit(X_train_vec, y_train)

    # Evaluation
    print("\nSignal Detection Performance:")
    print("-" * 40)
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    # Save as pipeline
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

    model_path = os.path.join(MODEL_DIR, "ghost_model.pkl")
    joblib.dump(pipeline, model_path)

    print(f"\nSignal Extractor saved at {model_path}")
    print("Note: This model is used for signal detection only.")
    print("Final classification is determined by resolver.py.")


if __name__ == "__main__":
    train_model()