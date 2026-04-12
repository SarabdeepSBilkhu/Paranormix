import json
import os
import sys
import joblib

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from src.ml.preprocessing import lemmatize_tokenizer

# ─── Configuration ────────────────────────────────────────────────────────────
CLASSES = ["material", "environmental", "immaterial", "rule_bound", "internal"]

TRAIN_DATA_PATH = os.path.join("data", "train_balanced.json")
TEST_DATA_PATH = os.path.join("data", "test_balanced.json")
FROZEN_CORPUS_PATH = os.path.join("data", "frozen_corpus.json")

MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    """
    Train the Signal Extractor model (Step 2 of pipeline).

    - Uses balanced training data
    - Uses frozen corpus for vocabulary expansion
    - Outputs probabilistic classifier (log_loss)
    """

    print("=" * 60)
    print("Paranormix Signal Extractor — Training")
    print(f"Classes: {CLASSES}")
    print("=" * 60)

    # ─── Load training data ───────────────────────────────────────────────────
    if not os.path.exists(TRAIN_DATA_PATH):
        print("ERROR: Training data not found.")
        return

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    X_train = [item["text"] for item in train_data]
    y_train = [item["label"] for item in train_data]

    print(f"Loaded {len(X_train)} training samples.")

    # Validate labels
    unique_labels = set(y_train)
    unexpected = unique_labels - set(CLASSES)
    if unexpected:
        print(f"WARNING: Unexpected labels found: {unexpected}")

    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # ─── Load frozen corpus (for vocabulary expansion) ────────────────────────
    if os.path.exists(FROZEN_CORPUS_PATH):
        with open(FROZEN_CORPUS_PATH, "r", encoding="utf-8") as f:
            frozen_data = json.load(f)
        frozen_texts = [item["text"] for item in frozen_data]
        print(f"Loaded {len(frozen_texts)} frozen corpus samples.")
    else:
        frozen_texts = []
        print("WARNING: Frozen corpus not found.")

    # ─── Vectorizer ───────────────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        tokenizer=lemmatize_tokenizer,
        token_pattern=None,
        sublinear_tf=True
    )

    print("Fitting TF-IDF on combined corpus...")
    vectorizer.fit(frozen_texts + X_train)

    X_train_vec = vectorizer.transform(X_train)

    # ─── Classifier ───────────────────────────────────────────────────────────
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=5e-4,
        random_state=42,
        max_iter=5000,
        class_weight="balanced"
    )

    print("Training signal extractor...")
    clf.fit(X_train_vec, y_train)

    # ─── Evaluation (balanced test set) ───────────────────────────────────────
    if os.path.exists(TEST_DATA_PATH):
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        X_test = [item["text"] for item in test_data]
        y_test = [item["label"] for item in test_data]

        X_test_vec = vectorizer.transform(X_test)
        y_pred = clf.predict(X_test_vec)

        print("\nSignal Detection Performance (Balanced Test):")
        print("-" * 50)
        print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))
    else:
        print("WARNING: Test dataset not found. Skipping evaluation.")

    # ─── Save pipeline ────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

    model_path = os.path.join(MODEL_DIR, "classifier.pkl")
    joblib.dump(pipeline, model_path)

    print(f"\nSignal Extractor saved at {model_path}")
    print("Model ready for inference (used by inference.py)")


if __name__ == "__main__":
    train_model()