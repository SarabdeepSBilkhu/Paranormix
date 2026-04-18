import json
import os
import sys
import joblib
import argparse
import numpy as np

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


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_model(train_path, test_path=None, corpus_path=None, model_dir="models"):
    """
    Train the Signal Extractor model (Step 2 of pipeline).

    - Uses balanced training data
    - Uses optional frozen corpus for vocabulary expansion
    - Outputs probabilistic classifier (log_loss)
    """

    np.random.seed(42)

    print("=" * 60)
    print("Paranormix Signal Extractor — Training")
    print(f"Classes: {CLASSES}")
    print("=" * 60)

    # ─── Load training data ───────────────────────────────────────────────────
    train_data = load_json(train_path)

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

    # ─── Load frozen corpus ───────────────────────────────────────────────────
    frozen_texts = []
    if corpus_path and os.path.exists(corpus_path):
        frozen_data = load_json(corpus_path)
        frozen_texts = [item["text"] for item in frozen_data]
        print(f"Loaded {len(frozen_texts)} frozen corpus samples.")
    else:
        print("WARNING: Frozen corpus not provided or not found.")

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

    # ─── Evaluation ───────────────────────────────────────────────────────────
    if test_path and os.path.exists(test_path):
        test_data = load_json(test_path)

        X_test = [item["text"] for item in test_data]
        y_test = [item["label"] for item in test_data]

        X_test_vec = vectorizer.transform(X_test)
        y_pred = clf.predict(X_test_vec)

        report = classification_report(y_test, y_pred, labels=CLASSES, zero_division=0)
        print("\nSignal Detection Performance (Balanced Test):")
        print("-" * 50)
        print(report)

        # Save to evaluation_results
        eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluation_results")
        os.makedirs(eval_dir, exist_ok=True)
        report_path = os.path.join(eval_dir, "metrics_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Metrics report saved at: {report_path}")
    else:
        print("WARNING: Test dataset not provided or not found. Skipping evaluation.")

    # ─── Save model + metadata ────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

    model_path = os.path.join(model_dir, "classifier.pkl")
    joblib.dump(pipeline, model_path)

    metadata = {
        "classes": CLASSES,
        "train_size": len(X_train),
        "features": vectorizer.max_features,
        "ngram_range": vectorizer.ngram_range,
        "random_state": 42
    }

    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved at: {model_path}")
    print(f"Metadata saved at: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=False)
    parser.add_argument("--corpus_path", required=False)
    parser.add_argument("--model_dir", default="models")

    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        test_path=args.test_path,
        corpus_path=args.corpus_path,
        model_dir=args.model_dir
    )