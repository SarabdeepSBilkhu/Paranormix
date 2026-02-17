import sys
import json
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Force UTF-8 output for Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = os.path.join("models", "ghost_model.pkl")
TEST_DATA_PATH = os.path.join("data", "processed", "creepypasta", "test.json")

def evaluate():
    print("Evaluating Model...")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model missing at {MODEL_PATH}. Train it first!")
        return
    
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded.")

    # 2. Load Test Data
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Test data missing at {TEST_DATA_PATH}. Run process_creepypasta.py first!")
        return

    with open(TEST_DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Handle list of dicts format
    if isinstance(data, list):
        df = pd.DataFrame(data)
        X_test = df['text']
        y_test = df['label']
    else:
        print("❌ Unknown data format.")
        return

    print(f"Testing on {len(X_test)} unseen samples...")

    # 3. Predict
    y_pred = pipeline.predict(X_test)

    # 4. Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAccuracy Score:")
    acc = accuracy_score(y_test, y_pred)
    print(f"{acc:.4f} ({acc*100:.2f}%)")

    print("\nConfusion Matrix:")
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

if __name__ == "__main__":
    evaluate()
