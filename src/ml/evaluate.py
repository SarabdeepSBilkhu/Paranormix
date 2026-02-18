import sys
import json
import os
import joblib

# Add project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd
from src.ml.preprocessing import lemmatize_tokenizer

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
    
    # Metrics
    print("\n--- PERFORMANCE SUMMARY (Macro-F1 Priority) ---")
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"HEADLINE METRIC (Macro F1): {macro_f1:.4f}")
    print(f"Diagnostic Accuracy:         {accuracy:.4f}")
    
    print("\n--- PER-CLASS METRICS ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- CONFUSION MATRIX (Semantic Validity) ---")
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(list(set(y_test)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)
    
    # Specific Semantic checks
    print("\n--- SEMANTIC INTEGRITY CHECKS ---")
    psych_recall = cm_df.loc['psychological', 'psychological'] / cm_df.loc['psychological'].sum()
    print(f"Psychological Recall: {psych_recall:.2%} (Target: High to avoid False Positives)")
    
    ghost_psych_overlap = cm_df.loc['apparition', 'psychological'] / cm_df.loc['apparition'].sum()
    print(f"Apparition → Psych Leakage: {ghost_psych_overlap:.2%} (Ambiguous Ghost/Psych stories)")

if __name__ == "__main__":
    evaluate()
