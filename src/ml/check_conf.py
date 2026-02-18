import json
import joblib
import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.preprocessing import lemmatize_tokenizer

MODEL_PATH = os.path.join("models", "ghost_model.pkl")
TEST_DATA_PATH = os.path.join("data", "processed", "creepypasta", "test.json")

def check_confidence():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_DATA_PATH):
        print("Missing files.")
        return

    pipeline = joblib.load(MODEL_PATH)
    with open(TEST_DATA_PATH, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    X_test = df['text']
    
    probs = pipeline.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    
    avg_conf = np.mean(max_probs)
    print(f"Average Confidence: {avg_conf:.4f} ({avg_conf*100:.2f}%)")
    print(f"Median Confidence:  {np.median(max_probs):.4f}")
    print(f"90th Percentile:    {np.percentile(max_probs, 90):.4f}")
    print(f"10th Percentile:    {np.percentile(max_probs, 10):.4f}")

if __name__ == "__main__":
    check_confidence()
