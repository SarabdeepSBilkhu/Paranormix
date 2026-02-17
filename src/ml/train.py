import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

PROCESSED_DATA_PATH = os.path.join("data", "processed", "creepypasta", "train.json")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    print("Initializing model...")
    
    # Load data
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Ectoplasm missing! {PROCESSED_DATA_PATH} not found.")
        return

    with open(PROCESSED_DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Check if data is list of dicts (new format) or dict of lists (old format)
    if isinstance(data, list):
        X = [item['text'] for item in data]
        y = [item['label'] for item in data]
    else:
        X = data['text']
        y = data['labels']
    
    # Check if we have enough data
    if len(X) < 5:
        print("WARNING: Not enough data to train.")
        return

    print(f"Training on {len(X)} spectral remnants from Creepypasta dataset...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None, class_weight='balanced')),
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save
    model_path = os.path.join(MODEL_DIR, "ghost_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model()
