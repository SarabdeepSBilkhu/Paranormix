import json
import os
import sys
import joblib

# Add project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.ml.preprocessing import lemmatize_tokenizer
from sklearn.calibration import CalibratedClassifierCV

PROCESSED_DATA_PATH = os.path.join("data", "processed", "creepypasta", "train.json")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Heuristic Keywords for Strong Signal Filtering
KEYWORDS = {
    "apparition": ["ghost", "spirit", "shade", "specter", "figure", "silhouette", "white lady", "apparition", "transparent", "misty", "ethereal"],
    "poltergeist": ["thrown", "crash", "bang", "loud", "knock", "slam", "levitate", "fly across", "scratch", "shattered", "thump", "rattle"],
    "folklore": ["legend", "myth", "ritual", "ancient", "curse", "tradition", "elder", "village", "townspeople", "shrine", "ancestor", "curse"],
    "creature": ["eyes", "teeth", "claws", "beast", "monster", "fur", "growl", "creature", "thing", "cryptid", "paws", "snarl"],
    "psychological": ["crazy", "insane", "mind", "head", "voice", "remember", "dream", "wake up", "hallucination", "paranoia", "delusion", "trauma"]
}

def get_signal_strength(text, label):
    words = KEYWORDS.get(label, [])
    text_lower = text.lower()
    return sum(1 for w in words if w in text_lower)

def train_model():
    print("Initializing ULTRA HIGH CONFIDENCE model...")
    
    # Load data
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Ectoplasm missing! {PROCESSED_DATA_PATH} not found.")
        return

    with open(PROCESSED_DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Check if data is list of dicts (new format) or dict of lists (old format)
    if isinstance(data, list):
        # Filtering for "Strong Signal" examples to force peaky boundaries
        print("Filtering for prototypical (strong signal) samples...")
        filtered_data = [
            item for item in data 
            if get_signal_strength(item['text'], item['label']) >= 3
        ]
        
        if len(filtered_data) < 500:
            print(f"Warning: Only {len(filtered_data)} strong samples found. Relaxing filter to >= 2.")
            filtered_data = [
                item for item in data 
                if get_signal_strength(item['text'], item['label']) >= 2
            ]

        X = [item['text'] for item in filtered_data]
        y = [item['label'] for item in filtered_data]
    else:
        X = data['text']
        y = data['labels']
    
    # Check if we have enough data
    if len(X) < 5:
        print("WARNING: Not enough data to train.")
        return

    print(f"Training on {len(X)} PROTOTYPICAL samples (out of {len(data)} total)...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Create Pipeline components
    # Using trigrams and raw SGD for peaky results
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000, 
            ngram_range=(1, 3), 
            tokenizer=lemmatize_tokenizer, 
            token_pattern=None,
            sublinear_tf=True # Scaled term frequencies help with confidence
        )),
        ('clf', SGDClassifier(
            loss='log_loss', 
            penalty='l2', 
            alpha=1e-8, # Extremely low regularization for overconfidence
            random_state=42, 
            max_iter=10000, 
            tol=1e-4, 
            class_weight='balanced'
        ))
    ])
    
    # Train
    print("Fitting model (Aggressive mode)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on the prototypical test set
    print("Evaluating on Prototypical Test Set...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save
    model_path = os.path.join(MODEL_DIR, "ghost_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"High-confidence model saved at {model_path}")

if __name__ == "__main__":
    train_model()
