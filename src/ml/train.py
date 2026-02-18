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
    print("Initializing SEMANTIC STABILITY model...")
    
    # Load data
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Ectoplasm missing! {PROCESSED_DATA_PATH} not found.")
        return

    with open(PROCESSED_DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Use full dataset for better generalization
    print(f"Loading {len(data)} narratives for comprehensive training...")
    X = [item['text'] for item in data]
    y = [item['label'] for item in data]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # TARGETED CLASS WEIGHTS
    # We boost 'psychological' to increase RECALL (avoid false supernatural attribution)
    # We boost 'creature' to improve its Macro-F1 (low support)
    custom_weights = {
        'apparition': 1.0,
        'poltergeist': 1.0,
        'folklore': 1.0,
        'psychological': 1.8, # Stronger bias for "real-world" explanations
        'creature': 2.0       # Prevent collapse of sparse class
    }

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 2), # Bi-grams are more stable than tri-grams for general patterns
            tokenizer=lemmatize_tokenizer, 
            token_pattern=None,
            sublinear_tf=True
        )),
        ('clf', SGDClassifier(
            loss='log_loss', 
            penalty='l2', 
            alpha=5e-4, # Stable regularization for better F1 generalization
            random_state=42, 
            max_iter=5000, 
            class_weight=custom_weights
        ))
    ])
    
    # Train
    print("Fitting model (Semantic focus)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating stability...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save
    model_path = os.path.join(MODEL_DIR, "ghost_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Semantic-stable model saved at {model_path}")

if __name__ == "__main__":
    train_model()
