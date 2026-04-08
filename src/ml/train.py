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

PROCESSED_DATA_PATH = os.path.join("data", "processed", "train.json")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Heuristic Keywords for Strong Signal Filtering
KEYWORDS = {
    "apparition": [
        "ghost", "spirit", "shade", "specter", "apparition",
        "figure", "silhouette", "shadow", "reflection", "mirror",
        "transparent", "misty", "ethereal", "faded", "vanished", "appeared",
        "presence", "watching", "stare", "whisper", "breath",
        "cold", "chill", "icy", "freeze", "goosebumps"
    ],

    "poltergeist": [
        "thrown", "crash", "bang", "loud", "knock", "slam",
        "dragged", "pulled", "pushed", "levitate", "fly across",
        "scratch", "shattered", "thump", "rattle",
        "footsteps", "door slammed", "drawer", "cupboard",
        "again", "repeatedly", "every night", "on its own"
    ],

    "folklore": [
        "legend", "myth", "folklore", "ritual", "ancient",
        "curse", "cursed", "tradition", "custom", "taboo",
        "elder", "ancestors", "generations", "passed down",
        "warning", "forbidden", "spoken of", "written record",
        "annual", "seasonal",
        "village", "townspeople", "shrine", "temple",
        "well", "forest", "hill", "ruins", "burial ground"
    ],

    "creature": [
        "creature", "monster", "beast", "thing", "cryptid",
        "eyes", "glowing eyes", "teeth", "fangs", "claws", "paws",
        "fur", "scales", "tail",
        "growl", "snarl", "breathing", "heavy breathing",
        "footsteps", "chased", "stalked", "followed",
        "crouched", "looming", "lunged",
        "blood", "wounds", "scratches", "bite"
    ],

    "psychological": [
        "dream", "nightmare", "hallucination", "delusion",
        "paranoia", "anxiety", "panic", "stress", "trauma",
        "voice", "voices", "inside my head",
        "imagined", "felt unreal", "couldn't tell",
        "memory", "forgot", "can't recall", "memory gap",
        "sleep deprived", "insomnia", "exhausted",
        "therapy", "doctor", "medication",
        "crazy", "insane", "losing my mind"
    ]
}


def get_signal_strength(text, label):
    words = KEYWORDS.get(label, [])
    text_lower = text.lower()
    return sum(1 for w in words if w in text_lower)

def train_model():
    print("Initializing SEMANTIC STABILITY model...")
    
    # Load data
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Ectoplasm missing! {PROCESSED_DATA_PATH} not found.")
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
