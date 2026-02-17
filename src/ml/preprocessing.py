import json
import os
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Ensure you have the model: python -m spacy download en_core_web_sm
# If not, we fall back to simple regex

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: SpaCy model not found. Using simple regex tokenization.")
    nlp = None

RAW_DATA_PATH = os.path.join("data", "raw", "stories.json")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    if nlp:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)
    else:
        # Simple fallback
        tokens = text.split()
        return " ".join(tokens)

def preprocess():
    print("Preprocessing text data...")
    
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        stories = json.load(f)
    
    processed_stories = []
    labels = []
    
    for story in stories:
        clean = clean_text(story['text'])
        processed_stories.append(clean)
        labels.append(story.get('label', 'unknown'))
        
    # Save processed text/labels
    with open(os.path.join(PROCESSED_DATA_DIR, "clean_stories.json"), "w") as f:
        json.dump({"text": processed_stories, "labels": labels, "originals": stories}, f)
        
    print(f"Preprocessing complete. {len(processed_stories)} records ready.")

if __name__ == "__main__":
    preprocess()
