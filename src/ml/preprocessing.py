import os
import re
import spacy

# Load SpaCy for lemmatization
try:
    # Use en_core_web_sm if available
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    nlp = None

def lemmatize_tokenizer(text):
    """
    Custom tokenizer for TfidfVectorizer that handles cleaning and lemmatization.
    """
    if not isinstance(text, str):
        return []
        
    # Pre-clean: lower and remove non-alphas
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    if nlp:
        doc = nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
    else:
        # Fallback to simple split
        return text.split()

def clean_text_simple(text):
    """Helper for basic text cleaning without tokenization"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
