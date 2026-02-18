import pandas as pd
import re
import os
import json
import numpy as np
import spacy
from sklearn.model_selection import train_test_split

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    nlp = None

# Config
INPUT_FILE = "creepypastas.xlsx" # Assumed in project root
OUTPUT_DIR = os.path.join("data", "processed", "creepypasta")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = {
    "apparition": "Ghostly manifestations, visual spirits, spectral figures.",
    "poltergeist": "Physical disturbances, moving objects, noise, kinetic energy.",
    "folklore": "Traditional myths, legends, cultural stories, rituals.",
    "creature": "Monsters, cryptids, physical entities (not ghosts).",
    "psychological": "Hallucinations, madness, gaslighting, internal horror."
}

# Heuristic Keywords for Semi-Automated Labeling
KEYWORDS = {
    "apparition": ["ghost", "spirit", "shade", "specter", "figure", "silhouette", "white lady", "apparition", "transparent", "misty", "ethereal"],
    "poltergeist": ["thrown", "crash", "bang", "loud", "knock", "slam", "levitate", "fly across", "scratch", "shattered", "thump", "rattle"],
    "folklore": ["legend", "myth", "ritual", "ancient", "curse", "tradition", "elder", "village", "townspeople", "shrine", "ancestor", "curse"],
    "creature": ["eyes", "teeth", "claws", "beast", "monster", "fur", "growl", "creature", "thing", "cryptid", "paws", "snarl"],
    "psychological": ["crazy", "insane", "mind", "head", "voice", "remember", "dream", "wake up", "hallucination", "paranoia", "delusion", "trauma"]
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove HTML/Markdown
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) # MD links
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove repeated symbols
    text = re.sub(r'([!?.])\1+', r'\1', text) 
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # SpaCy processing
    if nlp:
        doc = nlp(text.lower())
        # Lemmatize and remove stop words / punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
        return " ".join(tokens)
    
    # Fallback to simple cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

def segment_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        if current_word_count + word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += word_count
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def assign_label(text):
    # Simple keyword counting
    scores = {k: 0 for k in LABELS.keys()}
    for label, words in KEYWORDS.items():
        for word in words:
            if word in text:
                scores[label] += 1
    
    # Return label with max score, default to 'psychological' if tie/zero (common in creepypasta)
    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        return "psychological" # Default fallback
    return best_label

def process_pipeline():
    print("Starting Creepypasta Data Pipeline...")
    
    # 1. Load Dataset
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found in root. Please ensure file exists.")
        return

    # 2. Drop irrelevant columns
    if 'body' not in df.columns:
        # Try finding a likely candidate if 'body' doesn't exist
        print(f"WARNING: 'body' column not found. Available: {df.columns.tolist()}")
        # Check for 'Story' or similar if needed, else fail
        possible = [c for c in df.columns if 'body' in c.lower() or 'text' in c.lower()]
        if possible:
            print(f"   -> Using '{possible[0]}' as text column.")
            df = df.rename(columns={possible[0]: 'text'})
        else:
            print("❌ Critical: No text body column found.")
            return
    else:
        df = df[['body']].rename(columns={'body': 'text'})

    # 3. Generate identifiers
    df['id'] = [f"cp_{i:04d}" for i in range(len(df))]
    
    # 4. Remove empty
    print("Removing empty rows...")
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    
    # 5. Clean text
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    # 6. Compute word counts
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    # 7. Length filtering (< 50 words)
    print(f"📉 Filtering short texts (Original: {len(df)})...")
    df = df[df['word_count'] >= 50]
    print(f"   -> Remaining: {len(df)}")
    
    # 8. Segment long stories
    print("Segmenting long stories (>500 words)...")
    new_rows = []
    
    for idx, row in df.iterrows():
        if row['word_count'] > 500:
            chunks = segment_text(row['text'])
            for i, chunk in enumerate(chunks):
                new_rows.append({
                    'id': f"{row['id']}_{i+1:02d}",
                    'text': chunk,
                    'is_chunk': True,
                    'original_id': row['id']
                })
        else:
            new_rows.append({
                'id': row['id'],
                'text': row['text'],
                'is_chunk': False,
                'original_id': row['id']
            })
            
    df_processed = pd.DataFrame(new_rows)
    print(f"   -> Post-segmentation size: {len(df_processed)}")
    
    # 9. Remove unsafe lead-ins (Simple rule-based)
    # Example: "trigger warning", "cw:", etc.
    df_processed['text'] = df_processed['text'].apply(lambda x: re.sub(r'^(trigger warning|tw:|cw:).*?(\n|\.)', '', x, flags=re.IGNORECASE).strip())

    # 10. Narrative validation (Drop lists/metadata)
    # Heuristic: Drop if > 50% numbers or special chars, or very short chunks
    # For now, just simplistic length check on chunks
    df_processed = df_processed[df_processed['text'].apply(lambda x: len(x.split()) > 20)]

    # 11. Normalize Schema
    df_processed['source'] = 'creepypasta'
    final_cols = ['id', 'text', 'source', 'original_id'] # Keeping original_id for split
    df_processed = df_processed[final_cols]
    
    # 12. Freeze Preprocessing
    frozen_path = os.path.join(OUTPUT_DIR, "frozen_corpus.json")
    df_processed.to_json(frozen_path, orient='records', indent=2)
    print(f"🥶 Frozen corpus saved to {frozen_path}")
    
    # 13. Labeling (Semi-Manual)
    print("Applying labels...")
    df_processed['label'] = df_processed['text'].apply(assign_label)
    
    # 14. Class Balance Handling
    print("Balancing Classes...")
    counts = df_processed['label'].value_counts()
    print("Original Counts:", counts)
    
    # Target size: Cap at 2000, Upsample to 800
    CAP = 2000
    FLOOR = 800
    
    balanced_dfs = []
    for label in counts.index:
        subset = df_processed[df_processed['label'] == label]
        n = len(subset)
        
        if n > CAP:
            # Downsample
            subset = subset.sample(n=CAP, random_state=42)
        elif n < FLOOR:
            # Upsample (duplicate)
            # Use replace=True to allow sampling more than n
            subset = subset.sample(n=FLOOR, replace=True, random_state=42)
            
        balanced_dfs.append(subset)
        
    df_processed = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print("Balanced Counts:", df_processed['label'].value_counts())
    
    # 15. Final Export
    final_path = os.path.join(OUTPUT_DIR, "labeled_dataset.json")
    df_processed.to_json(final_path, orient='records', indent=2)
    print(f"💾 Labeled dataset saved to {final_path}")
    
    # 16. Train-Test Split (Grouped by original story)
    # We want to avoid leakage: chunks from same story must be in same set
    unique_ids = df_processed['original_id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    
    train_df = df_processed[df_processed['original_id'].isin(train_ids)]
    test_df = df_processed[df_processed['original_id'].isin(test_ids)]
    
    train_df.to_json(os.path.join(OUTPUT_DIR, "train.json"), orient='records', indent=2)
    test_df.to_json(os.path.join(OUTPUT_DIR, "test.json"), orient='records', indent=2)
    
    print(f"Split Complete: Train ({len(train_df)}), Test ({len(test_df)})")
    print("Pipeline Execution Successful.")

if __name__ == "__main__":
    process_pipeline()
