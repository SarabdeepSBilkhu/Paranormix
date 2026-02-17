import requests
import json
import os
import time

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

SOURCES = [
    {
        "name": "Bidoofgoo Ghost Stories",
        "url": "https://raw.githubusercontent.com/bidoofgoo/Ghost-Story-Dataset/main/stories_transformed.json",
        "type": "json_list"
    },
    {
        "name": "The Legend of Sleepy Hollow",
        "url": "https://www.gutenberg.org/files/41/41-0.txt",
        "type": "text",
        "class": "folklore"
    },
    {
        "name": "The Turn of the Screw",
        "url": "https://www.gutenberg.org/files/209/209-0.txt",
        "type": "text",
        "class": "ghost_story"
    }
]

def fetch_data():
    print("Fetching ghost stories from external sources...")
    all_stories = []

    for source in SOURCES:
        try:
            print(f"Fetching {source['name']}...")
            response = requests.get(source['url'])
            response.raise_for_status()
            
            if source['type'] == 'json_list':
                data = response.json()
                # Normalize Bidoofgoo data
                for item in data:
                    all_stories.append({
                        "text": item.get("story_text", ""),
                        "title": item.get("story_title", "Unknown"),
                        "source": "bidoofgoo",
                        "label": "personal_experience" # Assuming these are personal accounts
                    })
            elif source['type'] == 'text':
                # For Gutenberg, we treat the whole book as one entry for now, 
                # or split by chapters in a real app. Here we just take a chunk.
                text = response.text
                # Simple extraction of a chunk to avoid huge files in one record
                start_idx = text.find("*** START OF")
                if start_idx != -1:
                    text = text[start_idx+500:start_idx+5500] # Take 5000 chars
                all_stories.append({
                    "text": text,
                    "title": source['name'],
                    "source": "gutenberg",
                    "label": source['class']
                })
                
        except Exception as e:
            print(f"❌ Failed to fetch {source['name']}: {e}")

    # Add some synthetic data to ensure we have enough classes for demo
    print("Generating synthetic folklore...")
    synthetics = [
        {"text": "The old manor on the hill is said to be haunted by a lady in white. She appears at midnight.", "label": "apparition"},
        {"text": "Objects moving on their own, plates smashing against the wall, loud knocks in groups of three.", "label": "poltergeist"},
        {"text": "A glowing orb was seen floating in the cemetery, dancing between the tombstones before vanishing.", "label": "orb"},
        {"text": "The locals say a demon protects the bridge, demanding a soul from anyone who crosses after dark.", "label": "cryptid"},
        {"text": "I felt a cold breath on my neck and heard a whisper saying 'get out', but the room was empty.", "label": "personal_experience"}
    ]
    all_stories.extend(synthetics)

    output_path = os.path.join(DATA_DIR, "stories.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stories, f, indent=2)
    
    print(f"Successfully collected {len(all_stories)} narratives into {output_path}")

if __name__ == "__main__":
    fetch_data()
