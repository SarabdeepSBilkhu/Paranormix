# Exhaustive Technical Documentation: Paranormix AI Investigator

## 1. Project Vision

Paranormix is a hybrid AI system designed for the analysis and investigation of paranormal narratives. It bridges the gap between deterministic Machine Learning (NLP) and generative Large Language Models (LLMs) to provide an "Explainable AI" experience.

---

## 2. Machine Learning Architecture (The Engine)

### A. Data Acquisition (`src/ml/fetch_data.py`)

- **Process**: Polls external JSON and Text sources (GitHub, Project Gutenberg).
- **Synthetic Generation**: Injects handcrafted "ground truth" examples for each class to ensure the model has a baseline understanding of "perfectly clean" narratives.
- **Library (requests)**: Used to fetch external data via HTTP.

### B. The Core Pipeline (`src/ml/process_creepypasta.py`)

This is the heart of the automated data engineering.

1. **Cleaning**: Uses **regex** to strip HTML tags, URLs, emails, and repeated symbols (!!!, ???).
2. **Segmentation**: Long stories (>500 words) are split into 300-word chunks. This ensures the TF-IDF vectorizer captures local linguistic nuances rather than getting lost in a massive document's noise.
3. **Semi-Automated Labeling**: Assigned using a priority keyword scoring system. If a narrative contains "ghost" and "figure", it scores for _Apparition_. If no keywords match, it defaults to _Psychological_.
4. **Class Balancing (pandas/numpy)**:
   - **Upsampling**: Minority classes (like Folklore) are sampled with replacement to reach a minimum floor (800 samples).
   - **Downsampling**: Majority classes are capped at 2,000 to prevent the model from becoming biased towards one entity type.
5. **Leakage Prevention**: Uses `train_test_split` but ensures that chunks belonging to the same original story ID stay together in either the train or test set, preventing "data leakage."

### C. Model Training (`src/ml/train.py`)

- **Workflow**:
  1. Loads `train.json`.
  2. Creates a scikit-learn **Pipeline**.
  3. **TfidfVectorizer**: Converts text to 5,000 numerical features (unigrams and bigrams).
  4. **SGDClassifier**:
     - `loss='log_loss'`: This is critical. It turns the linear classifier into a probabilistic one, allowing us to see "how sure" the model is.
     - `class_weight='balanced'`: Automatically adjusts weights in the cost function inversely proportional to class frequencies.
- **Library (joblib)**: Used to save the high-performance binary file `ghost_model.pkl`.

### D. Inference Engine (`src/ml/inference.py`)

- **Probability Distribution**: Returns specific percentages for all 5 classes.
- **Signal Extraction**: A non-ML pattern matcher that looks for specific narrative markers (e.g., "cold presence", "visual apparition") to explain the ML's decision.
- **Confusion Analysis**: Identifies classes with >15% probability that weren't the winner, highlighting "model doubt."

---

## 3. Backend Orchestration (The Brain)

### A. Session & Memory (`src/backend/session_store.py`)

- **Thread-Safety**: Uses `threading.Lock()` to prevent race conditions when multiple users are investigating simultaneously.
- **State Management**: Stores the original ML report and the message sequence history.
- **Cleanup**: Implements a time-based TTL (Time-To-Live) to clear memory of inactive sessions.

### B. Unified API (`src/backend/main.py`)

- **FastAPI**: Provides high-performance async endpoints.
- **Grounding Strategy**: When the LLM (Groq) is called, the system **injects** the raw ML probabilities as a hidden system message. This ensures the chatbot cannot lie about the model's findings.
- **Turn Enforcement**: Tracks turn counts (max 5) to prevent infinite loops or excessive API usage.

---

## 4. Library & Tool Usage Guide

| Library           | Purpose in Paranormix                                                            |
| :---------------- | :------------------------------------------------------------------------------- |
| **scikit-learn**  | Core ML framework for Vectorization, Classification, and Metrics.                |
| **FastAPI**       | Modern web framework for the backend API.                                        |
| **pandas**        | The "Swiss Army Knife" for tabular data manipulation during training.            |
| **numpy**         | Efficient handling of probability arrays and numerical transformations.          |
| **uvicorn**       | The ASGI server that runs the Python code in a production-ready loop.            |
| **groq**          | High-speed interface to Llama 3 for the conversational voice.                    |
| **joblib**        | Efficient disk-writing for the large ML model files.                             |
| **nltk/spacy**    | Natural Language Toolkits for text normalization and lemmatization.              |
| **pydantic**      | Ensures the data coming from the user (JSON) matches our expectations perfectly. |
| **python-dotenv** | Securely loads the `GROQ_API_KEY` from the hidden `.env` file.                   |
| **openpyxl**      | Allows Python to read the original `.xlsx` story database.                       |

---

## 5. Live Deployment Strategy (GitHub Pages)

### The Limitation

GitHub Pages is **Static Hosting**. It cannot run Python code (`main.py`) or ML models (`.pkl`).

### The Solution: Hybrid Deployment

1. **Frontend (GitHub Pages)**:
   - The files in `docs/` are served globally.
   - The `app.js` is modified to point to an external **Live Backend URL**.
2. **Backend (Render/Railway)**:
   - The Python code and ML model are deployed to a server.
   - This provides the "API" that the GitHub Pages site talks to.

### Steps for Live Deployment:

1. **Host the Backend**:
   - Push this repo to GitHub.
   - Connect the repo to [Render.com](https://render.com).
   - Add your `GROQ_API_KEY` in Render's "Environment Variables."
2. **Configure GitHub Pages**:
   - In GitHub Repo Settings, enable Pages and point it to the `/docs` folder.
   - Paranormix will then be live at `https://[your-username].github.io/[repo-name]`.
