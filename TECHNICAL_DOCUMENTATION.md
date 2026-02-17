# Exhaustive Technical Documentation: Paranormix AI Investigator

## 1. Project Vision

Paranormix is a research-oriented AI system designed for the analysis and investigation of paranormal narratives. It follows an "Explainable AI" (XAI) methodology, where a deterministic Machine Learning (NLP) engine handles classification, and a Large Language Model (LLM) provides conversational interpretation.

---

## 2. Machine Learning Architecture (The Engine)

### A. Data Acquisition (`src/ml/fetch_data.py`)

- **Process**: Polls external JSON and Text sources (GitHub, Project Gutenberg).
- **Synthetic Generation**: Injects handcrafted **high-confidence reference examples** for each class to ensure the model has a baseline understanding of prototypical linguistic markers.
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
- **Grounding Strategy**: When the conversational layer is invoked, the system **injects** the raw ML probabilities as a hidden system context. This prevents the LLM from hallucinating findings that contradict the predictive engine.
- **Turn Enforcement**: Tracks turn counts (max 5) to maintain state within free-tier resource constraints.

---

## 4. Library & Tool Usage Guide

| Library           | Purpose in Paranormix                                                             |
| :---------------- | :-------------------------------------------------------------------------------- |
| **scikit-learn**  | Core ML framework for Vectorization, Classification, and Metrics.                 |
| **FastAPI**       | Modern web framework for the backend API.                                         |
| **pandas**        | The "Swiss Army Knife" for tabular data manipulation during training.             |
| **numpy**         | Efficient handling of probability arrays and numerical transformations.           |
| **uvicorn**       | The ASGI server that runs the Python code in a production-ready loop.             |
| **LLM Provider**  | Pluggable support for OpenAI, Groq, or Gemini-compatible APIs for interpretation. |
| **joblib**        | Efficient disk-writing for the large ML model files.                              |
| **nltk/spacy**    | Natural Language Toolkits for text normalization and lemmatization.               |
| **pydantic**      | Ensures the data coming from the user (JSON) matches our expectations perfectly.  |
| **python-dotenv** | Securely loads the `GROQ_API_KEY` from the hidden `.env` file.                    |
| **openpyxl**      | Allows Python to read the original `.xlsx` story database.                        |

---

## 5. Live Deployment Strategy (Railway – Unified Full Stack)

### Deployment Model

Paranormix is deployed as a **single full-stack application** on Railway, serving both:

- **Frontend** (HTML, CSS, JavaScript) via FastAPI static mounting.
- **Backend** (FastAPI + ML inference engine).

This architecture eliminates cross-origin (CORS) complexity and ensures that both the predictive models and the interface reside in a synchronized runtime environment.

### Why GitHub Pages Was Not Used

GitHub Pages is a static hosting provider and cannot execute the server-side Python logic, handle session memory, or run the scikit-learn inference engine. While a hybrid model was considered, the final deployment consolidates all components into a single Railway service for maximum reliability.

### Backend & Frontend Access

- **Application URL**: `https://paranormix-production.up.railway.app`
- **Interactive Documentation**: `https://paranormix-production.up.railway.app/docs` (Swagger UI)

### Runtime Configuration

- **Server**: Uvicorn (ASGI)
- **Primary Command**: `uvicorn src.backend.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `GROQ_API_KEY`: Required for the conversational interpretation layer.
  - `MAX_CHAT_TURNS`: Enforces session length limits.

### Session Behavior

Session memory is volatile and stored in-process. On service hibernation or restart (typical on free tiers), active session history resets. This ensures data privacy and maintains a predictable memory footprint.
