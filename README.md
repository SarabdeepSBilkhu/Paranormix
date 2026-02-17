# Paranormix - AI Investigator Chatbot

A hybrid machine learning and LLM-powered system for analyzing and investigating paranormal narratives. Paranormix combines the predictive grounding of a scikit-learn classifier with the conversational intelligence of Groq's Llama 3 models.

## 🚀 Overview

Paranormix is designed to bridge the gap between "black-box" machine learning and human interpretability. It doesn't just predict whether a story is a ghost or a cryptid; it explains **why** based on narrative signals and conversational investigation.

- **Entity engine**: Classifies narratives into five distinct entity types.
- **Explainable AI**: Detects key signals (e.g., "object movement", "visage") used in the decision.
- **Conversational**: Chat with the "Investigator" to probe the model's logic.

## 📁 Project Structure

```
Paranormix/
├── data/                       # Pre-processed narrative datasets
├── models/                     # Trained ML model weights (.pkl)
├── src/
│   ├── ml/                     # ML Pipeline (Preprocessing, Training, Inference)
│   ├── backend/                # FastAPI & Chat Orchestration
│   │   ├── main.py             # Unified /chat endpoint
│   │   └── session_store.py    # Concurrency-safe memory management
│   ├── frontend/               # Modern chatbot interface
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── tests/                  # Automated API verification
├── .env                        # Configuration (API Keys, Session TTL)
├── requirements.txt            # System dependencies
└── TECHNICAL_DOCUMENTATION.md  # Deep dive into system internals
```

## 🛠️ Setup

### Prerequisites

- **Python 3.9+**
- **Groq API Key**: Obtain a free key from [Groq Console](https://console.groq.com/).

### Installation

1. **Clone the repository** and enter the directory.
2. **Setup Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Environment Variables**:
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_key_here
   MAX_CHAT_TURNS=5
   SESSION_TTL_MINUTES=30
   ```

## 🏃 Usage

1. **Start the Investigator**:
   ```bash
   uvicorn src.backend.main:app --reload
   ```
2. **Open the interface**:
   Launch `http://localhost:8000/app/index.html` in your browser.
3. **Investigate**:
   - Paste a story (e.g., "I heard footsteps in the attic and my door slammed shut.")
   - Review the **Investigation Report** card.
   - Ask follow-up questions: _"Why did you flag this as poltergeist?"_

## 🧪 Testing

Run endpoints verification:

```bash
python src/tests/test_api.py
```

## 📊 Model Performance

- **Accuracy**: ~58% (High for ambiguous narrative classification)
- **Engine**: TF-IDF Vectorization + SGD Logistic Regression.
- **Training Set**: 7,200+ labeled narrative segments derived from public domain creepypastas and folklore.

## ⚖️ Disclaimer

Paranormix is an educational tool for exploring **machine learning interpretability**. Its predictions are probabilistic and based on linguistic patterns, not factual verification of the supernatural.

---

Developed as part of AI Essentials and Machine Learning Coursework.
