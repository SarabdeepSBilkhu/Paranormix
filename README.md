# Paranormix - AI Diagnostic Terminal

A high-fidelity machine learning system for the multi-axial analysis of paranormal narratives. Paranormix combines a deterministic scikit-learn classifier with a conversational decoder layer to provide grounded, measurement-based insights without speculative interpretation.

## 🚀 Overview

Paranormix is a **Technical Diagnostic Terminal**. It bridges the gap between raw machine learning output and human understanding by serving as a conversational manual for the model's findings.

- **Entity Engine**: Classifies narratives into five distinct categorical types.
- **Visual Diagnostic Suite**: Real-time dashboard for class distributions, signal contributions, and classification margins.
- **Empirical Certainty**: Certainty metrics calibrated by model stability (Resolution Boundaries) rather than just scalar probability.
- **Conversational Decoder**: An AI assistant that translates internal signal weights into human-friendly, grounded insights using progressive disclosure.

## 📁 Project Structure

```
Paranormix/
├── data/                       # Prototypical narrative datasets
├── models/                     # Trained ML model binaries (.pkl)
├── src/
│   ├── ml/                     # ML Engine (Preprocessing, Training, Inference)
│   │   ├── preprocessing.py    # Unified lemmatization & tokenization
│   │   └── inference.py        # Pattern-based diagnostic analyzer
│   ├── backend/                # FastAPI & Chat Orchestration
│   │   ├── main.py             # Unified /chat and /diagnostic endpoints
│   │   └── session_store.py    # Memory-based context management
│   ├── frontend/               # Visual Diagnostic Suite
│   │   ├── index.html          # Split-screen dashboard layout
│   │   ├── style.css           # Dynamic charting styles
│   │   └── app.js              # State & Rendering logic
│   └── tests/                  # API & Conversation verification
├── .env                        # Configuration (API Keys, TTL)
├── requirements.txt            # System dependencies
└── TECHNICAL_DOCUMENTATION.md  # Deep dive into diagnostic internals
```

## 🛠️ Setup

### Prerequisites

- **Python 3.9+**
- **LLM API Key**: Supports Groq (default Llama-3-8b).

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
   ```

## 🏃 Usage

1. **Start the Terminal**:
   ```bash
   uvicorn src.backend.main:app --reload
   ```
2. **Open the interface**:
   Launch `http://localhost:8000` in your browser.
3. **Analyze**:
   - Paste a story of at least 50 characters.
   - Review the **Visual Diagnostic Report** for signal overlap and classification margins.
   - Use the **Conversational Panel** to ask specific questions about certainty or class definitions.

## 📊 Model Architecture

- **Engine**: TF-IDF (15,000 Trigrams) + SGD Logistic Regression.
- **Certainty Logic**: Empirical calibration based on class transition boundaries.
- **Signal Logic**: Pure-measurement pattern matching (Kinetic, Visual, Cognitive, Sensory).

## ⚖️ Operational Boundaries

Paranormix is a **non-interpretive** tool. The AI conversational layer is programmed to:

1. Lock identity to the model's dominant prediction.
2. Refuse case-level reasoning (speculating on "why" a story matters).
3. Translate raw measurements into conceptual definitions.

---

Developed for research in Machine Learning Interpretability and Diagnostic Visualization.
