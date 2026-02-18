# Paranormix - XAI Research Terminal

A high-fidelity Explainable AI (XAI) research terminal for the multi-axial analysis of paranormal narratives. Paranormix combines a deterministic scikit-learn engine with a human-centric Analyst Decoder to provide grounded, measurement-based insights through a dynamic research interface.

## 🚀 Overview

Paranormix is a **Technical Diagnostic Terminal** designed for researchers to decode the hidden patterns within subject narratives. It prioritizes **Progressive Disclosure**—revealing empirical metrics only when they serve to clarify the diagnostic axial capture.

### Key Features:

- **Humanized Analyst Persona**: A conversational decoder that speaks as a technical expert, weaving empirical findings into natural, explanatory dialogue.
- **Dynamic 50/50 Split Layout**: The interface begins as a focused full-page chat and smoothly transitions to a balanced split-screen view revealing the **Empirical Data Axis** on the right.
- **V3.01 Research Schema**:
  - **Confidence Bands**: High, Moderate, and Low certainty mapping for academic calibration.
  - **Stability Indicators**: Explicit reporting of historical class overlap and resolution boundaries.
  - **Grouped Signals**: Clear distinction between **Observed Patterns** and **Absent Indicators**.
- **Deterministic Grounding**: The Analyst logic is locked to a 0.0 temperature setting, ensuring maximum factual consistency with the underlying ML models.

## 📁 Project Structure

```
Paranormix/
├── src/
│   ├── ml/                     # The Engine (Pattern Extraction & Inference)
│   │   └── inference.py        # Research Logic & Measurement Layer
│   ├── backend/                # The Brain (Chat Orchestration & API)
│   │   ├── main.py             # Human Analyst Persona & Session Context
│   │   └── session_store.py    # Memory-based context management
│   ├── frontend/               # The Terminal (Research Interface)
│   │   ├── index.html          # Dynamic 50/50 split-screen layout
│   │   ├── style.css           # Humanized Research Aesthetic
│   │   └── app.js              # Axial reveal & Rendering logic
├── .env                        # Configuration (Groq API, TTL)
├── requirements.txt            # System dependencies
└── TECHNICAL_DOCUMENTATION.md  # Exhaustive deep dive into XAI internals
```

## 🛠️ Setup

### Prerequisites

- **Python 3.9+**
- **Groq API Key**: (Default: Llama-3-8b-instant) for the Analyst layer.

### Installation

1. **Clone the repository** and enter the directory.
2. **Setup Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Environment Variables**:
   Create a `.env` file with your `GROQ_API_KEY`.

## 🏃 Usage

1. **Start the Research Terminal**:
   ```bash
   uvicorn src.backend.main:app --reload
   ```
2. **Access the Terminal**: Launch `http://localhost:8000` in your browser.
3. **Perform Analysis**:
   - Provide a subject narrative (min 50 characters).
   - Observe the **Axial Reveal**: The chat moves left to reveal the diagnostic metrics on the right.
   - Interrogate the **Human Analyst** to decode the specific signals and stability indicators.

## ⚖️ Operational Protocols

To maintain research integrity, the Analyst is strictly constrained:

1. **No Speculation**: The Analyst refuses to explain _why_ a paranormal event occurred.
2. **Identity Lock**: Responses must never contradict the model's primary diagnosis.
3. **Technical Transparency**: All metadata (word counts, timestamps) is provided for academic traceability.

---

Developed for AI Interpretability and Machine Learning Diagnostic Research.
