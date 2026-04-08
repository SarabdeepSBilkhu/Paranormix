# Paranormix: Multi-Axial Research Terminal (V3.01)

Paranormix is a high-fidelity **Explainable AI (XAI)** research terminal designed for the empirical analysis and decoding of paranormal narratives. It separates deterministic machine learning detection from human-centric conversational interpretation to provide grounded, measurement-based insights.

---

## 🔬 Core Philosophy: The XAI Separation

Paranormix operates under a strict **Non-Interpretive XAI** methodology. Most AI systems conflate "finding a pattern" with "explaining the cause." Paranormix intentionally decouples these:

1.  **The Detection Layer (The Engine)**: A deterministic scikit-learn pipeline that identifies statistical clusters in text without bias or speculation.
2.  **The Interpretation Layer (The Analyst)**: An LLM-driven persona that *transcribes* and *decodes* the engine's measurements into natural dialogue, strictly adhering to the empirical data provided.

This separation ensures that the "AI" never imagines details not supported by the underlying mathematical model.

---

## 🏗️ System Architecture

The project is structured into three distinct but highly synchronized layers:

### 1. The Machine Learning Engine (`src/ml/`)
- **Preprocessing**: Utilizes a custom `lemmatize_tokenizer` (SpaCy-powered) to consolidate semantic variations (e.g., "ghosts" → "ghost") before vectorization.
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) with sublinear scaling and bi-gram support for stable pattern recognition.
- **Model**: An `SGDClassifier` using Log Loss (Logistic Regression equivalent) with custom class weights to balance sparse categories like `creature` and bias results towards `psychological` explanations when ambiguity is high.

### 2. The Backend Orchestration (`src/backend/`)
- **FastAPI**: A high-performance async API that manages the lifecycle of a diagnostic session.
- **Session Store**: An in-memory, volatile store that tracks narrative history and diagnostic artifacts.
- **Analyst Persona**: Powered by Llama-3.1-8b (via Groq) with a 0.0 temperature setting for maximum factual consistency.

### 3. The Visual Terminal (`src/frontend/`)
- **Progressive Disclosure**: The UI begins as a focused narrative entry point and transitionally reveals the "Empirical Data Axis" (dashboard) only after analysis is complete.
- **Axial Reveal**: A custom 50/50 split-screen animation that physically shifts the conversational context to make room for quantitative metrics.

---

## 📊 V3.01 Research Schema: Measurements In-Depth

The heart of Paranormix is its measurement logic, which provides researchers with several stability markers:

### Certainty & Confidence Bands
Diagnostics are mapped into three academic tiers based on the primary probability score:
- **High (≥60%)**: Robust signal alignment; low entropy.
- **Moderate (35–59%)**: Noticeable ambiguity; multiple competing signals.
- **Low (<35%)**: High entropy; results are purely hypothetical.

### Dominance Ranking
Classes are labeled based on their relative "pull" within the distribution:
- **DOMINANT**: The primary statistical match.
- **CONTENDER**: A secondary class with >15% probability.
- **TRACE**: Residual signals (>5%).
- **NOISE**: Statistically insignificant signals.

### Stability & Resolution Boundaries
The engine accounts for **Historical Overlap**. For instance, narratives involving "Apparitions" and "Folklore" often share similar keywords. If a result falls into an overlap zone, the `stability_status` triggers a **Resolution Limit**, automatically capping the reported certainty to prevent over-confidence in ambiguous data.

### Narrative Purity
This is a calculated metric representing the "distance" between the top two classes. A high gap indicates a "pure" narrative that fits a single profile perfectly.

---

## 🤖 The Paranormix Analyst Protocol

The Analyst persona is not a "Ghost Hunter." It is a technical auditor. Its behavior is governed by the following constraints:

- **Identity Lock**: The Analyst must never contradict the model's primary diagnosis.
- **Explanatory vs. Descriptive**: Instead of saying "Confidence is 80%," it explains, "The system shows high reliability in this classification due to the clear presence of..."
- **No Speculation**: The Analyst is programmed to refuse requests to explain *why* a paranormal event occurred (e.g., "Was it a demon?"). It only explains why the *system* labeled it as such.
- **Hidden Context**: Every Analyst turn is grounded by a hidden "Diagnostic Axial Capture" containing word counts, UTC timestamps, and raw probability arrays.

---

## 🛠️ Setup and Operational Protocols

### Prerequisites
- Python 3.9+
- Groq API Key (for the Analyst layer)

### Local Implementation
1. **Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Configuration**: Create a `.env` file with `GROQ_API_KEY=your_key_here`.
3. **Execution**:
   ```bash
   uvicorn src.backend.main:app --reload
   ```

### Operational Edge Cases
- **Input Threshold**: Narratives must be at least 50 characters to triggeraxial analysis. Input below this is rejected as "Inert."
- **Turn Limits**: To prevent session bloat and maintain focus, sessions are capped at 5 turns (adjustable via `MAX_CHAT_TURNS`).
- **Low Confidence Handling**: When confidence falls into the **Low** band, the Analyst is instructed to adopt a more cautious, hypothetical tone, emphasizing the ambiguity of the signals.
- **API Failures**: If the LLM layer fails, the terminal provides the raw ML data artifacts directly to ensures the research remains accessible.

---

## 📁 Project Map

- `src/ml/inference.py`: The primary engine and measurement layer.
- `src/ml/train.py`: Training script with custom class weights and keyword heuristics.
- `src/backend/main.py`: FastAPI endpoints and Analyst prompting logic.
- `src/frontend/app.js`: Implementation of the Axial Reveal and Chart.js rendering.

---

*Developed for AI Interpretability and Machine Learning Diagnostic Research.*
