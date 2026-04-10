# Paranormix: Multi-Axial Research Terminal (V4.1.2-xai)

Paranormix is a high-fidelity **Explainable AI (XAI)** research terminal designed for the empirical analysis and decoding of paranormal narratives. It separates deterministic machine learning detection from human-centric conversational interpretation to provide grounded, measurement-based insights.

---

## 🔬 Core Philosophy: The XAI Separation

Paranormix operates under a strict **Non-Interpretive XAI** methodology. Most AI systems conflate "finding a pattern" with "explaining the cause." Paranormix intentionally decouples these:

1.  **The Detection Layer (The Engine)**: A deterministic scikit-learn pipeline combined with a rule-based precedence engine that identifies and resolves statistical clusters.
2.  **The Interpretation Layer (The Analyst)**: An LLM-driven persona (Llama-3.1-8b) that *transcribes* and *decodes* the engine's measurements into natural dialogue, strictly adhering to the empirical precedence data.

This separation ensures the Analyst acknowledes all signals while resolving a single, definitive classification based on absolute priority.

---

## 🏗️ System Architecture

The project is structured into three distinct but highly synchronized layers:

### 1. The Machine Learning Engine (`src/ml/`)
- **Preprocessing**: Utilizes a custom `lemmatize_tokenizer` (SpaCy-powered) to consolidate semantic variations (e.g., "ghosts" → "ghost") before vectorization.
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) with sublinear scaling and bi-gram support for stable pattern recognition.
- **Model**: An `SGDClassifier` using Log Loss (Logistic Regression equivalent). Results are passed through a **Precedence Resolver** that prioritizes physical evidence over cognitive states.

### 2. The Backend Orchestration (`src/backend/`)
- **FastAPI**: A high-performance async API that manages the lifecycle of a diagnostic session.
- **Session Store**: An in-memory, volatile store that tracks narrative history and diagnostic artifacts.
- **Analyst Persona**: Powered by Llama-3.1-8b (via Groq) with a 0.0 temperature setting for maximum factual consistency.

### 3. The Visual Terminal (`src/frontend/`)
- **Progressive Disclosure**: The UI begins as a focused narrative entry point and transitionally reveals the "Empirical Data Axis" (dashboard) only after analysis is complete.
- **Axial Reveal**: A custom 50/50 split-screen animation that physically shifts the conversational context to make room for quantitative metrics.

---

## 📊 V4.x Research Schema: Measurement & Resolution

The heart of Paranormix is its resolution logic, which ensures a single, stable classification even in complex narratives:

### Precedence Resolution (The Priority Axis)
Paranormix resolves signals based on an absolute hierarchy of evidence:
1.  **material**: Physical residue, biological markers, and structural damage.
2.  **environmental**: Manipulation of physical space, sound, and thermal shift.
3.  **immaterial**: Visual presence without direct physical interaction.
4.  **rule_bound**: Narrative-governed constraints and ritualistic logic.
5.  **internal**: Subjective cognitive experience with no external manifestation.

*Resolution Rule: A higher-tier signal always overrides lower-tier signals.*

### Certainty & Confidence Bands
Diagnostics are mapped into three academic tiers based on signal density:
- **High (≥60%)**: Strong, unambiguous signal alignment.
- **Moderate (35–59%)**: Detectable signal with moderate interference.
- **Low (<35%)**: Trace signals only; results are considered hypothetical.

### Persistence & Context
The engine provides a `stability_status` which tracks if signals are consistent across the narrative or isolated incidents.

---

## 🤖 The Paranormix Analyst Protocol

The Analyst persona is not a "Ghost Hunter." It is a technical auditor. Its behavior is governed by the following constraints:

- **Precedence Transparency**: The Analyst explicitly acknowledges secondary signals but explains they were ignored due to the priority of the primary classification.
- **No Speculation**: It refuses to hypothesize about the "truth" behind the paranormal; it only reports what the system detected.
- **Deterministic Persona**: Powered by Llama-3.1-8b at 0.0 temperature for factual stability.

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

- `src/ml/inference.py`: Signal extraction and measurement engine.
- `src/ml/resolver.py`: Precedence-based classification resolution.
- `src/backend/main.py`: FastAPI implementation and Analyst protocols (V4.1.2).
- `src/frontend/app.js`: Implementation of the Axial Reveal and Chart.js UI.

---

*Developed for AI Interpretability and Machine Learning Diagnostic Research.*
