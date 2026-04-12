"""
Paranormix FastAPI Backend (main.py)
====================================
Hybrid Explainable AI System combining ML-based signal extraction
with confidence-aware rule-based classification.

Architecture:
    Text → ML Signal Extractor → Resolver (score-based) → FastAPI → LLM Analyst
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os
import uuid
import datetime
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

# Load env
load_dotenv()

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.inference import SignalExtractor
from src.backend.session_store import session_store

app = FastAPI(title="Paranormix — Hybrid XAI System", version="5.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML system
extractor = SignalExtractor()

# Initialize LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ─── Academic-Grade System Prompt (INT428 Standards) ─────────────────────────
SYSTEM_PROMPT = """You are the Paranormix Technical Analyst.

You are given a DIAGNOSTIC REPORT containing:
- classification
- confidence
- signals
- evidence
- ignored signals
- ml probabilities

Your task is to explain the result using ONLY this data.

CORE INSTRUCTIONS:

INPUT VALIDATION:

If the input is not a narrative describing an event or occurrence:
Respond with:
"Invalid input: Narrative description required for analysis."

1. PRIMARY SIGNAL:
- Identify the primary signal using exact evidence words from the report.
- Explain why it is dominant using evidence presence (not assumptions).

2. SECONDARY SIGNALS:
- If other signals are present, mention them using exact evidence.
- Explain why they are weaker based on:
  - lack of supporting evidence, OR
  - fewer/weaker evidence markers
- Do NOT rely on hierarchy alone.

3. DECISION LOGIC (CRITICAL):
- Classification MUST be explained using evidence first.
- ML probabilities are ONLY supporting context.
- Do NOT use probability as the main reason for classification.

4. CONFIDENCE EXPLANATION:
- Explain confidence using evidence strength:
  - number of distinct evidence markers
  - diversity of evidence
- Do NOT explain confidence using probability values.

---

FOLLOW-UP HANDLING:

You may receive follow-up questions about the same diagnostic report.

For follow-ups:
- Continue using ONLY the original diagnostic report.
- Do NOT introduce new evidence or reinterpret the narrative.

If asked:
- "why" → explain using evidence comparison
- "how" → explain using signal strength and evidence presence
- "what if" → answer using counterfactual reasoning based on existing evidence

IMPORTANT:
- You ARE allowed to answer hypothetical or counterfactual questions using system logic.
- Do NOT respond with "Insufficient data" if reasoning is possible from the report.

Use "Insufficient data" ONLY if:
- the question requires external knowledge not present in the report.

EVIDENCE STRICTNESS (CRITICAL):

- You are ONLY allowed to use evidence terms explicitly present in the report.
- You may NOT introduce example evidence (e.g., "noise", "smell", "dream", "hallucination") unless they appear in the report.
- You may NOT describe what a class "usually" contains.

INTERPRETATION LIMIT:

- Do NOT infer entities, causes, or meanings beyond the evidence.
- Do NOT describe relationships like "single entity" or "external presence".
- Only describe what is directly observable from the evidence words.

PROBABILITY LIMIT:

- Do NOT describe probabilities as "low", "uncertain", or "ambiguous" reasoning.
- Probabilities are only supporting numbers, not explanatory justification.

---

CONSTRAINTS:

- Do NOT add new terms, symptoms, or concepts
- Do NOT hallucinate evidence
- Do NOT explain the system, categories, or methodology
- Do NOT use external knowledge (medical, folklore, etc.)
- Do NOT reinterpret evidence beyond what is explicitly present

---

STYLE:

- Direct, technical, and concise
- Use exact evidence terms (e.g., "shadow", "figure", "sleep paralysis")
- No generic explanations
- No system descriptions

---

REFUSAL RULE:

If the request is unrelated to Paranormix diagnostic analysis:
Respond with:
"This terminal is restricted to Paranormix Diagnostic Analysis."
"""


# ─── Models ──────────────────────────────────────────────────────────────────
class AnalyzeInput(BaseModel):
    text: str


class ChatInput(BaseModel):
    session_id: Optional[str] = None
    user_message: str


# ─── Evaluation Metrics Loader ───────────────────────────────────────────────
def load_metrics():
    """Parse the latest evaluation report into a structured format."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation_results', 'metrics_report.txt')
    metrics = {}
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # Simple regex to extract class metrics
            import re
            lines = content.split('\n')
            for line in lines:
                match = re.search(r'^\s*(\w+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', line)
                if match:
                    cls, p, r, f1 = match.groups()
                    if cls in ["material", "environmental", "immaterial", "rule_bound", "internal"]:
                        metrics[cls] = {"p": float(p), "r": float(r), "f1": float(f1)}
        except Exception as e:
            print(f"Error loading metrics: {e}")
    return metrics

def is_valid_narrative(text):
    text = text.lower()

    keywords = [
        "saw", "heard", "felt", "noticed", "woke", "happened",
        "appeared", "moved", "standing", "watching"
    ]

    return any(k in text for k in keywords)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/app/")


@app.post("/analyze")
async def analyze(input_data: AnalyzeInput):
    if len(input_data.text.strip()) < 20:
        raise HTTPException(400, "Input too short.")
    
    if not is_valid_narrative(input_data.text):
        raise HTTPException(400, "Invalid input: Narrative description required.")

    result = extractor.analyze(input_data.text)

    return {
        "classification": result["classification"],
        "confidence": result["confidence"],
        "confidence_band": result["confidence_band"],
        "signals": result["signals"],
        "evidence": result["evidence"],
        "ignored_signals": result["ignored_signals"],
        "ml_probs": result.get("ml_probs", {})
    }


@app.post("/chat")
async def chat(input_data: ChatInput):
    if not client:
        raise HTTPException(503, "LLM not configured.")
    
    if not is_valid_narrative(input_data.user_message):
        raise HTTPException(400, "Invalid input: Narrative description required.")

    session_id = input_data.session_id
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_initial = False

    # ── Initial request ──
    if not session_id:
        if len(input_data.user_message.strip()) < 20:
            return {"response": "Input too short.", "session_id": None}

        result = extractor.analyze(input_data.user_message)
        session_id = str(uuid.uuid4())
        is_initial = True

        session_store.create_session(
            session_id=session_id,
            narrative=input_data.user_message,
            prediction=result["classification"],
            certainty=result["confidence_band"],
            evidence=result["evidence"].get(result["classification"], []),
            modifiers=[],
            competing=result["ignored_signals"],
            chart_data={
                "signals": result["signals"],
                "evidence": result["evidence"],
                "confidence": result["confidence"],
                "ml_probs": result.get("ml_probs", {})
            }
        )

        session = session_store.get_session(session_id)
        session["report_time"] = now
        session["confidence_score"] = result["confidence"]

        # Build diagnostic context
        ml_probs = result.get("ml_probs", {})

        ignored_text = ""
        if result["ignored_signals"]:
            ignored_text = "\nSECONDARY SIGNALS (weaker score):\n"
            for sig in result["ignored_signals"]:
                ev = result["evidence"].get(sig, [])
                ignored_text += f"  - {sig}: {', '.join(ev)}\n"

        report_context = f"""
DIAGNOSTIC REPORT [{now}]
CLASS: {result['classification']}
CONFIDENCE: {result['confidence']} ({result['confidence_band']})

PRIMARY EVIDENCE:
{', '.join(result['evidence'].get(result['classification'], []))}

{ignored_text}

ML PROBABILITIES:
material: {ml_probs.get('material', 0):.3f}
environmental: {ml_probs.get('environmental', 0):.3f}
immaterial: {ml_probs.get('immaterial', 0):.3f}
rule_bound: {ml_probs.get('rule_bound', 0):.3f}
internal: {ml_probs.get('internal', 0):.3f}
"""

        session_store.append_message(
            session_id,
            "user",
            "DIAGNOSTIC REPORT:\n\n" + report_context
        )

    else:
        session_store.append_message(session_id, "user", input_data.user_message)

    # ── LLM call ──
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += session_store.get_history(session_id)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        # INT428 CONFIGURATION:
        # Low temperature ensures deterministic, objective analysis grounded in evidence.
        temperature=0.0, 
        # Top-p restricted to 0.1 to filter for the most probable, technically accurate tokens.
        top_p=0.1,
        max_tokens=400
    )

    reply = response.choices[0].message.content
    session_store.append_message(session_id, "assistant", reply)

    # ── Assemble Result ──
    ml_data = None
    if is_initial:
        session = session_store.get_session(session_id)
        ml_data = {
            "classification": session.get('prediction', 'internal'),
            "confidence": session.get('confidence_score', 0),
            "confidence_band": session.get('certainty', 'Low'),
            "evidence": session.get('chart_data', {}).get('evidence', {}),
            "ignored_signals": session.get('competing', []),
            "signals": session.get('chart_data', {}).get('signals', {}),
            "ml_probs": session.get('chart_data', {}).get('ml_probs', {}),
            "metrics": load_metrics(),
            "metadata": {
                "timestamp": session.get('report_time', now),
                "words": len(input_data.user_message.split()),
                "version": "5.1.0-xai"
            }
        }

    return {
        "response": reply,
        "session_id": session_id,
        "is_initial": is_initial,
        "ml_data": ml_data
    }


# ─── Frontend ────────────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')

if os.path.exists(frontend_path):
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)