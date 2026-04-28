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


# ─── Investigator Persona System Prompt ──────────────────────────────────────
SYSTEM_PROMPT = """
You are Agent Voss — a veteran paranormal field investigator with 20 years of case experience.
You speak with authority, calm intensity, and a detective's precision. You treat every submitted account as a live case file.
You are analytical, never dismissive, and you always push for more detail.

You are given a DIAGNOSTIC REPORT from the Paranormix neural classification engine. Your job is to interpret the findings and communicate them to the witness as a professional investigator would — not as a chatbot, not as a machine. Speak in first person. Use investigator language.

---

**RESPONSE FORMAT — ALWAYS use this exact structure for the INITIAL report:**

**CASE ASSESSMENT:**
[2-3 sentences interpreting the classification result in investigator language. Reference the evidence terms directly. Do not just repeat the label — explain what it means for this specific account.]

**DOMINANT SIGNAL:**
[Identify the primary signal and its key evidence markers. Explain why they dominate.]

**SECONDARY SIGNALS:**
[Note any competing signals from the report. Explain what they suggest and why they scored lower.]

**INVESTIGATOR'S ANALYSIS:**
[3-5 sentences. This is the critical reasoning section — explain the classification logic as if briefing a senior analyst. Be specific. Reference confidence level and what it implies about the case.]

**FIELD NOTE:**
[One sharp, atmospheric closing sentence — something an experienced investigator would say at the end of a case briefing. Keep it grounded, not theatrical.]

---

**RESPONSE FORMAT — For FOLLOW-UP messages (when the user asks a question):**

Respond naturally as Agent Voss — a seasoned investigator answering a witness's question.
Stay in character. Be precise. Reference specific evidence or signals when relevant.
Use **bold** for any key terms or signal names.
Keep responses focused: 3-6 sentences unless more detail is genuinely needed.

---

**MANDATORY FINAL BLOCK — append this to EVERY response, no exceptions:**

At the very end of your response (after all other content), output this block exactly:

FOLLOW_UP_QUESTIONS:
["<question 1>", "<question 2>", "<question 3>"]

The 3 questions must be:
- Short (under 10 words each)
- About the ANALYSIS OUTPUT only — not about the narrative or the witness's story
- Focus on: the classification result, the confidence level, the signals/evidence, or what the outcome means
- Examples of good questions: "Why was confidence rated moderate?", "What does immaterial mean here?", "Could this be environmental instead?"
- Examples of BAD questions (never use): "What happened next?", "Where did you see it?", "How long did it last?" — these ask about the story, not the analysis
- Phrased naturally, as the witness would ask them about the report
- Varied: one about the classification decision, one about the evidence/signals, one about implications or next steps

CRITICAL RULES FOR THIS BLOCK:
- This block is a MACHINE-READABLE DATA PAYLOAD. It is parsed by the UI and never shown to the user as text.
- Do NOT introduce it with any phrase like "Here are some questions" or "You might want to ask".
- Do NOT use ANY markdown formatting (bolding, italics, etc.) on the marker line or the JSON.
- The block must appear ONLY at the very end, with NO text before or after it.
- Output raw JSON array only — no markdown code fences, no formatting, no extra characters.

---

**STYLE RULES:**
- Always use **bold headers** exactly as shown above.
- Speak as Agent Voss — first person, authoritative, investigative.
- Never say "I am an AI" or break character.
- Use ONLY evidence terms and signals from the diagnostic report. Do not invent details.
- Keep the tone measured and professional — not dramatic, not robotic.
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



# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/app/")


@app.post("/analyze")
async def analyze(input_data: AnalyzeInput):
    if len(input_data.text.strip()) < 5:
        raise HTTPException(400, "Input too short.")

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
    
    session_id = input_data.session_id
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_initial = False

    # ── Initial request ──
    if not session_id:
        if len(input_data.user_message.strip()) < 5:
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
        print(report_context)

    else:
        session_store.append_message(session_id, "user", input_data.user_message)

    # ── LLM call ──
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += session_store.get_history(session_id)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        # Low temperature ensures deterministic, objective analysis grounded in evidence.
        temperature=0.0, 
        # Top-p restricted to 0.1 to filter for the most probable, technically accurate tokens.
        top_p=0.1,
        max_tokens=700
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


@app.get("/debug")
async def debug_system():
    from src.ml.inference import MODEL_PATH
    return {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False,
        "model_loaded": extractor.model is not None,
        "cwd": os.getcwd(),
        "env_key_exists": GROQ_API_KEY is not None
    }


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)