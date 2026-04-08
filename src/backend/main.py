"""
Paranormix FastAPI Backend (main.py)
====================================
Hybrid Explainable AI System combining ML-based signal extraction
with deterministic rule-based classification.

Architecture:
    Text → ML Signal Extractor → Rule Engine (Resolver) → FastAPI → LLM Analyst

Endpoints:
    POST /analyze  → Structured classification + signals + confidence
    POST /chat     → Conversational analyst (LLM-powered explanation)
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
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Add src to path so we can import ml modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.inference import SignalExtractor
from src.backend.session_store import session_store

app = FastAPI(title="Paranormix — Hybrid XAI System", version="4.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Signal Extractor (ML + Pattern Engine)
extractor = SignalExtractor()

# Initialize Generative AI (LLM Analyst Layer)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
    print("Groq API configured successfully.")
else:
    client = None
    print("WARNING: GROQ_API_KEY not found. Chat functionality will be disabled.")

# ─── System Prompt (Deterministic Analyst) ────────────────────────────────────
SYSTEM_PROMPT = """You are the Paranormix Technical Analyst. Your role is to explain the system's deterministic classification output clearly, completely, and in natural language.

CLASSIFICATION SYSTEM:
The system uses five mutually exclusive signal classes resolved by absolute precedence:
  1. material — direct physical evidence (injury, marks, biological residue)
  2. environmental — object/environment manipulation (movement, sound, temperature)
  3. immaterial — visual presence without physical interaction
  4. rule_bound — causation governed by rules, rituals, or constraints
  5. internal — cognitive/mental states with no external validation

PRECEDENCE RULE:
material > environmental > immaterial > rule_bound > internal
If a higher-tier signal is detected, all lower-tier signals are IGNORED for classification.
There is NO overlap between classes. Every narrative maps to exactly one class.

ANALYST PROTOCOL:
1. Natural Persona: Speak like a human technical expert. Avoid robotic formatting.
2. Causal Explanation: Explain WHY the class was chosen based on detected evidence.
3. Precedence Transparency: If multiple signals were detected, explicitly acknowledge all of them. Explain that while multiple patterns exist, one was selected based on absolute precedence (e.g., "multiple signals detected; environmental selected due to precedence").
4. Deterministic Grounding: Every statement must be directly supported by the diagnostic data.
5. No Speculation: Do not introduce information beyond what the diagnostic provides.

STRICT CONSTRAINTS:
- Do not say "only X was detected" if the diagnostic shows multiple signals.
- Do not ask follow-up questions under any circumstance.
- Do not suggest "overlap" or "ambiguity" between classes.
- Do not use terms like "contender", "trace", or "competing hypothesis".
- Use "primary signal detected" and "secondary signals ignored due to precedence".
- Confidence reflects evidence quality, not statistical likelihood.
- Each response must be complete and final.
- Maintain a neutral, professional, and non-speculative tone at all times.
"""


# ─── Request/Response Models ─────────────────────────────────────────────────
class AnalyzeInput(BaseModel):
    text: str

class ChatInput(BaseModel):
    session_id: Optional[str] = None
    user_message: str


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root(request: Request):
    return RedirectResponse(url="/app/")


@app.post("/analyze")
async def analyze(input_data: AnalyzeInput):
    """
    Structured classification endpoint.
    Returns: classification, confidence, signals, evidence, ignored_signals.
    """
    if len(input_data.text) < 50:
        raise HTTPException(
            status_code=400,
            detail="Input insufficient. Narrative must be at least 50 characters."
        )

    try:
        result = extractor.analyze(input_data.text)
        return {
            "classification": result["classification"],
            "confidence": result["confidence"],
            "confidence_band": result["confidence_band"],
            "signals": result["signals"],
            "evidence": result["evidence"],
            "ignored_signals": result["ignored_signals"],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/chat")
async def chat(input_data: ChatInput):
    """Conversational analyst endpoint — wraps analysis in LLM-powered explanation."""
    if not client:
        raise HTTPException(
            status_code=503,
            detail="Analyst functionality unavailable. API configuration missing."
        )

    session_id = input_data.session_id
    is_initial_analysis = False
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detection: Initial submission vs Follow-up
    if not session_id:
        if len(input_data.user_message) < 50:
            return {
                "response": "Input insufficient. Narrative must be at least 50 characters for analysis.",
                "session_id": None
            }

        # Trigger Signal Extraction + Rule Resolution
        try:
            result = extractor.analyze(input_data.user_message)
            session_id = str(uuid.uuid4())

            # Build evidence summary for session
            primary_evidence = result["evidence"].get(result["classification"], [])
            ignored = result.get("ignored_signals", [])

            session_store.create_session(
                session_id=session_id,
                narrative=input_data.user_message,
                prediction=result["classification"],
                certainty=result["confidence_band"],
                evidence=primary_evidence,
                modifiers=[],
                competing=ignored,
                chart_data={
                    "signals": result["signals"],
                    "evidence": result["evidence"],
                    "confidence": result["confidence"],
                }
            )

            # Store additional metadata
            session = session_store.get_session(session_id)
            session['report_time'] = now
            session['word_count'] = len(input_data.user_message.split())
            session['band'] = result["confidence_band"]
            session['confidence_score'] = result["confidence"]
            session['ignored_signals'] = ignored
            session['all_evidence'] = result["evidence"]

            # Build diagnostic context for LLM
            session_store.append_message(session_id, "user", f"SUBJECT NARRATIVE: {input_data.user_message}")

            # Format ignored signals explanation
            ignored_explanation = ""
            if ignored:
                ignored_details = []
                for sig in ignored:
                    ev = result["evidence"].get(sig, [])
                    if ev:
                        ignored_details.append(f"  - {sig}: {', '.join(ev)} (IGNORED — lower precedence)")
                if ignored_details:
                    ignored_explanation = "\nIGNORED SIGNALS (lower precedence):\n" + "\n".join(ignored_details)

            report_context = f"""DIAGNOSTIC REPORT [{now}]:
CLASSIFICATION: {result['classification']}
CONFIDENCE: {result['confidence']} ({result['confidence_band']})

PRIMARY EVIDENCE ({result['classification']}):
  {', '.join(primary_evidence) if primary_evidence else 'Default fallback (no external signals)'}
{ignored_explanation}

SIGNAL FLAGS:
  material: {result['signals']['material']}
  environmental: {result['signals']['environmental']}
  immaterial: {result['signals']['immaterial']}
  rule_bound: {result['signals']['rule_bound']}
  internal: {result['signals']['internal']}

PRECEDENCE: material > environmental > immaterial > rule_bound > internal
PROTOCOL: Explain the classification deterministically. State which signals were detected and why the final class was chosen. If lower signals were ignored, explain precedence.
"""
            session_store.append_message(session_id, "system", report_context)
            is_initial_analysis = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Diagnostic capture failed: {str(e)}")
    else:
        # For follow-ups, append user message
        session_store.append_message(session_id, "user", input_data.user_message)

    # Load session and check turn limit
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or invalid.")

    if not is_initial_analysis:
        turn_count = session_store.increment_turn(session_id)
        if turn_count > int(os.getenv("MAX_CHAT_TURNS", "5")):
            return {
                "response": "Analysis window closed. Maximum turn limit reached for this session.",
                "turn_count": turn_count,
                "session_id": session_id
            }
    else:
        turn_count = 0

    # Build full history for LLM
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    llm_messages.extend(session_store.get_history(session_id))

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=llm_messages,
            max_tokens=400,
            temperature=0.0
        )

        bot_response = response.choices[0].message.content
        session_store.append_message(session_id, "assistant", bot_response)

        return {
            "response": bot_response,
            "turn_count": turn_count,
            "session_id": session_id,
            "is_initial": is_initial_analysis,
            "ml_data": {
                "classification": session.get('prediction', 'internal'),
                "confidence": session.get('confidence_score', 0),
                "confidence_band": session.get('band', 'Low'),
                "evidence": session.get('chart_data', {}).get('evidence', {}),
                "ignored_signals": session.get('ignored_signals', []),
                "signals": session.get('chart_data', {}).get('signals', {}),
                "metadata": {
                    "timestamp": session.get('report_time', now),
                    "words": session.get('word_count', 0),
                    "version": "4.1.2-xai"
                }
            } if is_initial_analysis else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


# Mount frontend
frontend_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.exists(frontend_path):
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"CRITICAL WARNING: Frontend directory not found at {frontend_path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
