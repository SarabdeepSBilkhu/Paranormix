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

from src.ml.inference import ParanormalInvestigator
from src.backend.session_store import session_store

app = FastAPI(title="Paranormix - AI Investigator", version="3.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Investigator
investigator = ParanormalInvestigator()

# Initialize Generative AI
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
    print("Groq API configured successfully.")
else:
    client = None
    print("WARNING: GROQ_API_KEY not found. Chat functionality will be disabled.")

# System prompt for analyst chat
SYSTEM_PROMPT = """You are the Paranormix Technical Analyst. Your role is to decode and explain the system's Machine Learning diagnostics for research purposes.

ANALYST PROTOCOL:
1. Research Tone: Use professional, neutral, and precise language. Avoid speculative "investigator" roleplay.
2. Progressive Disclosure:
   - Initial Response: Confirm successful diagnostic axial capture. State primary diagnosis, confidence band (High/Moderate/Low), and model stability status.
   - Follow-up: Reveal specific observed signals or resolution boundary details ONLY when the user interrogates that specific metric.
3. Grounded Interpretation: Translate technical signals (e.g., "Kinetic disturbance") into conceptual definitions. Do NOT speculate on the "truth" or "haunting" of the story.
4. Transparency: If asked "why," focus on the statistical presence of patterns in the narrative and class overlap boundaries.

STRICT CONSTRAINTS:
- No emojis, flair, or robotic meta-prefixes ("Based on my analysis...").
- Identity Lock: You must never contradict the DOMINANT class reported in the diagnostic reference.
- Refuse case-level reasoning: You explain *what* the system detected, not *why* the actual paranormal event occurred.
"""

class ChatInput(BaseModel):
    session_id: Optional[str] = None
    user_message: str

@app.get("/")
async def root(request: Request):
    # Redirect to the frontend app
    return RedirectResponse(url="/app/")

@app.post("/chat")
async def chat(input_data: ChatInput):
    """Unified conversational analyst endpoint for narrative diagnostic decoding."""
    if not client:
        raise HTTPException(
            status_code=503, 
            detail="Analyst functionality unavailable. API configuration missing."
        )

    session_id = input_data.session_id
    is_initial_analysis = False
    now = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detection: Initial submission vs Follow-up
    if not session_id:
        if len(input_data.user_message) < 50:
            return {
                "response": "Input insufficient. Narrative must be at least 50 characters for axial analysis.",
                "session_id": None
            }
        
        # Trigger ML Analysis
        try:
            result = investigator.analyze(input_data.user_message)
            session_id = str(uuid.uuid4())
            session_store.create_session(
                session_id=session_id,
                narrative=input_data.user_message,
                prediction=result["prediction"],
                certainty=result["certainty"],
                evidence=result.get("observed_signals", []),
                modifiers=[], # Legacy
                competing=[h['class'] for h in result.get("ranked_matches", []) if h['label'] != 'DOMINANT'],
                chart_data=result["chart_data"]
            )

            # Store additional metadata locally in session
            session = session_store.get_session(session_id)
            session['report_time'] = now
            session['word_count'] = len(input_data.user_message.split())
            session['band'] = result.get("confidence_band", "Low")
            session['stability'] = result.get("stability_status", "Unknown")
            session['absent'] = result.get("absent_signals", [])
            session['ranked_matches_raw'] = result.get("ranked_matches", [])

            # Initialize history with the story and hidden diagnostic report
            session_store.append_message(session_id, "user", f"SUBJECT NARRATIVE: {input_data.user_message}")
            report_context = f"""DIAGNOSTIC AXIAL CAPTURE [{now}]:
PRIMARY_DIAGNOSIS: {result['prediction']}
CONFIDENCE_BAND: {result['confidence_band']}
STABILITY_INDEX: {result['stability_status']}

EMPIRICAL SIGNALS:
- OBSERVED: {', '.join(result.get('observed_signals', ['None']))}
- ABSENT: {', '.join(result.get('absent_signals', ['None']))}

DISTRIBUTION_METRICS:
{chr(10).join([f"{h['class']}: {h['p']:.2%} ({h['label']})" for h in result.get('ranked_matches', [])])}

PROTOCOL: Confirm results on turn 1. Use details to answer specific follow-up questions only.
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
            temperature=0.0 # Extreme grounding for research validity
        )

        bot_response = response.choices[0].message.content
        session_store.append_message(session_id, "assistant", bot_response)

        return {
            "response": bot_response,
            "turn_count": turn_count,
            "session_id": session_id,
            "is_initial": is_initial_analysis,
            "ml_data": {
                "prediction": session['prediction'],
                "certainty": session['certainty'],
                "band": session.get('band'),
                "stability": session.get('stability'),
                "observed": session['evidence'],
                "absent": session.get('absent'),
                "ranked_matches": session.get('ranked_matches_raw', []),
                "chart_data": session['chart_data'],
                "metadata": {
                    "timestamp": session.get('report_time'),
                    "words": session.get('word_count'),
                    "version": "3.0.1-research"
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
