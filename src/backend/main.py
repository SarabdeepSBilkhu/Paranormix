from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os
import uuid
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

# System prompt for chatbot
SYSTEM_PROMPT = """You are Paranormix, a conversational investigator's assistant. Your job is to translate complex ML analysis into human-friendly, grounded insights.

CONVERSATIONAL PROTOCOL:
1. No Meta-Language: Never say "The analysis report shows," "Selected class is," or "The data indicates." Instead, anchor descriptions in the user's narrative (e.g., "This means for your account...", "Your story aligns most with...").
2. Progressive Disclosure:
   - Turn 1 (Initial Report): ONLY confirm the analysis is done, state the primary result and confidence, and invite specific questions. Do NOT list signals, competitors, or boundaries yet.
   - Subsequent Turns: Reveal details ONLY when asked. If asked "why," explain the conceptual influence of signals rather than listing technical pattern IDs.
3. Natural Translation: Translate internal codes (like Pattern_A) into human terms (like "physical disturbance") using the provided Translation Key.
4. Anchoring: Always link findings back to the user's specific story details to make the interaction feel personal and grounded.

STRICT BOUNDARIES:
- You do NOT decide the classification; you decode it.
- You do NOT claim insight into model weights.
- You do NOT judge the truth of the narrative.

Tone: Professional, empathetic, direct, and conversational.
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
    """Unified conversational endpoint for narrative analysis and investigating."""
    if not client:
        raise HTTPException(
            status_code=503, 
            detail="Chat functionality unavailable. GROQ_API_KEY not configured."
        )

    session_id = input_data.session_id
    is_initial_analysis = False

    # Detection: Initial submission vs Follow-up
    if not session_id:
        if len(input_data.user_message) < 50:
            return {
                "response": "Please share a narrative of at least 50 characters so I can begin the investigation.",
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
                evidence=result.get("detected_patterns", []),
                modifiers=result.get("modifiers", []),
                competing=[h['class'] for h in result.get("ranked_matches", [])],
                chart_data=result["chart_data"]
            )

            # Initialize history with the story and hidden diagnostic report
            session_store.append_message(session_id, "user", f"SUBJECT NARRATIVE: {input_data.user_message}")
            report_context = f"""INTERNAL DIAGNOSTIC REFERENCE (NOT FOR RECITATION):

TRANSLATION KEY:
- Pattern_A: Physical disturbance / Kinetic energy
- Pattern_B: Sensory distortion / Temperature shift
- Pattern_C: Information-based / Historical matching
- Pattern_D: Visual anomaly / Residual echo
- resolution_boundary: Historical model overlap area

MEASUREMENT DATA:
Selected Class: {result['prediction']}
Certainty: {result['certainty']}
Detected Patterns: {', '.join(result.get('detected_patterns', ['None']))}
Absent Patterns: {', '.join(result.get('constraints', ['None']))}
Competitors: {', '.join([f"{h['class']} ({h['label']})" for h in result.get('ranked_matches', [])])}
Resolution Limit: {result.get('resolution_limit', 'None')}

Note: Use these details ONLY when asked. Initial turn should be a Confirm-Result-Invite response.
"""
            session_store.append_message(session_id, "system", report_context)
            is_initial_analysis = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ML Diagnostic failed: {str(e)}")
    else:
        # For follow-ups, append user message
        session_store.append_message(session_id, "user", input_data.user_message)

    # Load session and check turn limit
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired. Please restart.")

    if not is_initial_analysis:
        turn_count = session_store.increment_turn(session_id)
        if turn_count > int(os.getenv("MAX_CHAT_TURNS", "5")):
            return {
                "response": "Investigation concluded. Please reset to start a fresh analysis.",
                "turn_count": turn_count,
                "session_id": session_id
            }
    else:
        turn_count = 0

    # Build full history for LLM
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    llm_messages.extend(session_store.get_history(session_id))

    try:
        # Lower temperature for diagnostic reliability
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=llm_messages,
            max_tokens=400,
            temperature=0.1
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
                "evidence": session['evidence'],
                "modifiers": session['modifiers'],
                "competing": session['competing'],
                "chart_data": session['chart_data']
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
