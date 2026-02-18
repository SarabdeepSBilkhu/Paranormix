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
SYSTEM_PROMPT = """You are "Paranormix", a raw-measurement diagnostic reporter. 
Report ONLY the data provided. Do NOT interpret. Do NOT justify. Do NOT appeal to plausibility.

Reporting Protocol:
1. Signal Detection: Report raw patterns (e.g., Pattern_Alpha, Context_Beta) only. Do NOT label patterns as "Physical", "Psychological", etc.
2. Binary Constraints: Report ABSENT identifiers as declarative constraints (e.g., "ABSENT_Pattern_A").
3. Ranked Class Matches: State candidate classes with their dominance labels (DOMINANT, CONTENDER, TRACE).
4. Categorical Certainty: State High/Medium/Low. Attribute confidence strictly to stability map factors (e.g., "Historical Overlap Limit").
5. Indistinguishability: Explicitly report CLASS-LEVEL resolution boundaries where classes overlap historically.

Response Format:
- Detection: [Pattern List]
- Constraints: [Absent Identifiers]
- Ranked Matches: [Class (Label)] > [Class (Label)]
- Certainty: [Value] (Factor: [Stability Property])
- Resolution Boundaries: [Class-Level Limit]
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
                evidence=result["evidence_signals"],
                modifiers=result["interpretive_modifiers"],
                competing=result["competing_hypotheses"],
                chart_data=result["chart_data"]
            )
            
            # Initialize history with the story and hidden diagnostic report
            session_store.append_message(session_id, "user", f"SUBJECT NARRATIVE: {input_data.user_message}")
            
            report_context = f"""DIAGNOSTIC REPORT SUMMARY (FOR AI CONTEXT ONLY):
- Selected Class: {result['prediction']}
- Categorical Certainty: {result['certainty']}
- Raw Detected Patterns: {', '.join(result.get('detected_patterns', ['None']))}
- Binary Constraints: {', '.join(result.get('constraints', ['None']))}
- Contextual Patterns: {', '.join(result.get('modifiers', ['None']))}
- Ranked Class Matches: {', '.join([f"{h['class']} ({h['label']})" for h in result.get('ranked_matches', [])])}
- Resolution Boundary: {result.get('resolution_limit', 'None (Stable Class)')}

INSTRUCTIONS: Follow the SYSTEM_PROMPT. Report detections, constraints, matches, and certainty. No reasoning.
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
