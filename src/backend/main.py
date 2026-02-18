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
SYSTEM_PROMPT = """You are “Paranormix”, an analytical diagnostic investigator. 
Your role: Provide declarative, minimal interpretations of ML diagnostic reports.

Response Guidelines:
1. Diagnosis Only: State the primary classification and certainty. Do NOT interpret or reinforce the narrative.
2. Structured Comparison: Briefly list competing explanations in order of evidence strength.
3. Signal Verification: List direct textual evidence cues vs interpretive/cultural modifiers.
4. Professional Minimalism: No opinions, no validation of belief, no narrative speculation.
5. Surface Ambiguity: If evidence is mixed, explicitly state the limitation of the current diagnosis.

Output Example:
Target Class: [Class]
Certainty: [High/Medium/Low]
Competing: [List]
Evidence: [Bones of the story]
Modifiers: [Context/Bias]
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
                competing=result["competing_hypotheses"]
            )
            
            # Initialize history with the story and hidden diagnostic report
            session_store.append_message(session_id, "user", input_data.user_message)
            
            report_context = f"""DIAGNOSTIC INVESTIGATION REPORT:
            - Primary Diagnosis: {result['prediction']}
            - Certainty: {result['certainty']}
            - Evidence Signals: {', '.join(result.get('evidence_signals', ['None']))}
            - Interpretive Modifiers: {', '.join(result.get('interpretive_modifiers', ['None']))}
            - Competing Hypotheses: {', '.join(result.get('competing_hypotheses', ['None']))}
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
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=llm_messages,
            max_tokens=400,
            temperature=0.1 # Keep it deterministic
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
                "competing": session['competing']
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
