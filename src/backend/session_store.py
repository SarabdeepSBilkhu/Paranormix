# Session Store for Chatbot Functionality
# Manages in-memory session storage for conversational context

from datetime import datetime, timedelta
from typing import Dict, Optional
import threading

class SessionStore:
    def __init__(self, ttl_minutes: int = 30, max_sessions: int = 100):
        self.sessions: Dict[str, dict] = {}
        self.ttl_minutes = ttl_minutes
        self.max_sessions = max_sessions
        self.lock = threading.Lock()
    
    def create_session(self, session_id: str, narrative: str, prediction: str, 
                      certainty: str, evidence: list, modifiers: list,
                      competing: list) -> None:
        """Store analysis results in session."""
        with self.lock:
            # Clean expired sessions if at capacity
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_expired()
            
            self.sessions[session_id] = {
                "narrative": narrative,
                "prediction": prediction,
                "certainty": certainty,
                "evidence": evidence,
                "modifiers": modifiers,
                "competing": competing,
                "timestamp": datetime.now(),
                "turn_count": 0,
                "messages": [] # Conversation history
            }
    
    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["messages"].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> list:
        """Get the message history for a session."""
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id].get("messages", [])
            return []
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session data if it exists and hasn't expired."""
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check if expired
            if datetime.now() - session["timestamp"] > timedelta(minutes=self.ttl_minutes):
                del self.sessions[session_id]
                return None
            
            return session
    
    def increment_turn(self, session_id: str) -> int:
        """Increment turn count and return new count."""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["turn_count"] += 1
                return self.sessions[session_id]["turn_count"]
            return 0
    
    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = datetime.now()
        expired = [
            sid for sid, data in self.sessions.items()
            if now - data["timestamp"] > timedelta(minutes=self.ttl_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]
    
    def clear_session(self, session_id: str) -> None:
        """Manually clear a session."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

# Global session store instance
session_store = SessionStore()
