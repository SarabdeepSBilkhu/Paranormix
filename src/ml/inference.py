import joblib
import os
import sys
import numpy as np

# Add project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.preprocessing import lemmatize_tokenizer

MODEL_PATH = os.path.join("models", "ghost_model.pkl")

class ParanormalInvestigator:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("Investigator ready.")
        else:
            self.model = None
            print("WARNING: No trained model found.")

    def analyze(self, text):
        if not self.model:
            return {
                "prediction": "Unknown (Investigator Off-Duty)",
                "certainty": "Low",
                "evidence_signals": [],
                "interpretive_modifiers": ["Model missing from deployment"],
                "competing_hypotheses": []
            }
        
        # 1. Extract Signals (Evidence vs Modifiers)
        signals = self._extract_diagnostic_signals(text)
        
        # 2. Get Model Probabilities for Hypothesis Ranking
        try:
            probabilities = self.model.predict_proba([text])[0]
            class_names = self.model.classes_
            prob_dict = {name: float(prob) for name, prob in zip(class_names, probabilities)}
            
            # Rank all hypotheses by probability
            ranked_hypotheses = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Derive Categorical Certainty from dominance gap
            primary_prob = ranked_hypotheses[0][1]
            secondary_prob = ranked_hypotheses[1][1] if len(ranked_hypotheses) > 1 else 0
            gap = primary_prob - secondary_prob
            
            if gap > 0.4:
                certainty = "High"
            elif gap > 0.15:
                certainty = "Medium"
            else:
                certainty = "Low"

            # 3. Apply Decision Hierarchy (Override model if hierarchy signal is present)
            # Hierarchy: Psychological > Physical (Polter) > Visual (Appar) > Cultural (Folk)
            prediction = ranked_hypotheses[0][0]
            
            if "psychological indicators" in signals["evidence"]:
                prediction = "psychological"
                certainty = "High" if len(signals["evidence"]) == 1 else "Medium"
            elif "physical disturbance" in signals["evidence"] and prediction not in ["psychological", "poltergeist"]:
                prediction = "poltergeist"
            elif "visual apparition" in signals["evidence"] and prediction not in ["psychological", "poltergeist", "apparition"]:
                prediction = "apparition"

        except Exception as e:
            print(f"ERROR in diagnostic analysis: {e}")
            prediction = "unknown"
            certainty = "Low"
            ranked_hypotheses = []

        return {
            "prediction": prediction,
            "certainty": certainty,
            "evidence_signals": signals["evidence"],
            "interpretive_modifiers": signals["modifiers"],
            "competing_hypotheses": [h[0] for h in ranked_hypotheses if h[0] != prediction][:2]
        }
    
    def _extract_diagnostic_signals(self, text):
        """Rigidly separate hard evidence from interpretive bias"""
        text_lower = text.lower()
        evidence = []
        modifiers = []
        
        # Evidence (Direct textual cues)
        patterns_evidence = {
            "physical disturbance": ["thrown", "moved", "crash", "bang", "slam", "rattle"],
            "visual apparition": ["saw", "figure", "silhouette", "white lady", "ghost", "apparition"],
            "psychological indicators": ["voice in head", "insane", "hallucination", "dream", "wake up", "remembering"],
            "sensory anomaly": ["cold", "smell", "chill", "touch"]
        }
        
        # Modifiers (Contextual bias)
        patterns_modifiers = {
            "folklore context": ["legend", "myth", "curse", "ancient", "ritual"],
            "belief/expectation": ["i think", "i believe", "i know it was", "spirits"]
        }
        
        for name, keywords in patterns_evidence.items():
            if any(k in text_lower for k in keywords):
                evidence.append(name)
        
        for name, keywords in patterns_modifiers.items():
            if any(k in text_lower for k in keywords):
                modifiers.append(name)
                
        return {"evidence": evidence, "modifiers": modifiers}

if __name__ == "__main__":
    bot = ParanormalInvestigator()
    print(bot.analyze("I saw a floating orb in the kitchen."))
