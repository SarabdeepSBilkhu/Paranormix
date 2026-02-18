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
                "competing_hypotheses": [],
                "chart_data": {}
            }
        
        # 1. Extract Signals & Categorized Weights
        signals = self._extract_diagnostic_signals(text)
        category_weights = self._calculate_category_weights(text)
        
        # 2. Get Model Probabilities for Hypothesis Ranking
        try:
            probabilities = self.model.predict_proba([text])[0]
            class_names = self.model.classes_
            prob_dict = {name: float(prob) for name, prob in zip(class_names, probabilities)}
            
            # Rank all hypotheses by probability
            ranked_hypotheses = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            winner = ranked_hypotheses[0][0]
            
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

            # 3. Apply Decision Hierarchy Override
            prediction = winner
            if "psychological indicators" in signals["evidence"]:
                prediction = "psychological"
                certainty = "High" if len(signals["evidence"]) == 1 else "Medium"
            elif "physical disturbance" in signals["evidence"] and prediction not in ["psychological", "poltergeist"]:
                prediction = "poltergeist"

            # 4. Prepare Chart Data
            # Normalizing prob_dict for Class Score Bar Chart
            max_val = max(prob_dict.values()) if prob_dict else 1
            normalized_scores = {k: v/max_val for k, v in prob_dict.items()}

            # Competing Margin
            margins = {k: primary_prob - v for k, v in prob_dict.items() if k != winner}

            # Certainty Drivers
            drivers = {
                "multi_class_overlap": gap < 0.2,
                "modifier_presence": len(signals["modifiers"]) > 0,
                "signal_contradiction": len(signals["evidence"]) > 2 and gap < 0.25
            }

        except Exception as e:
            print(f"ERROR in diagnostic analysis: {e}")
            prediction = "unknown"
            certainty = "Low"
            ranked_hypotheses = []
            normalized_scores = {}
            margins = {}
            drivers = {}

        return {
            "prediction": prediction,
            "certainty": certainty,
            "evidence_signals": signals["evidence"],
            "interpretive_modifiers": signals["modifiers"],
            "competing_hypotheses": [h[0] for h in ranked_hypotheses if h[0] != prediction][:3],
            "chart_data": {
                "class_scores": normalized_scores,
                "signal_contributions": category_weights,
                "margins": margins,
                "certainty_drivers": drivers,
                "global_cm": {
                    "labels": ["Apparition", "Creature", "Folklore", "Poltergeist", "Psychological"],
                    "matrix": [
                        [76, 163, 1, 12, 138],
                        [2, 266, 1, 8, 85],
                        [7, 99, 8, 3, 80],
                        [9, 175, 1, 104, 125],
                        [4, 88, 0, 9, 274]
                    ]
                }
            }
        }
    
    def _calculate_category_weights(self, text):
        """Calculate weight contributions per category"""
        text_lower = text.lower()
        cats = {
            "psychological": ["voice", "insane", "hallucination", "mind", "remember", "dream"],
            "sensory": ["cold", "smell", "touch", "chill", "freeze"],
            "physical": ["thrown", "crash", "bang", "slam", "moved", "rattle"]
        }
        
        weights = {}
        for cat, keywords in cats.items():
            weights[cat] = sum(1 for k in keywords if k in text_lower)
            
        return weights

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
