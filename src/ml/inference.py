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
                "prediction": "Unknown (Terminal_Inert)",
                "certainty": "Low",
                "evidence_signals": [],
                "interpretive_modifiers": ["Hardware_Missing"],
                "competing_hypotheses": [],
                "chart_data": {}
            }
        
        # --- LAYER 1: RAW SIGNAL DETECTION ---
        signals = self._extract_diagnostic_signals(text)
        category_weights = self._calculate_category_weights(text)
        
        # --- LAYER 2: CLASS SELECTION (MEASUREMENT RANKING) ---
        try:
            probabilities = self.model.predict_proba([text])[0]
            class_names = self.model.classes_
            prob_dict = {name: float(prob) for name, prob in zip(class_names, probabilities)}
            
            # Rank Classes with Dominance Labels
            raw_ranked = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Labeling by Relative Pull
            ranked_with_labels = []
            for i, (name, prob) in enumerate(raw_ranked):
                if i == 0: label = "DOMINANT"
                elif i == 1 and prob > 0.15: label = "CONTENDER"
                elif prob > 0.05: label = "TRACE"
                else: label = "NOISE"
                ranked_with_labels.append({"class": name, "label": label, "p": prob})

            winner = raw_ranked[0][0]
            primary_prob = raw_ranked[0][1]
            secondary_prob = raw_ranked[1][1] if len(raw_ranked) > 1 else 0
            gap = primary_prob - secondary_prob

            # Mandatory Hierarchy Constraints
            prediction = winner
            if "Pattern_C" in signals["evidence"]: # Previously Psychological
                prediction = "psychological"
            elif "Pattern_A" in signals["evidence"] and prediction not in ["psychological", "poltergeist"]: # Previously Physical
                prediction = "poltergeist"

            # --- LAYER 3: EMPIRICAL CALIBRATION (STABILITY CAP) ---
            STABILITY_INDEX = {
                "apparition": 0.19,    # Historically High Overlap
                "folklore": 0.04,      # Historically Indistinguishable
                "poltergeist": 0.25,   # Moderate Instability
                "creature": 0.73,      # Stable Profile
                "psychological": 0.73  # Stable Profile
            }
            
            stability = STABILITY_INDEX.get(prediction.lower(), 0.5)
            
            # Narrative Purity (Intrinsic Measurement)
            purity = "High" if gap > 0.4 else "Medium" if gap > 0.15 else "Low"
            
            # Final Categorical Certainty (Capped by Stability)
            certainty = purity
            resolution_limit = None

            if stability < 0.3:
                resolution_limit = "CLASS_OVERLAP_BOUNDARY (High Historical Confusion)"
                if certainty == "High": certainty = "Medium"
                elif certainty == "Medium": certainty = "Low"
            elif stability < 0.6:
                resolution_limit = "MODERATE_RESOLUTION_LIMIT (Data Constraint)"
                if certainty == "High": certainty = "Medium"

            # Prepare Data Artifacts
            max_val = max(prob_dict.values()) if prob_dict else 1
            normalized_scores = {k: v/max_val for k, v in prob_dict.items()}
            margins = {k: primary_prob - v for k, v in prob_dict.items() if k != winner}

            drivers = {
                "multi_class_overlap": gap < 0.2,
                "resolution_boundary": resolution_limit is not None,
                "signal_conflict": len(signals["evidence"]) > 2 and gap < 0.25
            }

        except Exception as e:
            print(f"ERROR: Measurement Failure: {e}")
            prediction = "unknown"
            certainty = "Low"
            resolution_limit = "Computation_Error"
            normalized_scores = {}
            margins = {}
            drivers = {}
            ranked_with_labels = []

        return {
            "prediction": prediction,
            "certainty": certainty,
            "resolution_limit": resolution_limit,
            "detected_patterns": signals["evidence"],
            "modifiers": signals["modifiers"],
            "constraints": signals["absent"],
            "ranked_matches": ranked_with_labels[:3],
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
        """Quantify raw cluster density"""
        text_lower = text.lower()
        clusters = {
            "Pattern_C": ["voice", "insane", "hallucination", "mind", "remember", "dream"],
            "Pattern_D": ["cold", "smell", "touch", "chill", "freeze"],
            "Pattern_A": ["thrown", "crash", "bang", "slam", "moved", "rattle"]
        }
        
        weights = {}
        for cluster, keywords in clusters.items():
            weights[cluster] = sum(1 for k in keywords if k in text_lower)
            
        return weights

    def _extract_diagnostic_signals(self, text):
        """Raw Detection Layer (Pattern Matching)"""
        text_lower = text.lower()
        evidence = []
        modifiers = []
        absent = []
        
        # Raw Detection Patterns (No Class Labeling)
        patterns_detection = {
            "Pattern_A": ["thrown", "moved", "crash", "bang", "slam", "rattle"], # Impact
            "Pattern_B": ["saw", "figure", "silhouette", "white lady", "ghost", "apparition"], # Visual
            "Pattern_C": ["voice in head", "insane", "hallucination", "dream", "wake up", "remembering"], # Cognitive
            "Pattern_D": ["cold", "smell", "chill", "touch"] # Sensory
        }
        
        patterns_context = {
            "Context_Alpha": ["legend", "myth", "curse", "ancient", "ritual"],
            "Context_Beta": ["i think", "i believe", "i know it was", "spirits"]
        }
        
        for name, keywords in patterns_detection.items():
            if any(k in text_lower for k in keywords):
                evidence.append(name)
            else:
                absent.append(f"ABSENT_{name}")
        
        for name, keywords in patterns_context.items():
            if any(k in text_lower for k in keywords):
                modifiers.append(name)
            else:
                absent.append(f"ABSENT_{name}")
                
        return {"evidence": evidence, "modifiers": modifiers, "absent": absent}

if __name__ == "__main__":
    bot = ParanormalInvestigator()
    print(bot.analyze("I saw a floating orb in the kitchen."))
