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
                "confidence": 0.0,
                "probabilities": {"Unknown": 1.0},
                "key_signals": ["Model missing from deployment"],
                "likely_confusions": []
            }
        
        # Predict class
        prediction = self.model.predict([text])[0]
        
        # Get probabilities (now available with log_loss)
        try:
            probabilities = self.model.predict_proba([text])[0]
            class_names = self.model.classes_
            
            # Create probability dictionary
            prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
            
            # Confidence is the max probability
            confidence = float(max(probabilities))
            
            # Extract key signals (simple keyword detection)
            key_signals = self._extract_signals(text)
            
            # Identify likely confusions (classes with >15% probability)
            confusions = [cls for cls, prob in prob_dict.items() if prob > 0.15 and cls != prediction]
            
        except Exception as e:
            # Fallback if predict_proba fails
            print(f"WARNING: Error in predict_proba: {e}")
            import traceback
            traceback.print_exc()
            prob_dict = {}
            confidence = 0.0
            key_signals = []
            confusions = []
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "key_signals": key_signals,
            "likely_confusions": confusions
        }
    
    def _extract_signals(self, text):
        """Extract key narrative signals from text"""
        signals = []
        text_lower = text.lower()
        
        # Define signal patterns
        patterns = {
            "object movement": ["thrown", "moved", "flying", "levitate", "float"],
            "loud sounds": ["bang", "crash", "knock", "slam", "thud"],
            "visual apparition": ["saw", "figure", "shadow", "silhouette", "ghost"],
            "cold presence": ["cold", "chill", "freeze", "icy"],
            "first-person account": ["i saw", "i heard", "i felt", "we saw"],
            "folklore elements": ["legend", "myth", "ancient", "curse", "ritual"]
        }
        
        for signal_name, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                signals.append(signal_name)
        
        return signals[:5]  # Limit to top 5 signals

if __name__ == "__main__":
    bot = ParanormalInvestigator()
    print(bot.analyze("I saw a floating orb in the kitchen."))
