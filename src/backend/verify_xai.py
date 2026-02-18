import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.inference import ParanormalInvestigator

def verify_xai_schema():
    investigator = ParanormalInvestigator()
    
    test_cases = [
        "I saw a floating white lady in the hallway and heard a crash.", # Multi-axial
        "I just feel like someone is watching me but nothing happens.", # Low certainty
        "The legend says the ancient curse was real and I saw a figure." # Folklore + Visual
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\n--- VERIFYING TEST CASE {i+1} ---")
        result = investigator.analyze(text)
        
        # Check Core Fields
        print(f"Prediction: {result.get('prediction')}")
        print(f"Band: {result.get('confidence_band')}")
        print(f"Stability: {result.get('stability_status')}")
        
        # Check Signals
        observed = result.get('observed_signals', [])
        print(f"Observed ({len(observed)}): {observed}")
        
        # Check Chart Data
        sorted_dist = result.get('chart_data', {}).get('sorted_distribution', [])
        print(f"Distribution Items: {len(sorted_dist)}")
        if sorted_dist:
            print(f"Top Prob: {sorted_dist[0]['p']:.2%}")
        
        # Validation
        required = ['prediction', 'certainty', 'confidence_band', 'stability_status', 'observed_signals', 'absent_signals']
        missing = [f for f in required if f not in result]
        if missing:
            print(f"FAILED: Missing fields {missing}")
        else:
            print("PASSED: Schema integrity verified.")

if __name__ == "__main__":
    verify_xai_schema()
