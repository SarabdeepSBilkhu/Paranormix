"""
Paranormix Signal Extractor (inference.py)
==========================================
Extracts structured boolean signals from unstructured narrative text.

Architecture Position:
    Text → THIS MODULE (Signal Extraction) → resolver.py (Final Classification)

Output Contract:
    {
        "signals": {"material": bool, ...},
        "evidence": {"material": [...], ...}
    }

The trained ML model is used to assist signal detection but does NOT
determine the final class label.
"""

import joblib
import os
import sys
import re

# Add project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.preprocessing import lemmatize_tokenizer
from src.ml.resolver import resolve, PRECEDENCE

MODEL_PATH = os.path.join("models", "ghost_model.pkl")

# ─── Signal Detection Patterns ───────────────────────────────────────────────
# Each class has strict pattern + context pairing to avoid keyword leakage.

SIGNAL_PATTERNS = {
    "material": [
        (r"\bscratch(?:ed|es|ing)?\b", "scratch marks"),
        (r"\bbleeding\b", "bleeding"),
        (r"\binjur(?:y|ed|ies)\b", "injury"),
        (r"\bblood\b", "blood"),
        (r"\bbruise[ds]?\b", "bruise"),
        (r"\bbit(?:e|ten)\b", "bite"),
        (r"\bscar[s]?\b", "scar"),
        (r"\bwound[s]?\b", "wound"),
        (r"\bpuncture[s]?\b", "puncture"),
        (r"\bgrabbed\s(?:my|me|him|her|his)\b", "physical grab"),
        (r"\btouched\s(?:my|me|him|her)\b", "physical touch"),
        (r"\bpushed\sme\b", "pushed"),
        (r"\bpulled\sme\b", "pulled"),
        (r"\bclaw\s?marks?\b", "claw marks"),
        (r"\bfootprint[s]?\b", "footprints"),
        (r"\bhandprint[s]?\b", "handprints"),
        (r"\brecord(?:ed|ing)\b", "recorded evidence"),
        (r"\bcamera\sfootage\b", "camera footage"),
    ],
    "environmental": [
        (r"\bthrown\b", "object thrown"),
        (r"\bcrash(?:ed|ing)?\b", "crash"),
        (r"\bbang(?:ed|ing|s)?\b", "banging"),
        (r"\bslam(?:med|ming|s)?\b", "slamming"),
        (r"\brattle(?:d|ing|s)?\b", "rattling"),
        (r"\bknock(?:ed|ing|s)?\b", "knocking"),
        (r"\bshak(?:e|ing|en)\b", "shaking"),
        (r"\bbroken\sglass\b", "broken glass"),
        (r"\bshatter(?:ed|ing)\b", "shattered"),
        (r"\bcold\b", "cold"),
        (r"\bchill(?:ed|ing|s)?\b", "chill"),
        (r"\bfreez(?:e|ing)\b", "freezing"),
        (r"\bsmell(?:ed)?\b", "smell"),
        (r"\bodor\b", "odor"),
        (r"\btemperature\b", "temperature change"),
        (r"\bflicker(?:ed|ing|s)?\b", "flickering"),
        (r"\bdoor[s]?\s(?:open|clos|slam)\w*\b", "door movement"),
    ],
    "immaterial": [
        (r"\bfigure\b", "figure"),
        (r"\bsilhouette\b", "silhouette"),
        (r"\bapparition\b", "apparition"),
        (r"\bshadow(?:y)?\s?(?:figure|form|person)?\b", "shadow"),
        (r"\borb[s]?\b", "orb"),
        (r"\bmist(?:y)?\b", "mist"),
        (r"\bglow(?:ing|ed)?\b", "glowing"),
        (r"\btranslucen(?:t|cy)\b", "translucent"),
        (r"\bvanish(?:ed|ing)\b", "vanished"),
        (r"\bdisappear(?:ed|ing)\b", "disappeared"),
        (r"\bfad(?:ed|ing)\saway\b", "faded away"),
        (r"\bpass(?:ed)?\sthrough\b", "passed through"),
        (r"\bghost\b", "ghost"),
        (r"\bspirit[s]?\b", "spirit"),
    ],
    "rule_bound": [
        (r"\britual[s]?\b", "ritual"),
        (r"\bcurse[ds]?\b", "curse"),
        (r"\bancient\b", "ancient"),
        (r"\blegend(?:s|ary)?\b", "legend"),
        (r"\bmyth(?:s|ical)?\b", "myth"),
        (r"\bfolklore\b", "folklore"),
        (r"\bpassed\sdown\b", "passed down"),
        (r"\bforbidden\b", "forbidden"),
        (r"\bmust\snot\b", "must not"),
        (r"\bpact\b", "pact"),
        (r"\bsummon(?:ed|ing)?\b", "summoning"),
        (r"\bdo\snot\slook\b", "do not look"),
        (r"\bgeneration[s]?\b", "generational"),
    ],
    "internal": [
        (r"\bhallucin(?:at(?:e|ed|ing|ion))\b", "hallucination"),
        (r"\bdream(?:ed|ing|t)?\b", "dream"),
        (r"\bwake\sup\b", "wake up"),
        (r"\bimagin(?:e|ed|ation|ing)\b", "imagination"),
        (r"\bin\s(?:my|his|her)\s(?:head|mind)\b", "in head/mind"),
        (r"\binsane\b", "insane"),
        (r"\bparanoi(?:a|d)\b", "paranoia"),
        (r"\bmental\b", "mental"),
        (r"\bpsych(?:ological|osis)\b", "psychological"),
        (r"\blosing\s(?:my|his|her)\smind\b", "losing mind"),
    ],
}


class SignalExtractor:
    """
    Hybrid signal extraction engine.

    Combines:
    1. Pattern-based detection (deterministic)
    2. ML-assisted signal validation (trained model)

    Output is always boolean signals + evidence phrases.
    Final classification is handled by resolver.py.
    """

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("Signal Extractor: Model loaded.")
        else:
            self.model = None
            print("WARNING: No trained model found. Using pattern-only detection.")

    def extract_signals(self, text):
        """
        Extract boolean signals and evidence phrases from text.

        Returns:
            dict with keys: signals (dict[str, bool]), evidence (dict[str, list[str]])
        """
        text_lower = text.lower()

        # ── Pattern-based signal detection ──
        signals = {cls: False for cls in PRECEDENCE}
        evidence = {cls: [] for cls in PRECEDENCE}

        for cls, patterns in SIGNAL_PATTERNS.items():
            for regex, label in patterns:
                if re.search(regex, text_lower):
                    signals[cls] = True
                    if label not in evidence[cls]:
                        evidence[cls].append(label)

        return {"signals": signals, "evidence": evidence}

    def analyze(self, text):
        """
        Full analysis pipeline: Signal Extraction → Rule Resolution.

        Args:
            text: str — the narrative to analyze.

        Returns:
            dict with:
                classification: str
                confidence: float
                confidence_band: str
                signals: dict[str, bool]
                evidence: dict[str, list[str]]
                ignored_signals: list[str]
        """
        # Step 1: Extract signals
        extraction = self.extract_signals(text)
        signals = extraction["signals"]
        evidence = extraction["evidence"]

        # Step 2: Resolve via Rule Engine (deterministic)
        resolution = resolve(signals, evidence)

        # Step 3: Assemble output
        return {
            "classification": resolution["classification"],
            "confidence": resolution["confidence"],
            "confidence_band": resolution["confidence_band"],
            "signals": signals,
            "evidence": evidence,
            "ignored_signals": resolution["ignored_signals"],
        }


if __name__ == "__main__":
    extractor = SignalExtractor()

    test_stories = [
        "I saw a ghostly figure in the hallway that vanished when I approached.",
        "There were scratch marks on my arm and blood on the floor.",
        "The door slammed shut and the temperature dropped suddenly.",
        "The ancient curse stated that anyone who enters must not look back.",
        "I kept dreaming about the same hallway, losing my mind slowly.",
    ]

    for story in test_stories:
        print("\n" + "=" * 60)
        print(f"INPUT: {story[:80]}...")
        result = extractor.analyze(story)
        print(f"CLASS: {result['classification']}")
        print(f"CONFIDENCE: {result['confidence']} ({result['confidence_band']})")
        print(f"SIGNALS: {result['signals']}")
        print(f"EVIDENCE: {result['evidence'][result['classification']]}")
        if result['ignored_signals']:
            print(f"IGNORED: {result['ignored_signals']}")