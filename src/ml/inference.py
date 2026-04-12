"""
Paranormix Signal Extractor (inference.py)
==========================================

Architecture:
1. Signal Extraction (Regex)
2. ML-assisted validation (soft, not destructive)
3. Rule Resolver (resolver.py)
"""

import joblib
import os
import sys
import re

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.resolver import resolve, PRECEDENCE

def find_model_path():
    """Search for the classifier in multiple potential locations."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(base_dir, "models", "classifier.pkl"),
        os.path.join(os.getcwd(), "models", "classifier.pkl"),
        "models/classifier.pkl",
        "/app/models/classifier.pkl", # Common container path
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

MODEL_PATH = find_model_path()

# ─── Normalized thresholds (no class bias) ───────────────────────────────────
VALIDATION_THRESHOLDS = {
    "material": 0.08,
    "environmental": 0.08,
    "immaterial": 0.08,
    "rule_bound": 0.08,
    "internal": 0.08,
}

# ─── Exclusion patterns ──────────────────────────────────────────────────────
EXCLUSION_PATTERNS = {
    "material": [
        r"\bblood\s(?:relation|pressure|line|stream|cell|thirsty)\b",
        r"\bscar\stissue\b",
    ],
    "environmental": [
        r"\bcold\s(?:war|shoulder|feet|beer|coffee|case|front)\b",
        r"\bshaking\shands\b",
        r"\bsmell\sa\srat\b",
    ],
}

# ─── Signal patterns ─────────────────────────────────────────────────────────
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
        (r"\bfelt\s(?:a|someone|something)\s(?:watching|presence)\b", "felt presence"),
        (r"\bcouldn't\smove\b", "sleep paralysis"),
        (r"\bparalyzed\b", "paralysis"),
        (r"\bi\s(?:thought|felt)\s(?:it\swas|someone\swas)\b", "uncertain perception"),
        (r"\bwas\sit\sreal\b", "reality doubt"),
    ],
}


class SignalExtractor:
    def __init__(self):
        print(f"Signal Extractor: Initializing. Search result: {MODEL_PATH}")
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print("Signal Extractor: Model loaded successfully.")
            except Exception as e:
                print(f"Signal Extractor: ERROR loading model: {e}")
                self.model = None
        else:
            self.model = None
            print(f"WARNING: No valid model file found in search paths.")

    def extract_signals(self, text):
        text_lower = text.lower()

        candidate_signals = {cls: False for cls in PRECEDENCE}
        evidence = {cls: [] for cls in PRECEDENCE}

        # ── Pattern detection ──
        for cls, patterns in SIGNAL_PATTERNS.items():
            if cls in EXCLUSION_PATTERNS:
                if any(re.search(p, text_lower) for p in EXCLUSION_PATTERNS[cls]):
                    continue

            for regex, label in patterns:
                if re.search(regex, text_lower):
                    candidate_signals[cls] = True
                    if label not in evidence[cls]:
                        evidence[cls].append(label)

        # ── ML validation (soft, non-destructive) ──
        validated_signals = candidate_signals.copy()
        ml_probs = {}

        if self.model:
            try:
                probs = self.model.predict_proba([text])[0]
                ml_probs = {str(k): float(v) for k, v in zip(self.model.classes_, probs)}

                for cls in PRECEDENCE:
                    if candidate_signals[cls]:
                        threshold = VALIDATION_THRESHOLDS.get(cls, 0.08)
                        if ml_probs.get(cls, 0) < threshold:
                            # do NOT remove signal, just keep it weaker
                            validated_signals[cls] = True
            except Exception as e:
                print(f"Validation error: {e}")

        return {
            "signals": validated_signals,
            "evidence": evidence,
            "ml_probs": ml_probs
        }

    def analyze(self, text):
        extraction = self.extract_signals(text)

        signals = extraction["signals"]
        evidence = extraction["evidence"]
        ml_probs = extraction["ml_probs"]

        resolution = resolve(signals, evidence)

        return {
            "classification": resolution["classification"],
            "confidence": resolution["confidence"],
            "confidence_band": resolution["confidence_band"],
            "signals": signals,
            "evidence": evidence,
            "ignored_signals": resolution["ignored_signals"],
            "ml_probs": ml_probs,
        }


if __name__ == "__main__":
    extractor = SignalExtractor()

    samples = [
        "I saw a shadow figure watching me but I couldn't move.",
        "The door slammed and the temperature dropped suddenly.",
        "There were scratches and blood on my arm.",
        "The ritual said you must not look back.",
        "I thought I was imagining things and losing my mind."
    ]

    for s in samples:
        print("\n" + "=" * 60)
        print("INPUT:", s)
        r = extractor.analyze(s)
        print("CLASS:", r["classification"])
        print("CONF:", r["confidence"], r["confidence_band"])
        print("SIGNALS:", r["signals"])
        print("EVIDENCE:", r["evidence"][r["classification"]])