"""
Paranormix Rule Resolver
========================
Deterministic classification engine using Absolute Precedence Override.

This module is ISOLATED from:
  - ML models
  - Probability values
  - Any statistical inference

It reads ONLY boolean signals and outputs a single deterministic label.

Architecture Position:
    Text → ML Signal Extractor → [signals] → THIS MODULE → Final Label
"""


# ─── Precedence Order (Highest to Lowest) ─────────────────────────────────────
PRECEDENCE = ["material", "environmental", "immaterial", "rule_bound", "internal"]

# ─── Class Tier Weights (for confidence calculation) ──────────────────────────
TIER_WEIGHT = {
    "material": 0.9,
    "environmental": 0.8,
    "immaterial": 0.65,
    "rule_bound": 0.55,
    "internal": 0.4,
}

# ─── Evidence Thresholds (minimum evidence count for full confidence) ─────────
EVIDENCE_THRESHOLD = {
    "material": 3,
    "environmental": 3,
    "immaterial": 2,
    "rule_bound": 2,
    "internal": 2,
}


def resolve_label(signals):
    """
    Absolute Precedence Override.

    The highest-tier detected signal ALWAYS becomes the final class.
    No voting, no tie resolution, no dominance scoring.

    Args:
        signals: dict of {class_name: bool}

    Returns:
        str: The resolved class label.
    """
    if signals.get("material"):
        return "material"
    if signals.get("environmental"):
        return "environmental"
    if signals.get("immaterial"):
        return "immaterial"
    if signals.get("rule_bound"):
        return "rule_bound"
    return "internal"


def calculate_confidence(final_class, evidence):
    """
    Confidence = min(0.9, tier_weight[class] * evidence_strength)

    Where evidence_strength = min(1.0, len(unique_evidence) / threshold[class])

    This ensures:
      - Repeated identical signals do not inflate confidence.
      - Maximum confidence is capped at 0.9 (no 100% certainty).

    Args:
        final_class: str — the resolved label
        evidence: dict of {class_name: list_of_str}

    Returns:
        float: confidence score between 0.0 and 0.9
    """
    tier = TIER_WEIGHT.get(final_class, 0.4)
    threshold = EVIDENCE_THRESHOLD.get(final_class, 2)
    
    # Require unique signal types, not just raw count
    evidence_list = evidence.get(final_class, [])
    unique_evidence = len(set(evidence_list))
    
    strength = min(1.0, unique_evidence / threshold)
    
    # Cap at 0.9
    confidence = min(0.9, tier * strength)

    return round(confidence, 3)


def get_confidence_band(confidence):
    """
    Map confidence score to a human-readable band.

    Args:
        confidence: float between 0.0 and 1.0

    Returns:
        str: "High", "Moderate", or "Low"
    """
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Moderate"
    return "Low"


def get_ignored_signals(signals, final_class):
    """
    Identify all detected signals that were ignored for final classification
    due to precedence override.

    Args:
        signals: dict of {class_name: bool}
        final_class: str — the resolved label

    Returns:
        list of str: class names that were detected but overridden
    """
    return [
        k for k, v in signals.items()
        if v and k != final_class
    ]


def resolve(signals, evidence):
    """
    Full resolution pipeline.

    Args:
        signals: dict of {class_name: bool}
        evidence: dict of {class_name: list_of_str}

    Returns:
        dict with keys:
            classification: str
            confidence: float
            confidence_band: str
            signals: dict
            ignored_signals: list
    """
    final_class = resolve_label(signals)
    confidence = calculate_confidence(final_class, evidence)
    band = get_confidence_band(confidence)
    ignored = get_ignored_signals(signals, final_class)

    return {
        "classification": final_class,
        "confidence": confidence,
        "confidence_band": band,
        "signals": signals,
        "ignored_signals": ignored,
    }
