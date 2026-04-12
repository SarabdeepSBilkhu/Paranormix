"""
Paranormix Rule Resolver (resolver.py)
======================================
Step 3: Decision Logic.

Improved version:
- Soft precedence (not absolute)
- Uses evidence strength to compete across classes
- Prevents suppression of valid lower-tier signals
"""

# ─── Precedence Order (Highest to Lowest) ─────────────────────────────────────
PRECEDENCE = ["material", "environmental", "immaterial", "rule_bound", "internal"]

# ─── Class Tier Weights ───────────────────────────────────────────────────────
TIER_WEIGHT = {
    "material": 0.9,
    "environmental": 0.8,
    "immaterial": 0.65,
    "rule_bound": 0.55,
    "internal": 0.5,   # increased slightly (was 0.4)
}

# ─── Evidence Thresholds ──────────────────────────────────────────────────────
EVIDENCE_THRESHOLD = {
    "material": 3,
    "environmental": 3,
    "immaterial": 2,
    "rule_bound": 2,
    "internal": 2,
}


def compute_score(cls, signals, evidence):
    """
    Compute a score for each class using:
    score = tier_weight * evidence_strength

    Evidence strength is normalized.
    """
    if not signals.get(cls):
        return 0.0

    evidence_list = evidence.get(cls, [])
    unique_evidence = len(set(evidence_list))

    threshold = EVIDENCE_THRESHOLD.get(cls, 2)
    strength = min(1.0, unique_evidence / threshold)

    tier = TIER_WEIGHT.get(cls, 0.5)

    return tier * strength


def resolve_label(signals, evidence):
    """
    Soft precedence resolution:
    - Compute score for all detected classes
    - Select highest score
    - If tie → fallback to precedence order
    """
    scores = {cls: compute_score(cls, signals, evidence) for cls in PRECEDENCE}

    # Filter only active signals
    active_scores = {k: v for k, v in scores.items() if v > 0}

    if not active_scores:
        return "internal", scores

    # Find best score
    max_score = max(active_scores.values())

    # All classes with max score
    candidates = [cls for cls, val in active_scores.items() if val == max_score]

    # Tie-break using precedence
    for cls in PRECEDENCE:
        if cls in candidates:
            return cls, scores

    return "internal", scores


def calculate_confidence(final_class, scores):
    """
    Confidence derived directly from score.

    Capped at 0.9
    """
    confidence = min(0.9, scores.get(final_class, 0.0))
    return round(confidence, 3)


def get_confidence_band(confidence):
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Moderate"
    return "Low"


def get_ignored_signals(signals, final_class):
    return [
        k for k, v in signals.items()
        if v and k != final_class
    ]


def resolve(signals, evidence):
    """
    Full resolution pipeline.
    """

    final_class, scores = resolve_label(signals, evidence)

    confidence = calculate_confidence(final_class, scores)
    band = get_confidence_band(confidence)
    ignored = get_ignored_signals(signals, final_class)

    return {
        "classification": final_class,
        "confidence": confidence,
        "confidence_band": band,
        "signals": signals,
        "ignored_signals": ignored,
        "scores": scores,  # added for transparency
    }