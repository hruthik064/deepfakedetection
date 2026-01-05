# ============================================================
# confidence_utils.py
# Confidence-Aware Decision Reporting (IEEE-Ready)
# ============================================================

"""
This module converts model probabilities into
human-interpretable forensic confidence and risk levels.

NO model retraining required.
NO dependency on model architecture.
"""

def confidence_to_risk(prob_fake: float) -> str:
    """
    Maps probability score to forensic risk level.
    """
    if prob_fake >= 0.85:
        return "HIGH RISK"
    elif prob_fake >= 0.65:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def format_confidence_output(prob_fake: float) -> dict:
    """
    Formats confidence output for UI and reports.

    Args:
        prob_fake (float): Probability of fake class (0–1)

    Returns:
        dict: Structured confidence output
    """
    risk = confidence_to_risk(prob_fake)

    return {
        "probability_fake": round(float(prob_fake), 4),
        "confidence_percent": round(float(prob_fake) * 100, 2),
        "risk_level": risk
    }
