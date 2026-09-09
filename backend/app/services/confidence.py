"""Standardized AI confidence model.

Confidence levels are coarse and honest: HIGH / MEDIUM / LOW / UNKNOWN.
Raw model probabilities are never presented as factual certainty.
"""

from typing import Optional

CONFIDENCE_LEVELS = ("HIGH", "MEDIUM", "LOW", "UNKNOWN")


def level_from_score(score: Optional[float]) -> str:
    """Map a 0..1 score to a confidence level (UNKNOWN when unavailable)."""
    if score is None:
        return "UNKNOWN"
    if score >= 0.8:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    if score >= 0.3:
        return "LOW"
    return "UNKNOWN"


def combine(scores: list[float]) -> str:
    """Combine multiple evidence scores into one confidence level."""
    if not scores:
        return "UNKNOWN"
    return level_from_score(sum(scores) / len(scores))


def level_rank(level: str) -> int:
    return {"HIGH": 4, "MEDIUM": 3, "LOW": 2, "UNKNOWN": 1}.get(level, 1)


def confidence_dict(level: str, reasons: Optional[list[str]] = None, score: Optional[float] = None) -> dict:
    """Build a standard confidence envelope.

    Args:
        level: HIGH/MEDIUM/LOW/UNKNOWN
        reasons: human-readable factors
        score: optional 0..1 heuristic (never presented as certainty)
    """
    return {
        "level": level if level in CONFIDENCE_LEVELS else "UNKNOWN",
        "score": round(score, 3) if score is not None else None,
        "reasons": reasons or [],
    }