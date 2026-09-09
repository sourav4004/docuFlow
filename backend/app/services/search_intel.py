"""Phase 20 — Search intelligence.

Search quality scoring (relevance, zero results, reformulation, clicks,
feedback), controlled ranking experiments (reusing the experiment platform),
safe search explanations (matched metadata, semantic similarity, freshness,
source authority — never hidden model reasoning), and personalization safety
that can never cross tenant/workspace boundaries.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import SearchQualityEvent


def _hash(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:16]


def quality_score(db: Session, *, workspace_id: int,
                  limit: int = 500) -> dict:
    """Deterministic search quality score from recorded events."""
    rows = db.query(SearchQualityEvent).filter_by(
        workspace_id=workspace_id)\
        .order_by(SearchQualityEvent.created_at.desc()).limit(limit).all()
    total = len(rows)
    if not total:
        return {"score": None, "total_events": 0}
    zero = sum(1 for r in rows if r.event_type == "zero_result")
    reformulated = sum(1 for r in rows if r.event_type == "reformulated")
    clicked = sum(1 for r in rows if r.event_type == "clicked")
    abandoned = sum(1 for r in rows if r.event_type == "abandoned")
    zero_rate = zero / total
    click_rate = clicked / total if total else 0.0
    reform_rate = reformulated / total if total else 0.0
    abandon_rate = abandoned / total if total else 0.0
    score = max(0.0, 1.0 - (zero_rate * 0.5 + abandon_rate * 0.3) +
                click_rate * 0.1)
    return {
        "score": round(min(score, 1.0), 4),
        "total_events": total,
        "zero_result_rate": round(zero_rate, 4),
        "click_rate": round(click_rate, 4),
        "reformulation_rate": round(reform_rate, 4),
        "abandoned_rate": round(abandon_rate, 4),
    }


def ranking_experiment(db: Session, *, name: str, config: dict,
                       dataset_id: Optional[int] = None,
                       created_by: Optional[int] = None) -> dict:
    """Create a controlled ranking experiment via the experiment platform.
    The configuration is immutable; evaluation must happen before any
    promotion."""
    from ..services.improvement_platform import create_experiment
    exp = create_experiment(
        db, name=name, domain="search_ranking", config=config,
        dataset_id=dataset_id, created_by=created_by)
    return {"experiment_id": exp.id, "status": exp.status,
            "config_fingerprint": exp.config_fingerprint}


def explain(result: dict) -> dict:
    """Safe, non-chain-of-thought explanation of a search result."""
    explanations = []
    if result.get("matched_metadata"):
        explanations.append({"factor": "matched_metadata",
                             "value": result["matched_metadata"]})
    if result.get("semantic_similarity") is not None:
        explanations.append({
            "factor": "semantic_similarity",
            "value": round(float(result["semantic_similarity"]), 4)})
    if result.get("freshness_days") is not None:
        explanations.append({
            "factor": "freshness",
            "value": f"{int(result['freshness_days'])} days old"})
    if result.get("source_authority"):
        explanations.append({"factor": "source_authority",
                             "value": result["source_authority"]})
    if result.get("scope"):
        explanations.append({"factor": "scope", "value": result["scope"]})
    return {"explanations": explanations, "rank": result.get("rank")}


def personalization_safe(user_scope: dict, workspace_id: int,
                         organization_id: Optional[int]) -> bool:
    """Personalization may only apply within the user's own scope: the
    workspace/org must match, and at least one of them must be present."""
    if user_scope.get("workspace_id") != workspace_id:
        return False
    if user_scope.get("organization_id") is not None:
        if organization_id is not None and \
                user_scope["organization_id"] != organization_id:
            return False
    return True