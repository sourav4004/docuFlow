"""Phase 21 — Personal AI control + artifact intelligence + search autopilot.

Per-user autonomy settings within policy bounds, user memory controls
(inspect/suppress/reset/delete where permitted), AI activity feed with
evidence/policy-only explanations, artifact quality scoring + version
comparison + regression detection + provenance, and search autopilot
(quality monitoring, drift, candidate ranking changes, evaluation-gated
promotion).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.phase20 import SearchQualityEvent
from ..models.phase21 import (
    AIActivityItem, ArtifactQualityScore, AdaptiveCandidate,
    PersonalAutonomySetting,
)
from . import autonomy

# Autonomy levels a user may self-select (never above workspace policy).
_USER_LEVELS = ("OBSERVE", "RECOMMEND", "AUTO_LOW_RISK")


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


# ---------------------------------------------------------------------------
# Personal AI control
# ---------------------------------------------------------------------------

def get_personal_settings(db: Session, workspace_id: int, user_id: int) \
        -> PersonalAutonomySetting:
    """Fetch-or-create personal autonomy settings (defaults are restrictive)."""
    row = (db.query(PersonalAutonomySetting)
           .filter_by(workspace_id=workspace_id, user_id=user_id).first())
    if row is None:
        row = PersonalAutonomySetting(workspace_id=workspace_id,
                                      user_id=user_id)
        db.add(row)
        db.commit()
    return row


def set_personal_autonomy(db: Session, workspace_id: int, user_id: int,
                          requested_level: str,
                          workspace_policy_level: str) -> dict:
    """Users may configure autonomy where policy allows — never above it.

    Returns the effective level; requests above the workspace policy are
    clamped to the policy (recorded, not silently granted).
    """
    if requested_level not in _USER_LEVELS:
        raise ValueError(f"invalid personal autonomy level: {requested_level}")
    order = ("OBSERVE", "RECOMMEND", "AUTO_LOW_RISK", "AUTO_APPROVAL",
             "MANUAL_ONLY")
    effective = requested_level
    clamped = False
    if workspace_policy_level in order and requested_level in order:
        if order.index(requested_level) > order.index(workspace_policy_level):
            effective = workspace_policy_level
            clamped = True
    row = get_personal_settings(db, workspace_id, user_id)
    row.autonomy_level = effective
    db.commit()
    return {"effective_level": effective, "clamped_to_policy": clamped,
            "requested_level": requested_level}


def memory_control(db: Session, workspace_id: int, user_id: int,
                   action: str, memory_id: Optional[int] = None) -> dict:
    """User memory controls: inspect / suppress / reset / delete.

    Each action is permission-gated by the user's personal settings, and every
    permitted action is audited through the memory autonomy trail.
    """
    from . import infra_heal
    settings = get_personal_settings(db, workspace_id, user_id)
    allowed = {
        "inspect": True,
        "suppress": settings.allow_memory_suppression,
        "reset": settings.allow_memory_reset,
        "delete": settings.allow_memory_delete,
    }
    if action not in allowed:
        raise ValueError(f"unknown memory action: {action}")
    if not allowed[action]:
        return {"action": action, "performed": False,
                "reason": "not permitted by personal settings"}
    if action != "inspect" and memory_id is not None:
        kind = {"suppress": "suppressed", "reset": "audit",
                "delete": "audit"}[action]
        from ..models.phase21 import MemoryAutonomyEvent
        db.add(MemoryAutonomyEvent(
            workspace_id=workspace_id, memory_id=memory_id,
            event_kind=kind, automatic=False,
            detail=f"user {user_id} {action}"))
        db.commit()
    return {"action": action, "performed": True}


def record_activity(db: Session, workspace_id: int, user_id: int,
                    item_kind: str, title: str,
                    explanation: Optional[str] = None,
                    ref_kind: Optional[str] = None,
                    ref_id: Optional[int] = None) -> AIActivityItem:
    """Record a personal AI activity feed item.

    Explanations must contain only evidence/citations/confidence/policy —
    hidden reasoning is never stored or shown.
    """
    if item_kind not in ("SUGGESTION", "ACTION", "DECISION", "APPROVAL",
                         "RECOVERY"):
        raise ValueError(f"invalid activity kind: {item_kind}")
    row = AIActivityItem(
        workspace_id=workspace_id, user_id=user_id, item_kind=item_kind,
        title=title[:200],
        explanation=(explanation or "")[:2000],
        ref_kind=ref_kind, ref_id=ref_id)
    db.add(row)
    db.commit()
    return row


def activity_feed(db: Session, workspace_id: int, user_id: int,
                  limit: int = 50, offset: int = 0) -> list[AIActivityItem]:
    """Bounded personal activity feed."""
    limit = max(1, min(int(limit), 200))
    return (db.query(AIActivityItem)
            .filter_by(workspace_id=workspace_id, user_id=user_id)
            .order_by(AIActivityItem.id.desc())
            .offset(max(0, int(offset))).limit(limit).all())


# ---------------------------------------------------------------------------
# AI artifact intelligence
# ---------------------------------------------------------------------------

def score_artifact(db: Session, workspace_id: int, artifact_id: int,
                   artifact_kind: str = "REPORT", version: int = 1,
                   groundedness: float = 0.0,
                   citation_coverage: float = 0.0,
                   completeness: float = 0.0,
                   provenance_ok: bool = True) -> ArtifactQualityScore:
    """Score an artifact from groundedness/citations/completeness/provenance.

    quality = weighted mean; provenance failure caps the score at 50.
    """
    quality = (groundedness * 0.4 + citation_coverage * 0.3
               + completeness * 0.3) * 100.0
    if not provenance_ok:
        quality = min(quality, 50.0)
    row = ArtifactQualityScore(
        workspace_id=workspace_id, artifact_id=artifact_id,
        artifact_kind=artifact_kind, version=version,
        quality_score=round(quality, 2),
        groundedness=round(groundedness, 4),
        citation_coverage=round(citation_coverage, 4),
        completeness=round(completeness, 4), provenance_ok=provenance_ok)
    db.add(row)
    db.commit()
    return row


def compare_artifact_versions(db: Session, workspace_id: int,
                              artifact_id: int) -> dict:
    """Compare the two most recent artifact versions; detect regression."""
    rows = (db.query(ArtifactQualityScore)
            .filter_by(workspace_id=workspace_id, artifact_id=artifact_id)
            .order_by(ArtifactQualityScore.version.desc()).limit(2).all())
    if len(rows) < 2:
        return {"comparable": False, "versions_available": len(rows)}
    newer, older = rows
    delta = newer.quality_score - older.quality_score
    regression = delta < -5.0
    newer.regression = regression
    newer.comparison = _bounded_json({
        "older_version": older.version, "older_score": older.quality_score,
        "newer_version": newer.version, "newer_score": newer.quality_score,
        "delta": round(delta, 2)})
    db.commit()
    return {"comparable": True, "delta": round(delta, 2),
            "regression": regression,
            "newer_version": newer.version, "older_version": older.version}


# ---------------------------------------------------------------------------
# Search autopilot
# ---------------------------------------------------------------------------

def search_quality_snapshot(db: Session, workspace_id: int,
                            events: Optional[list[dict]] = None) -> dict:
    """Search quality from recorded analytics events or a provided batch.

    Each event: {reformulated?, zero_results?, feedback? 'up'|'down'}.
    """
    if events is None:
        rows = (db.query(SearchQualityEvent)
                .filter_by(workspace_id=workspace_id)
                .order_by(SearchQualityEvent.id.desc()).limit(200).all())
        events = [{"reformulated": bool(getattr(r, "reformulated", False)),
                   "zero_results": bool(getattr(r, "zero_results", False)),
                   "feedback": getattr(r, "feedback", None)} for r in rows]
    total = len(events)
    if not total:
        return {"quality": 100.0, "zero_result_rate": 0.0,
                "reformulation_rate": 0.0, "positive_feedback_rate": 0.0,
                "events": 0}
    zeros = sum(1 for e in events if e.get("zero_results"))
    reform = sum(1 for e in events if e.get("reformulated"))
    feedbacks = [e.get("feedback") for e in events
                 if e.get("feedback") in ("up", "down")]
    positive = sum(1 for f in feedbacks if f == "up")
    zero_rate = zeros / total
    reform_rate = reform / total
    pos_rate = (positive / len(feedbacks)) if feedbacks else 0.5
    quality = max(0.0, 100.0 * (1.0 - 0.5 * zero_rate - 0.3 * reform_rate)
                  * (0.5 + 0.5 * pos_rate))
    return {"quality": round(quality, 2),
            "zero_result_rate": round(zero_rate, 4),
            "reformulation_rate": round(reform_rate, 4),
            "positive_feedback_rate": round(pos_rate, 4),
            "events": total}


def detect_search_drift(db: Session, workspace_id: int, baseline: float,
                        current: float,
                        degrade_threshold_percent: float = 10.0) -> dict:
    """Search ranking drift detection."""
    drift = 0.0 if baseline <= 0 else (baseline - current) / baseline * 100.0
    return {"baseline": round(baseline, 2), "current": round(current, 2),
            "drift_percent": round(drift, 1),
            "degraded": drift >= degrade_threshold_percent}


def propose_search_candidate(db: Session, workspace_id: int,
                             change_kind: str, proposed_value: Any,
                             rationale: str) -> AdaptiveCandidate:
    """Generate a candidate ranking change (evaluation required)."""
    allowed = {"ranking_weights", "synonym_expansion", "personalization",
               "freshness_boost"}
    if change_kind not in allowed:
        raise ValueError(f"unsupported search tunable: {change_kind}")
    candidate = AdaptiveCandidate(
        workspace_id=workspace_id, domain="search",
        change_kind=change_kind, proposed_value=_bounded_json(proposed_value),
        rationale=rationale[:1000],
        idempotency_key=f"search:{change_kind}:{workspace_id}")
    db.add(candidate)
    db.commit()
    return candidate
