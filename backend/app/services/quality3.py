"""Phase 20 — AI quality platform 3.0.

Unified quality scorecards (retrieval/rag/citations/extraction/summarization/
search/agents/workflows), quality dimensions (correctness, completeness,
groundedness, citation coverage, citation correctness, confidence, latency,
cost, refusal accuracy), configurable thresholds, automatic regression
detection against baseline, trending, and deduplicated quality alerts.

Regression detection raises alerts but never deploys fixes automatically.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import QualityScorecard, QualityTrend, QualityAlert

VALID_DOMAINS = {
    "retrieval", "rag", "citations", "extraction", "summarization",
    "search", "agents", "workflows", "knowledge",
}

DIMENSIONS = [
    "correctness", "completeness", "groundedness", "citation_coverage",
    "citation_correctness", "confidence", "latency", "cost",
    "refusal_accuracy",
]


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def compute_scorecard(db: Session, *, domain: str,
                      dimensions: dict,
                      workspace_id: Optional[int] = None,
                      organization_id: Optional[int] = None,
                      period: str = "daily",
                      thresholds: Optional[dict] = None) -> QualityScorecard:
    """Persist a unified scorecard. `dimensions` maps dimension -> 0..1 value.
    Unknown dimensions are ignored; thresholds are configurable."""
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    cleaned = {k: float(v) for k, v in dimensions.items()
               if k in DIMENSIONS}
    overall = sum(cleaned.values()) / len(cleaned) if cleaned else 0.0
    card = QualityScorecard(
        domain=domain, workspace_id=workspace_id,
        organization_id=organization_id, period=period,
        scorecard_json=_dumps({"dimensions": cleaned, "overall": overall,
                               "thresholds": thresholds or {}}),
        overall_score=overall)
    db.add(card)
    db.flush()
    return card


def latest_scorecard(db: Session, *, domain: str,
                     workspace_id: Optional[int] = None) -> Optional[dict]:
    q = db.query(QualityScorecard).filter_by(domain=domain)
    if workspace_id is not None:
        q = q.filter(QualityScorecard.workspace_id == workspace_id)
    else:
        q = q.filter(QualityScorecard.workspace_id.is_(None))
    row = q.order_by(QualityScorecard.created_at.desc(),
                     QualityScorecard.id.desc()).first()
    if row is None:
        return None
    payload = json.loads(row.scorecard_json or "{}")
    return {"id": row.id, "overall_score": row.overall_score,
            "dimensions": payload.get("dimensions", {}),
            "period": row.period, "created_at": row.created_at}


def detect_regression(db: Session, *, domain: str,
                      workspace_id: Optional[int] = None,
                      min_delta: float = 0.05) -> dict:
    """Compare the latest scorecard against the previous one. Returns a
    regression verdict; raises a deduplicated alert when significant."""
    q = db.query(QualityScorecard).filter_by(domain=domain)
    if workspace_id is not None:
        q = q.filter(QualityScorecard.workspace_id == workspace_id)
    else:
        q = q.filter(QualityScorecard.workspace_id.is_(None))
    rows = q.order_by(QualityScorecard.created_at.desc(),
                      QualityScorecard.id.desc()).limit(2).all()
    if len(rows) < 2:
        return {"regressed": False, "reason": "insufficient history"}
    latest, previous = rows[0], rows[1]
    prev_score = previous.overall_score or 0.0
    curr_score = latest.overall_score or 0.0
    delta = curr_score - prev_score
    regressed = delta <= -min_delta
    if regressed:
        fingerprint = hashlib.sha256(
            f"{domain}:{workspace_id}".encode()).hexdigest()[:16]
        existing = db.query(QualityAlert).filter_by(
            fingerprint=fingerprint).first()
        if existing is None:
            db.add(QualityAlert(
                domain=domain, severity="HIGH", fingerprint=fingerprint,
                message=(f"Quality regression in {domain}: "
                         f"{prev_score:.3f} -> {curr_score:.3f}"),
                workspace_id=workspace_id))
            db.flush()
    return {"regressed": regressed, "delta": delta,
            "previous": prev_score, "current": curr_score}


def record_trend(db: Session, *, domain: str, period: str,
                 metrics: dict, score: Optional[float] = None,
                 window_start: Optional[datetime] = None) -> QualityTrend:
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    trend = QualityTrend(
        domain=domain, period=period,
        window_start=window_start or datetime.now(timezone.utc),
        metrics_json=_dumps(metrics), score=score)
    db.add(trend)
    db.flush()
    return trend


def quality_trends(db: Session, *, domain: Optional[str] = None,
                   period: Optional[str] = None,
                   days: int = 30, limit: int = 200) -> list[dict]:
    q = db.query(QualityTrend)
    if domain:
        q = q.filter(QualityTrend.domain == domain)
    if period:
        q = q.filter(QualityTrend.period == period)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = q.filter(QualityTrend.created_at >= cutoff)\
        .order_by(QualityTrend.created_at.desc()).limit(limit).all()
    return [{"id": t.id, "domain": t.domain, "period": t.period,
             "score": t.score, "metrics": json.loads(t.metrics_json or "{}"),
             "created_at": t.created_at} for t in rows]


def list_alerts(db: Session, *, domain: Optional[str] = None,
                resolved: Optional[bool] = None,
                limit: int = 100) -> list[dict]:
    q = db.query(QualityAlert)
    if domain:
        q = q.filter(QualityAlert.domain == domain)
    if resolved is not None:
        q = q.filter(QualityAlert.resolved == resolved)
    rows = q.order_by(QualityAlert.created_at.desc()).limit(limit).all()
    return [{"id": a.id, "domain": a.domain, "severity": a.severity,
             "message": a.message, "resolved": a.resolved,
             "workspace_id": a.workspace_id, "created_at": a.created_at}
            for a in rows]


def resolve_alert(db: Session, alert_id: int) -> dict:
    alert = db.get(QualityAlert, alert_id)
    if alert is None:
        raise KeyError("alert not found")
    alert.resolved = True
    return {"resolved": True, "alert_id": alert_id}


def scorecard_summary(db: Session, *, workspace_id: Optional[int] = None,
                      limit: int = 50) -> list[dict]:
    """Latest scorecard per domain for the ops dashboard."""
    out = []
    for domain in sorted(VALID_DOMAINS):
        card = latest_scorecard(db, domain=domain,
                                workspace_id=workspace_id)
        if card is not None:
            out.append({"domain": domain, **card})
    return out[:limit]