"""Phase 20 — SLO / Reliability 2.0.

Persisted SLO history, error-budget calculation per SLO, an error-budget
policy (surface warning, recommend freeze, require operator review — never
automatically block unrelated operations), and service reliability scores.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import SloHistory, ErrorBudget, ReliabilityScore


def record_history(db: Session, *, name: str, measured_value: float,
                   target: Optional[float] = None,
                   window: str = "daily") -> SloHistory:
    met = None
    if target is not None:
        met = measured_value >= target
    row = SloHistory(name=name, window=window,
                     measured_value=measured_value, target=target, met=met)
    db.add(row)
    db.flush()
    return row


def slo_recent(db: Session, *, name: Optional[str] = None,
               days: int = 30, limit: int = 200) -> list[dict]:
    q = db.query(SloHistory)
    if name:
        q = q.filter(SloHistory.name == name)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = q.filter(SloHistory.created_at >= cutoff)\
        .order_by(SloHistory.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "name": r.name, "window": r.window,
             "measured_value": r.measured_value, "target": r.target,
             "met": r.met, "created_at": r.created_at} for r in rows]


def error_budget(db: Session, *, name: str, budget: float,
                 period: str = "monthly",
                 consumed_override: Optional[float] = None) -> ErrorBudget:
    """Create or update the error budget for an SLO. Consumed can be passed
    explicitly for deterministic tests or computed from history."""
    row = db.query(ErrorBudget).filter_by(name=name, period=period).first()
    if row is None:
        row = ErrorBudget(name=name, period=period, budget=budget)
        db.add(row)
    if consumed_override is not None:
        row.consumed = consumed_override
    else:
        rows = slo_recent(db, name=name, days=30)
        missed = [r for r in rows if r["met"] is False]
        total = len(rows)
        row.consumed = round((len(missed) / total) * 100.0, 2) \
            if total else 0.0
    consumed = row.consumed
    row.status = "EXHAUSTED" if consumed >= 99.0 else \
        ("WARNING" if consumed >= 80.0 else "OK")
    db.flush()
    return row


def error_budget_policy(row: ErrorBudget) -> dict:
    """Policy for an error-budget state: never automatically blocks
    unrelated operations — it surfaces warnings and recommends review."""
    if row.status == "EXHAUSTED":
        return {
            "level": "CRITICAL",
            "action": "require_operator_review",
            "recommend_freeze": True,
            "message": (f"Error budget for {row.name} exhausted "
                        f"({row.consumed:.1f}%). Operator review required; "
                        "freeze recommended for the SLO's own scope.")}
    if row.status == "WARNING":
        return {
            "level": "WARNING",
            "action": "surface_warning",
            "recommend_freeze": False,
            "message": (f"Error budget for {row.name} at {row.consumed:.1f}%."
                        " Monitor closely.")}
    return {
        "level": "OK",
        "action": "none",
        "recommend_freeze": False,
        "message": f"Error budget for {row.name} healthy ({row.consumed:.1f}%)."}


def reliability_score(db: Session, *, service: str,
                      factors: dict) -> ReliabilityScore:
    """Weighted reliability score 0..100 with detail."""
    weights = {"availability": 0.4, "latency": 0.2, "error_rate": 0.2,
               "recovery": 0.2}
    total = 0.0
    denom = 0.0
    detail = {}
    for k, w in weights.items():
        if k in factors:
            v = max(0.0, min(100.0, float(factors[k])))
            detail[k] = round(v, 2)
            total += v * w
            denom += w
    score = round(total / denom, 2) if denom else 0.0
    row = ReliabilityScore(service=service, score=score,
                           detail_json=json.dumps(detail))
    db.add(row)
    db.flush()
    return row


def reliability_report(db: Session, *, service: Optional[str] = None,
                       limit: int = 100) -> list[dict]:
    q = db.query(ReliabilityScore)
    if service:
        q = q.filter(ReliabilityScore.service == service)
    rows = q.order_by(ReliabilityScore.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "service": r.service, "score": r.score,
             "detail": json.loads(r.detail_json or "{}"),
             "created_at": r.created_at} for r in rows]