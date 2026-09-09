"""Operational alert engine — Phase 17.

Rules watch named metrics (queue_depth, error_rate, provider_down,
budget_exhausted, worker_failure, db_latency). ``evaluate`` is deterministic
and enforces a per-rule cooldown so alerts are never spammed. Alert events
are tenant-tagged and exposed to authorized operators.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase17 import AlertRule, AlertEvent

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def create_rule(db: Session, *, name: str, metric: str, operator: str,
                threshold: float, severity: str = "WARNING",
                cooldown_minutes: int = 30,
                workspace_id: Optional[int] = None,
                organization_id: Optional[int] = None) -> AlertRule:
    if operator not in (">", ">=", "<", "<="):
        raise ValueError("operator must be one of > >= < <=")
    if severity not in ("INFO", "WARNING", "CRITICAL"):
        raise ValueError("invalid severity")
    rule = AlertRule(
        name=name, metric=metric, operator=operator, threshold=threshold,
        severity=severity, cooldown_minutes=max(1, cooldown_minutes),
        workspace_id=workspace_id, organization_id=organization_id)
    db.add(rule)
    db.flush()
    return rule


def list_rules(db: Session, workspace_id: Optional[int] = None,
               limit: int = 100) -> dict:
    q = db.query(AlertRule)
    if workspace_id is not None:
        q = q.filter((AlertRule.workspace_id == workspace_id)
                     | (AlertRule.workspace_id.is_(None)))
    items = q.order_by(AlertRule.id.desc()).limit(min(limit, 500)).all()
    return {"items": [
        {"id": r.id, "name": r.name, "metric": r.metric,
         "operator": r.operator, "threshold": r.threshold,
         "severity": r.severity, "enabled": r.enabled,
         "workspace_id": r.workspace_id,
         "last_fired_at": r.last_fired_at}
        for r in items], "total": len(items), "limit": min(limit, 500)}


def list_events(db: Session, workspace_id: Optional[int] = None,
                limit: int = 100) -> dict:
    q = db.query(AlertEvent)
    if workspace_id is not None:
        q = q.filter(AlertEvent.workspace_id == workspace_id)
    items = q.order_by(AlertEvent.fired_at.desc()).limit(min(limit, 200)).all()
    return {"items": [
        {"id": e.id, "rule_id": e.rule_id, "severity": e.severity,
         "message": e.message, "metric_value": e.metric_value,
         "fired_at": e.fired_at, "resolved_at": e.resolved_at}
        for e in items], "total": len(items), "limit": min(limit, 200)}


def evaluate(db: Session, metrics: dict,
             now: Optional[datetime] = None) -> list[dict]:
    """Evaluate enabled rules against a metric snapshot.

    ``metrics`` maps metric name → float. Rules fire only when the operator
    holds AND the cooldown has elapsed since the last fire. Fires are
    persisted; the list of fired rules is returned.
    """
    now = now or _utcnow()
    fired = []
    rules = db.query(AlertRule).filter(AlertRule.enabled.is_(True)).all()
    for rule in rules:
        if rule.metric not in metrics:
            continue
        value = float(metrics[rule.metric])
        hit = False
        if rule.operator == ">":
            hit = value > rule.threshold
        elif rule.operator == ">=":
            hit = value >= rule.threshold
        elif rule.operator == "<":
            hit = value < rule.threshold
        elif rule.operator == "<=":
            hit = value <= rule.threshold
        if not hit:
            continue
        last = _as_utc(rule.last_fired_at)
        if last is not None and (now - last).total_seconds() < \
                rule.cooldown_minutes * 60:
            continue  # cooldown — never spam
        rule.last_fired_at = now
        db.add(AlertEvent(
            rule_id=rule.id, workspace_id=rule.workspace_id,
            severity=rule.severity,
            message=(f"{rule.name}: {rule.metric} {rule.operator} "
                     f"{rule.threshold} (now {value})"),
            metric_value=value, fired_at=now))
        fired.append({"rule_id": rule.id, "name": rule.name,
                      "severity": rule.severity, "value": value})
    db.flush()
    return fired


def resolve_events(db: Session, rule_id: int,
                   operator_user_id: int) -> dict:
    """Mark open events for a rule resolved (operator action)."""
    from ..models.phase17 import AlertEvent
    now = _utcnow()
    rows = db.query(AlertEvent).filter(
        AlertEvent.rule_id == rule_id,
        AlertEvent.resolved_at.is_(None)).limit(200).all()
    for row in rows:
        row.resolved_at = now
    db.flush()
    return {"resolved": len(rows)}
