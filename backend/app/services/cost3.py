"""Phase 18 cost control — attribution drill-down, labeled forecasting,
budget enforcement (soft/hard/approval limits).

Estimates are ALWAYS labeled as estimates; never presented as exact.
Attribution is computed from durable AIExecution records (tenant-scoped).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _execution_rows(db: Session, *, organization_id: Optional[int] = None,
                    workspace_id: Optional[int] = None,
                    since: Optional[datetime] = None,
                    limit: int = 5000):
    from ..models.ai_execution import AIExecution
    q = db.query(AIExecution)
    if organization_id is not None:
        q = q.filter(AIExecution.organization_id == organization_id)
    if workspace_id is not None:
        q = q.filter(AIExecution.workspace_id == workspace_id)
    if since is not None:
        q = q.filter(AIExecution.created_at >= since)
    return q.order_by(AIExecution.created_at.desc()).limit(limit).all()


def attribution(db: Session, *, organization_id: int,
                days: int = 30) -> dict:
    """Cost drill-down by workspace, user, feature, model."""
    since = _utcnow() - timedelta(days=days)
    rows = _execution_rows(db, organization_id=organization_id, since=since)
    by_workspace: dict = {}
    by_user: dict = {}
    by_feature: dict = {}
    by_model: dict = {}
    total_cost = 0.0
    total_tokens = 0
    for row in rows:
        cost = float(row.estimated_cost or 0.0)
        tokens = int(row.total_tokens or 0)
        total_cost += cost
        total_tokens += tokens
        by_workspace[row.workspace_id] = by_workspace.get(
            row.workspace_id, 0.0) + cost
        by_user[row.user_id] = by_user.get(row.user_id, 0.0) + cost
        feat = row.execution_type or row.task_type or "unknown"
        by_feature[feat] = by_feature.get(feat, 0.0) + cost
        model = row.model or "unknown"
        by_model[model] = by_model.get(model, 0.0) + cost
    return {
        "days": days,
        "total_cost": round(total_cost, 4),
        "total_tokens": total_tokens,
        "execution_count": len(rows),
        "by_workspace": _ranked(by_workspace),
        "by_user": _ranked(by_user),
        "by_feature": _ranked(by_feature),
        "by_model": _ranked(by_model),
    }


def _ranked(counter: dict, top: int = 10) -> list[dict]:
    return [{"key": k, "cost": round(v, 4)}
            for k, v in sorted(counter.items(), key=lambda kv: -kv[1])[:top]]


def forecast(db: Session, *, organization_id: int,
             lookahead_days: int = 30) -> dict:
    """Projected usage — explicitly labeled as an estimate."""
    today = _utcnow()
    rows_30 = _execution_rows(db, organization_id=organization_id,
                              since=today - timedelta(days=30))
    recent_7 = [r for r in rows_30
                if _as_utc(r.created_at) is not None
                and _as_utc(r.created_at) >= today - timedelta(days=7)]
    cost_30 = sum(float(r.estimated_cost or 0.0) for r in rows_30)
    cost_7 = sum(float(r.estimated_cost or 0.0) for r in recent_7)
    daily_rate = (cost_7 / 7.0 if recent_7
                  else (cost_30 / 30.0 if rows_30 else 0.0))
    return {
        "is_estimate": True,
        "assumptions": ["linear projection of last 7 days",
                        "does not include provider price changes"],
        "last_30d_cost": round(cost_30, 4),
        "daily_rate_est": round(daily_rate, 4),
        "projected_30d_cost_est": round(daily_rate * lookahead_days, 4),
        "confidence": "low" if not rows_30 else "medium",
    }


def enforce_budget(db: Session, *, organization_id: int,
                   workspace_id: int, feature: str,
                   estimated_cost: float,
                   soft_limit: Optional[float] = None,
                   hard_limit: Optional[float] = None) -> dict:
    """Budget enforcement: BLOCK above hard, approval above soft."""
    used = attribution(db, organization_id=organization_id)["total_cost"]
    projected = used + max(0.0, estimated_cost)
    if hard_limit is not None and projected > hard_limit:
        return {"action": "BLOCK", "reason": "hard budget limit exceeded",
                "used": round(used, 4), "projected": round(projected, 4),
                "hard_limit": hard_limit}
    if soft_limit is not None and projected > soft_limit:
        return {"action": "REQUIRE_APPROVAL",
                "reason": "soft budget threshold crossed",
                "used": round(used, 4), "projected": round(projected, 4),
                "soft_limit": soft_limit}
    return {"action": "ALLOW", "reason": "within budget",
            "used": round(used, 4), "projected": round(projected, 4)}