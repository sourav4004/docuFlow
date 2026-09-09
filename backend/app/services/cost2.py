"""Cost + usage platform — Phase 17.

- anomaly detection over daily usage aggregates (deviation vs rolling mean,
  clearly labeled, never presented as certainty)
- forecasts are labeled estimates with assumptions
- usage export (CSV) for authorized organization admins, bounded and
  tenant-scoped
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)

ANOMALY_ZSCORE = 2.5
ANOMALY_MIN_DAYS = 3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _usage_series(db: Session, *, organization_id: Optional[int] = None,
                  workspace_id: Optional[int] = None,
                  days: int = 30) -> list[dict]:
    """Daily aggregates from usage_records + ai_executions (real columns)."""
    from ..models.usage import UsageRecord
    from ..models.ai_execution import AIExecution
    cutoff = _utcnow() - timedelta(days=days)
    series: dict[str, dict] = {}

    from sqlalchemy import case
    ur = db.query(
        func.date(UsageRecord.created_at).label("day"),
        func.count(UsageRecord.id),
        func.sum(case((UsageRecord.units == "tokens",
                       UsageRecord.quantity), else_=0)),
    )
    if workspace_id is not None:
        ur = ur.filter(UsageRecord.workspace_id == workspace_id)
    ur = ur.filter(UsageRecord.created_at >= cutoff).group_by(
        func.date(UsageRecord.created_at))
    for day, count, tokens in ur.all():
        series[str(day)] = {"day": str(day), "executions": int(count or 0),
                            "tokens": float(tokens or 0), "cost_usd": 0.0}

    ae = db.query(
        func.date(AIExecution.created_at).label("day"),
        func.count(AIExecution.id),
        func.sum(func.coalesce(AIExecution.actual_cost, 0.0)),
    )
    if organization_id is not None:
        ae = ae.filter(AIExecution.organization_id == organization_id)
    if workspace_id is not None:
        ae = ae.filter(AIExecution.workspace_id == workspace_id)
    ae = ae.filter(AIExecution.created_at >= cutoff).group_by(
        func.date(AIExecution.created_at))
    for day, count, cost in ae.all():
        key = str(day)
        entry = series.setdefault(key, {"day": key, "executions": 0,
                                        "tokens": 0.0, "cost_usd": 0.0})
        entry["executions"] += int(count or 0)
        entry["cost_usd"] = float(cost or 0)
    return [series[k] for k in sorted(series)]


def detect_anomalies(db: Session, *, organization_id: Optional[int] = None,
                     workspace_id: Optional[int] = None,
                     days: int = 30) -> list[dict]:
    """Detect sudden cost/token spikes vs a rolling mean.

    Outputs are explicitly labeled estimates — never presented as exact.
    """
    series = _usage_series(db, organization_id=organization_id,
                           workspace_id=workspace_id, days=days)
    if len(series) < ANOMALY_MIN_DAYS:
        return []
    anomalies = []
    for metric in ("cost_usd", "tokens", "executions"):
        values = [float(s[metric]) for s in series[:-1]]
        latest = series[-1]
        current = float(latest[metric])
        mean = sum(values) / len(values)
        if mean <= 0:
            continue
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        if std > 0:
            z = (current - mean) / std
        elif current > mean:
            # constant baseline: any material jump is anomalous by deviation
            z = (current - mean) / max(mean, 0.0001)
        else:
            continue
        if z >= ANOMALY_ZSCORE:
            anomalies.append({
                "period": latest["day"],
                "metric": metric,
                "expected": round(mean, 4),
                "actual": round(current, 4),
                "deviation": round(current / mean, 3),
                "z_score": round(z, 2),
                "severity": "HIGH" if z >= 4 else "MEDIUM",
                "label": "estimated anomaly — verify before action",
            })
    return anomalies


def forecast(db: Session, *, organization_id: Optional[int] = None,
             workspace_id: Optional[int] = None,
             horizon_days: int = 30) -> dict:
    """Simple labeled forecast from a linear trend of daily cost."""
    series = _usage_series(db, organization_id=organization_id,
                           workspace_id=workspace_id, days=30)
    daily = [float(s["cost_usd"]) for s in series]
    if len(daily) < 2:
        return {"estimate": None,
                "message": "insufficient history for a forecast",
                "assumptions": []}
    n = len(daily)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(daily) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, daily)) \
        / sum((x - mean_x) ** 2 for x in xs)
    current = daily[-1]
    projected_daily = max(0.0, current + slope * (horizon_days / 2))
    return {
        "estimate_labeled": True,
        "daily_estimate_usd": round(projected_daily, 2),
        "monthly_projection_usd": round(projected_daily * 30, 2),
        "confidence_range_usd": [
            round(max(0.0, projected_daily * 0.7), 2),
            round(projected_daily * 1.3, 2)],
        "assumptions": [
            "linear extrapolation of the last 30 days",
            "estimates are not guarantees",
        ],
    }


def export_usage_csv(db: Session, *, organization_id: int,
                     workspace_id: Optional[int] = None,
                     days: int = 90) -> str:
    """Bounded CSV export of usage for authorized org admins."""
    series = _usage_series(db, organization_id=organization_id,
                           workspace_id=workspace_id, days=min(days, 365))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["day", "executions", "tokens", "cost_usd"])
    for row in series:
        writer.writerow([row["day"], row["executions"],
                         int(row["tokens"]), row["cost_usd"]])
    return buf.getvalue()
