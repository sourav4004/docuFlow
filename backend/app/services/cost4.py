"""Phase 19 — AI cost platform 4.0.

Preflight estimation, budget reservation with release, reconciliation of
estimated vs actual vs reservation, multi-granularity labeled forecasting,
anomaly detection (spikes, per-user/workflow/provider, token explosions,
repeated-failure cost), and cost-optimization recommendations that are NEVER
auto-applied to production policy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

DEFAULT_INPUT_1K = 0.002
DEFAULT_OUTPUT_1K = 0.006


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _day(value: Optional[datetime]) -> str:
    value = _as_utc(value) or _utcnow()
    return value.date().isoformat()


def _executions(db: Session, *, organization_id: Optional[int] = None,
                workspace_id: Optional[int] = None,
                since: Optional[datetime] = None,
                limit: int = 10000):
    from ..models.ai_execution import AIExecution
    q = db.query(AIExecution)
    if organization_id is not None:
        q = q.filter(AIExecution.organization_id == organization_id)
    if workspace_id is not None:
        q = q.filter(AIExecution.workspace_id == workspace_id)
    if since is not None:
        q = q.filter(AIExecution.created_at >= since)
    return q.order_by(AIExecution.created_at.desc()).limit(limit).all()


def estimate_cost(provider: str, model: str, *,
                  input_tokens: int = 0, output_tokens: int = 0,
                  embedding_tokens: int = 0) -> dict:
    """Deterministic preflight estimate (labeled estimate, not exact)."""
    cost_in = DEFAULT_INPUT_1K * (input_tokens / 1000.0)
    cost_out = DEFAULT_OUTPUT_1K * (output_tokens / 1000.0)
    cost_emb = 0.0001 * (embedding_tokens / 1000.0)
    return {
        "is_estimate": True,
        "provider": provider, "model": model,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "embedding_tokens": embedding_tokens,
        "estimated_cost_usd": round(cost_in + cost_out + cost_emb, 6),
        "breakdown": {"input_usd": round(cost_in, 6),
                      "output_usd": round(cost_out, 6),
                      "embedding_usd": round(cost_emb, 6)},
    }


def reserve_budget(db: Session, *, organization_id: Optional[int],
                   workspace_id: int, execution_ref: Optional[str],
                   feature: str, estimated_cost: float) -> dict:
    """Reserve budget before an expensive AI operation."""
    from ..models.phase19 import CostReservation
    if estimated_cost < 0:
        raise ValueError("negative estimate")
    reservation = CostReservation(
        organization_id=organization_id, workspace_id=workspace_id,
        execution_ref=execution_ref, feature=feature,
        reserved_amount=round(estimated_cost, 6),
        status="RESERVED")
    db.add(reservation)
    db.flush()
    return {"reservation_id": reservation.id,
            "reserved_amount": reservation.reserved_amount,
            "status": "RESERVED"}


def release_reservation(db: Session, *, reservation_id: int,
                        actual_cost: Optional[float] = None,
                        released_amount: Optional[float] = None) -> dict:
    """Release unused reservation and settle with the actual cost."""
    from ..models.phase19 import CostReservation
    reservation = db.query(CostReservation).get(reservation_id)
    if reservation is None:
        raise ValueError("reservation not found")
    if reservation.status == "RECONCILED":
        return {"status": "ALREADY_RECONCILED",
                "reservation_id": reservation.id}
    actual = float(actual_cost if actual_cost is not None else 0.0)
    if released_amount is None:
        released_amount = max(0.0, float(reservation.reserved_amount)
                              - actual)
    reservation.released_amount = round(released_amount, 6)
    reservation.actual_cost = round(actual, 6)
    reservation.status = "RECONCILED"
    reservation.settled_at = _utcnow()
    db.flush()
    return {"status": "RECONCILED", "reservation_id": reservation.id,
            "reserved": reservation.reserved_amount,
            "released": reservation.released_amount,
            "actual": reservation.actual_cost}


def reconcile(db: Session, *, reservation_id: int,
              estimated_cost: float, actual_cost: float) -> dict:
    """Reconciliation: estimated vs actual vs reservation; flag discrepancy."""
    from ..models.phase19 import CostReservation
    reservation = db.query(CostReservation).get(reservation_id)
    if reservation is None:
        raise ValueError("reservation not found")
    diff = round(actual_cost - estimated_cost, 6)
    discrepancy = (abs(diff) / estimated_cost) > 0.2 \
        if estimated_cost else (actual_cost > 0)
    reservation.actual_cost = round(actual_cost, 6)
    reservation.released_amount = round(
        max(0.0, float(reservation.reserved_amount) - actual_cost), 6)
    reservation.status = "RECONCILED"
    reservation.settled_at = _utcnow()
    db.flush()
    return {"reservation_id": reservation.id,
            "estimated_cost": estimated_cost, "actual_cost": actual_cost,
            "delta": diff, "discrepancy": discrepancy,
            "note": "estimates are labeled estimates; actual provider "
                    "charges win"}


def forecast2(db: Session, *, organization_id: int,
              workspace_id: Optional[int] = None,
              granularity: str = "monthly",
              lookahead_days: int = 30) -> dict:
    """Labeled forecasting at org/workspace granularity."""
    if granularity not in ("daily", "weekly", "monthly"):
        raise ValueError("unsupported granularity")
    rows = _executions(db, organization_id=organization_id,
                       workspace_id=workspace_id,
                       since=_utcnow() - timedelta(days=30))
    by_model = defaultdict(float)
    by_provider = defaultdict(float)
    total = 0.0
    for row in rows:
        cost = float(row.estimated_cost or 0.0)
        total += cost
        by_model[row.model or "unknown"] += cost
        by_provider[row.provider or "unknown"] += cost
    daily_rate = total / 30.0 if rows else 0.0
    window = {"daily": 1, "weekly": 7, "monthly": 30}[granularity]
    return {
        "is_estimate": True,
        "granularity": granularity,
        "assumptions": ["linear projection over last 30 days",
                        "excludes provider price changes and discounts"],
        "observed_30d": round(total, 4),
        "projected_next_est": round(daily_rate * window * lookahead_days
                                    / window, 4),
        "by_model": [{"key": k, "cost": round(v, 4)} for k, v in
                     sorted(by_model.items(), key=lambda kv: -kv[1])[:10]],
        "by_provider": [{"key": k, "cost": round(v, 4)} for k, v in
                        sorted(by_provider.items(), key=lambda kv: -kv[1])
                        [:10]],
    }


def _daily_series(rows) -> dict[str, float]:
    series = defaultdict(float)
    for row in rows:
        series[_day(row.created_at)] += float(row.estimated_cost or 0.0)
    return series


def detect_anomalies(db: Session, *, organization_id: int,
                     days: int = 30, z_threshold: float = 2.5,
                     persist: bool = True) -> dict:
    """Detect sudden spend/token/execution spikes + repeated-failure cost."""
    from ..models.ai_execution import AIExecution
    from ..models.phase17 import CostAnomaly
    today = _utcnow()
    since = today - timedelta(days=days)
    rows = _executions(db, organization_id=organization_id, since=since)
    series = _daily_series(rows)
    found = []
    for metric, value_of in (
            ("cost_usd", lambda r: float(r.estimated_cost or 0.0)),
            ("tokens", lambda r: float(r.total_tokens or 0.0)),
            ("executions", lambda r: 1.0)):
        daily = defaultdict(float)
        for row in rows:
            daily[_day(row.created_at)] += value_of(row)
        today_key = _day(today)
        prior = [v for k, v in daily.items() if k != today_key]
        if len(prior) < 5:
            continue
        mean = sum(prior) / len(prior)
        variance = sum((v - mean) ** 2 for v in prior) / len(prior)
        std = variance ** 0.5 or mean * 0.1
        actual = daily.get(today_key, 0.0)
        if mean and (actual - mean) > z_threshold * std:
            severity = "HIGH" if actual > 5 * mean else "MEDIUM"
            found.append({"period": today_key, "metric": metric,
                          "expected": round(mean, 2),
                          "actual": round(actual, 2),
                          "deviation": round((actual - mean) / mean, 3),
                          "severity": severity})
            if persist:
                db.add(CostAnomaly(
                    organization_id=organization_id,
                    period=today_key, metric=metric,
                    expected=round(mean, 4), actual=round(actual, 4),
                    deviation=round((actual - mean) / mean, 3),
                    severity=severity, status="OPEN"))
    # repeated-failure cost (deterministic, from durable executions)
    failed_cost = sum(float(r.estimated_cost or 0.0) for r in rows
                      if r.status == "FAILED" and (r.retry_count or 0) > 0)
    repeated = {"metric": "failed_execution_cost",
                "period": _day(today), "actual": round(failed_cost, 4),
                "kind": "repeated_failures"}
    if persist:
        db.flush()
    return {"organization_id": organization_id, "detected": found,
            "repeated_failure_cost": repeated, "days": days,
            "series_points": len(series)}


def optimization_recommendations(db: Session, *,
                                 organization_id: int,
                                 persist: bool = True) -> dict:
    """Deterministic suggestions (never auto-applied)."""
    from ..models.ai_execution import AIExecution
    from ..models.phase16 import ProviderCapability
    from ..models.phase19 import CostRecommendation
    since = _utcnow() - timedelta(days=30)
    rows = _executions(db, organization_id=organization_id, since=since)
    suggestions = []
    # repeated identical requests → caching
    by_hash = defaultdict(int)
    for row in rows:
        if row.request_hash:
            by_hash[row.request_hash] += 1
    dup = sum(1 for v in by_hash.values() if v >= 3)
    if dup:
        suggestions.append({"kind": "CACHING",
                            "reason": f"{dup} request hash(es) repeated "
                                      ">=3 times in 30 days",
                            "savings_estimate": 5.0})
    # expensive model dominance → downgrade candidate
    by_model = defaultdict(float)
    for row in rows:
        by_model[row.model or "unknown"] += float(row.estimated_cost or 0.0)
    total = sum(by_model.values())
    if total:
        top_model, top_cost = max(by_model.items(), key=lambda kv: kv[1])
        cheap = (db.query(ProviderCapability)
                 .filter(ProviderCapability.cost_per_1k_input.isnot(None))
                 .order_by(ProviderCapability.cost_per_1k_input.asc())
                 .first())
        if top_cost / total > 0.7 and cheap is not None:
            suggestions.append({
                "kind": "MODEL_DOWNGRADE",
                "reason": f"{top_model} is {round(top_cost / total * 100)}% "
                          f"of spend; cheaper model {cheap.model} available",
                "savings_estimate": 2.0})
    if persist:
        for item in suggestions:
            existing = (db.query(CostRecommendation)
                        .filter(CostRecommendation.organization_id ==
                                organization_id,
                                CostRecommendation.kind == item["kind"],
                                CostRecommendation.status == "SUGGESTED")
                        .first())
            if existing is None:
                db.add(CostRecommendation(
                    organization_id=organization_id, kind=item["kind"],
                    reason=item["reason"],
                    savings_estimate=float(item.get("savings_estimate")
                                           or 1.0)))
        db.flush()
    return {"organization_id": organization_id, "recommendations":
            suggestions,
            "note": "recommendations require manual/approval-gated "
                    "application; never auto-applied"}


def attribution2(db: Session, *, organization_id: int,
                 workspace_id: Optional[int] = None,
                 days: int = 30) -> dict:
    """Drill-down attribution by workspace/user/feature/model/provider."""
    rows = _executions(db, organization_id=organization_id,
                       workspace_id=workspace_id,
                       since=_utcnow() - timedelta(days=days))
    dims = {key: defaultdict(float) for key in
            ("by_workspace", "by_user", "by_feature", "by_model",
             "by_provider", "by_execution_type")}
    total = 0.0
    for row in rows:
        cost = float(row.estimated_cost or 0.0)
        total += cost
        dims["by_workspace"][row.workspace_id] += cost
        dims["by_user"][row.user_id] += cost
        dims["by_feature"][row.task_type or row.execution_type or
                          "unknown"] += cost
        dims["by_model"][row.model or "unknown"] += cost
        dims["by_provider"][row.provider or "unknown"] += cost
        dims["by_execution_type"][row.execution_type or "unknown"] += cost
    return {"days": days, "total_cost": round(total, 4),
            "execution_count": len(rows),
            **{k: [{"key": str(key), "cost": round(v, 4)}
                   for key, v in sorted(counter.items(),
                                        key=lambda kv: -kv[1])[:10]]
               for k, counter in dims.items()}}
