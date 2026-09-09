"""Phase 19 — observability 4.0.

Trace platform 2.0 (spans for API/queue/worker/provider/retrieval/RAG/agent/
workflow/tool/notification), trace correlation across distributed processes,
latency breakdown by phase, error correlation, a configurable SLO engine
with burn-rate-style windows, and alert deduplication to prevent storms.

Sensitive payloads are redacted at the boundary via observability2.redact —
only safe summaries are ever persisted.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SPAN_TYPES = ("request", "queue_job", "worker", "provider", "retrieval",
              "rag", "agent", "workflow", "tool", "notification")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    """SQLite returns naive datetimes even for timezone-aware columns."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _redact(value: str) -> str:
    from .observability2 import redact
    return redact(str(value))[:500]


# ---------------------------------------------------------------------------
# Trace platform 2.0
# ---------------------------------------------------------------------------

def start_span(db: Session, *, workspace_id: int,
               organization_id: Optional[int] = None,
               span_type: str, trace_id: Optional[str] = None,
               parent_span_id: Optional[str] = None,
               execution_id: Optional[str] = None,
               workflow_execution_id: Optional[str] = None,
               node_execution_id: Optional[str] = None,
               input_summary: Optional[str] = None,
               provider: Optional[str] = None,
               model: Optional[str] = None,
               metadata: Optional[dict] = None):
    from ..models.phase16 import TraceSpan
    if span_type not in SPAN_TYPES:
        raise ValueError("unknown span type")
    span_id = str(uuid.uuid4())[:32]
    row = TraceSpan(
        span_id=span_id, trace_id=trace_id or str(uuid.uuid4())[:32],
        parent_span_id=parent_span_id, workspace_id=workspace_id,
        organization_id=organization_id, execution_id=execution_id,
        workflow_execution_id=workflow_execution_id,
        node_execution_id=node_execution_id, span_type=span_type,
        status="OK", provider=provider, model=model,
        input_summary=_redact(input_summary) if input_summary else None,
        metadata_json=json.dumps(metadata or {}, default=str)[:4000])
    db.add(row)
    db.flush()
    return {"span_id": row.span_id, "trace_id": row.trace_id,
            "span_type": span_type, "started_at": row.started_at}


def end_span(db: Session, *, span_id: str, status: str = "OK",
             error_class: Optional[str] = None,
             input_tokens: Optional[int] = None,
             output_tokens: Optional[int] = None,
             cost_usd: Optional[float] = None,
             metadata: Optional[dict] = None) -> dict:
    from ..models.phase16 import TraceSpan
    if status not in ("OK", "ERROR", "TIMEOUT", "RATE_LIMITED",
                      "CANCELLED"):
        raise ValueError("invalid span status")
    row = (db.query(TraceSpan)
           .filter(TraceSpan.span_id == span_id).first())
    if row is None:
        raise KeyError("span not found")
    now = _utcnow()
    if row.completed_at is not None:
        return {"status": "ALREADY_COMPLETED", "span_id": span_id}
    row.status = status
    row.error_class = error_class
    row.completed_at = now
    started_at = _as_utc(row.started_at)
    if started_at is not None:
        row.latency_ms = round(
            (now - started_at).total_seconds() * 1000.0, 1)
    row.input_tokens = input_tokens
    row.output_tokens = output_tokens
    row.cost_usd = cost_usd
    db.flush()
    return {"span_id": span_id, "status": status,
            "latency_ms": row.latency_ms}


def trace_summary(db: Session, trace_id: str) -> dict:
    """Bounded, redacted trace readout (admin-only in the API layer)."""
    from ..models.phase16 import TraceSpan
    spans = (db.query(TraceSpan)
             .filter(TraceSpan.trace_id == trace_id)
             .order_by(TraceSpan.started_at.asc()).limit(500).all())
    return {"trace_id": trace_id, "spans": [
        {"span_id": s.span_id, "span_type": s.span_type, "status":
         s.status, "latency_ms": s.latency_ms, "provider": s.provider,
         "model": s.model, "error_class": s.error_class,
         "input_summary": s.input_summary}
        for s in spans], "span_count": len(spans)}


def latency_breakdown(db: Session, *, since_minutes: int = 60) -> dict:
    from ..models.phase16 import TraceSpan
    since = _utcnow() - timedelta(minutes=since_minutes)
    rows = (db.query(TraceSpan)
            .filter(TraceSpan.started_at >= since,
                    TraceSpan.latency_ms.isnot(None))
            .limit(5000).all())
    groups = {}
    for row in rows:
        entry = groups.setdefault(row.span_type,
                                  {"count": 0, "latency_ms": []})
        entry["count"] += 1
        entry["latency_ms"].append(row.latency_ms)
    breakdown = []
    for span_type, entry in groups.items():
        values = sorted(entry["latency_ms"])
        breakdown.append({
            "span_type": span_type, "count": entry["count"],
            "avg_ms": round(sum(values) / len(values), 1),
            "p95_ms": values[min(len(values) - 1,
                                 int(len(values) * 0.95))],
            "p99_ms": values[min(len(values) - 1,
                                 int(len(values) * 0.99))],
        })
    breakdown.sort(key=lambda b: -b["avg_ms"])
    return {"since_minutes": since_minutes, "breakdown": breakdown}


def correlate_errors(db: Session, *, execution_id: Optional[str] = None,
                     trace_id: Optional[str] = None,
                     job_id: Optional[str] = None,
                     limit: int = 100) -> dict:
    """Correlate failures across distributed processes by id chain."""
    from ..models.phase16 import TraceSpan
    q = db.query(TraceSpan).filter(TraceSpan.status == "ERROR")
    if trace_id:
        q = q.filter(TraceSpan.trace_id == trace_id)
    if execution_id:
        q = q.filter(TraceSpan.execution_id == execution_id)
    if job_id:
        q = q.filter(TraceSpan.metadata_json.like(f"%{job_id}%"))
    rows = q.order_by(TraceSpan.started_at.desc()).limit(min(limit,
                                                             200)).all()
    by_class = {}
    for row in rows:
        key = row.error_class or "UNKNOWN"
        by_class[key] = by_class.get(key, 0) + 1
    return {"failures": len(rows),
            "by_error_class": by_class,
            "sample_spans": [{"span_id": r.span_id, "span_type":
                              r.span_type, "trace_id": r.trace_id,
                              "execution_id": r.execution_id,
                              "error_class": r.error_class,
                              "latency_ms": r.latency_ms}
                             for r in rows[:20]]}


# ---------------------------------------------------------------------------
# SLO engine
# ---------------------------------------------------------------------------

def create_slo(db: Session, *, name: str, metric: str, operator: str,
               target_value: float, window_minutes: int = 60,
               burn_rate_threshold: float = 2.0,
               organization_id: Optional[int] = None) -> dict:
    from ..models.phase19 import SloDefinition
    row = SloDefinition(name=name, metric=metric, operator=operator,
                        target_value=target_value,
                        window_minutes=window_minutes,
                        burn_rate_threshold=burn_rate_threshold,
                        organization_id=organization_id)
    db.add(row)
    db.flush()
    return {"definition_id": row.id, "name": name, "metric": metric,
            "target": target_value}


def list_slos(db: Session, *, enabled_only: bool = True,
              limit: int = 100) -> list:
    from ..models.phase19 import SloDefinition
    q = db.query(SloDefinition)
    if enabled_only:
        q = q.filter(SloDefinition.enabled.is_(True))
    return q.order_by(SloDefinition.metric).limit(min(limit, 500)).all()


def evaluate_slo(db: Session, *, definition_id: int, metric_value: float,
                 ok_requests: int = 0, total_requests: int = 0,
                 now: Optional[datetime] = None) -> dict:
    """Evaluate one SLO window and persist the burn record.

    ``metric_value`` is compared against the target with the definition
    operator. Burn rate = consumed error budget per window (deterministic).
    """
    from ..models.phase19 import SloDefinition, SloBudgetWindow
    definition = db.query(SloDefinition).get(definition_id)
    if definition is None:
        raise KeyError("SLO definition not found")
    now = now or _utcnow()
    value = float(metric_value)
    target = float(definition.target_value)
    op = definition.operator
    compliant = {"<=": value <= target, ">=": value >= target,
                 "<": value < target, ">": value > target}.get(op, False)
    # consumed error budget: how far the metric is from the target
    if op in ("<=", "<"):
        budget = max(0.0, value - target)
        allowed = max(target * 0.05, 1e-9)
    else:
        budget = max(0.0, target - value)
        allowed = max(target * 0.05, 1e-9)
    burn_rate = round(budget / allowed, 3) if not compliant else 0.0
    if compliant:
        status = "COMPLIANT"
    else:
        # Burning = the window consumed error budget faster than the
        # configured threshold; below it the breach is contained.
        status = "BURNING" if burn_rate >= definition.burn_rate_threshold \
            else "BREACHED"
    row = SloBudgetWindow(
        definition_id=definition.id,
        window_start=now - timedelta(minutes=definition.window_minutes),
        window_end=now, budget_ratio=round(budget / allowed, 4),
        burn_rate=burn_rate)
    db.add(row)
    db.flush()
    return {"definition_id": definition.id, "metric": definition.metric,
            "value": value, "target": target, "operator": op,
            "status": status, "burn_rate": burn_rate}


def slo_health(db: Session, limit: int = 100) -> dict:
    from ..models.phase19 import SloBudgetWindow
    rows = list_slos(db)
    statuses = {}
    for definition in rows:
        latest = (db.query(SloBudgetWindow)
                  .filter(SloBudgetWindow.definition_id == definition.id)
                  .order_by(SloBudgetWindow.window_end.desc()).first())
        if latest is None:
            statuses[definition.name] = "NO_DATA"
        elif latest.budget_ratio <= 0:
            statuses[definition.name] = "COMPLIANT"
        elif latest.burn_rate >= definition.burn_rate_threshold:
            statuses[definition.name] = "BURNING"
        else:
            statuses[definition.name] = "BREACHED"
    return {"definitions": len(rows), "by_name": statuses}


# ---------------------------------------------------------------------------
# Alert deduplication
# ---------------------------------------------------------------------------

def dedupe_alerts(events: list[dict], *, cooldown_minutes: int = 30,
                  now: Optional[datetime] = None) -> dict:
    """Prevent alert storms: one alert per fingerprint per cooldown window.
    Pure function — deterministic across workers."""
    now = now or _utcnow()
    seen = {}
    kept = []
    for event in events:
        fingerprint = str(event.get("fingerprint")
                          or event.get("rule") or event.get("message"))
        fired_at = event.get("fired_at")
        if fired_at is None:
            fired_at = now
        prev = seen.get(fingerprint)
        if prev is not None and (fired_at - prev) < timedelta(
                minutes=cooldown_minutes):
            continue
        seen[fingerprint] = fired_at
        kept.append(event)
    return {"kept": kept, "suppressed": len(events) - len(kept),
            "cooldown_minutes": cooldown_minutes}
