"""AI trace platform 2.0 — spans, correlation ids, failure classification.

Trace spans never store raw prompts by default: only an optional truncated
``input_summary`` is persisted. Correlation ids flow

    request_id → execution_id → workflow_id → node_execution_id → provider_call_id

and are recorded on the relevant rows so operators can follow one request
across every subsystem.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase16 import TraceSpan, SPAN_TYPES, SPAN_STATUSES

logger = logging.getLogger(__name__)

RETRYABLE_CLASSES = (
    "provider", "timeout", "rate_limit", "dependency", "storage", "database",
)


def classify_failure(exc: Exception) -> str:
    """Normalize arbitrary exceptions into a stable failure class."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timeout" in message:
        return "timeout"
    if "ratelimit" in name or "rate limit" in message or "429" in message:
        return "rate_limit"
    if name in ("httpxstatuserror", "llmprovidererror", "providererror"):
        return "provider"
    if "status 5" in message or message.startswith(("50", "502", "503",
                                                     "504")):
        return "provider"
    if name in ("validationerror", "pydanticvalidationerror", "valueerror",
                "httpvalidationerror"):
        return "validation"
    if name in ("permissionerror", "forbidden", "authorizationerror",
                "notfounderror", "httpexception", "copilotscopeerror"):
        return "authorization"
    if "tool" in name:
        return "tool"
    if "stor" in name or "file" in name:
        return "storage"
    if "database" in name or "db" in name or "sqlalchemy" in name:
        return "database"
    if "dependency" in name or "retry" in name:
        return "dependency"
    return "unknown"


def is_retryable(error_class: str) -> bool:
    return error_class in RETRYABLE_CLASSES


def new_trace_id() -> str:
    return uuid.uuid4().hex[:32]


def new_span_id() -> str:
    return uuid.uuid4().hex[:32]


def start_span(
    db: Session,
    workspace_id: int,
    span_type: str,
    trace_id: Optional[str] = None,
    organization_id: Optional[int] = None,
    execution_id: Optional[str] = None,
    workflow_execution_id: Optional[str] = None,
    node_execution_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    input_summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> TraceSpan:
    if span_type not in SPAN_TYPES:
        raise ValueError(f"Unknown span type: {span_type}")
    import json
    span = TraceSpan(
        span_id=new_span_id(),
        trace_id=trace_id or new_trace_id(),
        parent_span_id=parent_span_id,
        workspace_id=workspace_id,
        organization_id=organization_id,
        execution_id=execution_id,
        workflow_execution_id=workflow_execution_id,
        node_execution_id=node_execution_id,
        span_type=span_type,
        status="OK",
        model=model,
        provider=provider,
        input_summary=(input_summary or "")[:500] or None,
        metadata_json=json.dumps(metadata, default=str) if metadata else None,
    )
    db.add(span)
    db.flush()
    return span


def end_span(
    db: Session,
    span: TraceSpan,
    status: str = "OK",
    error: Optional[Exception] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
) -> TraceSpan:
    if status not in SPAN_STATUSES or (error is not None and status == "OK"):
        status = "ERROR"
    now = datetime.now(timezone.utc)
    span.completed_at = now
    span.status = status
    if error is not None:
        span.error_class = classify_failure(error)
        span.input_summary = None  # never keep content on error paths
    if span.started_at is not None:
        start = span.started_at
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        span.latency_ms = round((now - start).total_seconds() * 1000, 1)
    if model:
        span.model = model
    if provider:
        span.provider = provider
    if input_tokens is not None:
        span.input_tokens = input_tokens
    if output_tokens is not None:
        span.output_tokens = output_tokens
    if cost_usd is not None:
        span.cost_usd = cost_usd
    db.flush()
    return span


def correlate(request_id: Optional[str], execution_id: Optional[str] = None,
              workflow_id: Optional[str] = None,
              node_execution_id: Optional[str] = None) -> dict:
    """Build a correlation chain (ids are safe to propagate/log)."""
    return {
        "request_id": request_id or new_trace_id(),
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "node_execution_id": node_execution_id,
    }


def list_traces(
    db: Session,
    workspace_id: int,
    trace_id: Optional[str] = None,
    span_type: Optional[str] = None,
    limit: int = 50,
) -> list[TraceSpan]:
    query = db.query(TraceSpan).filter(TraceSpan.workspace_id == workspace_id)
    if trace_id:
        query = query.filter(TraceSpan.trace_id == trace_id)
    if span_type:
        query = query.filter(TraceSpan.span_type == span_type)
    return query.order_by(TraceSpan.started_at.desc()).limit(min(limit, 200)).all()


def trace_summary(db: Session, workspace_id: Optional[int] = None) -> dict:
    """Tenant-scoped trace rollup (no content)."""
    query = db.query(TraceSpan)
    if workspace_id is not None:
        query = query.filter(TraceSpan.workspace_id == workspace_id)
    spans = query.all()
    by_type: dict = {}
    errors = 0
    ok = 0
    latency = []
    cost = 0.0
    for s in spans:
        by_type[s.span_type] = by_type.get(s.span_type, 0) + 1
        if s.status == "OK":
            ok += 1
        else:
            errors += 1
        if s.latency_ms is not None:
            latency.append(s.latency_ms)
        if s.cost_usd is not None:
            cost += s.cost_usd
    return {
        "spans": len(spans),
        "ok": ok,
        "errors": errors,
        "by_type": by_type,
        "avg_latency_ms": round(sum(latency) / len(latency), 1) if latency else 0.0,
        "total_cost_usd": round(cost, 4),
    }
