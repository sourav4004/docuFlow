"""Phase 19 — enterprise analytics.

Aggregate-only analytics at organization/workspace/user level. Analytics
NEVER expose private user data or document content — only counts, sums and
rates, and the ``privacy_guard`` strips any non-aggregate field defensively.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

AGGREGATE_ONLY = ("count", "total", "avg", "rate", "p95_ms")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def privacy_guard(payload: dict) -> dict:
    """Defense-in-depth: keep only aggregate fields in analytics payloads."""
    if isinstance(payload, dict):
        return {k: privacy_guard(v) for k, v in payload.items()
                if any(marker in str(k) for marker in
                       ("count", "total", "avg", "rate", "p95", "sum",
                        "by_", "days"))}
    if isinstance(payload, list):
        return [privacy_guard(item) for item in payload]
    return payload


def _workspace_ids(db: Session, organization_id: int) -> list[int]:
    from ..models.workspace import Workspace
    rows = db.query(Workspace.id).filter(
        Workspace.organization_id == organization_id).limit(5000).all()
    return [row[0] for row in rows]


def org_analytics(db: Session, *, organization_id: int,
                  days: int = 30) -> dict:
    """Aggregate-only organization analytics."""
    from ..models.document import Document
    from ..models.workspace import Workspace
    from ..models.ai_execution import AIExecution
    workspace_ids = _workspace_ids(db, organization_id)
    since = _utcnow() - timedelta(days=days)
    ws_rows = db.query(Workspace).filter(
        Workspace.id.in_(workspace_ids)).limit(1000).all()
    doc_count = (db.query(Document)
                 .filter(Document.workspace_id.in_(workspace_ids))
                 .count())
    storage_bytes = sum(int(d.file_size or 0)
                        for d in db.query(Document.file_size).filter(
                            Document.workspace_id.in_(workspace_ids))
                        .limit(10000).all())
    executions = (db.query(AIExecution)
                  .filter(AIExecution.organization_id == organization_id,
                          AIExecution.created_at >= since)
                  .limit(10000).all())
    total_tokens = sum(int(e.total_tokens or 0) for e in executions)
    total_cost = sum(float(e.estimated_cost or 0.0) for e in executions)
    succeeded = sum(1 for e in executions if e.status == "COMPLETED")
    result = {
        "days": days, "workspaces": len(ws_rows),
        "document_count": doc_count,
        "storage_bytes": storage_bytes,
        "ai_requests": len(executions),
        "ai_success_rate": round(succeeded / len(executions), 4)
        if executions else 0.0,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "avg_ai_latency_ms": round(
            sum(float(e.latency_ms or 0.0) for e in executions)
            / len(executions), 1) if executions else None,
    }
    return result


def workspace_analytics(db: Session, *, workspace_id: int,
                        days: int = 30) -> dict:
    from ..models.document import Document
    from ..models.ai_execution import AIExecution
    from ..models.phase18 import SearchAnalyticsEvent
    since = _utcnow() - timedelta(days=days)
    docs = db.query(Document).filter(
        Document.workspace_id == workspace_id).limit(10000).all()
    ready = sum(1 for d in docs if d.status == "READY")
    executions = (db.query(AIExecution)
                  .filter(AIExecution.workspace_id == workspace_id,
                          AIExecution.created_at >= since)
                  .limit(5000).all())
    search_events = (db.query(SearchAnalyticsEvent)
                     .filter(SearchAnalyticsEvent.workspace_id ==
                             workspace_id,
                             SearchAnalyticsEvent.created_at >= since)
                     .count())
    return {
        "days": days,
        "document_count": len(docs),
        "ready_rate": round(ready / len(docs), 4) if docs else 0.0,
        "storage_bytes": sum(int(d.file_size or 0) for d in docs),
        "ai_requests": len(executions),
        "ai_cost_usd": round(sum(float(e.estimated_cost or 0.0)
                                 for e in executions), 4),
        "search_events": search_events,
        "workflow_count": _count_workflows(db, workspace_id),
    }


def _count_workflows(db: Session, workspace_id: int) -> int:
    from ..models.phase17 import WorkflowRun
    return db.query(WorkflowRun).filter(
        WorkflowRun.workspace_id == workspace_id).count()


def user_analytics(db: Session, *, workspace_id: int, user_id: int,
                   days: int = 30) -> dict:
    """A user may see their OWN usage only — never other users'."""
    from ..models.ai_execution import AIExecution
    from ..models.document import Document
    since = _utcnow() - timedelta(days=days)
    executions = (db.query(AIExecution)
                  .filter(AIExecution.workspace_id == workspace_id,
                          AIExecution.user_id == user_id,
                          AIExecution.created_at >= since)
                  .limit(5000).all())
    docs = db.query(Document).filter(
        Document.workspace_id == workspace_id,
        Document.user_id == user_id).count()
    return {
        "days": days,
        "my_ai_requests": len(executions),
        "my_cost_usd": round(sum(float(e.estimated_cost or 0.0)
                                 for e in executions), 4),
        "my_document_count": docs,
        "my_avg_latency_ms": round(
            sum(float(e.latency_ms or 0.0) for e in executions)
            / len(executions), 1) if executions else None,
        "privacy_note": "user analytics are scoped to the requesting user",
    }
