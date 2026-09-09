"""AI quality platform 2.0 — quality metrics, feedback analytics, and the
AI quality dashboard. All metrics are tenant scoped; there is no cross-tenant
aggregation. Feedback analytics never trains models from private data.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import AIQualityMetric, ProviderHealth
from ..models.ai_execution import AIExecution
from ..models.search_intel import AIFeedback
from ..models.usage import UsageRecord


def record_quality_metric(
    db: Session,
    workspace_id: int,
    metric_type: str,
    value: float,
    feature: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    organization_id: Optional[int] = None,
) -> AIQualityMetric:
    metric = AIQualityMetric(
        workspace_id=workspace_id,
        organization_id=organization_id,
        metric_type=metric_type,
        feature=feature,
        model=model,
        provider=provider,
        value=value,
        period_start=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
    )
    db.add(metric)
    db.flush()
    return metric


def quality_dashboard(
    db: Session,
    workspace_id: int,
    days: int = 30,
) -> dict:
    """Aggregate AI quality signals for the admin dashboard."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    executions = (
        db.query(AIExecution)
        .filter(AIExecution.workspace_id == workspace_id, AIExecution.created_at >= since)
        .limit(5000)
        .all()
    )
    total = len(executions)
    failed = sum(1 for e in executions if e.status in ("FAILED", "TIMED_OUT"))
    success = total - failed
    failure_rate = round(failed / total, 4) if total else None

    latency = [e.latency_ms for e in executions if e.latency_ms is not None]
    avg_latency_ms = round(sum(latency) / len(latency), 1) if latency else None

    total_tokens = sum(e.total_tokens or 0 for e in executions)
    total_cost = round(sum(e.actual_cost or 0 for e in executions), 6)

    feedback_rows = (
        db.query(AIFeedback)
        .filter(AIFeedback.workspace_id == workspace_id, AIFeedback.created_at >= since)
        .limit(5000)
        .all()
    )
    positive = sum(1 for f in feedback_rows if f.rating == "thumbs_up")
    negative = sum(1 for f in feedback_rows if f.rating == "thumbs_down")
    acceptance = round(positive / len(feedback_rows), 4) if feedback_rows else None

    # Tool failures from execution steps
    tool_failures = 0
    for e in executions:
        for step in e.steps or []:
            if step.step_type == "tool_call" and step.status == "failed":
                tool_failures += 1

    confidence_metrics = (
        db.query(AIQualityMetric)
        .filter(
            AIQualityMetric.workspace_id == workspace_id,
            AIQualityMetric.created_at >= since,
            AIQualityMetric.metric_type.in_(("grounding_rate", "citation_correctness")),
        )
        .all()
    )
    grounding = None
    for m in confidence_metrics:
        if m.metric_type == "grounding_rate":
            grounding = m.value

    return {
        "period_days": days,
        "executions": {
            "total": total,
            "success": success,
            "failed": failed,
            "failure_rate": failure_rate,
            "avg_latency_ms": avg_latency_ms,
        },
        "cost": {
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
        },
        "feedback": {
            "total": len(feedback_rows),
            "positive": positive,
            "negative": negative,
            "acceptance_rate": acceptance,
        },
        "tool_failures": tool_failures,
        "quality_metrics": {
            "grounding_rate": grounding,
        },
        "note": "Metrics are workspace-scoped; no cross-tenant aggregation",
    }


def feedback_analytics(
    db: Session,
    workspace_id: int,
    days: int = 30,
) -> dict:
    """Workspace-level feedback analytics (never trains models from data)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (
        db.query(AIFeedback)
        .filter(AIFeedback.workspace_id == workspace_id, AIFeedback.created_at >= since)
        .limit(5000)
        .all()
    )
    by_rating: dict[str, int] = {}
    citation_issues = 0
    by_category: dict[str, int] = {}
    for f in rows:
        by_rating[f.rating] = by_rating.get(f.rating, 0) + 1
        category = (f.category or "").strip()
        if category:
            by_category[category] = by_category.get(category, 0) + 1
        if category == "citation_issue":
            citation_issues += 1

    return {
        "period_days": days,
        "total": len(rows),
        "by_rating": by_rating,
        "citation_issues": citation_issues,
        "by_category": by_category,
        "acceptance_rate": round(by_rating.get("thumbs_up", 0) / len(rows), 4) if rows else None,
    }


def provider_health_summary(db: Session) -> dict:
    """Provider health overview (shared infrastructure, no tenant data)."""
    providers = db.query(ProviderHealth).limit(100).all()
    return {
        "total": len(providers),
        "up": sum(1 for p in providers if p.status == "UP"),
        "degraded": sum(1 for p in providers if p.status == "DEGRADED"),
        "down": sum(1 for p in providers if p.status == "DOWN"),
        "circuit_open": sum(1 for p in providers if p.circuit_state == "OPEN"),
        "providers": [
            {
                "provider": p.provider,
                "model": p.model,
                "status": p.status,
                "circuit_state": p.circuit_state,
                "success_count": p.success_count,
                "failure_count": p.failure_count,
                "avg_latency_ms": p.avg_latency_ms,
                "last_error": (p.last_error or "")[:200],
            }
            for p in providers
        ],
    }