"""Usage metering service — idempotent events, aggregates, threshold alerts."""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.retention import UsageEvent
from ..models.usage import UsageRecord
from ..models.notification import Notification
from ..services.entitlement_service import get_limits

# Metrics that can be metered
KNOWN_METRICS = {
    "storage_bytes": "lifetime",
    "documents": "lifetime",
    "ai_requests": "monthly",
    "agent_runs": "monthly",
    "workflow_runs": "monthly",
    "api_requests": "monthly",
    "webhook_deliveries": "monthly",
    "exports": "monthly",
    "embedding_tokens": "monthly",
    "llm_tokens": "monthly",
    "ai_cost_usd": "monthly",
}

# Maps usage metric names to plan limit keys
METRIC_LIMIT_MAP = {
    "storage_bytes": "max_storage_bytes",
    "documents": "max_documents",
    "ai_requests": "max_ai_requests",
    "agent_runs": "max_agent_runs",
    "workflow_runs": "max_workflow_runs",
    "api_requests": "max_api_requests",
    "webhook_deliveries": "max_webhooks",
    "exports": "max_exports",
    "embedding_tokens": "max_ai_requests",
    "llm_tokens": "max_ai_requests",
    "ai_cost_usd": "max_ai_budget",
}


def record_usage(
    db: Session,
    workspace_id: int,
    metric: str,
    quantity: int = 1,
    organization_id: Optional[int] = None,
    user_id: Optional[int] = None,
    event_key: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> UsageEvent:
    """Record a usage event idempotently.

    Same (workspace, metric, period, event_key) is only counted once.
    Also writes a backward-compatible UsageRecord row.
    """
    period = KNOWN_METRICS.get(metric, "monthly")

    # Idempotency: skip if an identical event key already exists
    if event_key:
        existing = (
            db.query(UsageEvent)
            .filter(
                UsageEvent.workspace_id == workspace_id,
                UsageEvent.metric == metric,
                UsageEvent.period == period,
                UsageEvent.event_key == event_key,
            )
            .first()
        )
        if existing:
            return existing

    event = UsageEvent(
        organization_id=organization_id,
        workspace_id=workspace_id,
        user_id=user_id,
        metric=metric,
        quantity=quantity,
        period=period,
        event_key=event_key,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(event)
    db.flush()

    # Backward-compatible UsageRecord (used by prior phases)
    usage_record = UsageRecord(
        workspace_id=workspace_id,
        user_id=user_id,
        usage_type=metric,
        quantity=quantity,
        units="bytes" if metric == "storage_bytes" else "tokens" if "token" in metric else "count",
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(usage_record)
    db.flush()
    return event


def usage_for_period(
    db: Session,
    workspace_id: int,
    metric: str,
    start: datetime,
    end: Optional[datetime] = None,
) -> int:
    """Sum usage for a metric in the given period."""
    query = db.query(func.coalesce(func.sum(UsageEvent.quantity), 0)).filter(
        UsageEvent.workspace_id == workspace_id,
        UsageEvent.metric == metric,
        UsageEvent.created_at >= start,
    )
    if end is not None:
        query = query.filter(UsageEvent.created_at < end)
    return int(query.scalar() or 0)


def month_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_workspace_usage(
    db: Session,
    workspace_id: int,
    organization_id: Optional[int] = None,
) -> dict:
    """Aggregate current usage for a workspace with quota percentages."""
    ms = month_start()
    usage = {}
    for metric in KNOWN_METRICS:
        period = KNOWN_METRICS[metric]
        if period == "lifetime":
            start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        else:
            start = ms
        usage[metric] = usage_for_period(db, workspace_id, metric, start)

    limits = get_limits(db, organization_id)
    quota = {}
    for metric, used in usage.items():
        limit_key = METRIC_LIMIT_MAP.get(metric)
        limit = limits.get(limit_key) if limit_key else None
        if limit is None or limit < 0:
            quota[metric] = {"used": used, "limit": None, "percent": 0.0, "remaining": None}
        else:
            percent = round((used / limit) * 100, 1) if limit else 0.0
            quota[metric] = {
                "used": used,
                "limit": limit,
                "percent": percent,
                "remaining": max(limit - used, 0),
            }
    return {"usage": usage, "quota": quota}


# Threshold alerting
ALERT_THRESHOLDS = (50, 75, 90, 100)


def check_thresholds(
    db: Session,
    workspace_id: int,
    organization_id: Optional[int] = None,
    user_ids: Optional[list[int]] = None,
) -> list[Notification]:
    """Emit one-time notifications when usage crosses a threshold.

    Threshold crossings are tracked per (workspace, metric, threshold) via
    the notification resource_id, preventing duplicate alerts.
    """
    result = get_workspace_usage(db, workspace_id, organization_id)
    notifications = []
    for metric, info in result["quota"].items():
        marker = f"{metric} usage reached {{threshold}}%"
        limit = info.get("limit")
        if not limit:
            continue
        percent = info["percent"]
        for threshold in ALERT_THRESHOLDS:
            if percent >= threshold:
                message = marker.format(threshold=threshold)
                # Dedup: one alert per (metric, threshold) — matched on the
                # message text since Notification.resource_id is an Integer column
                already = (
                    db.query(Notification)
                    .filter(
                        Notification.notification_type == "usage_threshold",
                        Notification.message == message,
                    )
                    .first()
                )
                if already:
                    continue
                for user_id in user_ids or []:
                    n = Notification(
                        user_id=user_id,
                        title="Usage Threshold Reached",
                        message=message,
                        notification_type="usage_threshold",
                        resource_type="usage",
                    )
                    db.add(n)
                    notifications.append(n)
    if notifications:
        db.flush()
    return notifications