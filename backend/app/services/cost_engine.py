"""AI cost engine 2.0 — token/cost tracking, forecasting, budget enforcement.

Aggregates cost by user/workspace/organization/workflow/feature/model,
forecasts future spend from historical usage (clearly labeled as estimates),
and enforces budgets with configurable actions (BLOCK / REQUIRE_APPROVAL /
DOWNGRADE_MODEL / QUEUE). Never silently incurs unlimited cost.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.ai_execution import AIExecution
from ..models.billing import Plan, Subscription
from ..models.usage import UsageRecord
from ..services.audit_service import log_audit_event

BUDGET_ACTIONS = ("BLOCK", "REQUIRE_APPROVAL", "DOWNGRADE_MODEL", "QUEUE")
DEFAULT_BUDGET_ACTIONS = {"BLOCK": 1.0, "REQUIRE_APPROVAL": 0.8, "DOWNGRADE_MODEL": 0.9}


def record_cost(
    db: Session,
    execution: AIExecution,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> AIExecution:
    """Record token/cost on an execution (idempotent — single completion)."""
    if execution.actual_cost and execution.actual_cost > 0:
        return execution  # already costed
    execution.input_tokens = input_tokens
    execution.output_tokens = output_tokens
    execution.total_tokens = input_tokens + output_tokens
    execution.actual_cost = cost_usd
    if provider:
        execution.provider = provider
    if model:
        execution.model = model
    db.flush()
    return execution


def cost_summary(
    db: Session,
    workspace_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    since: Optional[datetime] = None,
) -> dict:
    """Aggregate cost by dimension (workspace/feature/model)."""
    from sqlalchemy import func
    query = db.query(AIExecution)
    if workspace_id is not None:
        query = query.filter(AIExecution.workspace_id == workspace_id)
    if organization_id is not None:
        query = query.filter(AIExecution.organization_id == organization_id)
    if since is not None:
        query = query.filter(AIExecution.created_at >= since)

    executions = query.limit(5000).all()
    total_cost = sum(e.actual_cost or 0 for e in executions)
    total_tokens = sum(e.total_tokens or 0 for e in executions)
    input_tokens = sum(e.input_tokens or 0 for e in executions)
    output_tokens = sum(e.output_tokens or 0 for e in executions)

    by_model: dict[str, float] = {}
    by_feature: dict[str, float] = {}
    for e in executions:
        model = e.model or "unknown"
        by_model[model] = by_model.get(model, 0) + (e.actual_cost or 0)
        feature = e.execution_type or "unknown"
        by_feature[feature] = by_feature.get(feature, 0) + (e.actual_cost or 0)

    return {
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "execution_count": len(executions),
        "by_model": by_model,
        "by_feature": by_feature,
    }


def forecast_spend(
    db: Session,
    workspace_id: int,
    lookback_days: int = 30,
) -> dict:
    """Forecast future spend from historical usage (estimate, not guarantee)."""
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    historical = cost_summary(db, workspace_id=workspace_id, since=since)
    daily = historical["total_cost_usd"] / max(lookback_days, 1)
    monthly = daily * 30
    return {
        "lookback_days": lookback_days,
        "historical_cost_usd": round(historical["total_cost_usd"], 6),
        "daily_estimate_usd": round(daily, 6),
        "monthly_estimate_usd": round(monthly, 6),
        "confidence_range": {
            "low_usd": round(monthly * 0.7, 6),
            "high_usd": round(monthly * 1.5, 6),
        },
        "estimate_is_guaranteed": False,
    }


def get_budget(db: Session, workspace_id: int) -> Optional[dict]:
    """Resolve the effective AI budget for a workspace (plan or custom).

    Plans store limits in ``limits_json``; the ``ai_budget_usd`` key is the
    monthly AI cost ceiling. Subscriptions are organization-scoped, so the
    workspace's organization is resolved first.
    """
    from ..models.workspace import Workspace
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        return None
    subscription = (
        db.query(Subscription)
        .filter(
            Subscription.organization_id == workspace.organization_id,
            Subscription.status.in_(("ACTIVE", "TRIALING")),
        )
        .first()
    )
    if subscription:
        plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
        if plan:
            limits = plan.limits
            budget = limits.get("ai_budget_usd") or limits.get("max_ai_budget_usd")
            if budget:
                return {"source": "plan", "name": plan.name, "monthly_limit_usd": float(budget)}
    return None


def enforce_budget(
    db: Session,
    workspace_id: int,
    estimated_cost_usd: float,
    actor_id: int,
) -> dict:
    """Enforce budget before execution.

    Returns {decision, action, remaining, message}. When the limit would be
    exceeded, returns a policy action (BLOCK/REQUIRE_APPROVAL/DOWNGRADE_MODEL/QUEUE)
    — the caller must honor it; execution never silently proceeds.
    """
    budget = get_budget(db, workspace_id)
    if budget is None or not budget.get("monthly_limit_usd"):
        return {"decision": "ALLOW", "action": None, "remaining": None, "message": "No budget configured"}

    limit = float(budget["monthly_limit_usd"])
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    used = cost_summary(db, workspace_id=workspace_id, since=month_start)["total_cost_usd"]
    remaining = limit - used

    if remaining >= estimated_cost_usd:
        return {"decision": "ALLOW", "action": None, "remaining": round(remaining, 6), "message": "Within budget"}

    # Exceeded → pick action by how far over
    over_ratio = (used + estimated_cost_usd) / limit
    if over_ratio >= 1.2:
        action = "BLOCK"
        message = f"Budget exceeded — execution blocked (estimated ${estimated_cost_usd:.4f})"
    elif over_ratio >= 1.0:
        action = "REQUIRE_APPROVAL"
        message = f"Budget would be exceeded — approval required (estimated ${estimated_cost_usd:.4f})"
    else:
        action = "DOWNGRADE_MODEL"
        message = "Budget tight — downgrade model requested"

    log_audit_event(
        db, event_type="budget", event_action="enforce",
        user_id=actor_id, resource_type="workspace", resource_id=workspace_id,
        details=f"Budget enforcement: {action} (used ${used:.4f}/{limit:.4f}, est. ${estimated_cost_usd:.4f})",
    )
    return {
        "decision": "RESTRICTED",
        "action": action,
        "remaining": round(remaining, 6),
        "message": message,
        "used_usd": round(used, 6),
        "limit_usd": limit,
    }