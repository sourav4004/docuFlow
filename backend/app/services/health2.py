"""Health 2.0 — organization health aggregation (admin-scoped, aggregate only).

Organization health combines workspace health, AI spend, automation health,
security signals, knowledge risks, usage, and provider health. It never
exposes document contents across workspaces.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.workspace import Workspace
from ..models.phase15 import ProviderHealth, KnowledgeGap, PolicyConflict
from ..models.ai_action import AIAction
from ..models.security_event import SecurityEvent
from ..services.health_service import workspace_health, ai_usage_summary
from ..services.cost_engine import cost_summary
from ..services.model_policy import get_effective_policy


def organization_health(
    db: Session,
    organization_id: int,
    admin_user_id: Optional[int] = None,
) -> dict:
    """Aggregate organization health (requires org-admin; caller checks role)."""
    workspaces = (
        db.query(Workspace)
        .filter(Workspace.organization_id == organization_id)
        .limit(500)
        .all()
    )

    per_workspace = []
    total_docs = 0
    total_conflicts = 0
    total_gaps = 0
    total_actions = 0
    ai_spend = 0.0
    for ws in workspaces:
        try:
            health = workspace_health(db, ws.id)
        except Exception:  # noqa: BLE001 — isolated workspace must not break the org view
            health = {}
        per_workspace.append({
            "workspace_id": ws.id,
            "name": ws.name,
            "health": health,
        })
        total_docs += health.get("total_documents", 0)
        total_conflicts += (
            db.query(PolicyConflict)
            .filter(PolicyConflict.workspace_id == ws.id, PolicyConflict.status == "OPEN")
            .count()
        )
        total_gaps += (
            db.query(KnowledgeGap)
            .filter(KnowledgeGap.workspace_id == ws.id, KnowledgeGap.status == "OPEN")
            .count()
        )
        total_actions += (
            db.query(AIAction)
            .filter(AIAction.workspace_id == ws.id, AIAction.status.in_(("SUGGESTED", "APPROVAL_REQUIRED", "APPROVED")))
            .count()
        )
        ai_spend += cost_summary(db, workspace_id=ws.id)["total_cost_usd"]

    # Provider health (global — providers are shared infrastructure)
    providers = db.query(ProviderHealth).limit(100).all()
    provider_status = {
        "degraded": sum(1 for p in providers if p.status == "DEGRADED"),
        "down": sum(1 for p in providers if p.status == "DOWN"),
        "up": sum(1 for p in providers if p.status == "UP"),
        "total": len(providers),
    }

    # Security signals (org-scoped)
    security_events = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.organization_id == organization_id,
            SecurityEvent.created_at >= datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        )
        .count()
    )

    policy = get_effective_policy(db, organization_id)

    return {
        "organization_id": organization_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspaces": per_workspace,
        "totals": {
            "workspaces": len(workspaces),
            "documents": total_docs,
            "open_conflicts": total_conflicts,
            "open_gaps": total_gaps,
            "pending_ai_actions": total_actions,
            "ai_spend_usd": round(ai_spend, 6),
        },
        "provider_health": provider_status,
        "security_signals_month": security_events,
        "model_policy_active": bool(policy.allowed_providers or policy.blocked_providers or policy.allowed_models),
        "note": "Aggregate metrics only — document contents never cross workspaces",
    }