"""AI governance API — model policy, provider health, quality dashboard,
organization health, sensitivity routing, and data minimization."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..models.organization import Organization
from ..services.permission_service import (
    require_permission,
    require_workspace_membership,
    get_organization_role,
)
from ..services.model_policy import (
    get_effective_policy,
    save_policy,
    route_model,
    minimize_for_external_call,
    ModelPolicyError,
)
from ..services.quality_service import (
    quality_dashboard,
    feedback_analytics,
    provider_health_summary,
    record_quality_metric,
)
from ..services.health2 import organization_health
from ..services.workflow_reliability import record_provider_call, allow_provider_call
from ..services.copilot2 import _require_org_admin, CopilotScopeError


def _require_org_admin_or_403(db, organization_id, principal):
    """Org-admin gate mapping the domain error to HTTP 403."""
    try:
        _require_org_admin(db, organization_id, principal.user.id)
    except CopilotScopeError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

router = APIRouter(prefix="/governance", tags=["governance"])


class ModelPolicyUpdate(BaseModel):
    allowed_providers: list = []
    blocked_providers: list = []
    allowed_models: list = []
    blocked_models: list = []
    max_cost_usd: Optional[float] = None
    max_context_tokens: Optional[int] = None
    sensitivity_policy: dict = {}
    require_approval_for: list = []


class RouteRequest(BaseModel):
    workspace_id: int
    sensitivity: str
    provider: str
    model: str
    estimated_cost_usd: float = 0.0


class ProviderCallRecord(BaseModel):
    provider: str
    model: str
    success: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@router.get("/organizations/{organization_id}/policy")
def org_policy(
    organization_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Effective model policy for an organization (admin)."""
    _require_org_admin_or_403(db, organization_id, principal)
    policy = get_effective_policy(db, organization_id)
    return {
        "allowed_providers": policy.allowed_providers,
        "blocked_providers": policy.blocked_providers,
        "allowed_models": policy.allowed_models,
        "blocked_models": policy.blocked_models,
        "max_cost_usd": policy.max_cost_usd,
        "max_context_tokens": policy.max_context_tokens,
        "sensitivity_policy": policy.sensitivity_policy,
        "require_approval_for": policy.require_approval_for,
    }


@router.put("/organizations/{organization_id}/policy")
def update_org_policy(
    organization_id: int,
    data: ModelPolicyUpdate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_org_admin_or_403(db, organization_id, principal)
    policy = {
        "allowed_providers": data.allowed_providers,
        "blocked_providers": data.blocked_providers,
        "allowed_models": data.allowed_models,
        "blocked_models": data.blocked_models,
        "max_cost_usd": data.max_cost_usd,
        "max_context_tokens": data.max_context_tokens,
        "sensitivity_policy": data.sensitivity_policy,
        "require_approval_for": data.require_approval_for,
    }
    save_policy(db, organization_id, policy, principal.user.id)
    db.commit()
    return policy


@router.post("/route")
def route(
    data: RouteRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Server-side model routing honoring org policy + sensitivity."""
    workspace = db.query(Workspace).filter(Workspace.id == data.workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_workspace_membership(db, data.workspace_id, principal.user.id)
    try:
        return route_model(
            db, workspace.organization_id, data.workspace_id, data.sensitivity,
            data.provider, data.model, estimated_cost_usd=data.estimated_cost_usd,
            actor_id=principal.user.id,
        )
    except ModelPolicyError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/minimize")
def minimize(
    data: dict,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Return the data-minimized payload (allow-listed keys only)."""
    return {"minimized": minimize_for_external_call(data)}


@router.post("/provider-calls", status_code=201)
def record_provider_call_endpoint(
    data: ProviderCallRecord,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Record a provider call outcome for health/circuit tracking (admin)."""
    require_permission(db, principal.user.id, "ai:manage_settings")
    health = record_provider_call(
        db, data.provider, data.model, data.success,
        latency_ms=data.latency_ms, error=data.error,
    )
    db.commit()
    return {
        "provider": health.provider,
        "model": health.model,
        "status": health.status,
        "circuit_state": health.circuit_state,
        "consecutive_failures": health.consecutive_failures,
        "avg_latency_ms": health.avg_latency_ms,
    }


@router.get("/provider-health")
def provider_health(
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "ai:manage_settings")
    return provider_health_summary(db)


@router.get("/quality/{workspace_id}")
def quality(
    workspace_id: int,
    days: int = 30,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "ai:manage_settings", workspace_id=workspace_id)
    return quality_dashboard(db, workspace_id, days=days)


@router.get("/feedback/{workspace_id}")
def feedback(
    workspace_id: int,
    days: int = 30,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "usage:view", workspace_id=workspace_id)
    return feedback_analytics(db, workspace_id, days=days)


@router.get("/organizations/{organization_id}/health")
def org_health(
    organization_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Organization health dashboard (aggregates only, org admin)."""
    _require_org_admin_or_403(db, organization_id, principal)
    return organization_health(db, organization_id, admin_user_id=principal.user.id)


@router.post("/quality/{workspace_id}/metrics")
def add_quality_metric(
    workspace_id: int,
    metric_type: str,
    value: float,
    feature: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "ai:manage_settings", workspace_id=workspace_id)
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    metric = record_quality_metric(
        db, workspace_id, metric_type, value, feature=feature,
        model=model, provider=provider,
        organization_id=workspace.organization_id if workspace else None,
    )
    db.commit()
    return {"id": metric.id, "metric_type": metric.metric_type, "value": metric.value}