"""Usage analytics and quota API."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..services.permission_service import require_permission
from ..services.usage_service import get_workspace_usage, check_thresholds
from ..services.entitlement_service import (
    get_limits,
    get_features,
    remaining_quota,
    ensure_default_plan,
    get_plan_for_organization,
)

router = APIRouter(prefix="/usage", tags=["usage"])


def _require_workspace(db: Session, workspace_id: int, principal: AuthPrincipal) -> Workspace:
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "usage:view", workspace_id=workspace_id)
    return workspace


@router.get("/workspaces/{workspace_id}")
def workspace_usage(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Current usage and quota percentages for a workspace."""
    workspace = _require_workspace(db, workspace_id, principal)
    result = get_workspace_usage(db, workspace_id, workspace.organization_id)
    return result


@router.get("/workspaces/{workspace_id}/limits")
def workspace_limits(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Effective plan limits and features for a workspace."""
    workspace = _require_workspace(db, workspace_id, principal)
    org_id = workspace.organization_id
    return {
        "limits": get_limits(db, org_id),
        "features": get_features(db, org_id),
        "plan": get_plan_for_organization(db, org_id).code,
    }


@router.get("/workspaces/{workspace_id}/quota/{metric}")
def metric_quota(
    workspace_id: int,
    metric: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Remaining quota for a single metric."""
    workspace = _require_workspace(db, workspace_id, principal)
    usage = get_workspace_usage(db, workspace_id, workspace.organization_id)
    used = usage["usage"].get(metric, 0)
    info = remaining_quota(db, metric, used, workspace.organization_id)
    if info.get("limit") is None and metric not in usage["usage"]:
        raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")
    return info


@router.post("/workspaces/{workspace_id}/check-thresholds")
def run_threshold_check(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Re-check usage thresholds and emit one-time alerts."""
    workspace = _require_workspace(db, workspace_id, principal)
    # Ensure default plan exists so limits resolve
    ensure_default_plan(db)
    notifications = check_thresholds(db, workspace_id, workspace.organization_id, user_ids=[principal.user.id])
    db.commit()
    return {"notifications_created": len(notifications)}