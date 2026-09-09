"""AI cost engine API — cost aggregation, forecasting, budget enforcement."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..services.permission_service import require_workspace_membership, require_permission
from ..services.cost_engine import (
    cost_summary,
    forecast_spend,
    enforce_budget,
    get_budget,
    BUDGET_ACTIONS,
)

router = APIRouter(prefix="/costs", tags=["costs"])


class BudgetCheck(BaseModel):
    workspace_id: int
    estimated_cost_usd: float


def _require_workspace(db, workspace_id, principal):
    require_workspace_membership(db, workspace_id, principal.user.id)
    from ..models.workspace import Workspace
    return db.query(Workspace).filter(Workspace.id == workspace_id).first()


@router.get("/workspaces/{workspace_id}/summary")
def summary(
    workspace_id: int,
    days: int = 30,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Cost summary by model/feature for a workspace."""
    _require_workspace(db, workspace_id, principal)
    from datetime import datetime, timezone, timedelta
    since = datetime.now(timezone.utc) - timedelta(days=days)
    return cost_summary(db, workspace_id=workspace_id, since=since)


@router.get("/workspaces/{workspace_id}/forecast")
def forecast(
    workspace_id: int,
    lookback_days: int = 30,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    return forecast_spend(db, workspace_id, lookback_days=lookback_days)


@router.get("/workspaces/{workspace_id}/budget")
def budget(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    return {"budget": get_budget(db, workspace_id)}


@router.post("/enforce")
def check_budget(
    data: BudgetCheck,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Enforce budget BEFORE an expensive operation (BLOCK/REQUIRE_APPROVAL/
    DOWNGRADE_MODEL/QUEUE)."""
    _require_workspace(db, data.workspace_id, principal)
    return enforce_budget(db, data.workspace_id, data.estimated_cost_usd, principal.user.id)