"""Policy semantic intelligence API — normalize, compare, scan conflicts."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..services.permission_service import (
    require_permission, require_workspace_membership,
)
from ..services.policy2 import (
    normalize_statement, policy_conflict_verdict, detect_and_record_conflicts,
)
from ..models.phase15 import PolicyConflict

router = APIRouter(prefix="/policy-intelligence", tags=["policy-intelligence"])


class NormalizeRequest(BaseModel):
    statement: str


@router.post("/semanticize")
def semanticize_endpoint(
    data: NormalizeRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Deterministic semantic normalization of one policy statement."""
    return normalize_statement(data.statement)


class CompareRequest(BaseModel):
    statement_a: str
    statement_b: str


@router.post("/compare")
def compare_endpoint(
    data: CompareRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Verdict for two policy statements (never claims unsupported
    reasoning)."""
    return policy_conflict_verdict(data.statement_a, data.statement_b)


class ScanRequest(BaseModel):
    workspace_id: int


@router.post("/scan")
def scan_endpoint(
    data: ScanRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Detect real conflicts across workspace policy statements and create
    review items. Never auto-resolves."""
    require_permission(db, principal.user.id, "ai:execute",
                       workspace_id=data.workspace_id)
    result = detect_and_record_conflicts(db, data.workspace_id)
    db.commit()
    return result


@router.get("/conflicts")
def conflicts_endpoint(
    workspace_id: int,
    status: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    query = db.query(PolicyConflict).filter(
        PolicyConflict.workspace_id == workspace_id)
    if status:
        query = query.filter(PolicyConflict.status == status)
    rows = query.order_by(PolicyConflict.created_at.desc()).limit(
        min(limit, 200)).all()
    return {
        "items": [
            {
                "id": c.id,
                "conflict_type": c.conflict_type,
                "severity": c.severity,
                "description": c.description,
                "conditions": c.conditions_json,
                "policy_a_id": c.policy_a_id,
                "policy_b_id": c.policy_b_id,
                "status": c.status,
                "created_at": c.created_at,
            }
            for c in rows
        ],
        "count": len(rows),
    }
