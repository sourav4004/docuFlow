"""Automation 2.0 API — simulator, version diff, risk engine, node executions,
compensation, and failure recovery plans."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workflow_version import WorkflowVersion
from ..services.permission_service import require_workspace_membership, require_permission
from ..services.workflow_intel import (
    validate_definition,
    create_workflow_version,
    activate_workflow,
    list_versions,
    WorkflowValidationError,
)
from ..services.workflow_reliability import (
    simulate_workflow,
    workflow_risk,
    version_diff,
    begin_node_execution,
    complete_node_execution,
    fail_node_execution,
    record_side_effect,
    compensate_side_effect,
    recovery_plan,
    WorkflowReliabilityError,
)

router = APIRouter(prefix="/automation", tags=["automation"])


class WorkflowSimulate(BaseModel):
    definition: dict


class NodeExecutionStart(BaseModel):
    workspace_id: int
    workflow_execution_id: str
    workflow_id: str
    node_id: str
    input_data: Optional[dict] = None


class NodeExecutionComplete(BaseModel):
    output_reference: Optional[str] = None
    error: Optional[str] = None


class SideEffectRecord(BaseModel):
    workspace_id: int
    workflow_execution_id: str
    node_id: str
    side_effect_type: str
    metadata: Optional[dict] = None


@router.post("/simulate")
def simulate(
    data: WorkflowSimulate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Side-effect-free workflow simulation with cost/risk estimates."""
    try:
        return simulate_workflow(data.definition)
    except WorkflowValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/validate")
def validate_workflow_definition(
    data: WorkflowSimulate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Validate a workflow definition (permissions, cycles, inputs)."""
    try:
        definition = validate_definition(data.definition)
        risk = workflow_risk(definition)
    except WorkflowValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"valid": True, "risk": risk}


@router.get("/risk")
def risk_classify(
    definition: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    import json
    try:
        parsed = json.loads(definition)
    except ValueError:
        raise HTTPException(status_code=400, detail="definition must be JSON")
    try:
        parsed = validate_definition(parsed)
        return workflow_risk(parsed)
    except WorkflowValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/workflows/{workflow_id}/versions/diff")
def workflow_version_diff(
    workflow_id: str,
    from_version: int,
    to_version: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    versions = list_versions(db, workflow_id)
    by_version = {v.version: v for v in versions}
    v_old = by_version.get(from_version)
    v_new = by_version.get(to_version)
    if not v_old or not v_new:
        raise HTTPException(status_code=404, detail="Version not found")
    require_workspace_membership(db, v_old.workspace_id, principal.user.id)
    return version_diff(v_old, v_new)


@router.post("/node-executions/begin", status_code=201)
def begin_node(
    data: NodeExecutionStart,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Begin a node attempt (unique execution+node+attempt — never double-runs)."""
    require_workspace_membership(db, data.workspace_id, principal.user.id)
    try:
        record = begin_node_execution(
            db, data.workspace_id, data.workflow_execution_id,
            data.workflow_id, data.node_id, data.input_data,
        )
    except WorkflowReliabilityError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.commit()
    return {
        "id": record.id,
        "attempt": record.attempt,
        "status": record.status,
        "input_hash": record.input_hash,
        "started_at": record.started_at,
    }


@router.post("/node-executions/{record_id}/complete")
def complete_node(
    record_id: int,
    data: NodeExecutionComplete,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..models.phase15 import WorkflowNodeExecution
    record = db.query(WorkflowNodeExecution).filter(WorkflowNodeExecution.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Node execution not found")
    require_workspace_membership(db, record.workspace_id, principal.user.id)
    if data.error:
        record = fail_node_execution(db, record, data.error)
    else:
        record = complete_node_execution(db, record, output_reference=data.output_reference)
    db.commit()
    return {"id": record.id, "status": record.status, "attempt": record.attempt}


@router.post("/side-effects", status_code=201)
def record_side_effect_endpoint(
    data: SideEffectRecord,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Record a workflow side effect with explicit reversibility."""
    require_workspace_membership(db, data.workspace_id, principal.user.id)
    record = record_side_effect(
        db, data.workspace_id, data.workflow_execution_id,
        data.node_id, data.side_effect_type, data.metadata,
    )
    db.commit()
    return {
        "id": record.id,
        "side_effect_type": record.side_effect_type,
        "reversible": record.reversible,
        "status": record.status,
    }


@router.post("/side-effects/{record_id}/compensate")
def compensate_side_effect_endpoint(
    record_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..models.phase15 import WorkflowCompensation
    record = db.query(WorkflowCompensation).filter(WorkflowCompensation.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Side effect record not found")
    require_workspace_membership(db, record.workspace_id, principal.user.id)
    try:
        record = compensate_side_effect(db, record)
    except WorkflowReliabilityError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return {"id": record.id, "status": record.status}


@router.post("/recovery-plan")
def plan_recovery(
    workspace_id: int,
    workflow_execution_id: str,
    node_id: str,
    error: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Determine retry/dead-letter/manual-retry for a failed node."""
    require_workspace_membership(db, workspace_id, principal.user.id)
    return recovery_plan(db, workspace_id, workflow_execution_id, error, node_id)