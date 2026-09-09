"""Phase 17 knowledge API — knowledge graph 4.0, memory 3.0, RAG 5.0,
agent plan graphs/handoffs, and workflow run orchestration. Every route is
workspace-scoped; org aggregates require an org admin."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..models.organization import Organization
from ..services.permission_service import (
    require_workspace_membership, get_organization_role,
)
from ..services import kg4, memory3, rag5, agent3, workflow3 as wf3
from ..services.worker_platform import JobNotFoundError

router = APIRouter(tags=["knowledge17"])


def _ws(db: Session, workspace_id: int, user_id: int) -> Workspace:
    return require_workspace_membership(db, workspace_id, user_id)


def _org_admin(db: Session, org_id: int, user_id: int) -> None:
    role = get_organization_role(db, org_id, user_id)
    if role not in ("OWNER", "ADMIN"):
        raise HTTPException(status_code=403,
                            detail="Organization admin required")


# ---------------------------------------------------------------------------
# Knowledge graph 4.0
# ---------------------------------------------------------------------------

@router.post("/graph/entities/{entity_id}/candidates")
def entity_candidates(entity_id: int, workspace_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        results = kg4.suggest_entity_candidates(db, workspace_id, entity_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return {"entity_id": entity_id, "candidates": results}


class CandidateDecisionBody(BaseModel):
    decision: str  # APPROVED / REJECTED


@router.post("/graph/candidates/{candidate_id}/decide")
def decide_candidate(candidate_id: int, workspace_id: int,
                     body: CandidateDecisionBody,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = kg4.resolve_candidate(db, workspace_id, candidate_id,
                                       body.decision, principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return result


@router.post("/graph/entities/{entity_id}/relationship-suggestions")
def relationship_suggestions(entity_id: int, workspace_id: int,
                             principal: AuthPrincipal = Depends(get_current_principal),
                             db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        suggestions = kg4.suggest_relationships(db, workspace_id, entity_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return {"entity_id": entity_id, "suggestions": suggestions}


class ApplySuggestionBody(BaseModel):
    relation_type: str = "related_to"


@router.post("/graph/relationship-suggestions/{suggestion_id}/apply")
def apply_suggestion(suggestion_id: int, workspace_id: int,
                     body: ApplySuggestionBody,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = kg4.apply_relationship_suggestion(
            db, workspace_id, suggestion_id, body.relation_type,
            principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return result


@router.get("/graph/entities/{entity_id}/conflicts")
def graph_conflicts(entity_id: int, workspace_id: int,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    return {"conflicts": kg4.detect_relationship_conflicts(
        db, workspace_id, entity_id)}


@router.get("/graph/search")
def graph_search(entity_id: int, workspace_id: int,
                 max_depth: int = 3,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        return kg4.graph_search(db, workspace_id, entity_id,
                                max_depth=max_depth)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/organizations/{organization_id}/graph-summary")
def org_graph_summary(organization_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    org = db.query(Organization).filter(
        Organization.id == organization_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    _org_admin(db, organization_id, principal.user.id)
    ws_ids = [w.id for w in db.query(Workspace.id).filter(
        Workspace.organization_id == organization_id).all()]
    return kg4.organization_graph_summary(db, organization_id, ws_ids)


# ---------------------------------------------------------------------------
# Memory 3.0
# ---------------------------------------------------------------------------

class ConsolidateBody(BaseModel):
    memory_ids: list[int]


@router.post("/memory/consolidate")
def consolidate_memory(workspace_id: int, body: ConsolidateBody,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = memory3.consolidate(db, body.memory_ids, workspace_id,
                                     principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return result


class SupersedeBody(BaseModel):
    old_memory_id: int
    new_content: str
    memory_type: str = "WORKSPACE_FACT"
    source: str = "agent"
    reason: str = ""
    scope: str = "WORKSPACE"


@router.post("/memory/supersede")
def supersede_memory(workspace_id: int, body: SupersedeBody,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = memory3.supersede(
            db, body.old_memory_id, body.new_content, workspace_id,
            body.memory_type, body.source, reason=body.reason,
            user_id=principal.user.id, scope=body.scope)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return result


@router.get("/memory/retrieve")
def retrieve_memory(workspace_id: int,
                    memory_type: Optional[str] = None,
                    limit: int = 50,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    return memory3.retrieve_for_user(
        db, workspace_id, user_id=principal.user.id,
        organization_id=ws.organization_id,
        memory_type=memory_type, limit=limit)


@router.post("/memory/expire-due")
def expire_memory_due(workspace_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    result = memory3.expire_due(db)
    db.commit()
    return result


# ---------------------------------------------------------------------------
# RAG 5.0
# ---------------------------------------------------------------------------

@router.get("/rag5/plan")
def rag_plan(query: str,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    if not principal:
        raise HTTPException(status_code=401, detail="Authentication required")
    return rag5.retrieval_plan(query)


class EvaluateBody(BaseModel):
    dataset: list[dict]


@router.post("/rag5/evaluate")
def rag_evaluate(workspace_id: int, body: EvaluateBody,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    if not body.dataset or len(body.dataset) > 200:
        raise HTTPException(status_code=422,
                            detail="dataset must contain 1..200 rows")
    result = rag5.run_evaluation(db, workspace_id, body.dataset)
    db.commit()
    return result


# ---------------------------------------------------------------------------
# Agent plans + handoffs
# ---------------------------------------------------------------------------

class PlanBody(BaseModel):
    execution_id: str
    objective: str
    steps: list[dict]
    budgets: Optional[dict] = None
    risk: str = "LOW"


@router.post("/agent-plans")
def create_agent_plan(workspace_id: int, body: PlanBody,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        plan = agent3.create_agent_plan(
            db, workspace_id=workspace_id,
            execution_id=body.execution_id,
            objective=body.objective, plan={"steps": body.steps},
            budgets=body.budgets, risk=body.risk)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return {"plan_id": plan.id, "status": plan.status, "risk": plan.risk}


class HandoffBody(BaseModel):
    execution_id: str
    question: str
    context_ref: Optional[str] = None


@router.post("/agent-handoffs")
def request_handoff(workspace_id: int, body: HandoffBody,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        handoff = agent3.request_handoff(
            db, workspace_id=workspace_id, execution_id=body.execution_id,
            question=body.question, context_ref=body.context_ref,
            requested_by=principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return {"handoff_id": handoff.id, "status": handoff.status}


class AnswerHandoffBody(BaseModel):
    answer: str


@router.post("/agent-handoffs/{handoff_id}/answer")
def answer_handoff(handoff_id: int, workspace_id: int,
                   body: AnswerHandoffBody,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = agent3.answer_handoff(
            db, handoff_id=handoff_id, workspace_id=workspace_id,
            answer=body.answer, answered_by=principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return result


@router.post("/agent-executions/{execution_id}/cancel")
def cancel_agent_execution(execution_id: str, workspace_id: int,
                           principal: AuthPrincipal = Depends(get_current_principal),
                           db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = agent3.cancel_execution_durable(
            db, execution_id, workspace_id, principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return result


# ---------------------------------------------------------------------------
# Workflow runs (orchestration 3.0)
# ---------------------------------------------------------------------------

class WorkflowRunBody(BaseModel):
    definition: dict
    workflow_version_id: Optional[int] = None
    timeout_seconds: Optional[int] = None


@router.post("/workflow-runs")
def start_workflow_run(workspace_id: int, body: WorkflowRunBody,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    try:
        run = wf3.start_workflow_run(
            db, workspace_id=workspace_id,
            organization_id=ws.organization_id,
            definition=body.definition,
            workflow_version_id=body.workflow_version_id,
            timeout_seconds=body.timeout_seconds,
            created_by=principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return {"run_id": run.id, "status": run.status,
            "definition_hash": run.definition_hash}


@router.post("/workflow-runs/{run_id}/advance")
def advance_workflow_run(run_id: int, workspace_id: int,
                         principal: AuthPrincipal = Depends(get_current_principal),
                         db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = wf3.advance_run(db, run_id, workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return result


class PauseBody(BaseModel):
    reason: str = ""


@router.post("/workflow-runs/{run_id}/pause")
def pause_workflow_run(run_id: int, workspace_id: int, body: PauseBody,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = wf3.pause_run(db, run_id, workspace_id,
                               principal.user.id, reason=body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return result


@router.post("/workflow-runs/{run_id}/resume")
def resume_workflow_run(run_id: int, workspace_id: int,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = wf3.resume_run(db, run_id, workspace_id,
                                principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return result


@router.get("/workflow-runs")
def list_workflow_runs(workspace_id: int, status: Optional[str] = None,
                       limit: int = 50,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    q = db.query(wf3.WorkflowRun).filter(
        wf3.WorkflowRun.workspace_id == workspace_id)
    if status:
        q = q.filter(wf3.WorkflowRun.status == status)
    items = q.order_by(wf3.WorkflowRun.id.desc()).limit(min(limit, 200)).all()
    return {"items": [
        {"run_id": r.id, "status": r.status,
         "control_state": r.control_state,
         "definition_hash": r.definition_hash,
         "started_at": r.started_at, "completed_at": r.completed_at,
         "error": r.error}
        for r in items], "total": len(items), "limit": min(limit, 200)}
