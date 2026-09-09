"""Phase 21 operations API — Autonomous Enterprise AI + Self-Healing Cloud.

Surfaces the governed-autonomy platform: autonomy policies/levels/transition
audit, zero-side-effect operation simulation, the autonomous action guard,
autonomous operation history, system health aggregation + snapshots, recovery
playbooks/attempts (idempotent, cooldown-aware, escalating), diagnosis
reports, knowledge health/incidents/recovery plans, ingestion quality +
anomalies + adaptation candidates, adaptive retrieval/RAG candidates with
promotion gates, model/provider drift + routing simulation, cost
forecasts/anomalies/guards/optimizations, agent plan risk/simulation/
recovery/dead letters, workflow risk assessments, platform events + bounded
replay, evaluation schedules/runs/gates, personal autonomy settings + memory
controls + activity feed, artifact quality, search autopilot, tool safety +
injection/exfiltration/abuse corpora, emergency stop, security health +
incidents, data governance snapshots/impacts, region health/failover
simulation/residency guard, worker health/quarantine/capacity, broker/
scheduler/db/cache healing, graph repair + memory autonomy, incidents 2.0
with postmortems + learning, maintenance plans with dry-run, backup/DR +
chaos/load records.

Every endpoint requires authentication; workspace-scoped writes require an
owner/org-admin role. Never returns secrets or chain-of-thought.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..services.permission_service import (
    require_workspace_membership, get_organization_role,
)
from ..services import (
    autonomy as au,
    selfheal as sh,
    diagnosis as dg,
    knowledge_heal as kh,
    adaptive_ai as ad,
    autopilot as ap,
    agent_autonomy as aa,
    event_eval as ev,
    safety10 as s10,
    infra_heal as ih,
    personal_ai as pa,
    incident_ops as io_,
)

router = APIRouter(prefix="/ops21", tags=["ops21"])


def _owner(db: Session, principal: AuthPrincipal,
           workspace_id: int) -> Workspace:
    ws = require_workspace_membership(db, workspace_id,
                                      principal.user.id)
    if ws.organization_id is not None:
        role = get_organization_role(db, ws.organization_id,
                                     principal.user.id)
        if role in ("OWNER", "ADMIN"):
            return ws
    elif ws.owner_id == principal.user.id:
        return ws
    raise HTTPException(status_code=403,
                        detail="Organization admin or workspace owner "
                               "required")


def _member(db: Session, principal: AuthPrincipal,
            workspace_id: int) -> Workspace:
    return require_workspace_membership(db, workspace_id,
                                        principal.user.id)


# ===========================================================================
# Autonomy control plane
# ===========================================================================

class PolicyIn(BaseModel):
    workspace_id: int
    operation_type: str = Field(..., max_length=64)
    risk_level: str = "LOW"
    autonomy_level: str = "RECOMMEND"
    requires_approval: bool = True
    budget_limit_usd: Optional[float] = None
    execution_limit_per_hour: Optional[int] = None
    cooldown_seconds: int = 300
    requires_audit: bool = True


class LevelIn(BaseModel):
    workspace_id: int
    policy_id: int
    new_level: str
    reason: Optional[str] = None


class SimulateIn(BaseModel):
    workspace_id: int
    operation_type: str = Field(..., max_length=64)
    risk_level: str = "LOW"
    estimated_cost_usd: float = 0.0


class GuardIn(BaseModel):
    workspace_id: int
    operation_type: str = Field(..., max_length=64)
    risk_level: str = "LOW"
    actor: str = "operator"
    source: str = "OPERATOR"
    input_payload: Optional[dict] = None
    estimated_cost_usd: float = 0.0
    idempotency_key: Optional[str] = Field(None, max_length=128)


@router.get("/autonomy/policies")
def list_policies(workspace_id: int,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = (db.query(au.AutonomyPolicy)
            .filter_by(workspace_id=workspace_id)
            .order_by(au.AutonomyPolicy.id.asc()).limit(200).all())
    return {"policies": [
        {"id": r.id, "operation_type": r.operation_type,
         "risk_level": r.risk_level, "autonomy_level": r.autonomy_level,
         "requires_approval": r.requires_approval,
         "budget_limit_usd": r.budget_limit_usd,
         "execution_limit_per_hour": r.execution_limit_per_hour,
         "cooldown_seconds": r.cooldown_seconds,
         "requires_audit": r.requires_audit}
        for r in rows]}


@router.post("/autonomy/policies")
def upsert_policy(body: PolicyIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        policy = au.create_policy(
            db, body.workspace_id, body.operation_type, body.risk_level,
            body.autonomy_level, body.requires_approval,
            body.budget_limit_usd, body.execution_limit_per_hour,
            body.cooldown_seconds, body.requires_audit,
            actor=principal.user.email)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"policy_id": policy.id,
            "autonomy_level": policy.autonomy_level}


@router.post("/autonomy/level")
def set_level(body: LevelIn,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    policy = (db.query(au.AutonomyPolicy)
              .filter_by(workspace_id=body.workspace_id,
                         id=body.policy_id).first())
    if policy is None:
        raise HTTPException(status_code=404, detail="policy not found")
    try:
        au.set_autonomy_level(db, policy, body.new_level,
                              actor=principal.user.email,
                              reason=body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"policy_id": policy.id, "autonomy_level": policy.autonomy_level}


@router.get("/autonomy/transitions")
def list_transitions(workspace_id: int, limit: int = 50, offset: int = 0,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    limit = max(1, min(limit, 200))
    rows = (db.query(au.AutonomyTransition)
            .filter_by(workspace_id=workspace_id)
            .order_by(au.AutonomyTransition.id.desc())
            .offset(max(0, offset)).limit(limit).all())
    return {"transitions": [
        {"id": r.id, "policy_id": r.policy_id,
         "previous_level": r.previous_level, "new_level": r.new_level,
         "actor": r.actor, "reason": r.reason, "created_at": r.created_at}
        for r in rows]}


@router.post("/autonomy/simulate")
def simulate(body: SimulateIn,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    """Zero-side-effect: would this operation be automatically allowed?"""
    _member(db, principal, body.workspace_id)
    return au.simulate_operation(db, body.workspace_id, body.operation_type,
                                 body.risk_level, body.estimated_cost_usd)


@router.post("/autonomy/guard")
def guard(body: GuardIn,
          principal: AuthPrincipal = Depends(get_current_principal),
          db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    op = au.guard_operation(
        db, body.workspace_id, body.operation_type, body.risk_level,
        actor=body.actor[:120], source=body.source[:32],
        input_payload=body.input_payload,
        estimated_cost_usd=body.estimated_cost_usd,
        idempotency_key=body.idempotency_key)
    return {"operation_id": op.id, "decision": op.decision,
            "reason": op.decision_reason, "status": op.status,
            "policy_level": op.policy_level}


@router.get("/autonomy/operations")
def list_operations(workspace_id: int, limit: int = 50, offset: int = 0,
                    operation_type: Optional[str] = None,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = au.list_operations(db, workspace_id, limit, offset,
                              operation_type)
    return {"operations": [
        {"id": r.id, "operation_type": r.operation_type,
         "risk_level": r.risk_level, "decision": r.decision,
         "status": r.status, "actor": r.actor, "source": r.source,
         "simulated": r.simulated, "created_at": r.created_at}
        for r in rows]}


# ===========================================================================
# Self-healing platform
# ===========================================================================

class HealthIn(BaseModel):
    workspace_id: int
    component_states: dict


class PlaybookIn(BaseModel):
    workspace_id: int
    name: str = Field(..., max_length=120)
    trigger: str = Field(..., max_length=64)
    actions: list
    risk_level: str = "LOW"
    cooldown_seconds: int = 300
    max_attempts: int = 3
    detection_criteria: Optional[dict] = None
    rollback_strategy: Optional[str] = None
    approved: bool = False


class RecoveryIn(BaseModel):
    workspace_id: int
    trigger: str = Field(..., max_length=64)
    playbook_id: Optional[int] = None
    idempotency_key: Optional[str] = Field(None, max_length=128)


@router.post("/health/aggregate")
def aggregate(body: HealthIn,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return sh.aggregate_health(db, body.workspace_id,
                               body.component_states)


@router.get("/health/latest")
def latest_health(workspace_id: int,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    snap = sh.latest_snapshot(db, workspace_id)
    if snap is None:
        return {"snapshot": None}
    import json as _json
    return {"snapshot": {
        "id": snap.id, "overall_state": snap.overall_state,
        "components": _json.loads(snap.components or "{}"),
        "unhealthy_count": snap.unhealthy_count,
        "degraded_count": snap.degraded_count,
        "created_at": snap.created_at}}


@router.post("/health/failures")
def detect_failures(workspace_id: int, signals: dict,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return {"failures": sh.detect_failures(db, workspace_id, signals)}


@router.post("/recovery/playbooks")
def create_playbook(body: PlaybookIn,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        pb = sh.create_playbook(
            db, body.workspace_id, body.name, body.trigger,
            [str(a) for a in body.actions][:10], body.risk_level,
            body.cooldown_seconds, body.max_attempts,
            body.detection_criteria, body.rollback_strategy,
            body.approved)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"playbook_id": pb.id, "approved": pb.approved}


@router.post("/recovery/playbooks/{playbook_id}/approve")
def approve_playbook(playbook_id: int, workspace_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    pb = (db.query(sh.RecoveryPlaybook)
          .filter_by(workspace_id=workspace_id, id=playbook_id).first())
    if pb is None:
        raise HTTPException(status_code=404, detail="playbook not found")
    sh.approve_playbook(db, pb, actor=principal.user.email)
    return {"playbook_id": pb.id, "approved": True}


@router.post("/recovery/attempt")
def recovery_attempt(body: RecoveryIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    playbook = None
    if body.playbook_id is not None:
        playbook = (db.query(sh.RecoveryPlaybook)
                    .filter_by(workspace_id=body.workspace_id,
                               id=body.playbook_id).first())
        if playbook is None:
            raise HTTPException(status_code=404, detail="playbook not found")
    att = sh.attempt_recovery(db, body.workspace_id, body.trigger, playbook,
                              body.idempotency_key)
    return {"attempt_id": att.id, "status": att.status,
            "attempts_so_far": att.attempts_so_far,
            "auto_applied": att.auto_applied}


@router.get("/recovery/attempts")
def list_attempts(workspace_id: int, limit: int = 50, offset: int = 0,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = sh.list_attempts(db, workspace_id, limit, offset)
    return {"attempts": [
        {"id": r.id, "trigger": r.trigger, "status": r.status,
         "risk_level": r.risk_level, "auto_applied": r.auto_applied,
         "attempts_so_far": r.attempts_so_far,
         "escalated_incident_id": r.escalated_incident_id,
         "created_at": r.created_at} for r in rows]}


# ===========================================================================
# Self-diagnosis engine
# ===========================================================================

class DiagnoseIn(BaseModel):
    workspace_id: int
    symptom: str = Field(..., max_length=200)
    signals: dict
    failures: Optional[list] = None
    incident_id: Optional[int] = None


@router.post("/diagnosis")
def diagnose(body: DiagnoseIn,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    report = dg.diagnose(body.symptom, body.signals, body.failures)
    row = dg.persist_diagnosis(db, body.workspace_id, body.symptom, report,
                               body.incident_id)
    return {"report_id": row.id, "top_cause": report["top_cause"],
            "top_confidence": report["top_confidence"],
            "summary": report["summary"],
            "hypotheses": report["hypotheses"],
            "correlated_clusters": report["correlated_clusters"]}


@router.get("/diagnosis")
def list_diagnoses(workspace_id: int, limit: int = 50, offset: int = 0,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = dg.list_reports(db, workspace_id, limit, offset)
    return {"reports": [
        {"id": r.id, "symptom": r.symptom, "top_cause": r.top_cause,
         "top_confidence": r.top_confidence,
         "correlated_failures": r.correlated_failures,
         "report": r.report, "created_at": r.created_at} for r in rows]}


# ===========================================================================
# Knowledge healing
# ===========================================================================

class KnowledgeHealthIn(BaseModel):
    workspace_id: int
    signals: dict


class RecoveryPlanIn(BaseModel):
    workspace_id: int
    issue_kind: str = Field(..., max_length=48)
    target_type: str = Field(..., max_length=32)
    target_id: Optional[int] = None


@router.post("/knowledge/health")
def knowledge_health(body: KnowledgeHealthIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return kh.knowledge_health(db, body.workspace_id, body.signals)


@router.post("/knowledge/incidents")
def knowledge_incidents(body: KnowledgeHealthIn,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return {"incidents": kh.detect_knowledge_incidents(body.signals)}


@router.post("/knowledge/recovery-plans")
def create_recovery_plan(body: RecoveryPlanIn,
                         principal: AuthPrincipal = Depends(get_current_principal),
                         db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        plan = kh.create_recovery_plan(db, body.workspace_id,
                                       body.issue_kind, body.target_type,
                                       body.target_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"plan_id": plan.id, "risk_level": plan.risk_level,
            "status": plan.status, "actions": plan.plan}


@router.post("/knowledge/recovery-plans/{plan_id}/execute")
def execute_recovery_plan(plan_id: int, workspace_id: int,
                          principal: AuthPrincipal = Depends(get_current_principal),
                          db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    plan = (db.query(kh.KnowledgeRecoveryPlan)
            .filter_by(workspace_id=workspace_id, id=plan_id).first())
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    result = kh.execute_recovery_plan(db, plan,
                                      actor=principal.user.email,
                                      source="OPERATOR")
    kh.audit_recovery(db, workspace_id, plan_id, result["decision"],
                      actor=principal.user.email)
    return result


@router.get("/knowledge/recovery-plans")
def list_recovery_plans(workspace_id: int, limit: int = 50, offset: int = 0,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = kh.list_plans(db, workspace_id, limit, offset)
    return {"plans": [
        {"id": r.id, "issue_kind": r.issue_kind, "target_type": r.target_type,
         "risk_level": r.risk_level, "status": r.status,
         "decision": r.decision} for r in rows]}


# ===========================================================================
# Adaptive ingestion / retrieval / RAG
# ===========================================================================

class IngestionSampleIn(BaseModel):
    workspace_id: int
    extraction_quality: Optional[float] = None
    ocr_quality: Optional[float] = None
    chunk_quality: Optional[float] = None
    metadata_completeness: Optional[float] = None
    embedding_coverage: Optional[float] = None
    processing_latency_ms: Optional[float] = None
    document_id: Optional[int] = None


class CandidateIn(BaseModel):
    workspace_id: int
    domain: str = Field(..., max_length=32)
    change_kind: str = Field(..., max_length=48)
    proposed_value: Optional[dict] = None
    rationale: str = ""


@router.post("/ingestion/samples")
def add_ingestion_sample(body: IngestionSampleIn,
                         principal: AuthPrincipal = Depends(get_current_principal),
                         db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = kh.record_ingestion_sample(
        db, body.workspace_id, body.extraction_quality, body.ocr_quality,
        body.chunk_quality, body.metadata_completeness,
        body.embedding_coverage, body.processing_latency_ms,
        body.document_id)
    return {"sample_id": row.id}


@router.post("/ingestion/anomalies")
def detect_ingestion_anomalies(workspace_id: int,
                               principal: AuthPrincipal = Depends(get_current_principal),
                               db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    anomalies = kh.detect_ingestion_anomalies(db, workspace_id)
    return {"anomalies": [
        {"id": a.id, "metric": a.metric, "severity": a.severity,
         "drop_percent": a.drop_percent, "recommendation": a.recommendation}
        for a in anomalies]}


@router.post("/candidates")
def create_candidate(body: CandidateIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    if body.domain == "retrieval":
        try:
            cand = ad.propose_retrieval_candidate(
                db, body.workspace_id, body.change_kind,
                body.proposed_value, body.rationale)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    elif body.domain == "rag":
        try:
            cand = ad.propose_rag_candidate(
                db, body.workspace_id, body.change_kind,
                body.proposed_value, body.rationale)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        raise HTTPException(status_code=422,
                            detail="domain must be retrieval or rag")
    return {"candidate_id": cand.id, "status": cand.status}


@router.post("/candidates/{candidate_id}/evaluate")
def evaluate_candidate(candidate_id: int,
                       workspace_id: int,
                       evaluation_score: float,
                       baseline_score: float,
                       thresholds: Optional[dict] = None,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    cand = (db.query(kh.AdaptiveCandidate)
            .filter_by(workspace_id=workspace_id, id=candidate_id)
            .first())
    if cand is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    row = kh.evaluate_candidate(db, cand, evaluation_score,
                                baseline_score, thresholds)
    return {"candidate_id": row.id, "gates_passed": row.gates_passed,
            "gates": row.gates, "status": row.status}


@router.post("/candidates/{candidate_id}/promote")
def promote_candidate(candidate_id: int, workspace_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    cand = (db.query(kh.AdaptiveCandidate)
            .filter_by(workspace_id=workspace_id, id=candidate_id).first())
    if cand is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return ad.promote_candidate(db, cand, actor=principal.user.email,
                                source="OPERATOR")


@router.get("/candidates")
def list_candidates(workspace_id: int, domain: Optional[str] = None,
                    limit: int = 50, offset: int = 0,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = kh.list_candidates(db, workspace_id, domain, limit, offset)
    return {"candidates": [
        {"id": r.id, "domain": r.domain, "change_kind": r.change_kind,
         "status": r.status, "gates_passed": r.gates_passed,
         "promoted": r.promoted} for r in rows]}


# ===========================================================================
# Model / cost autopilot
# ===========================================================================

class ModelSampleIn(BaseModel):
    workspace_id: int
    model: str = Field(..., max_length=120)
    provider: str = Field(..., max_length=64)
    quality_score: Optional[float] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    availability: Optional[float] = None
    tool_reliability: Optional[float] = None
    structured_output_reliability: Optional[float] = None


class RoutingSimIn(BaseModel):
    workspace_id: int
    description: str = Field(..., max_length=200)
    workload: list
    current_routing: dict
    candidate_routing: dict
    constraints: Optional[dict] = None


class CostGuardIn(BaseModel):
    workspace_id: int
    operation_type: str = Field(..., max_length=64)
    estimated_cost_usd: float
    remaining_budget_usd: Optional[float] = None
    idempotency_key: Optional[str] = Field(None, max_length=128)


class CostOptimIn(BaseModel):
    workspace_id: int
    optimization_kind: str = Field(..., max_length=48)
    estimated_savings_usd: float = 0.0


@router.post("/models/samples")
def add_model_sample(body: ModelSampleIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = ap.record_model_sample(
        db, body.workspace_id, body.model, body.provider,
        body.quality_score, body.latency_ms, body.cost_usd,
        body.availability, body.tool_reliability,
        body.structured_output_reliability)
    return {"sample_id": row.id}


@router.post("/models/drift")
def detect_model_drift(workspace_id: int, model: str, provider: str,
                       metric: str = "quality_score",
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    event = ap.detect_model_drift(db, workspace_id, model, provider, metric)
    if event is None:
        return {"drift_detected": False}
    return {"drift_detected": True, "metric": event.metric,
            "direction": event.direction,
            "drop_percent": event.drop_percent,
            "baseline": event.baseline_value, "current": event.current_value}


@router.post("/models/routing-simulate")
def routing_simulate(body: RoutingSimIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    if len(body.workload) > 100:
        raise HTTPException(status_code=422,
                            detail="workload limited to 100 items")
    import json as _json
    sim = ap.simulate_routing(
        db, body.workspace_id, body.description,
        [dict(w) for w in body.workload], body.current_routing,
        body.candidate_routing, body.constraints)
    return {"simulation_id": sim.id, "safe": sim.safe,
            "current_metrics": _json.loads(sim.current_metrics or "{}"),
            "simulated_metrics": _json.loads(sim.simulated_metrics or "{}"),
            "safety_checks": _json.loads(sim.safety_checks or "{}")}


@router.post("/cost/forecast")
def cost_forecast(workspace_id: int, current_spend_usd: float,
                  daily_run_rate_usd: float = 0.0,
                  budget_usd: Optional[float] = None,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    row = ap.record_cost_snapshot(db, workspace_id, current_spend_usd,
                                  daily_run_rate_usd, budget_usd=budget_usd)
    return {"forecast_id": row.id, "forecast_usd": row.forecast_usd,
            "over_budget": row.over_budget}


@router.post("/cost/guard")
def cost_guard(body: CostGuardIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = ap.cost_guard(db, body.workspace_id, body.operation_type,
                        body.estimated_cost_usd,
                        body.remaining_budget_usd, body.idempotency_key)
    return {"decision_id": row.id, "decision": row.decision,
            "reason": row.reason}


@router.post("/cost/optimize")
def cost_optimize(body: CostOptimIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    result = ap.apply_cost_optimization(
        db, body.workspace_id, body.optimization_kind,
        body.estimated_savings_usd, actor=principal.user.email,
        source="OPERATOR")
    return result


@router.post("/cost/anomalies")
def cost_anomaly(workspace_id: int, baseline_value: float,
                 observed_value: float,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    row = ap.detect_cost_anomaly(db, workspace_id, baseline_value,
                                 observed_value)
    if row is None:
        return {"anomaly_detected": False}
    return {"anomaly_detected": True, "severity": row.severity,
            "increase_percent": row.increase_percent,
            "recommendation": row.recommendation}


# ===========================================================================
# Agent / workflow autonomy
# ===========================================================================

class PlanIn(BaseModel):
    workspace_id: int
    plan: dict
    agent_run_id: Optional[int] = None


class RecoveryKindIn(BaseModel):
    workspace_id: int
    agent_run_id: int
    recovery_kind: str
    detail: Optional[str] = None


class WorkflowRiskIn(BaseModel):
    workspace_id: int
    workflow: dict
    workflow_id: Optional[int] = None


@router.post("/agents/plan-risk")
def plan_risk(body: PlanIn,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = aa.assess_plan(db, body.workspace_id, body.plan,
                         body.agent_run_id)
    import json as _json
    return {"assessment_id": row.id, "risk_level": row.risk_level,
            "risk_factors": _json.loads(row.risk_factors or "[]"),
            "decision": row.decision, "handed_off": row.handed_off}


@router.post("/agents/plan-simulate")
def plan_simulate(body: PlanIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return aa.simulate_plan(body.plan)


@router.post("/agents/plan-optimize")
def plan_optimize(body: PlanIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return aa.optimize_plan(body.plan)


@router.post("/agents/recover")
def agent_recover(body: RecoveryKindIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    try:
        event = aa.recover_agent_run(db, body.workspace_id,
                                     body.agent_run_id, body.recovery_kind,
                                     body.detail)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"event_id": event.id, "recovery_kind": event.recovery_kind,
            "status": event.status}


@router.post("/agents/dead-letter")
def agent_dead_letter(workspace_id: int, execution_id: str, reason: str,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    letter = aa.dead_letter_run(db, workspace_id, execution_id, reason)
    return {"dead_letter_id": letter.id, "status": letter.status}


@router.post("/workflows/risk")
def workflow_risk(body: WorkflowRiskIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = aa.assess_workflow_risk(db, body.workspace_id, body.workflow,
                                  body.workflow_id)
    import json as _json
    return {"assessment_id": row.id, "risk_level": row.risk_level,
            "risk_factors": _json.loads(row.risk_factors or "[]"),
            "autonomy_decision": row.autonomy_decision,
            "requires_approval": row.requires_approval}


@router.post("/workflows/simulate")
def workflow_simulate(body: WorkflowRiskIn,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return aa.simulate_workflow(db, body.workspace_id, body.workflow)


# ===========================================================================
# Events / evaluation / learning
# ===========================================================================

class EventIn(BaseModel):
    workspace_id: int
    event_kind: str = Field(..., max_length=48)
    event_key: str = Field(..., max_length=160)
    payload: Optional[dict] = None


class EvalScheduleIn(BaseModel):
    workspace_id: int
    domain: str = "retrieval"
    dataset_id: Optional[int] = None
    dataset_version: str = "v1"
    interval_minutes: int = 1440
    config: Optional[dict] = None


class FeedbackIn(BaseModel):
    workspace_id: int
    input_text: str = Field(..., max_length=4000)
    expected_output: Optional[str] = None
    dataset_id: Optional[int] = None


@router.post("/events")
def emit_event(body: EventIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row, created = ev.emit_event(db, body.workspace_id, body.event_kind,
                                 body.event_key, body.payload)
    return {"event_id": row.id, "created": created,
            "deduplicated": not created}


@router.post("/events/replay")
def replay_events(workspace_id: int, event_kind: str, max_events: int = 25,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return ev.replay_events(db, workspace_id, event_kind, max_events)


@router.post("/evaluation/schedules")
def create_eval_schedule(body: EvalScheduleIn,
                         principal: AuthPrincipal = Depends(get_current_principal),
                         db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        row = ev.create_schedule(db, body.workspace_id, body.domain,
                                 body.dataset_id, body.dataset_version,
                                 body.interval_minutes, body.config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"schedule_id": row.id, "domain": row.domain,
            "dataset_version": row.dataset_version}


@router.post("/evaluation/schedules/{schedule_id}/run")
def run_eval(schedule_id: int, workspace_id: int, metrics: dict,
             model: Optional[str] = None, provider: Optional[str] = None,
             baseline_metrics: Optional[dict] = None,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    try:
        run = ev.run_evaluation(db, workspace_id, schedule_id, metrics,
                                model, provider,
                                baseline_metrics=baseline_metrics)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"run_id": run.id, "regression_detected": run.regression_detected,
            "gate_passed": run.gate_passed, "metrics": run.metrics}


@router.post("/feedback")
def submit_feedback(body: FeedbackIn,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = ev.ingest_feedback(db, body.workspace_id, body.input_text,
                             body.expected_output, body.dataset_id)
    return {"candidate_id": row.id, "quality": row.quality}


# ===========================================================================
# AI safety 10.0 / security center 3.0 / governance 6.0
# ===========================================================================

@router.post("/safety/injection-corpus")
def injection_corpus(workspace_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    results = s10.run_injection_corpus()
    return {"results": results,
            "all_blocked": all(r["blocked"] for r in results)}


@router.post("/safety/exfiltration-corpus")
def exfiltration_corpus(workspace_id: int,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    results = s10.run_exfiltration_corpus()
    return {"results": results,
            "all_blocked": all(r["blocked"] for r in results)}


class ToolSafetyIn(BaseModel):
    workspace_id: int
    tool_name: str = Field(..., max_length=120)
    arguments: Optional[dict] = None
    allowlist: Optional[list] = None
    budget_remaining_usd: Optional[float] = None
    estimated_cost_usd: float = 0.0
    timeout_ms: int = 30000
    scope: str = "object"
    granted_scope: str = "object"


@router.post("/safety/tool-check")
def tool_check(body: ToolSafetyIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    result = s10.enforce_tool_safety(
        db, body.workspace_id, body.tool_name, body.arguments,
        set(body.allowlist) if body.allowlist else None,
        body.budget_remaining_usd, body.estimated_cost_usd,
        body.timeout_ms, body.scope, body.granted_scope)
    return result


@router.post("/safety/sanitize-output")
def sanitize_output(text: str,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    # Sanitization endpoint requires auth but no workspace write.
    return s10.sanitize_tool_output(text)


class EmergencyStopIn(BaseModel):
    workspace_id: int
    scope: str = "ALL"
    reason: Optional[str] = None


@router.post("/safety/emergency-stop")
def emergency_stop(body: EmergencyStopIn,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        row = s10.activate_emergency_stop(
            db, body.workspace_id, body.scope,
            actor=principal.user.email, reason=body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"stop_id": row.id, "scope": row.scope, "active": row.active}


@router.post("/safety/emergency-stop/{stop_id}/lift")
def lift_emergency_stop(stop_id: int, workspace_id: int,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    row = (db.query(s10.EmergencyStop)
           .filter_by(workspace_id=workspace_id, id=stop_id).first())
    if row is None:
        raise HTTPException(status_code=404, detail="emergency stop not found")
    s10.lift_emergency_stop(db, workspace_id, row,
                            actor=principal.user.email)
    return {"stop_id": row.id, "active": row.active}


class SecuritySignalsIn(BaseModel):
    workspace_id: int
    auth_failures: int = 0
    authz_failures: int = 0
    injection_attempts: int = 0
    ssrf_attempts: int = 0
    tool_abuse: int = 0
    exfiltration_attempts: int = 0
    suspicious_api: int = 0


@router.post("/security/signals")
def security_signals(body: SecuritySignalsIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = s10.record_security_signal(
        db, body.workspace_id, body.auth_failures, body.authz_failures,
        body.injection_attempts, body.ssrf_attempts, body.tool_abuse,
        body.exfiltration_attempts, body.suspicious_api)
    return {"score": row.score, "state": row.state}


@router.post("/security/incidents")
def security_incidents(body: SecuritySignalsIn,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    counts = {"auth": body.auth_failures, "injection": body.injection_attempts,
              "ssrf": body.ssrf_attempts, "tool_abuse": body.tool_abuse,
              "exfiltration": body.exfiltration_attempts}
    rows = s10.evaluate_security_incidents(db, body.workspace_id, counts)
    return {"incidents": [
        {"id": r.id, "severity": r.severity, "category": r.category,
         "summary": r.summary} for r in rows]}


class ClassificationIn(BaseModel):
    workspace_id: int
    counts: dict
    environment: Optional[dict] = None
    classification: str = "internal"


@router.post("/governance/classification")
def classification_snapshot(body: ClassificationIn,
                            principal: AuthPrincipal = Depends(get_current_principal),
                            db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = s10.snapshot_classification(db, body.workspace_id, body.counts)
    return {"snapshot_id": row.id, "drifted": row.drifted,
            "drift_detail": row.drift_detail}


@router.post("/governance/policy-impact")
def governance_impact(body: ClassificationIn,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = s10.data_policy_impact(db, body.workspace_id,
                                 body.classification, body.environment)
    import json as _json
    return {"impact_id": row.id,
            "minimization_required": row.minimization_required,
            "residency_violations": (
                _json.loads(row.residency_violations)
                if row.residency_violations else None)}


# ===========================================================================
# Multi-region operations
# ===========================================================================

class RegionHealthIn(BaseModel):
    organization_id: int
    region_id: str = Field(..., max_length=64)
    state: str = "HEALTHY"
    workers: int = 0
    queue_depth: int = 0
    provider_healthy: bool = True
    db_healthy: bool = True


class FailoverSimIn(BaseModel):
    organization_id: int
    from_region: str = Field(..., max_length=64)
    to_region: str = Field(..., max_length=64)
    residency_rules: Optional[dict] = None


@router.post("/regions/health")
def region_health(body: RegionHealthIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member_org(db, principal, body.organization_id)
    try:
        row = ih.record_region_health(
            db, body.organization_id, body.region_id, body.state,
            body.workers, body.queue_depth, body.provider_healthy,
            body.db_healthy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"snapshot_id": row.id, "state": row.state,
            "failover_ready": row.failover_ready}


def _member_org(db: Session, principal: AuthPrincipal,
                organization_id: int) -> None:
    role = get_organization_role(db, organization_id, principal.user.id)
    if role not in ("OWNER", "ADMIN", "MEMBER"):
        raise HTTPException(status_code=403,
                            detail="Organization membership required")


@router.post("/regions/residency-check")
def residency_check(organization_id: int, requested_region: str,
                    residency_rules: dict,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member_org(db, principal, organization_id)
    return ih.check_residency(db, organization_id, requested_region,
                              residency_rules)


@router.post("/regions/failover-simulate")
def failover_simulate(body: FailoverSimIn,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _member_org(db, principal, body.organization_id)
    target = ih.latest_region_health(db, body.organization_id,
                                     body.to_region)
    sim = ih.simulate_failover(db, body.organization_id, body.from_region,
                               body.to_region, target,
                               body.residency_rules)
    return {"simulation_id": sim.id, "ready": sim.ready,
            "residency_ok": sim.residency_ok, "capacity_ok": sim.capacity_ok,
            "simulated": True, "executed": False}


@router.post("/regions/failover-simulations/{simulation_id}/execute")
def failover_execute(simulation_id: int, organization_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _owner_org(db, principal, organization_id)
    sim = (db.query(ih.FailoverSimulation)
           .filter_by(organization_id=organization_id, id=simulation_id)
           .first())
    if sim is None:
        raise HTTPException(status_code=404,
                            detail="failover simulation not found")
    return ih.execute_failover(db, sim, actor=principal.user.email)


def _owner_org(db: Session, principal: AuthPrincipal,
               organization_id: int) -> None:
    role = get_organization_role(db, organization_id, principal.user.id)
    if role not in ("OWNER", "ADMIN"):
        raise HTTPException(status_code=403,
                            detail="Organization admin required")


# ===========================================================================
# Infrastructure healing (worker/broker/scheduler/db/cache)
# ===========================================================================

class WorkerHealthIn(BaseModel):
    workspace_id: int
    worker_id: str = Field(..., max_length=120)
    heartbeat_age_seconds: int = 0
    throughput: float = 0.0
    failure_count: int = 0
    queue_latency_ms: float = 0.0
    region: Optional[str] = None


class BrokerHealthIn(BaseModel):
    workspace_id: int
    depth: int = 0
    latency_ms: float = 0.0
    visibility_timeouts: int = 0
    error_count: int = 0
    reconnects: int = 0
    broker: str = "postgres"


class CacheHealthIn(BaseModel):
    workspace_id: int
    cache: str = Field(..., max_length=48)
    hits: int = 0
    misses: int = 0
    stale_entries: int = 0
    memory_usage_mb: float = 0.0


class CacheInvalidateIn(BaseModel):
    workspace_id: int
    cache_keys: list


class GraphRepairIn(BaseModel):
    workspace_id: int
    repair_kind: str = Field(..., max_length=48)
    target_entity_id: Optional[int] = None
    target_relationship_id: Optional[int] = None
    impact: Optional[dict] = None


@router.post("/workers/health")
def worker_health(body: WorkerHealthIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = ih.record_worker_health(
        db, body.workspace_id, body.worker_id, body.heartbeat_age_seconds,
        body.throughput, body.failure_count, body.queue_latency_ms,
        body.region)
    return {"score_id": row.id, "score": row.score, "state": row.state,
            "quarantined": row.quarantined}


@router.post("/workers/{score_id}/recover")
def worker_recover(score_id: int, workspace_id: int,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    row = (db.query(ih.WorkerHealthScore)
           .filter_by(workspace_id=workspace_id, id=score_id).first())
    if row is None:
        raise HTTPException(status_code=404, detail="worker score not found")
    row = ih.recover_worker(db, row, actor=principal.user.email)
    return {"score_id": row.id, "state": row.state,
            "quarantined": row.quarantined}


@router.post("/workers/capacity")
def worker_capacity(workspace_id: int, current_workers: int,
                    queue_depth: int, avg_worker_throughput: float = 10.0,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    row = ih.recommend_capacity(db, workspace_id, current_workers,
                                queue_depth, avg_worker_throughput)
    return {"recommendation_id": row.id,
            "recommended_workers": row.recommended_workers,
            "fairness_preserved": row.fairness_preserved}


@router.post("/broker/health")
def broker_health(body: BrokerHealthIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = ih.record_broker_health(
        db, body.workspace_id, body.depth, body.latency_ms,
        body.visibility_timeouts, body.error_count, body.reconnects,
        body.broker)
    plan = ih.broker_recovery_plan(row)
    return {"snapshot_id": row.id, "state": row.state, "plan": plan}


@router.post("/scheduler/health")
def scheduler_health(workspace_id: int, has_leader: bool = False,
                     leader_age_seconds: int = 0,
                     missed_schedules: int = 0,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    row = ih.record_scheduler_health(db, workspace_id, has_leader,
                                     leader_age_seconds, missed_schedules)
    return {"snapshot_id": row.id, "state": row.state,
            "recovered_schedules": row.recovered_schedules}


@router.post("/db/slow-queries")
def slow_query(workspace_id: int, statement: str, duration_ms: float,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    row = ih.record_slow_query(db, workspace_id, statement, duration_ms)
    if row is None:
        return {"recorded": False}
    return {"recorded": True, "record_id": row.id,
            "recommendation": row.recommendation,
            "auto_applied": False}


@router.post("/cache/health")
def cache_health(body: CacheHealthIn,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    row = ih.record_cache_health(db, body.workspace_id, body.cache,
                                 body.hits, body.misses,
                                 body.stale_entries, body.memory_usage_mb)
    return {"snapshot_id": row.id, "hit_rate": row.hit_rate,
            "anomaly": row.anomaly, "recommendation": row.recommendation}


@router.post("/cache/invalidate")
def cache_invalidate(body: CacheInvalidateIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    keys = [str(k) for k in body.cache_keys][:100]
    return ih.invalidate_cache_tenant_safe(keys, body.workspace_id)


@router.post("/graph/repairs")
def graph_repair(body: GraphRepairIn,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        row = ih.propose_graph_repair(db, body.workspace_id,
                                      body.repair_kind,
                                      body.target_entity_id,
                                      body.target_relationship_id,
                                      body.impact)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"proposal_id": row.id, "risk_level": row.risk_level,
            "status": row.status}


@router.post("/graph/repairs/{proposal_id}/execute")
def graph_repair_execute(proposal_id: int, workspace_id: int,
                         principal: AuthPrincipal = Depends(get_current_principal),
                         db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    row = (db.query(ih.GraphRepairProposal)
           .filter_by(workspace_id=workspace_id, id=proposal_id).first())
    if row is None:
        raise HTTPException(status_code=404,
                            detail="repair proposal not found")
    return ih.execute_graph_repair(db, row, actor=principal.user.email)


# ===========================================================================
# Personal AI control
# ===========================================================================

class PersonalSettingsIn(BaseModel):
    workspace_id: int
    autonomy_level: str
    workspace_policy_level: str = "RECOMMEND"


class MemoryControlIn(BaseModel):
    workspace_id: int
    action: str
    memory_id: Optional[int] = None


class ActivityIn(BaseModel):
    workspace_id: int
    item_kind: str
    title: str = Field(..., max_length=200)
    explanation: Optional[str] = None


@router.post("/personal/settings")
def personal_settings(body: PersonalSettingsIn,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    try:
        result = pa.set_personal_autonomy(
            db, body.workspace_id, principal.user.id, body.autonomy_level,
            body.workspace_policy_level)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


@router.post("/personal/memory")
def personal_memory(body: MemoryControlIn,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    try:
        return pa.memory_control(db, body.workspace_id, principal.user.id,
                                 body.action, body.memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/personal/activity")
def record_activity(body: ActivityIn,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    try:
        row = pa.record_activity(db, body.workspace_id, principal.user.id,
                                 body.item_kind, body.title,
                                 body.explanation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"item_id": row.id, "item_kind": row.item_kind}


@router.get("/personal/activity")
def activity_feed(workspace_id: int, limit: int = 50, offset: int = 0,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = pa.activity_feed(db, workspace_id, principal.user.id, limit,
                            offset)
    return {"items": [
        {"id": r.id, "item_kind": r.item_kind, "title": r.title,
         "explanation": r.explanation, "created_at": r.created_at}
        for r in rows]}


# ===========================================================================
# Incidents 2.0 / maintenance / DR / chaos
# ===========================================================================

class IncidentIn(BaseModel):
    workspace_id: int
    title: str = Field(..., max_length=200)
    severity: str = "SEV3"
    source: str = "OPERATOR"
    detail: Optional[str] = None


class PostmortemIn(BaseModel):
    workspace_id: int
    diagnosis_summary: str
    contributing_factors: list = []


class LearningIn(BaseModel):
    workspace_id: int
    learnings: list


class MaintenanceIn(BaseModel):
    workspace_id: int
    plan_kind: str = Field(..., max_length=48)
    targets: Optional[list] = None
    older_than_days: int = 90


class ChaosIn(BaseModel):
    workspace_id: int
    scenario: str = Field(..., max_length=48)
    passed: bool = True
    metrics: Optional[dict] = None
    detail: Optional[str] = None


@router.post("/incidents")
def create_incident(body: IncidentIn,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    try:
        row = io_.create_incident(db, body.workspace_id, body.title,
                                  body.severity, body.source, body.detail)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"incident_id": row.id, "severity": row.severity,
            "status": row.status}


@router.post("/incidents/{incident_id}/postmortem")
def draft_postmortem(incident_id: int, body: PostmortemIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    incident = (db.query(io_.IncidentP21)
                .filter_by(workspace_id=body.workspace_id, id=incident_id)
                .first())
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    row = io_.draft_postmortem(db, incident, body.diagnosis_summary,
                               [str(f) for f in body.contributing_factors])
    return {"incident_id": row.id,
            "postmortem_finalized": row.postmortem_finalized,
            "status": "DRAFT_REQUIRES_HUMAN_REVIEW"}


@router.post("/incidents/{incident_id}/finalize")
def finalize_postmortem(incident_id: int, body: PostmortemIn,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    incident = (db.query(io_.IncidentP21)
                .filter_by(workspace_id=body.workspace_id, id=incident_id)
                .first())
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    try:
        row = io_.finalize_postmortem(db, incident, principal.user.email)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"incident_id": row.id,
            "postmortem_finalized": row.postmortem_finalized}


@router.post("/incidents/{incident_id}/learning")
def incident_learning(incident_id: int, body: LearningIn,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    incident = (db.query(io_.IncidentP21)
                .filter_by(workspace_id=body.workspace_id, id=incident_id)
                .first())
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    try:
        row = io_.record_incident_learning(
            db, incident,
            [dict(l) if isinstance(l, dict) else {"kind": str(l),
                                                  "detail": ""} for l in body.learnings][:20])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"incident_id": row.id, "learnings_recorded": True}


@router.get("/incidents")
def list_incidents(workspace_id: int, status: Optional[str] = None,
                   limit: int = 50, offset: int = 0,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    rows = io_.list_incidents(db, workspace_id, status, limit, offset)
    return {"incidents": [
        {"id": r.id, "severity": r.severity, "title": r.title,
         "status": r.status, "source": r.source,
         "postmortem_finalized": r.postmortem_finalized,
         "created_at": r.created_at} for r in rows]}


@router.post("/maintenance/plans")
def maintenance_plan(body: MaintenanceIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        row = io_.create_maintenance_plan(
            db, body.workspace_id, body.plan_kind,
            [dict(t) if isinstance(t, dict) else {"id": t}
             for t in (body.targets or [])][:100],
            body.older_than_days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"plan_id": row.id, "destructive": row.destructive,
            "requires_approval": row.requires_approval,
            "status": row.status}


@router.post("/maintenance/plans/{plan_id}/dry-run")
def maintenance_dry_run(plan_id: int, body: MaintenanceIn,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    plan = (db.query(io_.MaintenancePlan)
            .filter_by(workspace_id=body.workspace_id, id=plan_id).first())
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    legal_holds = (db.query(io_.LegalHold)
                   .filter_by(workspace_id=body.workspace_id).all()
                   if hasattr(io_.LegalHold, "workspace_id") else [])
    protected = set()
    for hold in legal_holds:
        for attr in ("document_id", "entity_id", "target_id"):
            val = getattr(hold, attr, None)
            if val is not None:
                protected.add(val)
    return io_.maintenance_dry_run(db, plan, protected)


@router.post("/maintenance/plans/{plan_id}/approve")
def maintenance_approve(plan_id: int, body: MaintenanceIn,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    plan = (db.query(io_.MaintenancePlan)
            .filter_by(workspace_id=body.workspace_id, id=plan_id).first())
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    row = io_.approve_maintenance(db, plan, actor=principal.user.email)
    return {"plan_id": row.id, "status": row.status}


@router.post("/maintenance/plans/{plan_id}/execute")
def maintenance_execute(plan_id: int, body: MaintenanceIn,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    plan = (db.query(io_.MaintenancePlan)
            .filter_by(workspace_id=body.workspace_id, id=plan_id).first())
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    return io_.execute_maintenance(db, plan, actor=principal.user.email)


@router.post("/dr/backup-health")
def backup_health(workspace_id: int,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    row = io_.record_backup_health(db, workspace_id, None)
    return {"record_id": row.id, "status": row.status,
            "honest": "UNKNOWN without real backup infrastructure"}


@router.post("/dr/simulate")
def dr_simulate(workspace_id: int,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return io_.dr_simulation(db, workspace_id)


@router.post("/chaos/run")
def chaos_run(body: ChaosIn,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    try:
        row = io_.record_chaos_run(db, body.workspace_id, body.scenario,
                                   body.passed, simulated=True,
                                   metrics=body.metrics, detail=body.detail)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"run_id": row.id, "scenario": row.scenario,
            "simulated": row.simulated, "passed": row.passed,
            "bounded": row.bounded}


@router.post("/performance/api-load")
def api_load(workspace_id: int, request_count: int = 200,
             concurrency: int = 8,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return io_.api_load_test(request_count, concurrency)


@router.post("/performance/noisy-neighbor")
def noisy_neighbor(workspace_id: int, tenants: list,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    clean = [dict(t) if isinstance(t, dict) else {"id": t, "job_weight": 1}
             for t in tenants][:100]
    return io_.noisy_neighbor_test(clean)


@router.post("/performance/soak")
def soak(workspace_id: int, duration_hours: float = 1.0,
         memory_growth_mb: float = 0.0, error_rate: float = 0.0,
         principal: AuthPrincipal = Depends(get_current_principal),
         db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return io_.soak_result(duration_hours, memory_growth_mb, error_rate)


# ===========================================================================
# API abuse monitoring
# ===========================================================================

class AbuseIn(BaseModel):
    workspace_id: int
    abuse_kind: str = Field(..., max_length=48)
    subject: Optional[str] = Field(None, max_length=160)
    severity: str = "MEDIUM"


@router.post("/api-abuse/signals")
def api_abuse_signal(body: AbuseIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    try:
        row = s10.record_api_abuse(db, body.workspace_id, body.abuse_kind,
                                   body.subject, body.severity)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"signal_id": row.id,
            "rate_limit_recommendation": row.rate_limit_recommendation}


@router.post("/api-abuse/incidents")
def api_abuse_incidents(workspace_id: int,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    rows = s10.evaluate_api_abuse_incidents(db, workspace_id)
    return {"incidents": [
        {"id": r.id, "severity": r.severity, "summary": r.summary}
        for r in rows]}
