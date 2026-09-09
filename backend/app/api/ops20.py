"""Phase 20 operations API — Self-Improving Enterprise AI Knowledge OS.

Surfaces the governed self-improvement platform: improvement proposals with
audited lifecycle, experiments/datasets/runs/comparisons, AI quality
scorecards/trends/alerts, retrieval + RAG failure analysis and
recommendations, knowledge freshness/health/gaps, document change
intelligence, policy versioning/drift/simulation, provider model routing
recommendations + anomalies, cost baselines/drift/token efficiency,
agent/workflow/memory/graph/search intelligence, unified feedback (with
governed golden promotion), governance 5.0 versioning + simulation, safety
9.0 corpora + scorecard, incidents + SLO/error budgets + alert
prioritization + notification preferences, versioned reports, API health +
abuse detection, and database health.

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
    improvement_platform as imp,
    quality3 as q3,
    retrieval_intel as reti,
    rag_intel as ragi,
    knowledge_intel as ki,
    policy_intel as pi,
    provider_intel as pri,
    cost_intel as ci,
    agent_intel as ai2,
    workflow_intel2 as wi,
    memory_intel as mi,
    graph_intel as gi,
    search_intel as si,
    feedback_intel as fi,
    governance5 as g5,
    safety9 as s9,
    observability5 as ob5,
    slo2,
    notifications_intel as ni,
    reporting3 as rep,
    api_intel as apii,
    db_intel as dbi,
)

router = APIRouter(prefix="/ops20", tags=["ops20"])


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


# ===========================================================================
# Improvement platform
# ===========================================================================

class ProposalIn(BaseModel):
    domain: str = Field(..., max_length=40)
    title: str = Field(..., max_length=300)
    problem: str
    proposed_change: str
    evidence: Optional[str] = None
    expected_benefit: Optional[str] = None
    risk: str = "LOW"
    estimated_cost: Optional[float] = None
    evaluation_requirements: Optional[str] = None
    author_source: str = "human"
    workspace_id: Optional[int] = None


class TransitionIn(BaseModel):
    to_state: str
    reason: Optional[str] = None
    evidence: Optional[str] = None


@router.get("/improvements")
def list_improvements(domain: Optional[str] = None,
                      status: Optional[str] = None,
                      workspace_id: Optional[int] = None,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    if workspace_id is not None:
        _owner(db, principal, workspace_id)
    return imp.list_proposals(db, domain=domain, status=status,
                              workspace_id=workspace_id)


@router.post("/improvements")
def create_improvement(body: ProposalIn,
                       principal: AuthPrincipal = Depends(
                           get_current_principal),
                       db: Session = Depends(get_db)):
    if body.workspace_id is not None:
        _owner(db, principal, body.workspace_id)
    try:
        prop = imp.create_proposal(
            db, domain=body.domain, title=body.title,
            problem=body.problem, proposed_change=body.proposed_change,
            evidence=body.evidence,
            expected_benefit=body.expected_benefit, risk=body.risk,
            estimated_cost=body.estimated_cost,
            evaluation_requirements=body.evaluation_requirements,
            author_source=body.author_source,
            author_user_id=principal.user.id,
            workspace_id=body.workspace_id)
        db.commit()
        return {"proposal_id": prop.id, "status": prop.status}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/improvements/{proposal_id}/audit")
def improvement_audit(proposal_id: int,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    try:
        return imp.proposal_audit_trail(db, proposal_id=proposal_id)
    except Exception:
        raise HTTPException(status_code=404, detail="proposal not found")


@router.post("/improvements/{proposal_id}/transition")
def transition(body: TransitionIn, proposal_id: int,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    try:
        result = imp.transition(db, proposal_id, body.to_state,
                                actor_user_id=principal.user.id,
                                reason=body.reason, evidence=body.evidence)
        db.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail="proposal not found")


# ===========================================================================
# Experiments
# ===========================================================================

class DatasetIn(BaseModel):
    name: str = Field(..., max_length=120)
    kind: str = "golden"
    domain: str = "retrieval"
    items: list


class ExperimentIn(BaseModel):
    name: str = Field(..., max_length=200)
    domain: str
    config: dict
    dataset_id: Optional[int] = None
    proposal_id: Optional[int] = None
    workspace_id: Optional[int] = None


class RunIn(BaseModel):
    metrics: dict
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    cost: Optional[float] = None
    latency_ms: Optional[float] = None
    environment: Optional[str] = None


class CompareIn(BaseModel):
    baseline_run_id: int
    candidate_run_id: int
    metrics: list[str]


@router.get("/experiments")
def experiments(domain: Optional[str] = None,
                status: Optional[str] = None,
                workspace_id: Optional[int] = None,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    if workspace_id is not None:
        _owner(db, principal, workspace_id)
    return imp.list_experiments(db, domain=domain, status=status,
                                workspace_id=workspace_id)


@router.post("/experiments")
def create_experiment(body: ExperimentIn,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    if body.workspace_id is not None:
        _owner(db, principal, body.workspace_id)
    try:
        exp = imp.create_experiment(
            db, name=body.name, domain=body.domain, config=body.config,
            dataset_id=body.dataset_id, proposal_id=body.proposal_id,
            created_by=principal.user.id, workspace_id=body.workspace_id)
        db.commit()
        return {"experiment_id": exp.id,
                "config_fingerprint": exp.config_fingerprint}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/experiments/{experiment_id}/runs")
def experiment_runs(experiment_id: int,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    try:
        return imp.list_runs(db, experiment_id)
    except Exception:
        raise HTTPException(status_code=404, detail="experiment not found")


@router.post("/experiments/{experiment_id}/runs")
def create_run(experiment_id: int, body: RunIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    try:
        run = imp.record_run(
            db, experiment_id=experiment_id, metrics=body.metrics,
            dataset_id=body.dataset_id, dataset_name=body.dataset_name,
            model=body.model, provider=body.provider, cost=body.cost,
            latency_ms=body.latency_ms, environment=body.environment)
        db.commit()
        return {"run_id": run.id, "status": run.status}
    except Exception:
        raise HTTPException(status_code=404, detail="experiment not found")


@router.post("/experiments/{experiment_id}/compare")
def compare(body: CompareIn, experiment_id: int,
            principal: AuthPrincipal = Depends(get_current_principal),
            db: Session = Depends(get_db)):
    try:
        result = imp.compare_runs(db, experiment_id=experiment_id,
                                  baseline_run_id=body.baseline_run_id,
                                  candidate_run_id=body.candidate_run_id,
                                  metrics=body.metrics)
        db.commit()
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="run not found")


@router.get("/experiments/{experiment_id}/gate")
def promotion_gate(experiment_id: int, min_quality: float = 0.8,
                   max_cost: Optional[float] = None,
                   max_latency_ms: Optional[float] = None,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    try:
        return imp.promotion_eligible(
            db, experiment_id, min_quality=min_quality, max_cost=max_cost,
            max_latency_ms=max_latency_ms)
    except KeyError:
        raise HTTPException(status_code=404, detail="experiment not found")


@router.get("/datasets")
def datasets(domain: Optional[str] = None,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    return imp.list_datasets(db, domain=domain)


@router.post("/datasets")
def create_dataset(body: DatasetIn,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    try:
        ds = imp.create_dataset(db, name=body.name, kind=body.kind,
                                domain=body.domain, items=body.items,
                                created_by=principal.user.id)
        db.commit()
        return {"dataset_id": ds.id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ===========================================================================
# AI quality
# ===========================================================================

class ScorecardIn(BaseModel):
    domain: str
    dimensions: dict
    workspace_id: Optional[int] = None
    period: str = "daily"


@router.get("/quality/scorecards")
def scorecards(workspace_id: Optional[int] = None,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    return q3.scorecard_summary(db, workspace_id=workspace_id)


@router.post("/quality/scorecards")
def create_scorecard(body: ScorecardIn,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    if body.workspace_id is not None:
        _owner(db, principal, body.workspace_id)
    try:
        card = q3.compute_scorecard(
            db, domain=body.domain, dimensions=body.dimensions,
            workspace_id=body.workspace_id, period=body.period)
        db.commit()
        return {"scorecard_id": card.id, "overall_score": card.overall_score}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/quality/regression")
def regression(domain: str, workspace_id: Optional[int] = None,
               min_delta: float = 0.05,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    if workspace_id is not None:
        _owner(db, principal, workspace_id)
    result = q3.detect_regression(db, domain=domain,
                                  workspace_id=workspace_id,
                                  min_delta=min_delta)
    db.commit()
    return result


@router.get("/quality/trends")
def trends(domain: Optional[str] = None, period: Optional[str] = None,
           days: int = 30,
           principal: AuthPrincipal = Depends(get_current_principal),
           db: Session = Depends(get_db)):
    return q3.quality_trends(db, domain=domain, period=period, days=days)


@router.get("/quality/alerts")
def quality_alerts(domain: Optional[str] = None,
                   resolved: Optional[bool] = None,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    return q3.list_alerts(db, domain=domain, resolved=resolved)


# ===========================================================================
# Retrieval / RAG intelligence
# ===========================================================================

class RetrievalFailureIn(BaseModel):
    workspace_id: int
    query: str
    failure_class: str
    detail: Optional[str] = None


@router.get("/retrieval/failures")
def retrieval_failures(workspace_id: int, limit: int = 100,
                       principal: AuthPrincipal = Depends(
                           get_current_principal),
                       db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return reti.failure_summary(db, workspace_id=workspace_id,
                                limit=limit)


@router.post("/retrieval/failures")
def retrieval_failure(body: RetrievalFailureIn,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        f = reti.classify_failure(
            db, workspace_id=body.workspace_id,
            query=body.query, failure_class=body.failure_class,
            detail=body.detail)
        db.commit()
        return {"failure_id": f.id, "failure_class": f.failure_class}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/retrieval/recommendations")
def retrieval_recommendations(workspace_id: int,
                              principal: AuthPrincipal = Depends(
                                  get_current_principal),
                              db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return reti.list_recommendations(db, workspace_id=workspace_id)


@router.post("/retrieval/recommendations/generate")
def generate_retrieval_recos(workspace_id: int,
                             principal: AuthPrincipal = Depends(
                                 get_current_principal),
                             db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    created = reti.generate_recommendations(db, workspace_id=workspace_id)
    db.commit()
    return {"created": created}


@router.get("/retrieval/query-quality")
def query_quality(workspace_id: int,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return reti.query_quality_analysis(db, workspace_id=workspace_id)


class RagFailureIn(BaseModel):
    workspace_id: int
    failure_class: str
    claim: Optional[str] = None
    execution_id: Optional[str] = None
    detail: Optional[str] = None


@router.get("/rag/failures")
def rag_failures(workspace_id: int, limit: int = 200,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return ragi.failure_analysis(db, workspace_id=workspace_id, limit=limit)


@router.post("/rag/failures")
def rag_failure(body: RagFailureIn,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        f = ragi.record_failure(
            db, workspace_id=body.workspace_id,
            failure_class=body.failure_class, claim=body.claim,
            execution_id=body.execution_id, detail=body.detail)
        db.commit()
        return {"failure_id": f.id, "failure_class": f.failure_class}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/rag/recommendations")
def rag_recommendations(workspace_id: int,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return ragi.repair_recommendations(db, workspace_id=workspace_id)


class RagPipelineIn(BaseModel):
    config: dict
    proposal_id: Optional[int] = None
    dataset_id: Optional[int] = None


@router.post("/rag/pipelines")
def rag_pipeline(body: RagPipelineIn,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    try:
        pipe = ragi.run_evaluation_pipeline(
            db, config=body.config, proposal_id=body.proposal_id,
            dataset_id=body.dataset_id)
        db.commit()
        return {"pipeline_id": pipe.id, "gate_passed": pipe.gate_passed}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ===========================================================================
# Knowledge intelligence
# ===========================================================================

@router.get("/knowledge/freshness")
def knowledge_freshness(workspace_id: int, stale_days: int = 90,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return ki.freshness_report(db, workspace_id=workspace_id,
                               stale_days=stale_days)


@router.get("/knowledge/health")
def knowledge_health(workspace_id: int,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    ki.compute_health(db, workspace_id=workspace_id)
    db.commit()
    return ki.latest_health(db, scope_type="WORKSPACE",
                            scope_id=workspace_id)


@router.get("/knowledge/gaps")
def knowledge_gaps(workspace_id: int, min_attempts: int = 1,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return {"gaps": ki.list_gaps(db, workspace_id=workspace_id,
                                 min_attempts=min_attempts),
            "recommendations": ki.gap_recommendations(
                db, workspace_id=workspace_id)}


@router.post("/knowledge/gaps")
def record_gap(workspace_id: int, query: str,
               evidence_score: Optional[float] = None,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    gap = ki.record_gap(db, workspace_id=workspace_id, query=query,
                        evidence_score=evidence_score)
    db.commit()
    return {"gap_id": gap.id, "attempts": gap.attempts}


class DocChangeIn(BaseModel):
    document_id: int
    workspace_id: int
    change_class: str
    version_from: Optional[int] = None
    version_to: Optional[int] = None


@router.post("/documents/changes")
def doc_change(body: DocChangeIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        ev = ki.classify_change(
            db, document_id=body.document_id,
            workspace_id=body.workspace_id, change_class=body.change_class,
            version_from=body.version_from, version_to=body.version_to,
            impact=ki.change_impact(body.change_class))
        db.commit()
        return {"event_id": ev.id, "change_class": ev.change_class,
                "impact": ki.change_impact(body.change_class),
                "reprocessing_plan": ki.reprocessing_plan(
                    body.change_class)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/documents/{document_id}/health")
def document_health(document_id: int,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    try:
        return ki.document_health(db, document_id=document_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="document not found")


# ===========================================================================
# Policy intelligence
# ===========================================================================

class PolicyIn(BaseModel):
    scope_type: str
    scope_id: Optional[int] = None
    policy: dict
    reason: Optional[str] = None


@router.get("/policy/versions")
def policy_versions(scope_type: str, scope_id: Optional[int] = None,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    return pi.list_versions(db, scope_type=scope_type, scope_id=scope_id)


@router.post("/policy/versions")
def policy_version(body: PolicyIn,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    try:
        row = pi.version_policy(db, scope_type=body.scope_type,
                                scope_id=body.scope_id,
                                policy=body.policy,
                                actor_user_id=principal.user.id,
                                reason=body.reason)
        db.commit()
        return {"version": row.version, "diff": row.diff_json}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/policy/drift")
def policy_drift(scope_type: str, scope_id: Optional[int] = None,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    return pi.drift_report(db, scope_type=scope_type, scope_id=scope_id)


@router.post("/policy/simulate")
def policy_simulate(operation: dict, policy: dict,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    return pi.simulate(db, operation=operation, policy=policy)


@router.get("/policy/conflicts")
def policy_conflicts(policies: list[dict],
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    return pi.detect_conflicts(policies)


# ===========================================================================
# Provider / model intelligence
# ===========================================================================

class ProfileIn(BaseModel):
    provider: str
    model: str
    metrics: dict
    period: str = "daily"


@router.get("/providers/profiles")
def provider_profiles(provider: Optional[str] = None,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    return pri.profile_summary(db, provider=provider)


@router.post("/providers/profiles")
def provider_profile(body: ProfileIn,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    row = pri.record_profile(db, provider=body.provider, model=body.model,
                             metrics=body.metrics, period=body.period)
    db.commit()
    return {"profile_id": row.id}


@router.get("/providers/routing-recommendations")
def routing_recos(status: Optional[str] = None,
                  principal: AuthPrincipal = Depends(
                      get_current_principal),
                  db: Session = Depends(get_db)):
    return pri.list_routing_recommendations(db, status=status)


@router.post("/providers/routing-recommendations/generate")
def generate_routing_recos(principal: AuthPrincipal = Depends(
    get_current_principal),
    db: Session = Depends(get_db)):
    created = pri.routing_recommendations(db)
    db.commit()
    return {"created": created}


@router.post("/providers/simulate-routing")
def simulate_routing(workload: list[dict], candidates: list[dict],
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    return pri.simulate_routing(db, workload=workload,
                                candidates=candidates)


@router.post("/providers/anomalies/detect")
def provider_anomalies(provider: Optional[str] = None,
                       threshold: float = 3.0,
                       principal: AuthPrincipal = Depends(
                           get_current_principal),
                       db: Session = Depends(get_db)):
    anomalies = pri.detect_anomalies(db, provider=provider,
                                     threshold=threshold)
    db.commit()
    return {"anomalies": anomalies}


# ===========================================================================
# Cost intelligence
# ===========================================================================

class BaselineIn(BaseModel):
    scope_type: str
    scope_id: Optional[int] = None
    baseline: dict
    dimension: Optional[str] = None


@router.post("/cost/baselines")
def cost_baseline(body: BaselineIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    try:
        row = ci.set_baseline(db, scope_type=body.scope_type,
                              scope_id=body.scope_id,
                              baseline=body.baseline,
                              dimension=body.dimension)
        db.commit()
        return {"baseline_id": row.id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/cost/drift")
def cost_drift(scope_type: str, scope_id: Optional[int] = None,
               sample_cost: Optional[float] = None,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    return ci.detect_cost_drift(db, scope_type=scope_type,
                                scope_id=scope_id,
                                sample_cost=sample_cost)


@router.get("/cost/token-efficiency")
def token_efficiency(workspace_id: int,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return ci.token_efficiency_report(db, workspace_id=workspace_id)


@router.post("/cost/token-efficiency")
def record_token_efficiency(workspace_id: int, input_tokens: int,
                            output_tokens: int, context_tokens: int = 0,
                            repeated_tokens: int = 0,
                            execution_ref: Optional[str] = None,
                            principal: AuthPrincipal = Depends(
                                get_current_principal),
                            db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    row = ci.record_token_usage(
        db, workspace_id=workspace_id, input_tokens=input_tokens,
        output_tokens=output_tokens, context_tokens=context_tokens,
        repeated_tokens=repeated_tokens, execution_ref=execution_ref)
    db.commit()
    return {"token_id": row.id}


@router.get("/cost/optimization-candidates")
def cost_optimization(workspace_id: int,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return ci.cost_optimization_candidates(db, workspace_id=workspace_id)


# ===========================================================================
# Agent / workflow / memory / graph / search intelligence
# ===========================================================================

@router.get("/agents/intelligence")
def agent_intelligence(workspace_id: int,
                       principal: AuthPrincipal = Depends(
                           get_current_principal),
                       db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return {"success": ai2.success_metrics(db, workspace_id=workspace_id),
            "failures": ai2.failure_analysis(db, workspace_id=workspace_id),
            "safety": ai2.safety_score(db, workspace_id=workspace_id)}


@router.get("/workflows/intelligence")
def workflow_intelligence(workspace_id: int,
                          principal: AuthPrincipal = Depends(
                              get_current_principal),
                          db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return {"analytics": wi.success_analytics(db,
                                              workspace_id=workspace_id),
            "bottlenecks": wi.bottlenecks(db, workspace_id=workspace_id),
            "hotspots": wi.failure_hotspots(db, workspace_id=workspace_id),
            "recommendations": wi.optimization_recommendations(
                db, workspace_id=workspace_id)}


@router.get("/memory/quality")
def memory_quality(workspace_id: int,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return {"quality": mi.memory_quality(db, workspace_id=workspace_id),
            "decay": mi.decay_report(db, workspace_id=workspace_id),
            "conflict_queue": mi.conflict_queue(db,
                                                workspace_id=workspace_id)}


@router.post("/memory/{memory_id}/controls")
def memory_control(memory_id: int, workspace_id: int, action: str,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    try:
        result = mi.user_controls(db, memory_id=memory_id,
                                  workspace_id=workspace_id, action=action,
                                  user_id=principal.user.id)
        db.commit()
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="memory not found")
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=403 if isinstance(
            exc, PermissionError) else 400, detail=str(exc))


@router.get("/graph/health")
def graph_health(workspace_id: int,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    gi.graph_health(db, workspace_id=workspace_id)
    db.commit()
    return {"health": gi.health_summary(db, workspace_id=workspace_id),
            "drift": gi.graph_drift(db, workspace_id=workspace_id)}


@router.get("/search/intelligence")
def search_intelligence(workspace_id: int,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return si.quality_score(db, workspace_id=workspace_id)


@router.post("/search/events")
def search_event(workspace_id: int, query: str, event_type: str,
                 result_count: Optional[int] = None,
                 latency_ms: Optional[float] = None,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    try:
        ev = reti.record_search_event(
            db, workspace_id=workspace_id, query=query,
            event_type=event_type, result_count=result_count,
            latency_ms=latency_ms)
        db.commit()
        return {"event_id": ev.id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ===========================================================================
# Feedback intelligence
# ===========================================================================

class FeedbackIn(BaseModel):
    workspace_id: int
    source: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    target_id: Optional[str] = None


@router.post("/feedback")
def feedback(body: FeedbackIn,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    try:
        row = fi.record_feedback(db, workspace_id=body.workspace_id,
                                 source=body.source, rating=body.rating,
                                 comment=body.comment,
                                 target_id=body.target_id,
                                 user_id=principal.user.id)
        db.commit()
        return {"feedback_id": row.id, "status": row.status}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/feedback/quality")
def feedback_quality(workspace_id: int,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return fi.feedback_quality(db, workspace_id=workspace_id)


@router.post("/feedback/{feedback_id}/promote")
def feedback_promote(feedback_id: int, dataset_name: str,
                     reason: Optional[str] = None,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    try:
        result = fi.promote_to_golden(
            db, feedback_id=feedback_id, dataset_name=dataset_name,
            authorized_by=principal.user.id, reason=reason)
        db.commit()
        return result
    except (KeyError, ValueError, PermissionError) as exc:
        code = 404 if isinstance(exc, KeyError) else \
            (403 if isinstance(exc, PermissionError) else 400)
        raise HTTPException(status_code=code, detail=str(exc))


@router.get("/feedback/summary")
def feedback_summary(workspace_id: int,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return fi.feedback_summary(db, workspace_id=workspace_id)


# ===========================================================================
# Governance 5.0 + safety 9.0
# ===========================================================================

class GovChangeIn(BaseModel):
    policy_type: str
    scope_type: str
    scope_id: Optional[int] = None
    policy: dict
    reason: str
    evaluation_ref: Optional[str] = None
    approval_ref: Optional[str] = None


@router.post("/governance/changes")
def governance_change(body: GovChangeIn,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    try:
        row = g5.record_change(
            db, owner_user_id=principal.user.id,
            policy_type=body.policy_type, scope_type=body.scope_type,
            scope_id=body.scope_id, policy=body.policy,
            reason=body.reason, evaluation_ref=body.evaluation_ref,
            approval_ref=body.approval_ref)
        db.commit()
        return {"version": row.version}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/governance/effective")
def governance_effective(policies: list[dict],
                         principal: AuthPrincipal = Depends(
                             get_current_principal),
                         db: Session = Depends(get_db)):
    return g5.effective_policy(policies)


@router.post("/governance/simulate")
def governance_simulate(operation: dict, policy: dict,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    return g5.simulate(db, operation=operation, policy=policy)


@router.get("/governance/audit")
def governance_audit(scope_type: str, scope_id: Optional[int] = None,
                     principal: AuthPrincipal = Depends(
                         get_current_principal),
                     db: Session = Depends(get_db)):
    return g5.change_audit(db, scope_type=scope_type, scope_id=scope_id)


@router.get("/safety/injection-corpus")
def safety_injection_corpus(principal: AuthPrincipal = Depends(
    get_current_principal),
    db: Session = Depends(get_db)):
    return s9.run_injection_corpus()


@router.post("/safety/injection-corpus")
def safety_injection_cases(cases: list[dict],
                          principal: AuthPrincipal = Depends(
                              get_current_principal),
                          db: Session = Depends(get_db)):
    return s9.run_injection_corpus(corpus=cases)


@router.post("/safety/exfiltration")
def safety_exfiltration(queries: list[str],
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    return s9.run_exfiltration_suite(queries)


@router.post("/safety/tool-abuse")
def safety_tool_abuse(plan: dict,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    return s9.tool_abuse_checks(plan)


@router.post("/safety/output")
def safety_output(text: str,
                  principal: AuthPrincipal = Depends(
                      get_current_principal),
                  db: Session = Depends(get_db)):
    return s9.output_security(text)


# ===========================================================================
# Observability 5.0: incidents + SLO + alerts
# ===========================================================================

class IncidentIn(BaseModel):
    title: str
    severity: str = "MEDIUM"
    affected_systems: list[str] = []
    summary: Optional[str] = None


@router.get("/incidents")
def incidents(status: Optional[str] = None,
              severity: Optional[str] = None,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    return ob5.list_incidents(db, status=status, severity=severity)


@router.post("/incidents")
def create_incident(body: IncidentIn,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    inc = ob5.correlate_incident(db, title=body.title,
                                 severity=body.severity,
                                 affected_systems=body.affected_systems,
                                 summary=body.summary or "")
    db.commit()
    return {"incident_id": inc.id, "status": inc.status,
            "correlated": inc.affected_systems is not None}


@router.get("/incidents/{incident_id}/timeline")
def incident_timeline(incident_id: int,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    return ob5.incident_timeline(db, incident_id)


@router.post("/incidents/{incident_id}/transition")
def incident_transition(incident_id: int, to_status: str,
                        detail: Optional[str] = None,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    try:
        result = ob5.transition_incident(db, incident_id=incident_id,
                                         to_status=to_status,
                                         actor_user_id=principal.user.id,
                                         detail=detail)
        db.commit()
        return result
    except (KeyError, ValueError) as exc:
        code = 404 if isinstance(exc, KeyError) else 400
        raise HTTPException(status_code=code, detail=str(exc))


@router.get("/health/score")
def system_health(factors: str,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    """factors: JSON dict of 0..1 factors (availability, latency, quality,
    cost, error_rate, queue_health, provider_health)."""
    import json
    try:
        parsed = json.loads(factors)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400,
                            detail="factors must be a JSON dict")
    return ob5.ai_system_health(parsed)


@router.get("/dependency-graph")
def dependency_graph(principal: AuthPrincipal = Depends(
    get_current_principal),
    db: Session = Depends(get_db)):
    return ob5.dependency_graph()


@router.get("/slo/history")
def slo_history(name: Optional[str] = None, days: int = 30,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    return slo2.slo_recent(db, name=name, days=days)


@router.post("/slo/history")
def record_slo(name: str, measured_value: float,
               target: Optional[float] = None,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    row = slo2.record_history(db, name=name, measured_value=measured_value,
                              target=target)
    db.commit()
    return {"history_id": row.id, "met": row.met}


@router.get("/slo/error-budget")
def error_budget_status(name: str, budget: float,
                        consumed_override: Optional[float] = None,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    row = slo2.error_budget(db, name=name, budget=budget,
                            consumed_override=consumed_override)
    db.commit()
    policy = slo2.error_budget_policy(row)
    return {"name": row.name, "budget": row.budget,
            "consumed": row.consumed, "status": row.status, "policy": policy}


@router.get("/alerts")
def alerts(status: Optional[str] = None, severity: Optional[str] = None,
           category: Optional[str] = None,
           principal: AuthPrincipal = Depends(get_current_principal),
           db: Session = Depends(get_db)):
    return ni.list_alerts(db, status=status, severity=severity,
                          category=category)


@router.post("/alerts/raise")
def raise_alert(category: str, severity: str, message: str,
                workspace_id: Optional[int] = None,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    try:
        row = ni.raise_alert(db, category=category, severity=severity,
                             message=message, workspace_id=workspace_id)
        db.commit()
        return {"alert_id": row.id, "severity": row.severity,
                "occurrence_count": row.occurrence_count}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/alerts/summary")
def alerts_summary(principal: AuthPrincipal = Depends(
    get_current_principal),
    db: Session = Depends(get_db)):
    return ni.alert_summary(db)


@router.post("/notifications/preferences")
def notification_pref(user_id: int, category: str, enabled: bool,
                      workspace_id: Optional[int] = None,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    if principal.user.id != user_id:
        raise HTTPException(status_code=403,
                            detail="Cannot edit another user's preferences")
    try:
        row = ni.set_preference(db, user_id=user_id, category=category,
                                enabled=enabled,
                                workspace_id=workspace_id)
        db.commit()
        return {"preference_id": row.id, "enabled": row.enabled}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ===========================================================================
# Reports + API / DB health
# ===========================================================================

class ReportIn(BaseModel):
    kind: str
    workspace_id: Optional[int] = None
    content: Optional[dict] = None


@router.post("/reports")
def create_report(body: ReportIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    if body.content is None:
        raise HTTPException(status_code=400,
                            detail="content is required")
    try:
        row = rep.create_report(
            db, kind=body.kind, scope_type="WORKSPACE",
            scope_id=body.workspace_id, content=body.content,
            generated_by=principal.user.id)
        db.commit()
        return {"report_id": row.id, "version": row.version}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/reports")
def reports(kind: Optional[str] = None,
            principal: AuthPrincipal = Depends(get_current_principal),
            db: Session = Depends(get_db)):
    return rep.list_reports(db, kind=kind)


@router.get("/api/health")
def api_health(principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    return apii.api_health_report(db)


@router.post("/api/health")
def record_api_health(endpoint: str, requests: int = 1, errors: int = 0,
                      auth_failures: int = 0,
                      latency_p50_ms: Optional[float] = None,
                      latency_p95_ms: Optional[float] = None,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    row = apii.record_metric(db, endpoint=endpoint, requests=requests,
                             errors=errors, auth_failures=auth_failures,
                             latency_p50_ms=latency_p50_ms,
                             latency_p95_ms=latency_p95_ms)
    db.commit()
    return {"metric_id": row.id}


@router.get("/api/abuse")
def api_abuse(principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    return apii.abuse_detection(db)


@router.get("/database/health")
def database_health(principal: AuthPrincipal = Depends(
    get_current_principal),
    db: Session = Depends(get_db)):
    return {"health": dbi.health_report(db),
            "migration": dbi.migration_head(db),
            "growth": dbi.table_growth(db, counts=None)}


@router.post("/database/metrics")
def database_metric(metric: str, value: Optional[float] = None,
                    detail: Optional[dict] = None,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    row = dbi.record_metric(db, metric=metric, value=value, detail=detail)
    db.commit()
    return {"metric_id": row.id}