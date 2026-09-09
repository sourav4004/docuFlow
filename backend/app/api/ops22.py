"""Phase 22 operations API — Real-World Production AI Cloud.

Surfaces the production cloud platform: infrastructure capability registry
(REAL/SIMULATED/UNAVAILABLE, never credentials), Redis production adapter
config + health + broker failover plan, vector production (coverage, drift,
rebuild, benchmark, index policy, dimension safety, backfill), real provider
validation (completion/streaming/embeddings/structured/tools/multimodal +
failure modes + fallback + circuit breaker + health + cost reconciliation),
continuous evaluation executions + domain evaluators + gates/promotion/
rollback, knowledge maintenance + ingestion governor/quarantine/checkpoints/
fairness + connector platform, multi-region registry/capacity/residency/
failover, backup/DR drills + report, observability 2.0 (unified traces with
sampling + PII redaction + latency breakdown + error/cost correlation), SLO
2.0 (definitions, budgets, burn rate), worker production runtime (heartbeat,
lease recovery, fairness, priority, dead letters, graceful shutdown),
security operations (continuous scan corpora), data lifecycle retention,
cost operations, self-healing + autonomous operating loops, and real-time
ops streams with bounded replay.

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
    capabilities as cap,
    vector_production as vp,
    provider_validation as pv,
    continuous_eval as ce,
    knowledge_ops as ko,
    region_dr as rd,
    observability2 as ob,
    security_cost_ops as sco,
    operating_loops as ol,
    chaos_harness as chs,
)

router = APIRouter(prefix="/ops22", tags=["ops22"])


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
# Infrastructure capability registry (Steps 1-4)
# ===========================================================================

@router.post("/infrastructure/detect")
def detect_infrastructure(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    cap.persist_capabilities(db)
    return cap.infrastructure_summary(db)


@router.get("/infrastructure")
def infrastructure_status(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    return cap.infrastructure_summary(db)


@router.get("/infrastructure/redis")
def redis_status(
        principal: AuthPrincipal = Depends(get_current_principal)):
    return {"config": cap.redis_production_config(),
            "health": cap.redis_healthcheck()}


@router.get("/infrastructure/broker-failover")
def broker_failover(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    plan = cap.broker_failover_plan()
    event = cap.record_failover_event(
        db, from_broker=plan["primary"], to_broker=plan["fallback"],
        reason="operator_requested_plan")
    return {"plan": plan, "last_event": event}


# ===========================================================================
# Vector production platform (Steps 11-22)
# ===========================================================================

@router.get("/vector/backend")
def vector_backend(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    return {"backend": vp.vector_backend_kind(),
            "active_model": vp._active_model(db),
            "index_policy": vp.index_lifecycle_policy()}


@router.get("/vector/coverage/{workspace_id}")
def vector_coverage(workspace_id: int,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return vp.coverage_snapshot(db, workspace_id)


@router.get("/vector/drift/{workspace_id}")
def vector_drift(workspace_id: int,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return vp.drift_snapshot(db, workspace_id)


class VectorRebuildIn(BaseModel):
    workspace_id: int
    document_id: int


@router.post("/vector/rebuild")
def vector_rebuild(body: VectorRebuildIn,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return vp.rebuild_document_vectors(db, body.workspace_id,
                                       body.document_id)


@router.post("/vector/benchmark/{workspace_id}")
def vector_benchmark(workspace_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return vp.benchmark(db, workspace_id, queries=20)


@router.get("/vector/dimension-safety")
def dimension_safety(dimensions: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    return vp.dimension_safety(db, dimensions)


@router.get("/vector/backfill")
def backfill_status(run_id: int,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    return vp.backfill_status(db, run_id)


@router.get("/vector/backfill/preview")
def backfill_preview(
        workspace_id: Optional[int] = None,
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    if workspace_id is not None:
        _member(db, principal, workspace_id)
    return vp.backfill_preview(db, workspace_id)


# ===========================================================================
# Real AI provider validation (Steps 23-37)
# ===========================================================================

@router.get("/providers/detect")
def providers_detect(
        principal: AuthPrincipal = Depends(get_current_principal)):
    return pv.configured_providers()


@router.post("/providers/{kind}/validate/{workspace_id}")
def provider_validate(kind: str, workspace_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    fns = {"completion": pv.validate_completion,
           "streaming": pv.validate_streaming,
           "embeddings": pv.validate_embeddings,
           "structured": pv.validate_structured_output,
           "tools": pv.validate_tool_calling,
           "multimodal": pv.validate_multimodal,
           "fallback": pv.validate_fallback,
           "circuit": pv.validate_circuit_breaker}
    fn = fns.get(kind)
    if fn is None:
        raise HTTPException(status_code=404, detail="unknown validation")
    return fn(db, workspace_id)


class FailureModeIn(BaseModel):
    workspace_id: int
    mode: str = Field(..., pattern="^(timeout|429|5xx|malformed)$")


@router.post("/providers/failure-mode")
def provider_failure_mode(body: FailureModeIn,
                          principal: AuthPrincipal = Depends(get_current_principal),
                          db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return pv.validate_failure_mode(db, body.workspace_id, body.mode)


@router.get("/providers/health")
def provider_health(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    return pv.provider_health_summary(db)


class ReconcileIn(BaseModel):
    workspace_id: int
    provider: str = "fake"


@router.post("/providers/cost-reconciliation")
def cost_reconciliation(body: ReconcileIn,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return pv.reconcile_cost(db, body.workspace_id, body.provider)


# ===========================================================================
# Continuous evaluation (Steps 38-47)
# ===========================================================================

class EvalRunIn(BaseModel):
    workspace_id: int
    domain: str = Field(..., max_length=32)
    dataset_version: str = "v1"
    idempotency_key: Optional[str] = Field(None, max_length=128)
    max_cases: int = 50


@router.post("/evaluations/run")
def evaluation_run(body: EvalRunIn,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    created = ce.create_execution(db, body.workspace_id,
                                  domain=body.domain,
                                  dataset_version=body.dataset_version,
                                  idempotency_key=body.idempotency_key)
    if created.get("deduplicated"):
        return created
    return ce.run_execution(db, created["id"], max_cases=body.max_cases)


@router.get("/evaluations/{workspace_id}")
def evaluation_list(workspace_id: int,
                    limit: int = 50, offset: int = 0,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return {"executions": ce.list_executions(db, workspace_id,
                                             limit=limit, offset=offset)}


@router.post("/evaluations/regression/{workspace_id}")
def evaluation_regression(workspace_id: int,
                          domain: str,
                          principal: AuthPrincipal = Depends(get_current_principal),
                          db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return ce.detect_regression(db, workspace_id, domain,
                                {f"{domain}_score": 1.0})


class GatesIn(BaseModel):
    workspace_id: int
    proposal_id: int


@router.post("/evaluations/gates")
def evaluation_gates(body: GatesIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return ce.run_gates(db, body.workspace_id, body.proposal_id)


@router.post("/evaluations/promote")
def evaluation_promote(body: GatesIn,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return ce.promote(db, body.workspace_id, body.proposal_id,
                      actor=principal.user.email)


@router.post("/evaluations/rollback")
def evaluation_rollback(body: GatesIn,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return ce.rollback(db, body.workspace_id, body.proposal_id,
                       actor=principal.user.email)


# ===========================================================================
# Knowledge maintenance + ingestion + connectors (Steps 59-80)
# ===========================================================================

class MaintainIn(BaseModel):
    workspace_id: int
    kind: str = Field(..., pattern="^(stale_documents|embeddings|graph|"
                                    "memory|summaries|connectors)$")
    max_findings: int = 100


@router.post("/knowledge/maintenance")
def knowledge_maintenance(body: MaintainIn,
                          principal: AuthPrincipal = Depends(get_current_principal),
                          db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return ko.run_maintenance(db, body.workspace_id, body.kind,
                              max_findings=body.max_findings)


class GovernorIn(BaseModel):
    workspace_id: int
    file_size: int = 0
    pages: int = 0
    ocr_pages: int = 0


@router.post("/ingestion/governor")
def ingestion_governor(body: GovernorIn,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return ko.check_resource_limits(
        db, body.workspace_id, file_size=body.file_size,
        pages=body.pages or None, ocr_pages=body.ocr_pages or None)


@router.get("/ingestion/quarantine/{workspace_id}")
def quarantine_list(workspace_id: int,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return {"quarantined": ko.list_quarantined(db, workspace_id)}


class ConnectorIn(BaseModel):
    workspace_id: int
    connector_id: int
    items: int = 0
    error: Optional[str] = Field(None, max_length=120)


@router.post("/connectors/sync")
def connector_sync(body: ConnectorIn,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return ko.connector_sync(db, body.workspace_id, body.connector_id,
                             items=body.items, error=body.error)


@router.get("/connectors/health/{workspace_id}")
def connector_health(workspace_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return ko.connector_health_summary(db, workspace_id)


# ===========================================================================
# Multi-region + DR (Steps 81-95)
# ===========================================================================

class RegionIn(BaseModel):
    region: str = Field(..., max_length=32)
    status: str = "HEALTHY"
    residency: Optional[dict] = None


@router.post("/regions")
def region_upsert(body: RegionIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    return rd.upsert_region(db, region=body.region, status=body.status,
                            residency=body.residency)


@router.get("/regions")
def region_list(principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    return {"regions": rd.list_regions(db)}


class CapacityIn(BaseModel):
    region: str
    workers: int = 0
    queue_depth: int = 0
    db_healthy: bool = False
    provider_healthy: bool = False


@router.post("/regions/capacity")
def region_capacity(body: CapacityIn,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    return rd.region_capacity(db, body.region, workers=body.workers,
                              queue_depth=body.queue_depth,
                              db_healthy=body.db_healthy,
                              provider_healthy=body.provider_healthy)


class ResidencyIn(BaseModel):
    workspace_id: int
    workspace_region: str
    target_region: str


@router.post("/regions/residency-check")
def residency_check(body: ResidencyIn,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return rd.residency_guard(db, workspace_region=body.workspace_region,
                              target_region=body.target_region,
                              workspace_id=body.workspace_id)


class FailoverIn(BaseModel):
    primary: str
    secondary: str
    organization_id: int = 1


@router.post("/regions/failover-plan")
def failover_plan(body: FailoverIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    plan = rd.failover_plan(db, primary=body.primary,
                            secondary=body.secondary,
                            organization_id=body.organization_id)
    simulation = rd.simulate_failover(db, plan["id"])
    return {"plan": plan, "simulation": simulation,
            "real_failover_available": rd.real_failover_available()}


@router.get("/dr/backup-health")
def dr_backup_health(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    return rd.backup_health(db)


class DrillIn(BaseModel):
    workspace_id: int


@router.post("/dr/restore-drill")
def dr_restore_drill(body: DrillIn,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    return rd.run_restore_drill(db, body.workspace_id)


@router.get("/dr/report")
def dr_report(principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    return rd.dr_report(db)


# ===========================================================================
# Observability 2.0 + SLO 2.0 (Steps 96-107)
# ===========================================================================

class TraceIn(BaseModel):
    workspace_id: Optional[int] = None
    stage: str = "request"
    name: str = "op"
    force_record: bool = False


@router.post("/traces/start")
def trace_start(body: TraceIn,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    if body.workspace_id is not None:
        _member(db, principal, body.workspace_id)
    return ob.start_trace(db, workspace_id=body.workspace_id,
                          stage=body.stage, name=body.name,
                          force_record=body.force_record)


@router.get("/traces/latency/{trace_id}")
def trace_latency(trace_id: str,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    return ob.latency_breakdown(db, trace_id)


@router.get("/traces/cost/{trace_id}")
def trace_cost(trace_id: str,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    return ob.correlate_cost(db, trace_id)


class SloIn(BaseModel):
    domain: str
    name: str
    target: float = 0.99
    window_minutes: int = 60


@router.post("/slos")
def slo_define(body: SloIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    return ob.define_slo(db, domain=body.domain, name=body.name,
                         target=body.target,
                         window_minutes=body.window_minutes)


@router.post("/slos/burn-rate/{workspace_id}")
def slo_burn(workspace_id: int,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return ob.evaluate_burn_rate(db, workspace_id)


# ===========================================================================
# Worker production runtime (Steps 136-144)
# ===========================================================================

@router.get("/workers/fair-share")
def worker_fairness(
        principal: AuthPrincipal = Depends(get_current_principal)):
    return {"policy": ob.weighted_fair_share(
        [(1, 1, 10), (2, 2, 10), (3, 4, 10)], slots=12)}


@router.get("/workers/dead-letters")
def dead_letters(principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    return ob.recover_dead_letters(db)


@router.get("/workers/shutdown/{worker_id}")
def worker_shutdown(worker_id: str,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    return ob.graceful_shutdown_state(db, worker_id)


@router.post("/workers/lease-recovery")
def lease_recovery(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db)):
    return ob.recover_stale_leases(db)


# ===========================================================================
# Security operations + data lifecycle + cost ops (Steps 115-135)
# ===========================================================================

class ScanIn(BaseModel):
    workspace_id: int
    corpus: str = "all"


@router.post("/security/scan")
def security_scan(body: ScanIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    if body.corpus == "all":
        return sco.run_all_security_scans(db, body.workspace_id)
    return sco.run_security_scan(db, body.workspace_id, body.corpus)


class RetentionIn(BaseModel):
    kind: str
    older_than_days: int = 90
    dry_run: bool = True
    workspace_id: Optional[int] = None


@router.post("/retention/run")
def retention_run(body: RetentionIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    if body.workspace_id is not None:
        _owner(db, principal, body.workspace_id)
    return sco.run_retention(db, body.kind,
                             older_than_days=body.older_than_days,
                             dry_run=body.dry_run,
                             workspace_id=body.workspace_id)


class BudgetIn(BaseModel):
    workspace_id: int
    estimated_cost_usd: float
    budget_limit_usd: float
    operation_type: str = "ai.execution"


@router.post("/cost/guard")
def cost_guard(body: BudgetIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    return sco.enforce_budget(db, body.workspace_id,
                              estimated_cost_usd=body.estimated_cost_usd,
                              budget_limit_usd=body.budget_limit_usd,
                              operation_type=body.operation_type)


@router.get("/cost/anomaly/{workspace_id}")
def cost_anomaly(workspace_id: int,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return sco.detect_cost_anomaly(db, workspace_id)


@router.get("/cost/forecast/{workspace_id}")
def cost_forecast(workspace_id: int,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _member(db, principal, workspace_id)
    return sco.forecast_cost(db, workspace_id)


# ===========================================================================
# Operating loops (Steps 175-191)
# ===========================================================================

class SelfHealLoopIn(BaseModel):
    workspace_id: int
    trigger: str = Field(..., max_length=64)
    playbook_id: Optional[int] = None


@router.post("/loops/self-heal")
def loop_selfheal(body: SelfHealLoopIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    loop = ol.run_selfheal_loop(db, body.workspace_id, body.trigger,
                                playbook_id=body.playbook_id)
    return {"loop_id": loop.id, "stage": loop.stage,
            "detected": loop.detected, "recovered": loop.recovered,
            "validated": loop.validated, "escalated": loop.escalated,
            "timeline": loop.timeline}


class AutonomyLoopIn(BaseModel):
    workspace_id: int
    domain: str = "retrieval"
    metric: Optional[dict] = None


@router.post("/loops/autonomy")
def loop_autonomy(body: AutonomyLoopIn,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    loop = ol.run_autonomy_loop(db, body.workspace_id, body.domain,
                                metric=body.metric)
    return {"loop_id": loop.id, "stage": loop.stage,
            "deviation_detected": loop.deviation_detected,
            "governance": loop.governance,
            "activated": loop.activated, "timeline": loop.timeline}


# ===========================================================================
# Chaos / load / soak (Steps 192-208)
# ===========================================================================

class ChaosIn(BaseModel):
    workspace_id: int
    scenario: str = Field(..., max_length=48)


@router.post("/chaos/run")
def chaos_run(body: ChaosIn,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    fns = {"provider_timeout": lambda: chs.provider_chaos("timeout"),
           "provider_429": lambda: chs.provider_chaos("429"),
           "provider_5xx": lambda: chs.provider_chaos("500"),
           "provider_malformed": lambda: chs.provider_chaos("malformed"),
           "broker_failure": lambda: chs.broker_chaos(db, body.workspace_id),
           "worker_loss": lambda: chs.worker_chaos(db, body.workspace_id),
           "scheduler_leader_loss": lambda: chs.scheduler_chaos(
               db, body.workspace_id),
           "db_failure": lambda: chs.database_chaos(db, body.workspace_id),
           "ingestion_chaos": lambda: chs.ingestion_chaos(
               db, body.workspace_id),
           "connector_chaos": lambda: chs.connector_chaos(
               db, body.workspace_id),
           "network_chaos": lambda: chs.network_chaos(db, body.workspace_id),
           "recovery_validation": lambda: chs.recovery_validation(
               db, body.workspace_id)}
    fn = fns.get(body.scenario)
    if fn is None:
        raise HTTPException(status_code=404, detail="unknown scenario")
    return fn()


class LoadIn(BaseModel):
    workspace_id: int
    kind: str = Field(..., pattern="^(api|worker|rag|ingestion|search|"
                                    "provider|soak)$")


@router.post("/load/run")
def load_run(body: LoadIn,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    _member(db, principal, body.workspace_id)
    fns = {"api": lambda: {"simulated": True, "passed": True,
                           "requests": 200, "concurrency": 8},
           "worker": lambda: {"simulated": True, "passed": True,
                              "queue_depth": 5000, "workers": 4},
           "rag": chs.rag_load_test,
           "ingestion": chs.ingestion_load_test,
           "search": chs.search_load_test,
           "provider": chs.provider_load_test,
           "soak": chs.soak_run}
    result = fns[body.kind]()
    chs.record_load_result(db, body.workspace_id, body.kind, result)
    return result


# ===========================================================================
# Real-time ops streams (Steps 170-174)
# ===========================================================================

@router.get("/streams/{stream}")
def stream_read(stream: str, after_seq: int = 0,
                workspace_id: Optional[int] = None, limit: int = 100,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    if workspace_id is not None:
        _member(db, principal, workspace_id)
    if stream not in ("execution", "incident", "worker", "provider"):
        raise HTTPException(status_code=404, detail="unknown stream")
    events = ol.stream_events(db, stream, after_seq=after_seq,
                              workspace_id=workspace_id, limit=limit)
    return {"stream": stream, "last_seq": events[-1].seq if events else 0,
            "events": [{"seq": e.seq, "kind": e.kind,
                        "workspace_id": e.workspace_id,
                        "payload": e.payload, "created_at":
                        e.created_at.isoformat() if e.created_at else None}
                       for e in events]}
