"""Phase 19 operations API — enterprise cloud console 4.0 surfaces.

Control-plane snapshots/rollback, region + residency + failover, worker
autoscaling/quarantine/backpressure, scheduler leadership, broker
guarantees/recovery, provider capability matrix + routing + admission,
cost reservation/reconciliation/forecast/anomalies, vector coverage/
lifecycle/rebuild, ingestion governor + quality, connector conflicts,
graph consistency, memory explainability, RAG quality, agent/workflow ops,
analytics, consistency checks, DR readiness, SLO health, quality gates,
search quality, and security check endpoints.

Every surface requires authentication; tenant-scoped writes require a
workspace owner/org-admin role. Only safe, non-secret data is returned.
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
    control_plane as cp,
    worker_ops2 as wo,
    provider_ops2 as po,
    cost4,
    vector_ops2 as vo,
    ingestion_ops3 as io3,
    federation_ops2 as fo2,
    kg5,
    memory4,
    rag7,
    agent4,
    workflow4,
    action3,
    safety8,
    governance4 as gov4,
    observability4 as ob4,
    search5,
    analytics2 as an,
    data_consistency as dc,
    dr2,
    api3,
    quality_platform as qp,
    broker2,
    perf2,
)

router = APIRouter(prefix="/ops19", tags=["ops19"])


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
# Control plane
# ===========================================================================

class SnapshotIn(BaseModel):
    scope_type: str = Field(..., max_length=20)
    scope_id: Optional[int] = None
    config: dict
    reason: Optional[str] = None


class RollbackIn(BaseModel):
    scope_type: str
    scope_id: Optional[int] = None
    target_version: int
    reason: Optional[str] = None


@router.get("/health")
def global_health(principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    return cp.global_health(db)


@router.post("/config/snapshots")
def create_snapshot(body: SnapshotIn,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    if body.scope_type != "GLOBAL" and body.scope_id is None:
        raise HTTPException(status_code=400,
                            detail="scope_id required for tenant scopes")
    snap = cp.snapshot_config(db, scope_type=body.scope_type,
                              scope_id=body.scope_id,
                              config=body.config,
                              actor_user_id=principal.user.id,
                              reason=body.reason)
    db.commit()
    return {"snapshot_id": snap.id, "version": snap.version,
            "is_active": snap.is_active}


@router.post("/config/activate")
def activate_snapshot(snapshot_id: int,
                      principal: AuthPrincipal = Depends(
                          get_current_principal),
                      db: Session = Depends(get_db)):
    result = cp.activate_snapshot(db, snapshot_id,
                                  actor_user_id=principal.user.id)
    db.commit()
    return result


@router.post("/config/rollback")
def rollback(body: RollbackIn,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    try:
        result = cp.rollback_snapshot(db, scope_type=body.scope_type,
                                      scope_id=body.scope_id,
                                      target_version=body.target_version,
                                      actor_user_id=principal.user.id,
                                      reason=body.reason)
        db.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/config")
def config_overview(scope_type: str, scope_id: Optional[int] = None,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    active = cp.latest_snapshot(db, scope_type=scope_type,
                                scope_id=scope_id, only_active=True)
    snapshots = cp.list_snapshots(db, scope_type=scope_type,
                                  scope_id=scope_id, limit=20)
    return {"active_version": active.version if active else None,
            "snapshots": [{"id": s.id, "version": s.version,
                           "active": s.is_active,
                           "reason": s.reason,
                           "created_at": s.created_at}
                          for s in snapshots]}


# ---------------------------------------------------------------------------
# Regions / residency / failover
# ---------------------------------------------------------------------------

class RegionIn(BaseModel):
    organization_id: Optional[int] = None
    region_id: str
    name: Optional[str] = None
    deployment: Optional[str] = None
    status: str = "HEALTHY"
    provider_availability: Optional[dict] = None
    vector_availability: Optional[dict] = None


class ResidencyIn(BaseModel):
    organization_id: Optional[int] = None
    classification: str
    allowed_regions: Optional[list] = None
    prohibited_regions: Optional[list] = None
    default_region: Optional[str] = None


class FailoverIn(BaseModel):
    organization_id: Optional[int] = None
    region_from: str
    region_to: str
    reason: Optional[str] = None


@router.get("/regions")
def regions(organization_id: Optional[int] = None,
            principal: AuthPrincipal = Depends(get_current_principal),
            db: Session = Depends(get_db)):
    return [{"region_id": r.region_id, "name": r.name,
             "status": r.status, "health_score": r.health_score,
             "failover_to": r.failover_to}
            for r in cp.list_regions(db, organization_id=organization_id)]


@router.put("/regions")
def upsert_region(body: RegionIn,
                  principal: AuthPrincipal = Depends(
                      get_current_principal),
                  db: Session = Depends(get_db)):
    row = cp.upsert_region(
        db, organization_id=body.organization_id,
        region_id=body.region_id, name=body.name,
        deployment=body.deployment, status=body.status,
        provider_availability=body.provider_availability,
        vector_availability=body.vector_availability)
    db.commit()
    return {"region_id": row.region_id, "status": row.status}


@router.put("/residency")
def residency(body: ResidencyIn,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    cp.set_residency_rule(db, organization_id=body.organization_id,
                          classification=body.classification,
                          allowed_regions=body.allowed_regions,
                          prohibited_regions=body.prohibited_regions,
                          default_region=body.default_region)
    db.commit()
    return {"saved": True}


@router.post("/failover")
def failover(body: FailoverIn,
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    try:
        result = cp.initiate_failover(db, organization_id=body.organization_id,
                                      region_from=body.region_from,
                                      region_to=body.region_to,
                                      actor_user_id=principal.user.id,
                                      reason=body.reason)
        db.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ===========================================================================
# Workers / scheduler / broker
# ===========================================================================

@router.get("/workers/autoscale")
def autoscale(principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    return wo.autoscale_signals(db)


@router.post("/workers/{worker_id}/quarantine")
def quarantine(worker_id: str, reason: Optional[str] = None,
               auto_recover_minutes: Optional[int] = None,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    row = wo.quarantine_worker(db, worker_id=worker_id, reason=reason,
                               auto_recover_after_minutes=auto_recover_minutes)
    db.commit()
    return {"worker_id": row.worker_id, "status": row.status}


@router.post("/workers/{worker_id}/override")
def override_quarantine(worker_id: str,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    result = wo.operator_override(db, worker_id=worker_id,
                                  operator_user_id=principal.user.id)
    db.commit()
    return result


@router.get("/quarantines")
def quarantines(principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    return [{"worker_id": q.worker_id, "status": q.status,
             "reason": q.reason, "failure_count": q.failure_count,
             "auto_recover_after": q.auto_recover_after}
            for q in wo.list_quarantines(db)]


@router.post("/jobs/{job_id}/migrate")
def migrate(job_id: int, reason: str = "operator",
            target_worker: Optional[str] = None,
            principal: AuthPrincipal = Depends(get_current_principal),
            db: Session = Depends(get_db)):
    try:
        result = wo.migrate_job(db, job_id=job_id, reason=reason,
                                target_worker=target_worker)
        db.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/backpressure")
def backpressure(queue_name: str, workspace_id: int,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return wo.backpressure_decision(db, workspace_id=workspace_id,
                                    queue_name=queue_name)


@router.post("/scheduler/acquire")
def acquire(leader_id: str,
            principal: AuthPrincipal = Depends(get_current_principal),
            db: Session = Depends(get_db)):
    result = wo.acquire_leadership(db, leader_id=leader_id)
    db.commit()
    return result


@router.get("/scheduler/leader")
def scheduler_leader(principal: AuthPrincipal = Depends(
    get_current_principal),
    db: Session = Depends(get_db)):
    from ..models.phase19 import SchedulerLeader
    row = db.query(SchedulerLeader).filter(
        SchedulerLeader.status == "LEADER").first()
    if row is None:
        return {"leader": None}
    return {"leader": {"leader_id": row.leader_id,
                       "lease_until": row.lease_until}}


@router.get("/broker")
def broker_info(principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    return {"config": broker2.broker_config(),
            "guarantees": broker2.delivery_guarantees(),
            "outage_policy": broker2.outage_policy(),
            "recovery": broker2.broker_recovery_check(db),
            "pool": broker2.pool_config(),
            "redis_available": broker2.redis_available()}


# ===========================================================================
# Providers / cost
# ===========================================================================

class CapabilityIn(BaseModel):
    provider: str
    model: str
    supports_tools: bool = False
    supports_structured: bool = False
    supports_streaming: bool = False
    supports_vision: bool = False
    context_window: Optional[int] = None
    max_output: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    latency_class: Optional[str] = "medium"


@router.get("/providers/capabilities")
def capabilities(principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    return {"matrix": po.capability_matrix(),
            "registered": po.list_capabilities(db)}


@router.post("/providers/capabilities")
def register_capability(body: CapabilityIn,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    po.register_capability(db, provider=body.provider, model=body.model,
                           supports_tools=body.supports_tools,
                           supports_structured=body.supports_structured,
                           supports_streaming=body.supports_streaming,
                           supports_vision=body.supports_vision,
                           context_window=body.context_window,
                           max_output=body.max_output,
                           embedding_dimensions=body.embedding_dimensions,
                           cost_per_1k_input=body.cost_per_1k_input,
                           cost_per_1k_output=body.cost_per_1k_output,
                           latency_class=body.latency_class)
    db.commit()
    return {"registered": True}


@router.get("/providers/route")
def route(task: str, mode: str = "balanced",
          sensitivity: str = "INTERNAL",
          provider: Optional[str] = None,
          organization_id: Optional[int] = None,
          workspace_id: Optional[int] = None,
          principal: AuthPrincipal = Depends(get_current_principal),
          db: Session = Depends(get_db)):
    candidates = po.list_capabilities(db, provider=provider)
    return po.route_model(db, task=task, candidates=candidates, mode=mode,
                          sensitivity=sensitivity,
                          organization_id=organization_id,
                          workspace_id=workspace_id)


class ReservationIn(BaseModel):
    workspace_id: int
    organization_id: Optional[int] = None
    feature: str = "ai"
    estimated_cost: float


@router.post("/cost/reserve")
def reserve(body: ReservationIn,
            principal: AuthPrincipal = Depends(get_current_principal),
            db: Session = Depends(get_db)):
    _owner(db, principal, body.workspace_id)
    result = cost4.reserve_budget(
        db, organization_id=body.organization_id,
        workspace_id=body.workspace_id, execution_ref=None,
        feature=body.feature, estimated_cost=body.estimated_cost)
    db.commit()
    return result


@router.post("/cost/{reservation_id}/release")
def release(reservation_id: int, actual_cost: Optional[float] = None,
            principal: AuthPrincipal = Depends(get_current_principal),
            db: Session = Depends(get_db)):
    try:
        result = cost4.release_reservation(db, reservation_id=reservation_id,
                                           actual_cost=actual_cost)
        db.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/cost/forecast")
def forecast(organization_id: int, granularity: str = "monthly",
             principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    return cost4.forecast2(db, organization_id=organization_id,
                           granularity=granularity)


@router.post("/cost/anomalies/detect")
def anomalies(organization_id: int,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    result = cost4.detect_anomalies(db, organization_id=organization_id)
    db.commit()
    return result


@router.post("/cost/recommendations")
def recommendations(organization_id: int,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    result = cost4.optimization_recommendations(
        db, organization_id=organization_id)
    db.commit()
    return result


# ===========================================================================
# Vector / ingestion / connectors / KG / memory
# ===========================================================================

@router.get("/vector/coverage")
def vector_coverage(workspace_id: Optional[int] = None,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    return vo.compute_coverage(db, workspace_id=workspace_id)


@router.get("/vector/drift")
def vector_drift(principal: AuthPrincipal = Depends(
    get_current_principal),
    db: Session = Depends(get_db)):
    return vo.embedding_drift(db)


@router.post("/vector/lifecycle")
def lifecycle(embedding_model_id: int, lifecycle: str,
              reason: Optional[str] = None,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    result = vo.set_lifecycle(db, model_id=embedding_model_id,
                              lifecycle=lifecycle, reason=reason,
                              decided_by=principal.user.id)
    db.commit()
    return result


@router.get("/ingestion/governor")
def governor(principal: AuthPrincipal = Depends(get_current_principal),
             db: Session = Depends(get_db)):
    return io3.resource_governor()


@router.get("/connectors/{source_id}/conflicts")
def connector_conflicts(source_id: int,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    return [{"id": c.id, "external_id": c.external_id,
             "conflict_type": c.conflict_type, "status": c.status,
             "detail": c.detail}
            for c in fo2.list_conflicts(db, source_id=source_id)]


@router.get("/kg/consistency")
def kg_consistency(workspace_id: int,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return kg5.graph_consistency(db, workspace_id=workspace_id)


@router.get("/memory/{memory_id}/explain")
def memory_explain(memory_id: int, workspace_id: int,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    try:
        return memory4.explain_memory(db, memory_id=memory_id,
                                      workspace_id=workspace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ===========================================================================
# Agents / workflows / actions
# ===========================================================================

@router.post("/agents/{execution_id}/cancel")
def cancel_agent(execution_id: str, workspace_id: int,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    try:
        result = agent4.cancel_agent(db, execution_id=execution_id,
                                     workspace_id=workspace_id,
                                     user_id=principal.user.id)
        db.commit()
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/workflows/{run_id}/pause")
def pause(run_id: int, workspace_id: int,
          principal: AuthPrincipal = Depends(get_current_principal),
          db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    try:
        result = workflow4.pause_workflow(db, run_id=run_id,
                                          workspace_id=workspace_id)
        db.commit()
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ===========================================================================
# Observability / SLO / analytics / quality
# ===========================================================================

@router.get("/slo/health")
def slo_health(principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    return ob4.slo_health(db)


@router.get("/analytics/org")
def analytics_org(organization_id: int,
                  principal: AuthPrincipal = Depends(
                      get_current_principal),
                  db: Session = Depends(get_db)):
    return an.privacy_guard(an.org_analytics(db,
                                             organization_id=organization_id))


@router.get("/analytics/workspace")
def analytics_workspace(workspace_id: int,
                        principal: AuthPrincipal = Depends(
                            get_current_principal),
                        db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return an.workspace_analytics(db, workspace_id=workspace_id)


@router.post("/consistency/run")
def consistency(workspace_id: Optional[int] = None,
                dry_run: bool = True,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    result = dc.run_consistency_checks(db, workspace_id=workspace_id,
                                       dry_run=dry_run)
    db.commit()
    return result


@router.get("/dr/readiness")
def dr_readiness(principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    return dr2.readiness_score(db)


@router.post("/dr/backups")
def dr_backup(scope: str, backup_ref: str,
              principal: AuthPrincipal = Depends(get_current_principal),
              db: Session = Depends(get_db)):
    try:
        result = dr2.record_backup(db, scope=scope, backup_ref=backup_ref)
        db.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/dr/validate")
def dr_validate(principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    return dr2.restore_validation(db)


@router.get("/quality/evaluations")
def evaluations(dataset_name: Optional[str] = None,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    return [{"id": e.id, "dataset": e.dataset_name, "passed": e.passed,
             "created_at": e.created_at}
            for e in qp.list_evaluations(db,
                                         dataset_name=dataset_name)]


@router.get("/search/quality")
def search_quality(workspace_id: int,
                   principal: AuthPrincipal = Depends(
                       get_current_principal),
                   db: Session = Depends(get_db)):
    _owner(db, principal, workspace_id)
    return search5.quality_summary(db, workspace_id=workspace_id)


# ===========================================================================
# Security checks (pure, request-body inputs only)
# ===========================================================================

class InjectionIn(BaseModel):
    text: Optional[str] = None
    source: str = "document"
    metadata: Optional[dict] = None
    filename: Optional[str] = None


@router.post("/security/injection")
def injection_check(body: InjectionIn,
                    principal: AuthPrincipal = Depends(
                        get_current_principal),
                    db: Session = Depends(get_db)):
    return safety8.injection8(body.text, source=body.source,
                              metadata=body.metadata,
                              filename=body.filename)


class FileIn(BaseModel):
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    magic_hex: Optional[str] = None


@router.post("/security/file")
def file_check(body: FileIn,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    return safety8.file_security3(filename=body.filename,
                                  mime_type=body.mime_type,
                                  size_bytes=body.size_bytes,
                                  magic_hex=body.magic_hex)


@router.get("/error-format")
def error_format_example(request_id: str = "req_demo_1",
                         principal: AuthPrincipal = Depends(
                             get_current_principal),
                         db: Session = Depends(get_db)):
    return api3.error_payload("EXAMPLE", "structured error contract sample",
                              request_id=request_id, retryable=False)
