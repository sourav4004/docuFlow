"""Phase 23 operations API — Global AI Cloud Platform.

Surfaces the Phase 23 platform: global control plane (capability endpoints,
dependency graph 2.0, global health, readiness, degraded components),
broker migration safety + zero-downtime storage migration, provider
production gate + routing 3.0 + admission control + cost reconciliation,
vector activation + embedding-model migration + shadow search coexistence,
multi-region control plane (registry, residency enforcement + log, drain,
failover engine + failback), RAG 8.0 + quality gates + change impact 3.0,
API platform 4.0 (idempotency, cache stats, v2 compatibility), knowledge
maintenance 2.0 + freshness + drift, webhook reliability + scheduler dedup
+ review center, security operations 4.0 (findings + continuous scan),
continuous evaluation 2.0 + promotion gates, search 6.0 explainable ranking
+ self-evaluation, and real-time ops streams with sequence IDs.

Every endpoint requires authentication; workspace-scoped writes require
owner/org-admin. Never returns secrets or chain-of-thought.
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
    global_control as gc,
    platform_migration as pm,
    provider_gate as pg,
    vector_activation as va,
    region_control as rc,
    rag8,
    api_platform3 as ap3,
    knowledge_maintenance2 as km2,
    webhook_reliability as wr,
    security_eval3 as se3,
)

router = APIRouter(prefix="/ops23", tags=["ops23"])


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


def _ws_id(payload: dict) -> int:
    """Extract workspace_id with a structured 422 instead of a KeyError 500."""
    raw = payload.get("workspace_id")
    if raw is None:
        raise HTTPException(status_code=422,
                            detail="workspace_id is required")
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422,
                            detail="workspace_id must be an integer")


# ===========================================================================
# Global control plane (Steps 1, 18-19)
# ===========================================================================

@router.get("/capabilities")
def list_capabilities(db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    caps_block = gc.global_health(db)["capabilities"]
    return caps_block["components"]


@router.get("/capabilities/{name}")
def get_capability(name: str, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(
                       get_current_principal)):
    detail = gc.capability_detail(db, name)
    if detail is None:
        raise HTTPException(status_code=404, detail="unknown capability")
    return detail


@router.post("/capabilities/check")
def check_capabilities(payload: dict,
                       db: Session = Depends(get_db),
                       principal: AuthPrincipal = Depends(
                           get_current_principal)):
    return gc.run_capability_check(db, payload.get("component"))


@router.post("/capabilities/refresh")
def refresh_capabilities(db: Session = Depends(get_db),
                         principal: AuthPrincipal = Depends(
                             get_current_principal)):
    return gc.refresh_capabilities(db)


@router.get("/global-health")
def get_global_health(db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    return gc.global_health(db)


@router.get("/dependencies")
def get_dependencies(db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    return gc.dependencies_overview(db)


@router.get("/dependencies/{component}/impact")
def get_dependency_impact(component: str, db: Session = Depends(get_db),
                          principal: AuthPrincipal = Depends(
                              get_current_principal)):
    try:
        return gc.dependency_impact(db, component)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/readiness")
def get_readiness(db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(
                      get_current_principal)):
    return gc.readiness(db)


@router.get("/degraded-components")
def get_degraded(db: Session = Depends(get_db),
                 principal: AuthPrincipal = Depends(get_current_principal)):
    return gc.degraded_components(db)


# ===========================================================================
# Broker migration + storage migration (Steps 3-5)
# ===========================================================================

@router.post("/broker/migration/plan")
def plan_broker_migration(payload: dict, db: Session = Depends(get_db),
                          principal: AuthPrincipal = Depends(
                              get_current_principal)):
    return pm.broker_migration_plan(
        db, migration_id=payload.get("migration_id", "m-default"),
        to_backend=payload.get("to_backend", "redis"),
        dry_run=bool(payload.get("dry_run", True)))


@router.post("/broker/migration/transition")
def transition_broker_migration(payload: dict, db: Session = Depends(get_db),
                                principal: AuthPrincipal = Depends(
                                    get_current_principal)):
    try:
        return pm.BrokerMigrationRecord.transition(
            db, payload["migration_id"], new_state=payload["new_state"],
            from_backend=payload.get("from_backend", "postgres"),
            to_backend=payload.get("to_backend", "redis"),
            actor=principal.user.email)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/broker/migration/{migration_id}")
def get_broker_migration(migration_id: str,
                         db: Session = Depends(get_db),
                         principal: AuthPrincipal = Depends(
                             get_current_principal)):
    return pm.BrokerMigrationRecord.state(db, migration_id)


@router.get("/broker/drain-check")
def broker_drain_check(db: Session = Depends(get_db),
                       principal: AuthPrincipal = Depends(
                           get_current_principal)):
    return pm.broker_verify_queue_state(db)


@router.post("/storage-migration/plans")
def create_storage_plan(payload: dict, db: Session = Depends(get_db),
                        principal: AuthPrincipal = Depends(
                            get_current_principal)):
    _owner(db, principal, _ws_id(payload))
    try:
        return pm.create_storage_migration_plan(
            db, workspace_id=_ws_id(payload),
            batch_size=int(payload.get("batch_size", 25)),
            dry_run=bool(payload.get("dry_run", False)),
            created_by=principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/storage-migration/plans/{plan_id}/run")
def run_storage_batch(plan_id: int, db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    plan = pm.storage_migration_progress(db, plan_id)
    _owner(db, principal, int(plan["workspace_id"])
           if plan.get("workspace_id") else
           (db.query(Workspace).order_by(Workspace.id).first().id))
    try:
        return pm.run_storage_migration_batch(db, plan_id,
                                              actor=principal.user.email)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/storage-migration/plans/{plan_id}")
def storage_plan_progress(plan_id: int, db: Session = Depends(get_db),
                          principal: AuthPrincipal = Depends(
                              get_current_principal)):
    try:
        progress = pm.storage_migration_progress(db, plan_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    progress.setdefault("mode", "DRY_RUN" if progress.get("dry_run")
                        else "COMMIT")
    return progress


@router.get("/storage/integrity/{document_id}")
def storage_integrity(document_id: int, workspace_id: int,
                      db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    _member(db, principal, workspace_id)
    result = pm.verify_document_integrity(db, document_id, workspace_id)
    if result.get("found") is False:
        raise HTTPException(status_code=404, detail="document not found")
    return result


@router.get("/storage/orphans/{workspace_id}")
def storage_orphans(workspace_id: int, db: Session = Depends(get_db),
                    principal: AuthPrincipal = Depends(get_current_principal)):
    _owner(db, principal, workspace_id)
    return pm.orphan_object_candidates(db, workspace_id)


@router.get("/storage/capability")
def storage_capability(db: Session = Depends(get_db),
                       principal: AuthPrincipal = Depends(
                           get_current_principal)):
    return pm.storage_capability_report(db)


# ===========================================================================
# Provider production gate + routing + cost (Steps 6-9)
# ===========================================================================

@router.post("/providers/{kind}/readiness")
def validate_provider(kind: str, db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    return pg.validate_provider_readiness(db, provider_kind=kind)


@router.get("/providers/readiness")
def provider_readiness_matrix(db: Session = Depends(get_db),
                              principal: AuthPrincipal = Depends(
                                  get_current_principal)):
    return pg.readiness_matrix(db)


@router.post("/routing/decide")
def decide_routing(payload: dict, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    return pg.route_provider(
        db, workspace_id=_ws_id(payload),
        operation=payload.get("operation", "completion"),
        sensitivity=payload.get("sensitivity", "INTERNAL"),
        region=payload.get("region"),
        context_chars=int(payload.get("context_chars", 0)),
        actor=principal.user.email)


@router.post("/routing/admission")
def admission_control(payload: dict, db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    _member(db, principal, _ws_id(payload))
    return pg.admission_check(
        db, workspace_id=_ws_id(payload),
        operation=payload.get("operation", "completion"),
        sensitivity=payload.get("sensitivity", "INTERNAL"),
        region=payload.get("region"),
        context_chars=int(payload.get("context_chars", 0)),
        estimated_cost=float(payload.get("estimated_cost", 0.0)),
        budget_remaining=(float(payload["budget_remaining"])
                          if payload.get("budget_remaining") is not None
                          else None))


@router.post("/cost/reconcile")
def reconcile_cost(payload: dict, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    _owner(db, principal, _ws_id(payload))
    return pg.reconcile_provider_cost(
        db, workspace_id=_ws_id(payload),
        provider=payload.get("provider", "fake"),
        model=payload.get("model", "fake-model"),
        estimated_tokens=int(payload.get("estimated_tokens", 0)),
        actual_input_tokens=int(payload.get("actual_input_tokens", 0)),
        actual_output_tokens=int(payload.get("actual_output_tokens", 0)),
        estimated_cost=float(payload.get("estimated_cost", 0.0)),
        actual_cost=(float(payload["actual_cost"])
                     if payload.get("actual_cost") is not None else None),
        execution_id=payload.get("execution_id"),
        request_id=payload.get("request_id"))


@router.get("/cost/reconcile/{workspace_id}")
def cost_reconciliation(workspace_id: int, db: Session = Depends(get_db),
                        principal: AuthPrincipal = Depends(
                            get_current_principal)):
    _member(db, principal, workspace_id)
    summary = pg.reconciliation_summary(db, workspace_id=workspace_id)
    summary.setdefault("count", summary.get("total", 0))
    return summary


# ===========================================================================
# Vector activation + model migration + shadow search (Steps 10-12)
# ===========================================================================

@router.get("/vector/activation")
def vector_activation(db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    return va.activation_report(db)


@router.get("/vector/migration-readiness")
def vector_migration_ready(db: Session = Depends(get_db),
                           principal: AuthPrincipal = Depends(
                               get_current_principal)):
    return va.migration_readiness(db)


@router.post("/vector/models")
def register_vector_model(payload: dict, db: Session = Depends(get_db),
                          principal: AuthPrincipal = Depends(
                              get_current_principal)):
    return va.register_model_version(
        db, model_name=payload["model_name"],
        version=payload["version"], dimension=int(payload["dimension"]))


@router.post("/vector/model-migration/{workspace_id}/start")
def start_model_migration(workspace_id: int, payload: dict,
                          db: Session = Depends(get_db),
                          principal: AuthPrincipal = Depends(
                              get_current_principal)):
    _owner(db, principal, workspace_id)
    try:
        return va.start_model_migration(
            db, workspace_id=workspace_id,
            model_name=payload["model_name"], version=payload["version"],
            dimension=int(payload["dimension"]),
            batch_size=int(payload.get("batch_size", 20)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/vector/model-migration/{workspace_id}/run")
def run_model_migration(workspace_id: int, payload: dict,
                        db: Session = Depends(get_db),
                        principal: AuthPrincipal = Depends(
                            get_current_principal)):
    _owner(db, principal, workspace_id)
    return va.run_dual_generation_batch(
        db, workspace_id=workspace_id,
        model_version_id=int(payload["model_version_id"]),
        batch_size=int(payload.get("batch_size", 20)))


@router.get("/vector/model-migration/{workspace_id}/coverage/{mv_id}")
def model_migration_coverage(workspace_id: int, mv_id: int,
                             db: Session = Depends(get_db),
                             principal: AuthPrincipal = Depends(
                                 get_current_principal)):
    _member(db, principal, workspace_id)
    return va.verify_coverage(db, workspace_id=workspace_id,
                              model_version_id=mv_id)


@router.post("/vector/model-migration/{workspace_id}/quality")
def model_migration_quality(workspace_id: int, payload: dict,
                            db: Session = Depends(get_db),
                            principal: AuthPrincipal = Depends(
                                get_current_principal)):
    _owner(db, principal, workspace_id)
    return va.record_quality_comparison(
        db, workspace_id=workspace_id,
        model_version_id=int(payload["model_version_id"]),
        quality_delta=float(payload["quality_delta"]))


@router.post("/vector/model-migration/promote/{mv_id}")
def promote_model(mv_id: int, db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    return va.promote_model(db, model_version_id=mv_id,
                            actor=principal.user.email)


@router.post("/vector/model-migration/rollback/{mv_id}")
def rollback_model(mv_id: int, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    return va.rollback_model(db, model_version_id=mv_id,
                             actor=principal.user.email)


@router.get("/vector/model-migration/retirement/{mv_id}")
def model_retirement(mv_id: int, db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    return va.retirement_plan(db, model_version_id=mv_id)


@router.post("/vector/shadow-compare")
def shadow_compare(payload: dict, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    return va.shadow_compare(
        db, workspace_id=_ws_id(payload),
        query=payload.get("query", "")[:500],
        baseline_ranking=payload.get("baseline_ranking", []),
        candidate_ranking=payload.get("candidate_ranking", []))


@router.get("/vector/shadow-gate/{workspace_id}")
def shadow_gate(workspace_id: int, db: Session = Depends(get_db),
                principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, workspace_id)
    return va.promotion_gate(db, workspace_id=workspace_id)


# ===========================================================================
# Multi-region control plane (Steps 13-17)
# ===========================================================================

@router.post("/regions")
def register_region(payload: dict, db: Session = Depends(get_db),
                    principal: AuthPrincipal = Depends(get_current_principal)):
    return rc.register_region(
        db, region=payload["region"],
        status=payload.get("status", "HEALTHY"),
        failover_to=payload.get("failover_to"),
        providers=payload.get("providers"),
        vector=bool(payload.get("vector", False)),
        storage=bool(payload.get("storage", False)),
        broker=bool(payload.get("broker", False)),
        residency_policy=payload.get("residency_policy"),
        latency_ms=payload.get("latency_ms"))


@router.get("/regions")
def region_overview(db: Session = Depends(get_db),
                    principal: AuthPrincipal = Depends(
                        get_current_principal)):
    return rc.region_overview(db)


@router.get("/regions/{region}")
def region_health(region: str, db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    return rc.region_health(db, region)


@router.post("/residency/evaluate")
def evaluate_residency(payload: dict, db: Session = Depends(get_db),
                       principal: AuthPrincipal = Depends(
                           get_current_principal)):
    _member(db, principal, _ws_id(payload))
    return rc.evaluate_residency(
        db, workspace_id=_ws_id(payload),
        operation=payload.get("operation", "processing"),
        source_region=payload.get("source_region", "unknown"),
        destination_region=payload.get("destination_region", "unknown"),
        classification=payload.get("classification", "INTERNAL"),
        actor=principal.user.email)


@router.get("/residency/log/{workspace_id}")
def residency_log(workspace_id: int, db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, workspace_id)
    return rc.residency_log(db, workspace_id=workspace_id)


@router.post("/regions/{region}/drain")
def start_region_drain(region: str, payload: dict,
                       db: Session = Depends(get_db),
                       principal: AuthPrincipal = Depends(
                           get_current_principal)):
    return rc.start_region_drain(
        db, region=region, actor=principal.user.email,
        workspace_ids=payload.get("workspace_ids"))


@router.get("/regions/drain/{drain_id}")
def drain_progress(drain_id: int, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    try:
        return rc.drain_progress(db, drain_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/regions/drain/{drain_id}/complete")
def complete_drain(drain_id: int, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    try:
        return rc.complete_drain(db, drain_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/regions/drain/{drain_id}/recover")
def recover_region(drain_id: int, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    try:
        return rc.recover_region(db, drain_id, actor=principal.user.email)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/regions/failover/decision")
def failover_decision(payload: dict, db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    return rc.failover_decision(
        db, primary=payload["primary"], secondary=payload["secondary"],
        autonomy_allowed=bool(payload.get("autonomy_allowed", False)),
        actor=principal.user.email)


@router.get("/regions/failback")
def failback_plan(primary: str, secondary: str,
                  db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    return rc.failback_plan(db, primary=primary, secondary=secondary)


@router.get("/regions/failover-history")
def failover_history(db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    return rc.failover_history(db)


# ===========================================================================
# RAG 8.0 + change impact 3.0 (Steps 26-28)
# ===========================================================================

@router.post("/rag8/answer")
def rag8_answer(payload: dict, db: Session = Depends(get_db),
                principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    result = rag8.generate_answer(
        db, workspace_id=_ws_id(payload),
        query=payload.get("query", ""),
        evidence=payload.get("evidence", []))
    if payload.get("record_quality"):
        rag8.record_response_quality(db,
                                     workspace_id=_ws_id(payload),
                                     rag_result=result)
    return result


@router.post("/rag8/quality")
def rag8_quality(payload: dict, db: Session = Depends(get_db),
                 principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    result = payload.get("rag_result", {})
    return rag8.record_response_quality(
        db, workspace_id=_ws_id(payload), rag_result=result)


@router.get("/rag8/change-impact/{document_id}")
def change_impact(document_id: int, workspace_id: int, change_class: str,
                  db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, workspace_id)
    result = rag8.change_impact_3(db, document_id=document_id,
                                  workspace_id=workspace_id,
                                  change_class=change_class)
    if result.get("found") is False:
        raise HTTPException(status_code=404, detail=result.get("reason"))
    return result


# ===========================================================================
# API platform 4.0 (Steps 39-40, 44-45)
# ===========================================================================

@router.get("/api/compatibility")
def api_compatibility(principal: AuthPrincipal = Depends(
        get_current_principal)):
    return ap3.compatibility_metadata()


@router.get("/api/contract-matrix")
def api_contract_matrix(principal: AuthPrincipal = Depends(
        get_current_principal)):
    return ap3.contract_case_matrix()


@router.get("/cache/stats")
def cache_stats(principal: AuthPrincipal = Depends(get_current_principal)):
    return ap3.tenant_cache.stats()


@router.delete("/cache/{workspace_id}/{namespace}")
def cache_invalidate(workspace_id: int, namespace: str,
                     db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    _owner(db, principal, workspace_id)
    removed = ap3.tenant_cache.invalidate(workspace_id=workspace_id,
                                          namespace=namespace)
    return {"invalidated": removed, "workspace_id": workspace_id,
            "namespace": namespace}


# ===========================================================================
# Knowledge maintenance 2.0 + freshness + drift (Steps 23-25)
# ===========================================================================

@router.post("/knowledge/freshness/{workspace_id}")
def refresh_freshness(workspace_id: int, db: Session = Depends(get_db),
                      principal: AuthPrincipal = Depends(
                          get_current_principal)):
    _member(db, principal, workspace_id)
    return km2.refresh_freshness(db, workspace_id=workspace_id)


@router.get("/knowledge/freshness/{workspace_id}")
def freshness_overview(workspace_id: int, db: Session = Depends(get_db),
                       principal: AuthPrincipal = Depends(
                           get_current_principal)):
    _member(db, principal, workspace_id)
    return km2.freshness_overview(db, workspace_id=workspace_id)


@router.get("/knowledge/drift/{workspace_id}")
def knowledge_drift(workspace_id: int, db: Session = Depends(get_db),
                    principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, workspace_id)
    return km2.detect_drift(db, workspace_id=workspace_id)


@router.post("/knowledge/maintenance/{workspace_id}")
def run_maintenance(workspace_id: int, payload: dict,
                    db: Session = Depends(get_db),
                    principal: AuthPrincipal = Depends(get_current_principal)):
    _owner(db, principal, workspace_id)
    return km2.run_maintenance_scan(db, workspace_id=workspace_id,
                                    dimensions=payload.get("dimensions"))


@router.get("/knowledge/maintenance/{workspace_id}")
def maintenance_history(workspace_id: int, db: Session = Depends(get_db),
                        principal: AuthPrincipal = Depends(
                            get_current_principal)):
    _member(db, principal, workspace_id)
    return km2.maintenance_history(db, workspace_id=workspace_id)


# ===========================================================================
# Webhooks + scheduler dedup + review center (Steps 35, 48, 50-51)
# ===========================================================================

@router.post("/webhooks/sign-preview")
def webhook_sign_preview(payload: dict,
                         principal: AuthPrincipal = Depends(
                             get_current_principal)):
    """Demonstrate signing shape WITHOUT a real secret (doc/testing)."""
    preview = wr.sign_payload("preview-secret-not-real",
                              payload.get("payload", {}))
    return {"signature_shape": preview["signature"].split(",")[0],
            "timestamp": preview["timestamp"],
            "note": "real signature requires the configured endpoint secret"}


@router.post("/webhooks/delivery")
def webhook_delivery(payload: dict, db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    _owner(db, principal, _ws_id(payload))
    return wr.record_delivery_attempt(
        db, workspace_id=_ws_id(payload),
        endpoint_id=int(payload["endpoint_id"]),
        event_type=payload.get("event_type", "test"),
        ok=bool(payload.get("ok", False)),
        response_status=payload.get("response_status"),
        error=payload.get("error"),
        delivery_id=payload.get("delivery_id"))


@router.get("/webhooks/dead-letters/{workspace_id}")
def webhook_dead_letters(workspace_id: int, db: Session = Depends(get_db),
                         principal: AuthPrincipal = Depends(
                             get_current_principal)):
    _member(db, principal, workspace_id)
    return wr.dead_letter_deliveries(db, workspace_id=workspace_id)


@router.post("/webhooks/endpoints/{endpoint_id}/reenable")
def webhook_reenable(endpoint_id: int, db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    return wr.reenable_endpoint(db, endpoint_id=endpoint_id,
                                actor=principal.user.email)


@router.post("/scheduler/tasks/claim")
def scheduler_claim(payload: dict, db: Session = Depends(get_db),
                    principal: AuthPrincipal = Depends(
                        get_current_principal)):
    return wr.claim_scheduled_task(
        db, task_key=payload["task_key"],
        leader_id=payload.get("leader_id", "api-leader"),
        bucket=payload.get("bucket"))


@router.post("/scheduler/tasks/{run_id}/finish")
def scheduler_finish(run_id: int, payload: dict,
                     db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    return wr.finish_scheduled_task(db, run_id,
                                    ok=bool(payload.get("ok", True)))


@router.post("/reviews")
def create_review(payload: dict, db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    for required in ("item_type", "title"):
        if not payload.get(required):
            raise HTTPException(status_code=422,
                                detail=f"{required} is required")
    try:
        result = wr.create_unified_review(
            db, workspace_id=_ws_id(payload),
            item_type=payload["item_type"], title=payload["title"],
            user_id=principal.user.id,
            description=payload.get("description"),
            payload=payload.get("payload"),
            priority=payload.get("priority", "NORMAL"))
        result.setdefault("id", result.get("item_id"))
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/reviews/{item_id}/decide")
def decide_review(item_id: int, payload: dict,
                  db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    if not payload.get("decision"):
        raise HTTPException(status_code=422, detail="decision is required")
    try:
        result = wr.decide_review(
            db, workspace_id=_ws_id(payload), item_id=item_id,
            decision=payload["decision"],
            actor_user_id=principal.user.id,
            reason=payload.get("reason"),
            delegate_to=payload.get("delegate_to"))
        result.setdefault("id", result.get("item_id"))
        return result
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ===========================================================================
# Security operations 4.0 (Step 36)
# ===========================================================================

@router.post("/security/findings")
def create_finding(payload: dict, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    for required in ("category", "severity", "title"):
        if not payload.get(required):
            raise HTTPException(status_code=422,
                                detail=f"{required} is required")
    try:
        result = se3.create_security_finding(
            db, workspace_id=_ws_id(payload),
            category=payload["category"], severity=payload["severity"],
            title=payload["title"], evidence=payload.get("evidence"),
            dedup_key=payload.get("dedup_key"))
        result.setdefault("id", result.get("finding_id"))
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/security/findings/{workspace_id}")
def list_findings(workspace_id: int, db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, workspace_id)
    return se3.list_security_findings(db, workspace_id=workspace_id)


@router.post("/security/findings/{finding_id}/status")
def update_finding(finding_id: int, payload: dict,
                   db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    if not payload.get("new_status"):
        raise HTTPException(status_code=422, detail="new_status is required")
    try:
        return se3.update_security_finding(
            db, workspace_id=_ws_id(payload),
            finding_id=finding_id, new_status=payload["new_status"],
            actor=principal.user.email)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/security/scan/{workspace_id}")
def security_scan(workspace_id: int, db: Session = Depends(get_db),
                  principal: AuthPrincipal = Depends(get_current_principal)):
    _owner(db, principal, workspace_id)
    result = se3.run_continuous_security_scan(db, workspace_id=workspace_id)
    result["created"] = (result.get("injection_findings", 0)
                         + result.get("exfiltration_findings", 0))
    return result


@router.get("/security/severity-matrix")
def severity_matrix(principal: AuthPrincipal = Depends(
        get_current_principal)):
    return se3.security_severity_matrix()


# ===========================================================================
# Continuous evaluation 2.0 + promotion gates (Steps 56-57)
# ===========================================================================

@router.post("/evaluations")
def record_eval(payload: dict, db: Session = Depends(get_db),
                principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    domain = payload.get("domain")
    if domain is None:
        raise HTTPException(status_code=422, detail="domain is required")
    try:
        return se3.record_evaluation_run(
            db, workspace_id=_ws_id(payload),
            domain=domain,
            dataset_version=payload.get("dataset_version", "v1"),
            model=payload.get("model", "fake"),
            provider=payload.get("provider", "fake"),
            config=payload.get("config", {}),
            environment=payload.get("environment", "default"),
            metrics=payload.get("metrics", {}),
            latency_p95_ms=float(payload.get("latency_p95_ms", 0)),
            cost=float(payload.get("cost", 0.0)),
            security_findings=int(payload.get("security_findings", 0)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/evaluations/promotion-gates")
def promotion_gates(payload: dict, db: Session = Depends(get_db),
                    principal: AuthPrincipal = Depends(
                        get_current_principal)):
    result = se3.evaluate_promotion_gates(
        db, workspace_id=_ws_id(payload),
        quality=float(payload.get("quality", 0)),
        security_findings=int(payload.get("security_findings", 0)),
        latency_p95_ms=float(payload.get("latency_p95_ms", 0)),
        cost=float(payload.get("cost", 0)),
        regression=float(payload.get("regression", 0)))
    result["promote"] = result.get("eligible", False)
    return result


# ===========================================================================
# Search 6.0 (Steps 52-53)
# ===========================================================================

@router.post("/search/explain")
def search_explain(payload: dict, db: Session = Depends(get_db),
                   principal: AuthPrincipal = Depends(get_current_principal)):
    return se3.explain_ranking(db, candidates=payload.get("candidates", []),
                               weights=payload.get("weights"))


@router.post("/search/diversity")
def search_diversity(payload: dict,
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    ranked = se3.apply_diversity(payload.get("ranking", []))
    return {"ranking": ranked}


@router.post("/search/self-evaluation/{workspace_id}")
def search_self_eval(workspace_id: int, payload: dict,
                     db: Session = Depends(get_db),
                     principal: AuthPrincipal = Depends(
                         get_current_principal)):
    _member(db, principal, workspace_id)
    return se3.search_self_evaluation(
        db, workspace_id=workspace_id,
        zero_result_queries=int(payload.get("zero_result_queries", 0)),
        total_queries=int(payload.get("total_queries", 0)),
        reformulations=int(payload.get("reformulations", 0)),
        precision_estimate=float(payload.get("precision_estimate", 1.0)))


# ===========================================================================
# Real-time streams (Step 62)
# ===========================================================================

@router.post("/streams/{stream}")
def emit_event(stream: str, payload: dict, db: Session = Depends(get_db),
               principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, _ws_id(payload))
    if stream not in se3.STREAM_KINDS:
        raise HTTPException(status_code=422, detail="unknown stream")
    return se3.emit_stream_event(db, workspace_id=_ws_id(payload),
                                 stream=stream, kind=payload.get("kind",
                                                                 "event"),
                                 payload=payload.get("payload", {}))


@router.get("/streams/{stream}/{workspace_id}")
def read_events(stream: str, workspace_id: int, after_seq: int = 0,
                limit: int = 50, db: Session = Depends(get_db),
                principal: AuthPrincipal = Depends(get_current_principal)):
    _member(db, principal, workspace_id)
    if stream not in se3.STREAM_KINDS:
        raise HTTPException(status_code=422, detail="unknown stream")
    result = se3.read_stream(db, workspace_id=workspace_id, stream=stream,
                             after_seq=after_seq, limit=limit)
    result["events"] = result.get("items", [])
    return result
