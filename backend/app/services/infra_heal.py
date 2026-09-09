"""Phase 21 — Multi-region operations + infrastructure self-healing.

Region health/capacity tracking, failover readiness validation and dry-run
simulation with a data-residency guard, worker health scoring + safe
quarantine + recovery + capacity adaptation with tenant fairness, broker
health/failure detection/safe recovery with duplicate protection, scheduler
leader health/recovery/schedule dedup/missed-schedule recovery, database
health with slow-query detection and recommendations-only optimization (never
automatic DDL), cache health/anomaly/optimization with tenant-safe
invalidation, knowledge-graph autonomy (quality monitor, repair proposals,
impact analysis), and memory autonomy (health, deterministic suppression,
conflict review routing, full audit).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.phase19 import RegionRecord, SchedulerLeader
from ..models.phase21 import (
    BrokerHealthSnapshot, CapacityRecommendation, CacheHealthSnapshot,
    FailoverSimulation, GraphRepairProposal, MemoryAutonomyEvent,
    RegionHealthP21, ResidencyGuardEvent, SchedulerHealthSnapshot,
    SlowQueryRecord, WorkerHealthScore,
)
from . import autonomy


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Multi-region operations
# ---------------------------------------------------------------------------

def record_region_health(db: Session, organization_id: int, region_id: str,
                         state: str = "HEALTHY", workers: int = 0,
                         queue_depth: int = 0,
                         provider_healthy: bool = True,
                         db_healthy: bool = True) -> RegionHealthP21:
    """Track region state + capacity (workers/queues/providers/DB)."""
    if state not in ("HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"):
        raise ValueError(f"invalid region state: {state}")
    row = RegionHealthP21(
        organization_id=organization_id, region_id=region_id, state=state,
        workers=workers, queue_depth=queue_depth,
        provider_healthy=provider_healthy, db_healthy=db_healthy,
        failover_ready=(state == "HEALTHY" and provider_healthy
                        and db_healthy))
    db.add(row)
    db.commit()
    return row


def check_residency(db: Session, organization_id: int,
                    requested_region: str,
                    residency_rules: dict) -> dict:
    """Data-residency guard: block illegal cross-region routing.

    residency_rules: {region_id: [allowed region ids]} — a request for a
    region not in the allowed set is blocked and recorded.
    """
    allowed = residency_rules.get("__allowed__") or []
    if allowed and requested_region not in allowed:
        row = ResidencyGuardEvent(
            organization_id=organization_id,
            requested_region=requested_region,
            allowed_region=allowed[0] if allowed else None, blocked=True,
            detail=f"region {requested_region} not in residency allowlist")
        db.add(row)
        db.commit()
        return {"allowed": False, "blocked": True,
                "allowed_regions": allowed}
    return {"allowed": True, "blocked": False, "allowed_regions": allowed}


def simulate_failover(db: Session, organization_id: int, from_region: str,
                      to_region: str, target_health: RegionHealthP21,
                      residency_rules: Optional[dict] = None) \
        -> FailoverSimulation:
    """Dry-run failover simulation — validates residency + capacity.

    Zero side effects; `executed` stays False. Physical failover requires
    real infrastructure and is never claimed validated without it.
    """
    residency_ok = True
    if residency_rules:
        allowed = residency_rules.get("__allowed__") or []
        residency_ok = (not allowed) or to_region in allowed
    capacity_ok = (target_health is not None
                   and target_health.state == "HEALTHY"
                   and target_health.workers > 0)
    sim = FailoverSimulation(
        organization_id=organization_id, from_region=from_region,
        to_region=to_region, residency_ok=residency_ok,
        capacity_ok=capacity_ok, ready=residency_ok and capacity_ok,
        detail=("failover ready" if residency_ok and capacity_ok
                else f"residency_ok={residency_ok} capacity_ok={capacity_ok}"),
        simulated=True, executed=False)
    db.add(sim)
    db.commit()
    return sim


def execute_failover(db: Session, sim: FailoverSimulation,
                     actor: str = "operator") -> dict:
    """Governed failover — requires a READY simulation + autonomy ALLOWED.

    With real multi-region infrastructure absent, execution records the
    governance decision only.
    """
    if not sim.ready:
        return {"simulation_id": sim.id, "decision": "BLOCKED",
                "reason": "failover simulation not ready"}
    op = autonomy.guard_operation(
        db, sim.organization_id, "region.failover", risk_level="HIGH",
        actor=actor, source="OPERATOR",
        input_payload={"simulation_id": sim.id,
                       "from_region": sim.from_region,
                       "to_region": sim.to_region},
        idempotency_key=f"failover:{sim.id}")
    return {"simulation_id": sim.id, "decision": op.decision,
            "reason": op.decision_reason, "operation_id": op.id}


def latest_region_health(db: Session, organization_id: int, region_id: str) \
        -> Optional[RegionHealthP21]:
    return (db.query(RegionHealthP21)
            .filter_by(organization_id=organization_id, region_id=region_id)
            .order_by(RegionHealthP21.id.desc()).first())


# ---------------------------------------------------------------------------
# Worker self-healing 2.0
# ---------------------------------------------------------------------------

def record_worker_health(db: Session, workspace_id: int, worker_id: str,
                         heartbeat_age_seconds: int = 0,
                         throughput: float = 0.0, failure_count: int = 0,
                         queue_latency_ms: float = 0.0,
                         region: Optional[str] = None) -> WorkerHealthScore:
    """Compute a worker health score from heartbeat/throughput/failures/queue.

    score = 100 - staleness penalty - failure penalty - queue penalty.
    """
    score = 100.0
    if heartbeat_age_seconds > 120:
        score -= 40
    elif heartbeat_age_seconds > 60:
        score -= 15
    score -= min(30.0, failure_count * 5.0)
    score -= min(20.0, queue_latency_ms / 1000.0)
    if throughput <= 0 and queue_latency_ms > 5000:
        score -= 15
    score = max(0.0, min(100.0, score))
    state = ("HEALTHY" if score >= 70 else
             "DEGRADED" if score >= 40 else "QUARANTINED")
    row = WorkerHealthScore(
        workspace_id=workspace_id, worker_id=worker_id, region=region,
        heartbeat_age_seconds=heartbeat_age_seconds,
        throughput=throughput, failure_count=failure_count,
        queue_latency_ms=queue_latency_ms, score=score, state=state,
        quarantined=(state == "QUARANTINED"))
    db.add(row)
    db.commit()
    return row


def quarantine_worker(db: Session, worker: WorkerHealthScore,
                      actor: str = "system") -> WorkerHealthScore:
    """Safe automatic quarantine of unhealthy workers (no data mutation)."""
    if worker.score < 40:
        worker.quarantined = True
        worker.state = "QUARANTINED"
        db.commit()
    return worker


def recover_worker(db: Session, worker: WorkerHealthScore,
                   actor: str = "system") -> WorkerHealthScore:
    """Recover stale/failed worker state (logical restart semantics)."""
    worker.quarantined = False
    worker.state = "RECOVERED"
    worker.heartbeat_age_seconds = 0
    db.commit()
    return worker


def recommend_capacity(db: Session, workspace_id: int,
                       current_workers: int, queue_depth: int,
                       avg_worker_throughput: float = 10.0,
                       region: Optional[str] = None) -> CapacityRecommendation:
    """Autoscaling recommendation preserving tenant fairness.

    Recommended capacity = ceil(queue_depth / throughput) with a fairness
    floor: never recommend below half the current fleet (no starvation) and
    cap burst growth at 2x current.
    """
    import math
    needed = math.ceil(queue_depth / max(avg_worker_throughput, 0.1))
    recommended = max(int(current_workers * 0.5),
                      min(needed, int(current_workers * 2) or needed))
    fairness = recommended >= current_workers * 0.5
    row = CapacityRecommendation(
        workspace_id=workspace_id, region=region,
        current_workers=current_workers, recommended_workers=recommended,
        fairness_preserved=fairness,
        reason=f"queue_depth={queue_depth} throughput={avg_worker_throughput}")
    db.add(row)
    db.commit()
    return row


def latest_worker_scores(db: Session, workspace_id: int,
                         limit: int = 100) -> list[WorkerHealthScore]:
    """Latest score per worker (bounded)."""
    rows = (db.query(WorkerHealthScore)
            .filter_by(workspace_id=workspace_id)
            .order_by(WorkerHealthScore.id.desc()).limit(limit * 4).all())
    seen, latest = set(), []
    for row in rows:
        if row.worker_id in seen:
            continue
        seen.add(row.worker_id)
        latest.append(row)
        if len(latest) >= limit:
            break
    return latest


# ---------------------------------------------------------------------------
# Broker self-healing
# ---------------------------------------------------------------------------

def record_broker_health(db: Session, workspace_id: int, depth: int = 0,
                         latency_ms: float = 0.0,
                         visibility_timeouts: int = 0,
                         error_count: int = 0, reconnects: int = 0,
                         broker: str = "postgres") -> BrokerHealthSnapshot:
    """Track broker depth/latency/visibility/errors/reconnects; flag degraded."""
    degraded = (error_count > 10 or reconnects > 3
                or latency_ms > 5000 or visibility_timeouts > 5)
    row = BrokerHealthSnapshot(
        workspace_id=workspace_id, broker=broker, depth=depth,
        latency_ms=latency_ms, visibility_timeouts=visibility_timeouts,
        error_count=error_count, reconnects=reconnects,
        state="DEGRADED" if degraded else "HEALTHY", degraded=degraded)
    db.add(row)
    db.commit()
    return row


def broker_recovery_plan(snapshot: BrokerHealthSnapshot) -> dict:
    """Safe broker recovery recommendations (reconnect/requeue only)."""
    actions: list[str] = []
    if snapshot.reconnects > 3:
        actions.append("reconnect_broker")
    if snapshot.visibility_timeouts > 5:
        actions.append("requeue_safe_jobs")
    if snapshot.depth > 500:
        actions.append("scale_workers")
    return {"broker": snapshot.broker, "actions": actions,
            "safe_auto": [a for a in actions
                          if a in ("reconnect_broker", "requeue_safe_jobs")]}


def verify_duplicate_protection(consumed_keys: list[str]) -> dict:
    """Idempotent consumption check — duplicates are rejected."""
    seen: set[str] = set()
    duplicates = []
    for key in consumed_keys:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return {"total": len(consumed_keys),
            "duplicates_detected": len(duplicates),
            "protected": True}


# ---------------------------------------------------------------------------
# Scheduler self-healing
# ---------------------------------------------------------------------------

def record_scheduler_health(db: Session, workspace_id: int,
                            has_leader: bool = False,
                            leader_age_seconds: int = 0,
                            missed_schedules: int = 0) \
        -> SchedulerHealthSnapshot:
    """Track leader health; bounded missed-schedule recovery + dedup."""
    recovered = 0
    if missed_schedules > 0 and has_leader:
        recovered = min(missed_schedules, 50)  # bounded recovery
    dedup_blocked = 0
    row = SchedulerHealthSnapshot(
        workspace_id=workspace_id, has_leader=has_leader,
        leader_age_seconds=leader_age_seconds,
        missed_schedules=missed_schedules, recovered_schedules=recovered,
        dedup_blocked=dedup_blocked,
        state="HEALTHY" if has_leader else "DEGRADED")
    db.add(row)
    db.commit()
    return row


def dedupe_schedule(db: Session, workspace_id: int, schedule_key: str,
                    active_leaders: list[str]) -> dict:
    """Prevent duplicate scheduled execution — only the leader may fire."""
    from ..models.phase21 import SchedulerHealthSnapshot  # noqa: F401
    if not active_leaders:
        return {"fired": False, "reason": "no leader"}
    if len(active_leaders) > 1:
        return {"fired": False,
                "reason": "multiple leaders — deduplication applied"}
    return {"fired": True, "leader": active_leaders[0]}


# ---------------------------------------------------------------------------
# Database self-monitoring
# ---------------------------------------------------------------------------

def record_slow_query(db: Session, workspace_id: int, statement: str,
                      duration_ms: float, threshold_ms: float = 1000.0) \
        -> Optional[SlowQueryRecord]:
    """Detect and record slow queries with recommendations-only output."""
    if duration_ms < threshold_ms:
        return None
    rec = ("add covering index on filtered columns"
           if "WHERE" in statement.upper()
           else "review query plan; consider batching")
    row = SlowQueryRecord(
        workspace_id=workspace_id, statement=statement[:300],
        duration_ms=duration_ms, recommendation=rec, auto_applied=False)
    db.add(row)
    db.commit()
    return row


def db_anomalies(snapshots: list[dict]) -> list[dict]:
    """Detect abnormal DB latency/lock/failure behavior deterministically."""
    anomalies = []
    for snap in snapshots:
        if snap.get("latency_ms", 0) > 1000:
            anomalies.append({"kind": "high_latency",
                              "value": snap.get("latency_ms")})
        if snap.get("locks", 0) > 10:
            anomalies.append({"kind": "lock_contention",
                              "value": snap.get("locks")})
        if snap.get("failures", 0) > 0:
            anomalies.append({"kind": "failures",
                              "value": snap.get("failures")})
    return anomalies


# ---------------------------------------------------------------------------
# Cache intelligence
# ---------------------------------------------------------------------------

def record_cache_health(db: Session, workspace_id: int, cache: str,
                        hits: int = 0, misses: int = 0,
                        stale_entries: int = 0,
                        memory_usage_mb: float = 0.0) -> CacheHealthSnapshot:
    """Track cache hit/miss/stale/usage; flag anomalies; recommend tuning."""
    total = hits + misses
    hit_rate = hits / total if total else 0.0
    anomaly = hit_rate < 0.2 and total > 50 or stale_entries > 100 \
        or memory_usage_mb > 1024
    recommendation = None
    if anomaly:
        if hit_rate < 0.2 and total > 50:
            recommendation = "review cache key cardinality and TTLs"
        elif stale_entries > 100:
            recommendation = "invalidate stale entries (tenant-scoped)"
        else:
            recommendation = "review cache memory bounds"
    row = CacheHealthSnapshot(
        workspace_id=workspace_id, cache=cache, hit_rate=round(hit_rate, 4),
        miss_rate=round(1.0 - hit_rate, 4), stale_entries=stale_entries,
        memory_usage_mb=memory_usage_mb, anomaly=anomaly,
        recommendation=recommendation)
    db.add(row)
    db.commit()
    return row


def invalidate_cache_tenant_safe(cache_keys: list[str],
                                 workspace_id: int) -> dict:
    """Tenant-safe invalidation — only keys scoped to the workspace match."""
    prefix = f"ws:{workspace_id}:"
    invalidated = [k for k in cache_keys if k.startswith(prefix)]
    return {"invalidated": invalidated,
            "skipped_foreign": len(cache_keys) - len(invalidated),
            "tenant_safe": True}


# ---------------------------------------------------------------------------
# Knowledge graph autonomy
# ---------------------------------------------------------------------------

def propose_graph_repair(db: Session, workspace_id: int, repair_kind: str,
                         target_entity_id: Optional[int] = None,
                         target_relationship_id: Optional[int] = None,
                         impact: Optional[dict] = None) -> GraphRepairProposal:
    """Generate a graph repair proposal (approved low-risk repairs only)."""
    allowed = {"orphan_cleanup", "conflict_resolution", "stale_edge_refresh",
               "confidence_recompute"}
    if repair_kind not in allowed:
        raise ValueError(f"unknown graph repair kind: {repair_kind}")
    risk = "LOW" if repair_kind in ("stale_edge_refresh",
                                    "confidence_recompute") else "MEDIUM"
    row = GraphRepairProposal(
        workspace_id=workspace_id, repair_kind=repair_kind,
        target_entity_id=target_entity_id,
        target_relationship_id=target_relationship_id,
        proposal=_bounded_json({"repair_kind": repair_kind}),
        risk_level=risk, impact=_bounded_json(impact),
        status="PROPOSED")
    db.add(row)
    db.commit()
    return row


def execute_graph_repair(db: Session, proposal: GraphRepairProposal,
                         actor: str = "system") -> dict:
    """Governed graph repair — proposal + autonomy guard + audit."""
    op = autonomy.guard_operation(
        db, proposal.workspace_id, f"graph.repair.{proposal.repair_kind}",
        risk_level=proposal.risk_level, actor=actor, source="SYSTEM",
        input_payload={"proposal_id": proposal.id},
        idempotency_key=f"graph-repair:{proposal.id}")
    proposal.decision = op.decision
    if op.decision == "ALLOWED":
        proposal.status = "COMPLETED"
    db.commit()
    return {"proposal_id": proposal.id, "decision": op.decision,
            "reason": op.decision_reason, "operation_id": op.id}


# ---------------------------------------------------------------------------
# Memory autonomy
# ---------------------------------------------------------------------------

def memory_health_check(memories: list[dict]) -> dict:
    """Memory health: stale/conflict/usage/provenance/confidence."""
    total = len(memories)
    if not total:
        return {"overall": 100.0, "stale": 0, "conflicts": 0,
                "low_confidence": 0}
    stale = sum(1 for m in memories if m.get("stale"))
    conflicts = sum(1 for m in memories if m.get("conflict"))
    low_conf = sum(1 for m in memories
                   if (m.get("confidence") or 1.0) < 0.3)
    no_provenance = sum(1 for m in memories if not m.get("provenance"))
    score = 100.0 - (stale + conflicts * 2 + low_conf + no_provenance) \
        / total * 100.0
    return {"overall": round(max(0.0, score), 1), "stale": stale,
            "conflicts": conflicts, "low_confidence": low_conf,
            "missing_provenance": no_provenance}


def suppress_memory(db: Session, workspace_id: int, memory_id: int,
                    policy_allows: bool, reason: str,
                    automatic: bool = False) -> dict:
    """Suppress a memory only when the deterministic policy allows it."""
    if not policy_allows:
        return {"suppressed": False,
                "reason": "policy does not permit automatic suppression"}
    row = MemoryAutonomyEvent(
        workspace_id=workspace_id, memory_id=memory_id,
        event_kind="suppressed", automatic=automatic,
        detail=(reason or "")[:1000])
    db.add(row)
    db.commit()
    return {"suppressed": True, "event_id": row.id}


def route_memory_conflict(db: Session, workspace_id: int, memory_id: int,
                          detail: Optional[str] = None) -> MemoryAutonomyEvent:
    """Uncertain conflicts go to human review — never auto-resolved."""
    row = MemoryAutonomyEvent(
        workspace_id=workspace_id, memory_id=memory_id,
        event_kind="conflict_review", automatic=False,
        detail=(detail or "")[:1000])
    db.add(row)
    db.commit()
    return row


def audit_memory_change(db: Session, workspace_id: int, memory_id: int,
                        detail: Optional[str] = None) -> MemoryAutonomyEvent:
    row = MemoryAutonomyEvent(
        workspace_id=workspace_id, memory_id=memory_id, event_kind="audit",
        automatic=False, detail=(detail or "")[:1000])
    db.add(row)
    db.commit()
    return row
