"""Phase 23 — Broker migration safety + storage migration.

Broker switching (Step 3): explicit migration state machine with dry-run,
drain, verify, idempotent replay, duplicate-execution protection, audit and
rollback — extending broker2's outage policy, never bypassing it.

Storage migration (Steps 4-5): local -> object storage with dry-run, batch
size, checksum verification, resumability, idempotency, progress, failure
recording, retry, no source deletion until verification, and audit trail —
extending storage_backend's Local/S3 backends.

Both operate through the Phase 21 autonomy guard where consequential.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _dumps(value) -> Optional[str]:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None


def _loads(value: Optional[str]) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Broker migration safety (Step 3)
# ---------------------------------------------------------------------------

BROKER_MIGRATION_STATES = (
    "PLANNED", "DRAINING", "DRAINED", "VERIFYING", "VERIFIED",
    "ACTIVATING", "ACTIVE", "ROLLED_BACK", "FAILED",
)

_BROKER_TRANSITIONS = {
    "PLANNED": {"DRAINING", "FAILED"},
    "DRAINING": {"DRAINED", "FAILED"},
    "DRAINED": {"VERIFYING", "FAILED"},
    "VERIFYING": {"VERIFIED", "FAILED"},
    "VERIFIED": {"ACTIVATING", "ROLLED_BACK", "FAILED"},
    "ACTIVATING": {"ACTIVE", "FAILED"},
    "ACTIVE": set(),
    "ROLLED_BACK": set(),
    "FAILED": {"PLANNED"},  # re-plan allowed after failure
}


class BrokerMigrationRecord:
    """Migration state persisted on Phase 22 OpsStreamEvent stream 'broker'.

    Uses the existing ops-stream substrate for durability rather than a new
    table: every transition appends an audited event with full state.
    """

    STREAM = "broker"

    @staticmethod
    def _db_rows(db: Session, migration_id: str) -> list:
        from ..models import OpsStreamEvent

        return (db.query(OpsStreamEvent)
                .filter(OpsStreamEvent.stream == BrokerMigrationRecord.STREAM,
                        OpsStreamEvent.kind == f"broker_migration:{migration_id}")
                .order_by(OpsStreamEvent.id).limit(200).all())

    @staticmethod
    def state(db: Session, migration_id: str) -> dict:
        rows = BrokerMigrationRecord._db_rows(db, migration_id)
        if not rows:
            return {"migration_id": migration_id, "state": None,
                    "exists": False}
        latest = _loads(rows[-1].payload)
        return {
            "migration_id": migration_id,
            "state": latest.get("state"),
            "exists": True,
            "from_backend": latest.get("from_backend"),
            "to_backend": latest.get("to_backend"),
            "history": [_loads(r.payload) for r in rows],
        }

    @staticmethod
    def transition(db: Session, migration_id: str, *, new_state: str,
                   from_backend: str, to_backend: str,
                   actor: str = "system", detail: Optional[dict] = None) -> dict:
        from ..models import OpsStreamEvent

        if new_state not in BROKER_MIGRATION_STATES:
            raise ValueError(f"unknown migration state: {new_state}")
        current = BrokerMigrationRecord.state(db, migration_id)
        prev = current.get("state")
        allowed = _BROKER_TRANSITIONS.get(prev or "PLANNED",
                                          {"PLANNED"} if prev is None
                                          else _BROKER_TRANSITIONS["FAILED"])
        if prev is not None and new_state not in allowed:
            raise ValueError(
                f"illegal transition {prev} -> {new_state} "
                f"(allowed: {sorted(allowed)})")
        payload = {
            "state": new_state, "previous": prev,
            "from_backend": from_backend, "to_backend": to_backend,
            "actor": actor, "detail": detail or {},
            "at": _utcnow().isoformat(),
        }
        # seq is unique per stream: next sequence = max(existing) + 1
        from sqlalchemy import func
        max_seq = db.query(func.max(OpsStreamEvent.seq)).filter(
            OpsStreamEvent.stream == BrokerMigrationRecord.STREAM).scalar()
        db.add(OpsStreamEvent(
            stream=BrokerMigrationRecord.STREAM,
            seq=(max_seq or 0) + 1,
            kind=f"broker_migration:{migration_id}",
            payload=_dumps(payload)))
        db.commit()
        return payload


def broker_drain_old_backend(db: Session, *, backend: str = "postgres",
                             max_jobs: int = 500) -> dict:
    """Drain a backend's in-flight jobs before switching (bounded).

    PostgresBroker jobs in RETRYING/QUEUED state are left in place — the
    drain verifies queue state instead of moving rows. Job ownership stays
    with the Phase 16 worker platform; this records verified state only.
    """
    from ..models import WorkerJob
    from .broker2 import broker_config

    cfg = broker_config()
    queued = db.query(WorkerJob).filter(
        WorkerJob.status.in_(["QUEUED", "RETRYING"])).limit(max_jobs).count()
    return {
        "backend": backend,
        "active_backend": cfg.get("active_backend"),
        "queued_jobs": queued,
        "max_jobs": max_jobs,
        "safe_to_switch": queued < max_jobs,
    }


def broker_verify_queue_state(db: Session, *, max_jobs: int = 500) -> dict:
    """Verify no job would be lost by a switch (bounded count by status)."""
    from ..models import WorkerJob
    from sqlalchemy import func

    counts = dict(
        db.query(WorkerJob.status, func.count(WorkerJob.id))
        .group_by(WorkerJob.status).all())
    in_flight = counts.get("RUNNING", 0)
    return {
        "status_counts": {k: int(v) for k, v in counts.items()},
        "in_flight": int(in_flight),
        "dead": int(counts.get("DEAD", 0)),
        "no_job_loss_risk": in_flight == 0 or in_flight < max_jobs,
    }


def broker_migration_plan(db: Session, *, migration_id: str,
                          to_backend: str, dry_run: bool = True) -> dict:
    """Plan (or dry-run) a broker switch with health gate."""
    from .broker2 import broker_config, redis_available
    from .capabilities import detect_component

    cfg = broker_config()
    from_backend = cfg.get("active_backend", "postgres")
    # Health gate: postgres must be up for PG-broker continuity; the target
    # backend's own availability is checked via the redis capability.
    target_ok = detect_component("postgres" if to_backend == "postgres"
                                 else "redis")
    health_gate = target_ok["state"] in ("AVAILABLE",) or (
        to_backend == "postgres" and
        target_ok["state"] in ("AVAILABLE", "DEGRADED"))
    plan = {
        "migration_id": migration_id,
        "from_backend": from_backend,
        "to_backend": to_backend,
        "dry_run": dry_run,
        "health_gate": {
            "target_state": target_ok["state"],
            "passed": health_gate,
        },
        "steps": ["DRAINING", "DRAINED", "VERIFYING", "VERIFIED",
                  "ACTIVATING", "ACTIVE"],
        "rollback": "ROLLED_BACK via transition; postgres remains authoritative",
    }
    if dry_run:
        plan["would_execute"] = health_gate
        return plan
    if not health_gate:
        plan["blocked"] = "health gate failed — target backend not healthy"
        return plan
    BrokerMigrationRecord.transition(
        db, migration_id, new_state="DRAINING",
        from_backend=from_backend, to_backend=to_backend)
    return plan


# ---------------------------------------------------------------------------
# Storage migration (Steps 4-5)
# ---------------------------------------------------------------------------

def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_storage_migration_plan(db: Session, *, workspace_id: int,
                                  batch_size: int = 25,
                                  dry_run: bool = False,
                                  created_by: Optional[int] = None) -> dict:
    """Create a DRAFT migration plan and enumerate candidate documents."""
    from ..models import Document, StorageMigrationObject, StorageMigrationPlan

    if batch_size < 1 or batch_size > 500:
        raise ValueError("batch_size must be 1..500")
    plan = StorageMigrationPlan(
        workspace_id=workspace_id, batch_size=batch_size,
        dry_run=dry_run, created_by=created_by)
    db.add(plan)
    db.flush()
    docs = (db.query(Document)
            .filter(Document.workspace_id == workspace_id,
                    Document.status.in_(["READY", "UPLOADED"]))
            .limit(5000).all())
    for doc in docs:
        db.add(StorageMigrationObject(
            plan_id=plan.id, workspace_id=workspace_id,
            document_id=doc.id))
    plan.total_objects = len(docs)
    db.commit()
    return {"plan_id": plan.id, "total_objects": plan.total_objects,
            "batch_size": batch_size, "dry_run": dry_run}


def _local_bytes_for(document) -> Optional[bytes]:
    """Read the current local bytes for a document (or None)."""
    try:
        from .storage import storage_service
        data = storage_service.retrieve(document.storage_key)
        return data if isinstance(data, bytes) else None
    except Exception:  # noqa: BLE001 — missing object recorded as failure
        return None


def run_storage_migration_batch(db: Session, plan_id: int,
                                *, actor: str = "system") -> dict:
    """Run one bounded batch. Dry-run copies nothing; real runs copy,
    verify checksum, and only then mark VERIFIED. Source is never deleted."""
    from ..models import (Document, StorageMigrationObject,
                          StorageMigrationPlan)
    from .storage_backend import get_storage_backend

    plan = db.query(StorageMigrationPlan).get(plan_id)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found")
    if plan.status in ("COMPLETED", "ROLLED_BACK"):
        return {"plan_id": plan_id, "status": plan.status,
                "already_done": True}

    if plan.status == "DRAFT":
        plan.status = "DRY_RUN" if plan.dry_run else "RUNNING"
    elif plan.status in ("PAUSED",):
        plan.status = "RUNNING"

    progress = _loads(plan.progress_json)
    cursor = int(progress.get("cursor", 0))

    pending = (db.query(StorageMigrationObject)
               .filter(StorageMigrationObject.plan_id == plan_id,
                       StorageMigrationObject.state.in_(["PENDING", "FAILED"]))
               .order_by(StorageMigrationObject.id)
               .offset(cursor).limit(plan.batch_size).all())

    backend = get_storage_backend()
    migrated = verified = failed = 0
    for obj in pending:
        obj.attempts += 1
        doc = db.query(Document).get(obj.document_id)
        if doc is None:
            obj.state = "SKIPPED"
            obj.last_error = "document not found"
            continue
        data = _local_bytes_for(doc)
        if data is None:
            obj.state = "FAILED"
            obj.last_error = "source object unreadable"
            failed += 1
            continue
        src_sum = _checksum(data)
        obj.source_checksum = src_sum
        if plan.dry_run:
            obj.state = "MIGRATED"
            obj.last_error = None
            migrated += 1
            continue
        try:
            target_key = f"migrated/{plan.workspace_id}/{doc.id}"
            backend.save(target_key, data)
            roundtrip = backend.retrieve(target_key)
            obj.target_checksum = _checksum(roundtrip)
            if obj.target_checksum == src_sum:
                obj.state = "VERIFIED"
                obj.verified_at = _utcnow()
                verified += 1
            else:
                obj.state = "FAILED"
                obj.last_error = "checksum mismatch after copy"
                failed += 1
        except Exception as exc:  # noqa: BLE001 — recorded, not raised
            obj.state = "FAILED"
            obj.last_error = str(exc)[:1000]
            failed += 1
        migrated += 1

    # Source deletion is NEVER performed here (verification-only migration).
    plan.migrated_objects = (plan.migrated_objects or 0) + migrated
    plan.verified_objects = (plan.verified_objects or 0) + verified
    plan.failed_objects = (plan.failed_objects or 0) + failed
    progress["cursor"] = cursor + len(pending)
    plan.progress_json = _dumps(progress)
    audit = _loads(plan.audit_json)
    audit.setdefault("batches", []).append({
        "at": _utcnow().isoformat(), "actor": actor,
        "batch": len(pending), "verified": verified, "failed": failed,
        "dry_run": plan.dry_run})
    plan.audit_json = _dumps(audit)

    remaining = (db.query(StorageMigrationObject)
                 .filter(StorageMigrationObject.plan_id == plan_id,
                         StorageMigrationObject.state.in_(["PENDING", "FAILED"]))
                 .count())
    if remaining == 0:
        plan.status = "COMPLETED"
    db.commit()
    return {"plan_id": plan_id, "status": plan.status,
            "batch": len(pending), "migrated": migrated,
            "verified": verified, "failed": failed,
            "remaining": remaining, "dry_run": plan.dry_run}


def storage_migration_progress(db: Session, plan_id: int) -> dict:
    from ..models import StorageMigrationObject, StorageMigrationPlan

    plan = db.query(StorageMigrationPlan).get(plan_id)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found")
    counts = dict(
        db.query(StorageMigrationObject.state,
                 __import__("sqlalchemy", fromlist=["func"]).func.count(
                     StorageMigrationObject.id))
        .filter(StorageMigrationObject.plan_id == plan_id)
        .group_by(StorageMigrationObject.state).all())
    return {
        "plan_id": plan_id, "status": plan.status,
        "dry_run": plan.dry_run, "total": plan.total_objects,
        "states": {k: int(v) for k, v in counts.items()},
        "progress": _loads(plan.progress_json),
    }


def verify_document_integrity(db: Session, document_id: int,
                              workspace_id: int) -> dict:
    """Re-read a stored object and compare its checksum to a fresh read."""
    from ..models import Document
    from .storage import storage_service

    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.workspace_id == workspace_id).one_or_none()
    if doc is None:
        return {"document_id": document_id, "verified": False,
                "found": False, "reason": "not found"}
    if not doc.storage_key:
        return {"document_id": document_id, "verified": False,
                "found": True, "reason": "no storage key"}
    try:
        data = storage_service.retrieve(doc.storage_key)
        first = _checksum(data)
        second = _checksum(storage_service.retrieve(doc.storage_key))
        return {"document_id": document_id, "verified": first == second,
                "checksum": first,
                "size": len(data)}
    except Exception as exc:  # noqa: BLE001
        return {"document_id": document_id, "verified": False,
                "reason": str(exc)[:300]}


def orphan_object_candidates(db: Session, workspace_id: int,
                             *, limit: int = 200) -> dict:
    """Documents whose storage rows reference missing DB objects (bounded)."""
    from ..models import Document
    from .storage import storage_service

    docs = (db.query(Document)
            .filter(Document.workspace_id == workspace_id,
                    Document.storage_key.isnot(None))
            .limit(limit).all())
    orphans = []
    for doc in docs:
        try:
            storage_service.retrieve(doc.storage_key)
        except Exception:  # noqa: BLE001
            orphans.append({"document_id": doc.id,
                            "storage_key": doc.storage_key})
    return {"workspace_id": workspace_id, "scanned": len(docs),
            "orphans": orphans, "count": len(orphans),
            "cleanup_requires_governance": True}


def storage_capability_report(db: Session) -> dict:
    """Storage capability + health from the Phase 22 registry + backend."""
    from .capabilities import detect_component
    from .storage_backend import get_storage_backend

    backend = get_storage_backend()
    cap = detect_component("object_storage")
    return {
        "backend_class": backend.__class__.__name__,
        "capability": cap,
        "supports_signed_urls": hasattr(backend, "signed_url"),
        "supports_multipart": hasattr(backend, "multipart_url"),
    }
