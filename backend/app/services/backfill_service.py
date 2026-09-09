"""Legacy workspace backfill — deterministic, resumable, idempotent,
batch-based, auditable, dry-run capable.

Rules:
- A document/collection with NULL workspace is assigned ONLY when its owner
  belongs to exactly one workspace (or a unique owned workspace). Ambiguous
  cases are counted, reported and left unchanged — never guessed.
- Every batch writes audit records + a BackfillRun ledger row.
- ``--dry-run`` is the default posture of the CLI; no destructive default.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase16 import BackfillRun, BACKFILL_KINDS
from ..models.document import Document
from ..models.collection import Collection
from ..models.workspace import Workspace, WorkspaceMember
from ..services.audit_service import log_audit_event

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 500


class BackfillError(Exception):
    """Raised on invalid backfill configuration."""


def audit_legacy(db: Session) -> dict:
    """Count unassigned legacy records (never migrates anything)."""
    docs_null = db.query(Document).filter(Document.workspace_id.is_(None)).count()
    cols_null = db.query(Collection).filter(Collection.workspace_id.is_(None)).count()
    return {
        "documents_without_workspace": docs_null,
        "collections_without_workspace": cols_null,
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _user_workspaces(db: Session, user_id: int) -> list[Workspace]:
    memberships = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.user_id == user_id)
        .all()
    )
    member_ids = {m.workspace_id for m in memberships}
    owned = [w.id for w in db.query(Workspace).filter(
        Workspace.owner_id == user_id).all()]
    ids = sorted(set(member_ids) | set(owned))
    if not ids:
        return []
    return db.query(Workspace).filter(Workspace.id.in_(ids)).all()


def resolve_workspace_for_user(db: Session, user_id: int) -> tuple[Optional[int], str]:
    """Resolve the unambiguous workspace for a legacy owner.

    Returns (workspace_id, decision) where decision ∈ assigned / ambiguous /
    no_workspace. Ambiguity is never guessed.
    """
    workspaces = _user_workspaces(db, user_id)
    if not workspaces:
        return None, "no_workspace"
    # Ownership is the strongest signal: a workspace the user owns wins
    # outright; otherwise exactly one membership resolves cleanly.
    owned = [w for w in workspaces if w.owner_id == user_id]
    if len(owned) == 1:
        return owned[0].id, "assigned"
    if len(owned) > 1:
        return None, "ambiguous"
    if len(workspaces) == 1:
        return workspaces[0].id, "assigned"
    return None, "ambiguous"


def create_run(
    db: Session,
    created_by: int,
    kind: str,
    dry_run: bool = False,
    workspace_id: Optional[int] = None,
    user_id: Optional[int] = None,
    batch_size: int = 100,
    resume_from: Optional[int] = None,
) -> BackfillRun:
    if kind not in BACKFILL_KINDS:
        raise BackfillError(f"Unknown backfill kind: {kind}")
    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise BackfillError(f"batch_size must be 1..{MAX_BATCH_SIZE}")
    run = BackfillRun(
        kind=kind,
        status="DRY_RUN" if dry_run else "PENDING",
        dry_run=dry_run,
        workspace_id=workspace_id,
        user_id=user_id,
        batch_size=batch_size,
        cursor_id=resume_from,
        created_by=created_by,
        total=0,
    )
    db.add(run)
    db.flush()
    log_audit_event(
        db, event_type="backfill", event_action="create",
        user_id=created_by, resource_type="backfill_run",
        resource_id=run.id,
        details=f"kind={kind} dry_run={dry_run} batch_size={batch_size}",
    )
    return run


def preview_batch(
    db: Session,
    kind: str,
    batch_size: int = 100,
    cursor_id: Optional[int] = None,
) -> dict:
    """Dry preview of the next batch — includes decisions, changes nothing."""
    items = _next_candidates(db, kind, batch_size, cursor_id)
    rows = []
    for record in items:
        owner_id = record.user_id
        workspace_id, decision = resolve_workspace_for_user(db, owner_id)
        rows.append({
            "record_id": record.id,
            "record_type": kind.split("_")[0],
            "owner_id": owner_id,
            "decision": decision,
            "workspace_id": workspace_id,
        })
    return {"batch_size": len(rows), "rows": rows}


def _next_candidates(db: Session, kind: str, batch_size: int,
                     cursor_id: Optional[int],
                     user_id: Optional[int] = None):
    batch_size = min(batch_size, MAX_BATCH_SIZE)
    if kind == "document_workspace":
        query = db.query(Document).filter(Document.workspace_id.is_(None))
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
    elif kind == "collection_workspace":
        query = db.query(Collection).filter(Collection.workspace_id.is_(None))
    else:
        raise BackfillError(f"Unknown backfill kind: {kind}")
    if cursor_id:
        query = query.filter(record_pk(kind) > cursor_id)
    return query.order_by(record_pk(kind)).limit(batch_size).all()


def record_pk(kind: str):
    return Document.id if kind == "document_workspace" else Collection.id


def execute_batch(db: Session, run: BackfillRun) -> dict:
    """Execute one batch of an existing run (idempotent per row)."""
    if run.status in ("COMPLETED", "FAILED"):
        raise BackfillError(f"Run is already {run.status}")
    items = _next_candidates(db, run.kind, run.batch_size, run.cursor_id,
                              user_id=run.user_id)
    stats = {"processed": 0, "assigned": 0, "skipped": 0,
             "ambiguous": 0, "failed": 0}
    last_id = run.cursor_id
    for record in items:
        owner_id = record.user_id
        workspace_id, decision = resolve_workspace_for_user(db, owner_id)
        stats["processed"] += 1
        last_id = record.id
        if decision == "ambiguous":
            stats["ambiguous"] += 1
            continue
        if decision == "no_workspace":
            stats["skipped"] += 1
            continue
        if not run.dry_run:
            record.workspace_id = workspace_id
            log_audit_event(
                db, event_type="backfill", event_action="assign",
                user_id=run.created_by, resource_type=run.kind,
                resource_id=record.id,
                details=(f"workspace_id={workspace_id} dry_run={run.dry_run} "
                         f"run_id={run.id}"),
            )
        stats["assigned"] += 1
    if run.dry_run:
        run.status = "DRY_RUN"
    db.flush()
    run.processed += stats["processed"]
    run.assigned += stats["assigned"]
    run.skipped += stats["skipped"]
    run.ambiguous += stats["ambiguous"]
    run.failed += stats["failed"]
    run.cursor_id = last_id
    run.total = max(run.total, run.processed)
    if len(items) < run.batch_size:
        run.status = "COMPLETED" if not run.dry_run else "DRY_RUN"
        run.completed_at = datetime.now(timezone.utc)
    return {**stats, "batch_finished": len(items) < run.batch_size,
            "run_status": run.status}


def run_complete(db: Session, run_id: int, batch_limit: int = 10000) -> dict:
    """Drive a run to completion in bounded batches (used by API/CLI)."""
    run = db.query(BackfillRun).filter(BackfillRun.id == run_id).first()
    if run is None:
        raise BackfillError(f"Run {run_id} not found")
    total_stats = {"processed": 0, "assigned": 0, "skipped": 0,
                   "ambiguous": 0, "failed": 0}
    for _ in range(batch_limit // max(1, run.batch_size) + 1):
        stats = execute_batch(db, run)
        for key in total_stats:
            total_stats[key] += stats[key]
        db.commit()
        if stats["batch_finished"] or run.status in ("COMPLETED", "FAILED"):
            break
    return {**total_stats, "run_id": run.id, "status": run.status,
            "dry_run": run.dry_run}
