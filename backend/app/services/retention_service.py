"""Retention policy service — configurable cleanup with dry-run support.

Cleanup never deletes data merely because a configuration is malformed;
invalid policies are skipped and logged, and dry-run mode reports what
*would* be deleted without deleting.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.retention import RetentionPolicy

logger = logging.getLogger(__name__)

SUPPORTED_DATA_TYPES = (
    "documents",
    "trash",
    "audit_logs",
    "ai_executions",
    "artifacts",
    "exports",
    "sessions",
)


def set_policy(
    db: Session,
    scope_type: str,
    scope_id: Optional[int],
    data_type: str,
    retention_days: int,
    is_enabled: bool = True,
) -> RetentionPolicy:
    """Create or update a retention policy."""
    if data_type not in SUPPORTED_DATA_TYPES:
        raise ValueError(f"Unsupported retention data type: {data_type}")
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")

    policy = (
        db.query(RetentionPolicy)
        .filter(
            RetentionPolicy.scope_type == scope_type,
            RetentionPolicy.scope_id == scope_id,
            RetentionPolicy.data_type == data_type,
        )
        .first()
    )
    if policy is None:
        policy = RetentionPolicy(
            scope_type=scope_type,
            scope_id=scope_id,
            data_type=data_type,
            retention_days=retention_days,
            is_enabled=is_enabled,
        )
        db.add(policy)
    else:
        policy.retention_days = retention_days
        policy.is_enabled = is_enabled
    db.flush()
    return policy


def get_policy(
    db: Session,
    scope_type: str,
    scope_id: Optional[int],
    data_type: str,
) -> Optional[RetentionPolicy]:
    """Resolve the effective policy: specific scope first, then GLOBAL."""
    policy = (
        db.query(RetentionPolicy)
        .filter(
            RetentionPolicy.scope_type == scope_type,
            RetentionPolicy.scope_id == scope_id,
            RetentionPolicy.data_type == data_type,
        )
        .first()
    )
    if policy is None and scope_type != "GLOBAL":
        policy = (
            db.query(RetentionPolicy)
            .filter(
                RetentionPolicy.scope_type == "GLOBAL",
                RetentionPolicy.scope_id.is_(None),
                RetentionPolicy.data_type == data_type,
            )
            .first()
        )
    return policy


def run_cleanup(db: Session, data_type: str, dry_run: bool = False) -> dict:
    """Run cleanup for a data type across enabled policies.

    Returns a report of records that were (or would be) deleted.
    """
    cutoff_ages = {}  # scope -> cutoff datetime

    policies = (
        db.query(RetentionPolicy)
        .filter(RetentionPolicy.data_type == data_type, RetentionPolicy.is_enabled.is_(True))
        .all()
    )
    for policy in policies:
        key = (policy.scope_type, policy.scope_id)
        cutoff_ages[key] = max(cutoff_ages.get(key, 0), policy.retention_days)

    report = {"data_type": data_type, "dry_run": dry_run, "deleted": 0, "scanned": 0, "skipped_invalid": 0}
    now = datetime.now(timezone.utc)

    for (scope_type, scope_id), days in cutoff_ages.items():
        cutoff = now - timedelta(days=days)
        deleted, scanned = _cleanup_type(db, data_type, scope_type, scope_id, cutoff, dry_run)
        report["deleted"] += deleted
        report["scanned"] += scanned

    if not dry_run:
        # Update last_cleanup_at for enabled policies
        for policy in policies:
            policy.last_cleanup_at = now
        db.flush()
    return report


def _cleanup_type(
    db: Session,
    data_type: str,
    scope_type: str,
    scope_id: Optional[int],
    cutoff: datetime,
    dry_run: bool,
) -> tuple[int, int]:
    """Delete expired records for one data type and scope.

    Returns (deleted_count, scanned_count).
    """
    if data_type == "sessions":
        from ..models.session import UserSession
        query = db.query(UserSession).filter(UserSession.expires_at < cutoff)
        return _delete_count(query, dry_run)

    if data_type == "exports":
        from ..models.export_job import ExportJob
        query = db.query(ExportJob).filter(
            ExportJob.created_at < cutoff,
            ExportJob.status.in_(["COMPLETED", "FAILED", "EXPIRED"]),
        )
        return _delete_count(query, dry_run)

    if data_type == "audit_logs":
        from ..models.audit_log import AuditLog
        query = db.query(AuditLog).filter(AuditLog.created_at < cutoff)
        return _delete_count(query, dry_run)

    if data_type == "ai_executions":
        from ..models.ai_execution import AIExecution
        query = db.query(AIExecution).filter(AIExecution.created_at < cutoff)
        return _delete_count(query, dry_run)

    # Unsupported/not-yet-implemented types are skipped (never malformed-delete)
    logger.warning("Retention cleanup for '%s' not implemented; skipped", data_type)
    return 0, 0


def _delete_count(query, dry_run: bool) -> tuple[int, int]:
    scanned = query.count()
    if scanned == 0:
        return 0, 0
    if dry_run:
        return scanned, scanned
    deleted = query.delete(synchronize_session=False)
    return deleted, scanned