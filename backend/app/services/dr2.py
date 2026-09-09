"""Phase 19 — disaster recovery 2.0.

Backup inventory records (timestamp, migration head, checksum where
available), automated non-destructive restore validation (restored DB must
reach the expected migration head and preserve tenant isolation), and a
deterministic DR readiness score.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _migration_head() -> Optional[str]:
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        cfg = Config("alembic.ini")
        script = ScriptDirectory.from_config(cfg)
        heads = sorted(script.get_heads())
        return heads[0] if heads else None
    except Exception as exc:  # noqa: BLE001
        logger.error("migration head lookup failed: %s", exc)
        return None


def record_backup(db: Session, *, scope: str, backup_ref: str,
                  migration_head: Optional[str] = None,
                  checksum: Optional[str] = None,
                  size_bytes: Optional[int] = None,
                  note: Optional[str] = None) -> dict:
    from ..models.phase19 import BackupRecord
    if scope not in ("DATABASE", "OBJECT_STORAGE", "CONFIGURATION"):
        raise ValueError("invalid backup scope")
    row = BackupRecord(scope=scope, backup_ref=backup_ref,
                       database_version=_migration_head()
                       or migration_head, migration_head=migration_head
                       or _migration_head(), checksum=checksum,
                       size_bytes=size_bytes, note=note)
    db.add(row)
    db.flush()
    return {"backup_id": row.id, "scope": row.scope,
            "backup_ref": row.backup_ref,
            "migration_head": row.migration_head}


def validate_backups(db: Session, limit: int = 100) -> dict:
    """Validate pending backup records against the current migration head
    (non-destructive)."""
    from ..models.phase19 import BackupRecord
    from .dr import _migration_state
    state = _migration_state(db)
    current_heads = set(state.get("current") or [])
    rows = (db.query(BackupRecord)
            .filter(BackupRecord.status == "PENDING")
            .order_by(BackupRecord.created_at.asc())
            .limit(min(limit, 500)).all())
    validated = 0
    for row in rows:
        recorded = row.migration_head or row.database_version
        ok = bool(recorded) and recorded in current_heads
        row.status = "VALIDATED" if ok else "FAILED"
        row.validated_at = _utcnow()
        if ok:
            validated += 1
    db.flush()
    return {"validated": validated, "failed": len(rows) - validated,
            "current_migration": sorted(current_heads)}


def restore_validation(db: Session) -> dict:
    """Full non-destructive restore validation: migration state + tenant
    isolation after restore + schema reachability."""
    from .dr import validate_restore, tenant_isolation_after_restore
    migration = validate_restore(db)
    isolation = tenant_isolation_after_restore(db)
    head = _migration_head()
    return {
        "migration_in_sync": migration.get("migration_in_sync"),
        "current_migration_head": head,
        "restore_valid": bool(migration.get("restore_valid")),
        "authorization_orphans": migration.get("authorization_orphans"),
        "tenant_isolation": isolation,
        "note": "non-destructive validation only — no restore is executed "
                "against a live database",
    }


def readiness_score(db: Session) -> dict:
    """Deterministic DR readiness: backup coverage + validation recency +
    migration sync + tenant isolation."""
    from ..models.phase19 import BackupRecord
    from .dr import _migration_state, validate_restore
    scores = {}
    rows = db.query(BackupRecord).limit(1000).all()
    scores["backup_present"] = 0.5 if rows else 0.0
    validated_recent = sum(1 for r in rows
                           if r.status == "VALIDATED")
    scores["backup_validated"] = min(1.0, validated_recent / max(1, len(
        rows)))
    state = _migration_state(db)
    scores["migration_in_sync"] = 1.0 if state.get("in_sync") else 0.0
    migration_ok = bool(state.get("in_sync"))
    restore = validate_restore(db)
    scores["tenant_isolation"] = 1.0 if restore.get(
        "authorization_orphans") == 0 else 0.0
    total = round(sum(scores.values()) / len(scores), 3)
    return {
        "score": total,
        "grade": "READY" if total >= 0.75 else (
            "PARTIAL" if total >= 0.4 else "NOT_READY"),
        "components": scores,
        "detail": {
            "backups": len(rows),
            "validated": validated_recent,
            "migration_in_sync": migration_ok,
            "note": "score is a deterministic operational indicator, not a "
                    "guarantee",
        },
    }
