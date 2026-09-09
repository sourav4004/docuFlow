"""Phase 18 federation operations — connector runtime, health, rate limits.

Connector synchronization runs through the durable worker queue; sync state
is resumable (cursor + content hashes) and rate limits are enforced
deterministically per source kind.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase17 import ConnectorSource, ConnectorSync

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMITS = {
    "cloud_storage": 60,   # requests / 5 min
    "email": 30,
    "collaboration": 40,
    "knowledge_base": 50,
    "generic": 40,
}
SYNC_COOLDOWN_SECONDS = int(os.getenv("CONNECTOR_SYNC_COOLDOWN", "300"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def rate_limit_for(kind: str) -> int:
    return DEFAULT_RATE_LIMITS.get(kind, DEFAULT_RATE_LIMITS["generic"])


def can_sync_now(db: Session, source_id: int, *,
                 cooldown_seconds: int = SYNC_COOLDOWN_SECONDS,
                 now: Optional[datetime] = None) -> tuple[bool, str]:
    """Respect per-source cooldown so a hot connector cannot hammer the API."""
    now = now or _utcnow()
    last = (db.query(ConnectorSync)
            .filter(ConnectorSync.source_id == source_id)
            .order_by(ConnectorSync.started_at.desc()).first())
    if last is None:
        return True, "never synced"
    last_start = _as_utc(last.started_at)
    if last_start is None:
        return True, "no timestamp"
    elapsed = (now - last_start).total_seconds()
    if elapsed < cooldown_seconds:
        return False, (f"cooldown active ({int(cooldown_seconds - elapsed)}s "
                       "remaining)")
    return True, "ready"


def enqueue_connector_sync(db: Session, *, source_id: int,
                           workspace_id: int) -> dict:
    """Schedule a connector sync through the durable worker queue."""
    from .worker_platform import enqueue_job
    ok, reason = can_sync_now(db, source_id)
    if not ok:
        return {"enqueued": False, "reason": reason}
    job = enqueue_job(
        db, queue_name="CONNECTORS", job_type="CONNECTOR_SYNC",
        workspace_id=workspace_id, payload={"source_id": source_id},
        dedupe_key=f"conn-sync:{source_id}",
        max_attempts=3)
    return {"enqueued": True, "job_id": job.id, "reason": reason}


def connector_health(db: Session, workspace_id: Optional[int] = None,
                     limit: int = 100) -> dict:
    """Per-connector health: last sync, lag, errors, throughput."""
    q = db.query(ConnectorSource)
    if workspace_id is not None:
        q = q.filter(ConnectorSource.workspace_id == workspace_id)
    sources = q.limit(min(limit, 500)).all()
    items = []
    now = _utcnow()
    for src in sources:
        last = (db.query(ConnectorSync)
                .filter(ConnectorSync.source_id == src.id)
                .order_by(ConnectorSync.started_at.desc()).first())
        lag_s = None
        if last is not None and last.completed_at is not None:
            lag_s = round((now - _as_utc(last.completed_at)).total_seconds(), 1)
        total_items = (last.items_added + last.items_changed + last.items_deleted
                       if last else 0)
        items.append({
            "id": src.id, "name": src.name, "kind": src.kind,
            "enabled": src.enabled,
            "scopes": src.scopes_json,
            "last_sync_status": last.status if last else None,
            "last_sync_at": last.completed_at if last else None,
            "lag_s": lag_s,
            "last_items": total_items,
            "last_error": (last.error if last else None) or "",
            "rate_limit_per_5min": rate_limit_for(src.kind),
        })
    return {"connectors": items, "total": len(items)}


def connector_scope_summary(source: ConnectorSource) -> dict:
    """Declared scopes/permissions — never credentials."""
    import json
    def _loads(raw):
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return []
    return {
        "organization_scope": source.organization_id,
        "workspace_scope": source.workspace_id,
        "resource_scopes": _loads(source.scopes_json),
        "permissions": _loads(source.permissions_json),
        "allowed_domains": _loads(source.allowed_domains_json),
        "credential_ref": bool(source.credential_ref),
        "retention_days": source.retention_days,
    }


def sync_recovery_plan(db: Session, source_id: int) -> dict:
    """Resume plan after an interrupted sync: cursor + pending tombstones."""
    from ..models.phase17 import ConnectorItem
    src = db.get(ConnectorSource, source_id)
    if src is None:
        raise KeyError(f"connector source {source_id} not found")
    last = (db.query(ConnectorSync)
            .filter(ConnectorSync.source_id == source_id,
                    ConnectorSync.status == "RUNNING")
            .order_by(ConnectorSync.started_at.desc()).first())
    pending_deletes = (db.query(ConnectorItem)
                       .filter(ConnectorItem.source_id == source_id,
                               ConnectorItem.deleted.is_(False))
                       .count())
    return {
        "source_id": source_id,
        "resume_from_cursor": (last.cursor_json if last else None),
        "interrupted_sync_id": (last.id if last else None),
        "known_items": pending_deletes,
        "plan": ("resume from cursor; re-fetch changed since last seen"
                 if last else "full initial sync"),
    }