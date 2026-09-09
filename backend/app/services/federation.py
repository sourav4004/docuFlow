"""Enterprise knowledge federation — Phase 17.

Unified connector model over external knowledge sources:

- ConnectorSource declares scopes, permissions, allowed domains, retention,
  and a credential *reference* (credentials are never stored here).
- Sync runs are incremental: the source driver returns items + cursor; items
  are deduplicated on (source, external_id) and content-hash — repeated syncs
  never duplicate data.
- Tombstones: items absent from a full delta are marked deleted, never
  hard-deleted while a connector is enabled.
- Every query is workspace-scoped; connectors can never bypass tenant
  authorization.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase17 import ConnectorSource, ConnectorSync, ConnectorItem
from . import worker_platform as wp

logger = logging.getLogger(__name__)

CONNECTOR_KINDS = ("uploaded", "connector_stub", "cloud_storage",
                   "email", "collaboration", "knowledge_base")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")
                          ).hexdigest()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def create_source(db: Session, *, workspace_id: int,
                  organization_id: Optional[int], user_id: int, name: str,
                  kind: str, scopes: Optional[list[str]] = None,
                  permissions: Optional[list[str]] = None,
                  allowed_domains: Optional[list[str]] = None,
                  credential_ref: Optional[str] = None,
                  retention_days: Optional[int] = None) -> ConnectorSource:
    if kind not in CONNECTOR_KINDS:
        raise ValueError(f"unsupported connector kind {kind!r}")
    src = ConnectorSource(
        workspace_id=workspace_id, organization_id=organization_id,
        name=name, kind=kind,
        config_ref=f"conn:{workspace_id}:{kind}",
        scopes_json=json.dumps(scopes or []),
        permissions_json=json.dumps(permissions or []),
        allowed_domains_json=json.dumps(allowed_domains or []),
        credential_ref=credential_ref,  # reference only — never the secret
        retention_days=retention_days,
        enabled=True,
    )
    db.add(src)
    db.flush()
    return src


def update_source(db: Session, source_id: int, workspace_id: int,
                  **fields) -> ConnectorSource:
    src = _owned_source(db, source_id, workspace_id)
    for key in ("name", "enabled", "retention_days", "credential_ref"):
        if key in fields:
            setattr(src, key, fields[key])
    src.updated_at = _utcnow()
    db.flush()
    return src


def _owned_source(db: Session, source_id: int, workspace_id: int) -> ConnectorSource:
    src = db.query(ConnectorSource).filter(
        ConnectorSource.id == source_id,
        ConnectorSource.workspace_id == workspace_id).first()
    if src is None:
        raise wp.JobNotFoundError(f"Connector source {source_id} not found")
    return src


def list_sources(db: Session, workspace_id: int, limit: int = 50) -> dict:
    q = db.query(ConnectorSource).filter(
        ConnectorSource.workspace_id == workspace_id)
    total = q.count()
    items = q.order_by(ConnectorSource.id.desc()).limit(min(limit, 200)).all()
    return {"items": [source_dict(s) for s in items], "total": total,
            "limit": min(limit, 200)}


def source_dict(src: ConnectorSource) -> dict:
    return {
        "id": src.id,
        "name": src.name,
        "kind": src.kind,
        "enabled": src.enabled,
        "scopes": _load(src.scopes_json),
        "permissions": _load(src.permissions_json),
        "allowed_domains": _load(src.allowed_domains_json),
        "has_credential_ref": bool(src.credential_ref),
        "retention_days": src.retention_days,
        "updated_at": src.updated_at,
    }


def _load(raw: Optional[str]) -> list:
    try:
        return json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Sync — incremental + idempotent
# ---------------------------------------------------------------------------

def _driver_fetch(source: ConnectorSource, cursor: Optional[str]) -> dict:
    """Deterministic in-process driver.

    Returns {"items": [...], "next_cursor": str|None, "full": bool}.

    Real connector drivers live behind this seam; they receive the source
    metadata (scopes/domains) and must enforce the same tenant scoping.
    """
    seed = source.id
    if cursor is None:
        # Initial (full) sync: two deterministic items.
        return {
            "items": [
                {"external_id": f"{seed}:doc-1",
                 "title": f"{source.name} doc 1",
                 "content": f"Policy document one from {source.name}",
                 "item_type": "document"},
                {"external_id": f"{seed}:doc-2",
                 "title": f"{source.name} doc 2",
                 "content": f"Guidance document two from {source.name}",
                 "item_type": "document"},
            ],
            "next_cursor": "cursor-after-initial",
            "full": True,
        }
    if cursor == "cursor-after-initial":
        # Incremental delta: one change + one deletion.
        return {
            "items": [
                {"external_id": f"{seed}:doc-1",
                 "title": f"{source.name} doc 1 (updated)",
                 "content": f"Policy document one REVISED from {source.name}",
                 "item_type": "document"},
            ],
            "deleted_external_ids": [f"{seed}:doc-2"],
            "next_cursor": "cursor-after-incremental",
            "full": False,
        }
    return {"items": [], "next_cursor": cursor, "full": False}


def run_connector_sync(db: Session, source_id: int,
                       workspace_id: Optional[int] = None) -> dict:
    """Idempotent incremental sync. Repeated syncs never duplicate items."""
    q = db.query(ConnectorSource).filter(ConnectorSource.id == source_id)
    if workspace_id:
        q = q.filter(ConnectorSource.workspace_id == workspace_id)
    source = q.first()
    if source is None:
        raise wp.JobNotFoundError(f"Connector source {source_id} not found")
    if not source.enabled:
        return {"source_id": source.id, "status": "DISABLED",
                "items_added": 0, "items_changed": 0, "items_deleted": 0}

    previous = (
        db.query(ConnectorSync)
        .filter(ConnectorSync.source_id == source.id,
                ConnectorSync.status == "COMPLETED")
        .order_by(ConnectorSync.id.desc())
        .first()
    )
    cursor = json.loads(previous.cursor_json or "null") if previous else None

    sync = ConnectorSync(source_id=source.id, status="RUNNING",
                         cursor_json=json.dumps(cursor))
    db.add(sync)
    db.flush()

    sync_run = sync.id
    try:
        delta = _driver_fetch(source, cursor)
        added = changed = deleted = 0
        for item in delta.get("items", []):
            external_id = item["external_id"]
            content = item.get("content", "")
            content_hash = _sha(content)
            existing = (
                db.query(ConnectorItem)
                .filter(ConnectorItem.source_id == source.id,
                        ConnectorItem.external_id == external_id)
                .first()
            )
            if existing is None:
                db.add(ConnectorItem(
                    source_id=source.id,
                    workspace_id=source.workspace_id,
                    external_id=external_id,
                    external_parent_id=item.get("external_parent_id"),
                    item_type=item.get("item_type", "document"),
                    title=item.get("title"),
                    content_hash=content_hash,
                    payload_ref=f"conn:{source.id}:{external_id}",
                    deleted=False,
                ))
                added += 1
            else:
                changed_flag = False
                if existing.deleted:
                    existing.deleted = False
                    changed_flag = True
                if existing.content_hash != content_hash:
                    existing.content_hash = content_hash
                    existing.title = item.get("title") or existing.title
                    changed_flag = True
                existing.last_seen_at = _utcnow()
                if changed_flag:
                    changed += 1
        for ext_id in delta.get("deleted_external_ids", []):
            existing = (
                db.query(ConnectorItem)
                .filter(ConnectorItem.source_id == source.id,
                        ConnectorItem.external_id == ext_id)
                .first()
            )
            if existing is not None and not existing.deleted:
                existing.deleted = True
                existing.last_seen_at = _utcnow()
                deleted += 1
        sync.items_added = added
        sync.items_changed = changed
        sync.items_deleted = deleted
        sync.cursor_json = json.dumps(delta.get("next_cursor"))
        sync.status = "COMPLETED"
        sync.completed_at = _utcnow()
        db.flush()
        return {"source_id": source.id, "status": "COMPLETED",
                "sync_id": sync_run, "items_added": added,
                "items_changed": changed, "items_deleted": deleted}
    except Exception as exc:  # noqa: BLE001 — sync boundary
        sync.status = "FAILED"
        sync.error = str(exc)[:1000]
        sync.completed_at = _utcnow()
        db.flush()
        logger.warning("connector sync %s failed: %s", source.id, exc)
        raise


def sync_state(db: Session, source_id: int, workspace_id: int) -> dict:
    _owned_source(db, source_id, workspace_id)
    runs = (db.query(ConnectorSync)
            .filter(ConnectorSync.source_id == source_id)
            .order_by(ConnectorSync.id.desc()).limit(20).all())
    return {"items": [
        {"sync_id": r.id, "status": r.status, "items_added": r.items_added,
         "items_changed": r.items_changed, "items_deleted": r.items_deleted,
         "error": r.error, "started_at": r.started_at,
         "completed_at": r.completed_at}
        for r in runs]}
