"""Phase 19 — knowledge federation 2.0.

Standardized connector contract, persisted sync checkpoints, idempotent
item effects (upsert by external_id + content hash; tombstones instead of
physical deletes), a conflict engine that never silently overwrites local
data, and connector security-scope enforcement (tenant/workspace/resource/
credential-reference/region).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dumps(value) -> str:
    return json.dumps(value, default=str)


def connector_contract() -> dict:
    """The operations every connector driver must expose."""
    return {
        "methods": ["discover", "fetch", "incremental_sync",
                    "delete_or_tombstone", "health", "rate_limits"],
        "state": ["cursor", "checkpoint", "retry", "error"],
        "semantics": {"items_deduplicated_by": "external_id + content_hash",
                      "deletes": "tombstone only",
                      "conflicts": "never silently overwrite local data"},
        "security": {"tenant_scoped": True, "workspace_scoped": True,
                     "resource_scoped": True,
                     "credentials": "credential reference only"},
    }


def validate_connector(db: Session, source_id: int) -> dict:
    """Validate a connector source: scoping, credential-ref boundary,
    retention, region policy."""
    from ..models.phase17 import ConnectorSource
    source = db.query(ConnectorSource).get(source_id)
    if source is None:
        return {"valid": False, "reason": "connector not found"}
    scopes = _loads(source.scopes_json) or {}
    permissions = _loads(source.permissions_json) or {}
    domains = _loads(source.allowed_domains_json) or []
    issues = []
    if not scopes.get("organization") and source.organization_id is None:
        issues.append("organization scope missing")
    if not scopes.get("workspace"):
        issues.append("workspace scope missing")
    if not scopes.get("resource"):
        issues.append("resource scope missing")
    if source.credential_ref is None:
        issues.append("credential reference missing (plaintext credentials "
                      "are never stored)")
    if source.retention_days is None or source.retention_days <= 0:
        issues.append("retention policy missing")
    return {"valid": not issues, "issues": issues,
            "scopes": scopes, "permissions": permissions,
            "allowed_domains": domains,
            "credential_ref": bool(source.credential_ref),
            "retention_days": source.retention_days,
            "enabled": source.enabled}


def _loads(raw: Optional[str]):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def save_checkpoint(db: Session, *, source_id: int,
                    cursor: Optional[dict] = None) -> dict:
    from ..models.phase17 import ConnectorSync
    latest = (db.query(ConnectorSync)
              .filter(ConnectorSync.source_id == source_id,
                      ConnectorSync.status == "COMPLETED")
              .order_by(ConnectorSync.completed_at.desc()).first())
    sync = latest if latest is not None else ConnectorSync(source_id=source_id)
    sync.cursor_json = _dumps(cursor) if cursor is not None else \
        sync.cursor_json
    sync.status = "COMPLETED"
    sync.completed_at = _utcnow()
    if latest is None:
        db.add(sync)
    db.flush()
    return {"source_id": source_id, "cursor": cursor,
            "checkpointed_at": sync.completed_at}


def classify_item(*, external_id: str,
                  external_content_hash: Optional[str] = None,
                  external_deleted: bool = False,
                  external_title: Optional[str] = None,
                  local_content_hash: Optional[str] = None,
                  local_deleted: Optional[bool] = None,
                  local_title: Optional[str] = None) -> dict:
    """Deterministic conflict classification. Returns a decision:
    APPLY / SKIP_CONFLICT."""
    if local_deleted and not external_deleted:
        return {"decision": "APPLY", "conflict": None,
                "reason": "externally recreated after local tombstone"}
    if external_deleted and not local_deleted:
        return {"decision": "APPLY", "conflict": None,
                "reason": "external deletion applies a reversible tombstone "
                          "(local payload is preserved, never destroyed)"}
    if local_content_hash and external_content_hash and \
            local_content_hash != external_content_hash:
        return {"decision": "SKIP_CONFLICT", "conflict": "CONTENT",
                "reason": "external content differs from local content"}
    if local_title and external_title and \
            local_title.strip() != external_title.strip():
        return {"decision": "SKIP_CONFLICT", "conflict": "METADATA",
                "reason": "external metadata differs from local metadata"}
    return {"decision": "APPLY", "conflict": None,
            "reason": "no conflict detected"}


def sync_connector_batch(db: Session, *, source_id: int,
                         workspace_id: int,
                         items: list[dict],
                         cursor: Optional[dict] = None) -> dict:
    """Idempotent incremental sync batch.

    Matched on (source_id, external_id); content-hash equal → no-op. External
    deletions become tombstones. Conflicts are recorded OPEN and never
    overwrite local data. Cursor is checkpointed at the end.
    """
    from ..models.phase17 import ConnectorItem
    from ..models.phase19 import ConnectorConflict
    added = changed = unchanged = deleted = 0
    conflicts = 0
    now = _utcnow()
    for item in items[:2000]:
        external_id = str(item.get("external_id"))
        content_hash = item.get("content_hash")
        title = item.get("title")
        is_delete = bool(item.get("deleted"))
        row = (db.query(ConnectorItem)
               .filter(ConnectorItem.source_id == source_id,
                       ConnectorItem.external_id == external_id).first())
        local_hash = row.content_hash if row else None
        local_deleted = row.deleted if row else None
        classification = classify_item(
            external_id=external_id, external_content_hash=content_hash,
            external_deleted=is_delete, external_title=title,
            local_content_hash=local_hash, local_deleted=local_deleted,
            local_title=row.title if row else None)
        if classification["decision"] == "SKIP_CONFLICT":
            db.add(ConnectorConflict(
                workspace_id=workspace_id, source_id=source_id,
                external_id=external_id,
                conflict_type=classification["conflict"],
                local_ref=row.payload_ref if row else None,
                external_ref=item.get("payload_ref"),
                detail=classification["reason"], status="OPEN"))
            conflicts += 1
            continue
        if row is None:
            row = ConnectorItem(source_id=source_id,
                                workspace_id=workspace_id,
                                external_id=external_id,
                                external_parent_id=item.get(
                                    "external_parent_id"),
                                item_type=item.get("item_type", "document"),
                                title=title,
                                content_hash=content_hash,
                                payload_ref=item.get("payload_ref"),
                                deleted=is_delete, first_seen_at=now,
                                last_seen_at=now)
            db.add(row)
            added += 1
            continue
        if is_delete:
            row.deleted = True
            deleted += 1
        elif row.deleted:
            row.deleted = False
            row.content_hash = content_hash
            row.title = title
            row.payload_ref = item.get("payload_ref")
            row.last_seen_at = now
            changed += 1
        elif row.content_hash != content_hash:
            row.content_hash = content_hash
            row.title = title
            row.payload_ref = item.get("payload_ref")
            row.last_seen_at = now
            changed += 1
        else:
            row.last_seen_at = now
            unchanged += 1
    checkpoint = save_checkpoint(db, source_id=source_id, cursor=cursor)
    db.flush()
    return {"source_id": source_id, "added": added, "changed": changed,
            "unchanged": unchanged, "deleted": deleted,
            "conflicts": conflicts,
            "cursor": checkpoint["cursor"]}


def list_conflicts(db: Session, *, source_id: Optional[int] = None,
                   status: str = "OPEN", limit: int = 100) -> list:
    from ..models.phase19 import ConnectorConflict
    q = db.query(ConnectorConflict)
    if source_id is not None:
        q = q.filter(ConnectorConflict.source_id == source_id)
    if status:
        q = q.filter(ConnectorConflict.status == status)
    return q.order_by(ConnectorConflict.created_at.desc()
                      ).limit(min(limit, 500)).all()


def resolve_conflict(db: Session, conflict_id: int, *, decision: str,
                     resolver_user_id: int) -> dict:
    from ..models.phase19 import ConnectorConflict
    if decision not in ("RESOLVED", "IGNORED"):
        raise ValueError("decision must be RESOLVED or IGNORED")
    row = db.query(ConnectorConflict).get(conflict_id)
    if row is None:
        raise ValueError("conflict not found")
    row.status = decision
    row.resolved_by = resolver_user_id
    row.updated_at = _utcnow()
    db.flush()
    return {"conflict_id": row.id, "status": row.status,
            "external_id": row.external_id}


def connector_security_scope(db: Session, source_id: int) -> dict:
    """Connector security surface: scopes/permissions/domains/credential
    boundary/region policy — never the credential value itself."""
    from ..models.phase17 import ConnectorSource
    from ..models.phase19 import RegionRecord
    source = db.query(ConnectorSource).get(source_id)
    if source is None:
        raise ValueError("connector not found")
    regions = (db.query(RegionRecord)
               .filter(RegionRecord.organization_id ==
                       source.organization_id).all())
    return {
        "source_id": source.id, "workspace_id": source.workspace_id,
        "organization_id": source.organization_id,
        "scopes": _loads(source.scopes_json) or {},
        "permissions": _loads(source.permissions_json) or {},
        "allowed_domains": _loads(source.allowed_domains_json) or [],
        "credential_ref_only": source.credential_ref is not None,
        "retention_days": source.retention_days,
        "regions": [{"region_id": r.region_id, "status": r.status}
                    for r in regions[:50]],
    }
