"""Knowledge OS API — change detection, impact graph, policy intelligence,
temporal knowledge, snapshots, and the unified timeline."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..models.document import Document
from ..services.permission_service import require_permission, require_workspace_membership
from ..services.knowledge_engine import (
    detect_and_record_change,
    list_changes,
    build_impact_links,
    list_impact_links,
    extract_policy_statements,
    detect_policy_conflicts,
    record_temporal_fact,
    facts_as_of,
    current_facts,
    create_snapshot,
    list_snapshots,
)
from ..services.timeline_service import unified_timeline
from ..services.audit_service import log_audit_event

router = APIRouter(prefix="/knowledge-os", tags=["knowledge-os"])


class ChangeDetect(BaseModel):
    document_id: int
    version_from: Optional[int] = None
    version_to: int


class TemporalFactCreate(BaseModel):
    fact_type: str
    fact_value: str
    valid_from: datetime
    valid_until: Optional[datetime] = None
    document_id: Optional[int] = None
    entity_id: Optional[int] = None
    source: Optional[str] = None


def _require_workspace(db, workspace_id, principal) -> Workspace:
    workspace = require_workspace_membership(db, workspace_id, principal.user.id)
    return workspace


@router.post("/changes/detect", status_code=201)
def detect_change(
    data: ChangeDetect,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Detect and record a document change between versions."""
    doc = db.query(Document).filter(Document.id == data.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    workspace_id = doc.workspace_id
    if workspace_id is None:
        from ..services.workspace_service import resolve_document_workspace
        workspace_id = resolve_document_workspace(db, doc, principal.user.id)
    workspace = _require_workspace(db, workspace_id, principal)
    change = detect_and_record_change(
        db, workspace_id, data.document_id,
        data.version_from, data.version_to, workspace.organization_id,
    )
    db.commit()
    if change is None:
        return {"detected": False, "message": "No difference between versions"}
    return {
        "detected": True,
        "id": change.id,
        "change_type": change.change_type,
        "severity": change.severity,
        "confidence": change.confidence,
        "summary": change.summary,
        "old_evidence": change.old_evidence_json,
        "new_evidence": change.new_evidence_json,
        "affected_entities": change.affected_entities_json,
    }


@router.get("/changes")
def changes_list(
    workspace_id: int,
    change_type: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    items = list_changes(db, workspace_id, limit=min(limit, 200), change_type=change_type)
    return {
        "items": [
            {
                "id": c.id,
                "document_id": c.document_id,
                "version_from": c.version_from,
                "version_to": c.version_to,
                "change_type": c.change_type,
                "severity": c.severity,
                "confidence": c.confidence,
                "summary": c.summary,
                "created_at": c.created_at,
            }
            for c in items
        ]
    }


@router.post("/impact/{document_id}", status_code=201)
def impact_analysis(
    document_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Build the change impact graph for a document (explicit vs inferred)."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    workspace_id = doc.workspace_id
    if workspace_id is None:
        from ..services.workspace_service import resolve_document_workspace
        workspace_id = resolve_document_workspace(db, doc, principal.user.id)
    _require_workspace(db, workspace_id, principal)
    result = build_impact_links(db, workspace_id, document_id)
    db.commit()
    return result


@router.get("/impact")
def impact_links(
    workspace_id: int,
    source_type: Optional[str] = None,
    source_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    items = list_impact_links(db, workspace_id, source_type=source_type, source_id=source_id)
    return {
        "items": [
            {
                "id": l.id,
                "source_type": l.source_type,
                "source_id": l.source_id,
                "target_type": l.target_type,
                "target_id": l.target_id,
                "relation": l.relation,
                "confidence": l.confidence,
            }
            for l in items
        ]
    }


@router.post("/policies/extract/{document_id}", status_code=201)
def extract_policies(
    document_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    workspace_id = doc.workspace_id
    if workspace_id is None:
        from ..services.workspace_service import resolve_document_workspace
        workspace_id = resolve_document_workspace(db, doc, principal.user.id)
    workspace = _require_workspace(db, workspace_id, principal)
    statements = extract_policy_statements(
        db, workspace_id, document_id, workspace.organization_id
    )
    db.commit()
    return {
        "extracted": len(statements),
        "items": [
            {
                "id": s.id,
                "statement": s.statement,
                "requirement_type": s.requirement_type,
                "applicability": s.applicability,
                "effective_date": s.effective_date,
                "evidence_reference": s.evidence_reference,
            }
            for s in statements
        ],
    }


@router.get("/policies")
def list_policies(
    workspace_id: int,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    from ..models.phase15 import PolicyStatement
    items = (
        db.query(PolicyStatement)
        .filter(PolicyStatement.workspace_id == workspace_id)
        .order_by(PolicyStatement.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return {
        "items": [
            {
                "id": s.id,
                "document_id": s.document_id,
                "statement": s.statement,
                "requirement_type": s.requirement_type,
                "applicability": s.applicability,
                "effective_date": s.effective_date,
                "expiration_date": s.expiration_date,
                "responsible_party": s.responsible_party,
            }
            for s in items
        ]
    }


@router.post("/conflicts/detect", status_code=201)
def detect_conflicts(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Detect real policy conflicts (threshold/date rules, not text diffs)."""
    workspace = _require_workspace(db, workspace_id, principal)
    conflicts = detect_policy_conflicts(db, workspace_id, workspace.organization_id)
    db.commit()
    return {
        "detected": len(conflicts),
        "items": [
            {
                "id": c.id,
                "conflict_type": c.conflict_type,
                "severity": c.severity,
                "description": c.description,
                "policy_a_id": c.policy_a_id,
                "policy_b_id": c.policy_b_id,
                "status": c.status,
            }
            for c in conflicts
        ],
    }


@router.get("/conflicts")
def list_conflicts(
    workspace_id: int,
    status: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    from ..models.phase15 import PolicyConflict
    query = db.query(PolicyConflict).filter(PolicyConflict.workspace_id == workspace_id)
    if status:
        query = query.filter(PolicyConflict.status == status)
    items = query.order_by(PolicyConflict.created_at.desc()).limit(min(limit, 200)).all()
    return {
        "items": [
            {
                "id": c.id,
                "conflict_type": c.conflict_type,
                "severity": c.severity,
                "description": c.description,
                "conditions": c.conditions_json,
                "status": c.status,
                "policy_a_id": c.policy_a_id,
                "policy_b_id": c.policy_b_id,
            }
            for c in items
        ]
    }


@router.post("/conflicts/{conflict_id}/resolve")
def resolve_conflict(
    conflict_id: int,
    note: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..models.phase15 import PolicyConflict
    conflict = db.query(PolicyConflict).filter(PolicyConflict.id == conflict_id).first()
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")
    require_permission(db, principal.user.id, "document:write", workspace_id=conflict.workspace_id)
    conflict.status = "RESOLVED"
    conflict.resolution_note = note
    db.commit()
    log_audit_event(
        db, event_type="policy_conflict", event_action="resolve",
        user_id=principal.user.id, resource_type="policy_conflict", resource_id=conflict.id,
        details="Policy conflict resolved",
    )
    return {"message": "Conflict resolved", "id": conflict.id}


@router.post("/temporal-facts", status_code=201)
def create_temporal_fact(
    data: TemporalFactCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Record a temporal fact; supersedes current facts of the same type."""
    doc = None
    if data.document_id:
        doc = db.query(Document).filter(Document.id == data.document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
    workspace_id = doc.workspace_id if doc else None
    if workspace_id is None:
        raise HTTPException(status_code=400, detail="A workspace-scoped document is required")
    workspace = _require_workspace(db, workspace_id, principal)
    fact = record_temporal_fact(
        db, workspace_id, data.fact_type, data.fact_value, data.valid_from,
        valid_until=data.valid_until, document_id=data.document_id,
        entity_id=data.entity_id, source=data.source,
        organization_id=workspace.organization_id,
    )
    db.commit()
    return {
        "id": fact.id,
        "fact_type": fact.fact_type,
        "fact_value": fact.fact_value,
        "valid_from": fact.valid_from,
        "valid_until": fact.valid_until,
    }


@router.get("/temporal-facts/current")
def temporal_current(
    workspace_id: int,
    fact_type: Optional[str] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """What is currently true (stale facts are never 'current')."""
    _require_workspace(db, workspace_id, principal)
    facts = current_facts(db, workspace_id, fact_type)
    return {"items": [_fact_dict(f) for f in facts]}


@router.get("/temporal-facts/as-of")
def temporal_as_of(
    workspace_id: int,
    at: datetime,
    fact_type: Optional[str] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """What was true at a point in time."""
    _require_workspace(db, workspace_id, principal)
    facts = facts_as_of(db, workspace_id, at, fact_type)
    return {"items": [_fact_dict(f) for f in facts], "as_of": at}


def _fact_dict(f) -> dict:
    return {
        "id": f.id,
        "fact_type": f.fact_type,
        "fact_value": f.fact_value,
        "valid_from": f.valid_from,
        "valid_until": f.valid_until,
        "document_id": f.document_id,
        "entity_id": f.entity_id,
        "source": f.source,
    }


@router.post("/snapshots", status_code=201)
def create_knowledge_snapshot(
    workspace_id: int,
    name: str = "snapshot",
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    workspace = _require_workspace(db, workspace_id, principal)
    snapshot = create_snapshot(
        db, workspace_id, principal.user.id, workspace.organization_id, name=name
    )
    db.commit()
    return {"id": snapshot.id, "name": snapshot.name, "created_at": snapshot.created_at}


@router.get("/snapshots")
def list_snapshots_endpoint(
    workspace_id: int,
    limit: int = 50,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    items = list_snapshots(db, workspace_id, limit=limit)
    return {
        "items": [
            {
                "id": s.id,
                "name": s.name,
                "snapshot_type": s.snapshot_type,
                "created_by": s.created_by,
                "created_at": s.created_at,
            }
            for s in items
        ]
    }


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(
    workspace_id: int,
    snapshot_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    from ..models.phase15 import KnowledgeSnapshot
    snapshot = db.query(KnowledgeSnapshot).filter(
        KnowledgeSnapshot.id == snapshot_id,
        KnowledgeSnapshot.workspace_id == workspace_id,
    ).first()
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {
        "id": snapshot.id,
        "name": snapshot.name,
        "snapshot_type": snapshot.snapshot_type,
        "created_at": snapshot.created_at,
        "data": snapshot.data_json,
    }


@router.get("/timeline")
def timeline(
    workspace_id: int,
    event_type: Optional[str] = None,
    document_id: Optional[int] = None,
    entity_id: Optional[int] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Unified knowledge timeline across documents/versions/policies/entities/
    deadlines/AI actions/workflow events/approvals."""
    _require_workspace(db, workspace_id, principal)
    try:
        items = unified_timeline(
            db, workspace_id,
            event_type=event_type, document_id=document_id, entity_id=entity_id,
            severity=severity, limit=min(limit, 300),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"items": items, "count": len(items)}