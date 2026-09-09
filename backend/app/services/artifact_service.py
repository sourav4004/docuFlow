"""Artifact platform 2.0 — versioned AI artifacts with source references.

Artifacts (research briefs, reports, analyses, comparisons, extractions,
timelines, risk reports, workflow plans) are versioned under a stable
logical id (``family_id`` = the original artifact id). Every version is a
separate row; versions can be diffed to show changed claims,
recommendations, evidence, and confidence.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.ai_execution import AIArtifact
from ..services.audit_service import log_audit_event

ARTIFACT_TYPES = (
    "research_brief", "report", "analysis", "comparison", "extraction",
    "timeline", "risk_report", "workflow_plan", "summary",
)


class ArtifactError(Exception):
    """Raised on invalid artifact operations."""


def _row_dict(artifact: AIArtifact) -> dict:
    return {
        "id": artifact.id,
        "family_id": artifact.family_id,
        "version": artifact.version,
        "name": artifact.name,
        "artifact_type": artifact.artifact_type,
        "created_at": artifact.created_at,
        "updated_at": artifact.updated_at,
    }


def create_artifact(
    db: Session,
    workspace_id: int,
    user_id: int,
    artifact_type: str,
    name: str,
    content: dict,
    source_document_ids: Optional[list[int]] = None,
    execution_id: Optional[str] = None,
    model_used: Optional[str] = None,
    provider_used: Optional[str] = None,
) -> AIArtifact:
    """Create a new versioned artifact (v1). The artifact id is the family id."""
    if artifact_type not in ARTIFACT_TYPES:
        raise ArtifactError(f"Unknown artifact type: {artifact_type}")
    artifact_id = str(uuid.uuid4())
    artifact = AIArtifact(
        id=artifact_id,
        family_id=artifact_id,
        workspace_id=workspace_id,
        execution_id=execution_id,
        user_id=user_id,
        artifact_type=artifact_type,
        name=name,
        content_json=content,
        version=1,
        source_document_ids_json=source_document_ids or [],
        model_used=model_used,
        provider_used=provider_used,
        status="ai_generated",
    )
    db.add(artifact)
    db.flush()
    return artifact


def new_version(
    db: Session,
    workspace_id: int,
    artifact_id: str,
    content: dict,
    user_id: int,
    source_document_ids: Optional[list[int]] = None,
    model_used: Optional[str] = None,
) -> AIArtifact:
    """Create the next version of an artifact family (existing versions untouched).

    ``artifact_id`` is the stable family id (the original artifact's id);
    legacy pre-Phase-15 rows without a family_id are matched by their own id.
    """
    latest = _latest_version(db, workspace_id, artifact_id)
    if not latest:
        raise ArtifactError("Artifact not found in this workspace")

    family_id = latest.family_id or latest.id
    artifact = AIArtifact(
        id=str(uuid.uuid4()),
        family_id=family_id,
        workspace_id=workspace_id,
        execution_id=latest.execution_id,
        user_id=user_id,
        artifact_type=latest.artifact_type,
        name=latest.name,
        content_json=content,
        version=latest.version + 1,
        source_document_ids_json=source_document_ids or latest.source_document_ids_json,
        model_used=model_used or latest.model_used,
        provider_used=latest.provider_used,
        status="ai_generated",
    )
    db.add(artifact)
    db.flush()
    return artifact


def _family_filter(artifact_id: str):
    """Match version rows by family_id, falling back to legacy row ids."""
    from sqlalchemy import or_
    return or_(AIArtifact.family_id == artifact_id, AIArtifact.id == artifact_id)


def _latest_version(db: Session, workspace_id: int, artifact_id: str) -> Optional[AIArtifact]:
    return (
        db.query(AIArtifact)
        .filter(AIArtifact.workspace_id == workspace_id, _family_filter(artifact_id))
        .order_by(AIArtifact.version.desc())
        .first()
    )


def list_versions(db: Session, workspace_id: int, artifact_id: str) -> list[AIArtifact]:
    return (
        db.query(AIArtifact)
        .filter(AIArtifact.workspace_id == workspace_id, _family_filter(artifact_id))
        .order_by(AIArtifact.version.asc())
        .all()
    )


def list_artifacts(
    db: Session,
    workspace_id: int,
    artifact_type: Optional[str] = None,
    limit: int = 100,
) -> list[AIArtifact]:
    """List the latest version of each artifact family (bounded)."""
    versions = (
        db.query(AIArtifact)
        .filter(AIArtifact.workspace_id == workspace_id)
        .order_by(AIArtifact.created_at.desc())
        .limit(2000)
        .all()
    )
    latest: dict[str, AIArtifact] = {}
    for v in versions:
        key = v.family_id or v.id
        current = latest.get(key)
        if current is None or v.version > current.version:
            latest[key] = v
    result = list(latest.values())
    if artifact_type:
        result = [v for v in result if v.artifact_type == artifact_type]
    return result[:limit]


def version_diff(v_old: AIArtifact, v_new: AIArtifact) -> dict:
    """Diff two artifact versions: claims, recommendations, evidence, confidence."""
    old = v_old.content_json or {}
    new = v_new.content_json or {}

    changed_sections = []
    for key in ("claims", "recommendations", "evidence", "confidence", "summary", "findings", "uncertainties"):
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val != new_val:
            changed_sections.append({
                "section": key,
                "changed": True,
                "from": old_val,
                "to": new_val,
            })

    old_confidence = old.get("confidence")
    new_confidence = new.get("confidence")
    return {
        "artifact_id": v_old.family_id or v_old.id,
        "from_version": v_old.version,
        "to_version": v_new.version,
        "changed_sections": changed_sections,
        "confidence_changed": old_confidence != new_confidence,
        "confidence_from": old_confidence,
        "confidence_to": new_confidence,
        "evidence_changed": old.get("evidence") != new.get("evidence"),
    }


def delete_artifact(db: Session, workspace_id: int, artifact_id: str, actor_id: int) -> bool:
    """Delete ALL versions of an artifact family (audited)."""
    versions = (
        db.query(AIArtifact)
        .filter(AIArtifact.workspace_id == workspace_id, _family_filter(artifact_id))
        .all()
    )
    if not versions:
        return False
    for v in versions:
        db.delete(v)
    db.flush()
    log_audit_event(
        db, event_type="artifact", event_action="delete",
        user_id=actor_id, resource_type="artifact", resource_id=artifact_id,
        details=f"Artifact deleted ({len(versions)} version(s))",
    )
    return True