"""Knowledge gap detection.

Identifies POSSIBLE gaps (never confirmed facts): missing metadata,
missing effective dates, unreferenced entities, missing owners, etc.
"""

import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.knowledge_graph import Entity
from ..models.knowledge import KnowledgeInsight

# A gap is always labeled POSSIBLE_GAP — never presented as confirmed fact
GAP_KIND = "POSSIBLE_GAP"


def detect_document_gaps(db: Session, document: Document) -> list[dict]:
    """Detect metadata/knowledge gaps for a single document."""
    from ..services.health_service import document_metadata
    gaps = []
    meta = document_metadata(db, document)

    if not document.original_filename:
        gaps.append({"kind": GAP_KIND, "type": "missing_filename", "detail": "Document has no filename"})

    # Metadata completeness
    for field in ("author", "created", "classification", "summary"):
        if not meta.get(field):
            gaps.append({
                "kind": GAP_KIND,
                "type": "missing_metadata",
                "field": field,
                "detail": f"No {field} metadata available",
            })

    return gaps


def detect_entity_gaps(db: Session, workspace_id: int) -> list[dict]:
    """Detect entities without any supporting document relationship.

    An entity is considered supported when it appears in an
    EntityRelationship that carries a source_document_id. Entities with no
    such link are flagged as POSSIBLE_GAP — never as confirmed facts.
    """
    from ..models.knowledge_graph import EntityRelationship
    gaps = []
    entities = db.query(Entity).filter(Entity.workspace_id == workspace_id).all()
    if not entities:
        return gaps
    linked_ids = set()
    rows = (
        db.query(EntityRelationship.source_id, EntityRelationship.target_id)
        .filter(
            EntityRelationship.workspace_id == workspace_id,
            EntityRelationship.source_document_id.isnot(None),
        )
        .all()
    )
    for source_id, target_id in rows:
        if source_id:
            linked_ids.add(source_id)
        if target_id:
            linked_ids.add(target_id)
    for entity in entities:
        if entity.id in linked_ids:
            continue
        gaps.append({
            "kind": GAP_KIND,
            "type": "unsupported_entity",
            "entity_id": entity.id,
            "name": getattr(entity, "name", None),
            "detail": "Entity has no supporting document source",
        })
    return gaps


def detect_policy_gaps(db: Session, workspace_id: int) -> list[dict]:
    """Detect policy-like documents missing an effective date (when classified)."""
    gaps = []
    docs = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
        .limit(500)
        .all()
    )
    from ..services.health_service import document_metadata
    for doc in docs:
        meta = document_metadata(db, doc)
        classification = meta.get("classification") or getattr(doc, "classification", None)
        if classification and "polic" in str(classification).lower() and not meta.get("effective_date"):
            gaps.append({
                "kind": GAP_KIND,
                "type": "missing_effective_date",
                "document_id": doc.id,
                "title": getattr(doc, "title", None) or getattr(doc, "original_filename", None),
                "detail": "Policy document has no effective date",
            })
    return gaps


def record_gap_insights(db: Session, workspace_id: int, organization_id: Optional[int], user_id: Optional[int]) -> int:
    """Record detected gaps as knowledge insights (deduplicated by title)."""
    gaps = (
        detect_entity_gaps(db, workspace_id)
        + detect_policy_gaps(db, workspace_id)
    )
    created = 0
    for gap in gaps:
        title = f"{gap.get('type', 'gap')} — {gap.get('title') or gap.get('name') or gap.get('detail', '')[:60]}"
        existing = (
            db.query(KnowledgeInsight)
            .filter(
                KnowledgeInsight.workspace_id == workspace_id,
                KnowledgeInsight.insight_type == "knowledge_gap",
                KnowledgeInsight.title == title,
            )
            .first()
        )
        if existing:
            continue
        db.add(KnowledgeInsight(
            workspace_id=workspace_id,
            organization_id=organization_id,
            owner_id=user_id,
            insight_type="knowledge_gap",
            title=title,
            detail=gap.get("detail"),
            importance="MEDIUM",
            evidence_json=json.dumps(gap),
        ))
        created += 1
    if created:
        db.flush()
    return created