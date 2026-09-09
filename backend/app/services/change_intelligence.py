"""Document change intelligence — version comparison, change classification,
and impact analysis.

Uses deterministic text/version comparison; the LLM is not required.
"""

import re
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document_version import DocumentVersion
from ..models.document_content import DocumentContent
from ..models.knowledge import KnowledgeInsight
from ..services.audit_service import log_audit_event

CHANGE_TYPES = (
    "CONTENT_CHANGE",
    "DATE_CHANGE",
    "POLICY_CHANGE",
    "NUMERIC_CHANGE",
    "ENTITY_CHANGE",
    "STRUCTURAL_CHANGE",
)

DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b")
NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
ENTITY_HINTS = ("company", "inc", "llc", "ltd", "corp", "partnership", "agreement between")


def _text_for_version(db: Session, document_id: int, version_number: Optional[int]) -> str:
    """Retrieve text for a document version (current content when no version)."""
    if version_number is not None:
        version = (
            db.query(DocumentVersion)
            .filter(
                DocumentVersion.document_id == document_id,
                DocumentVersion.version_number == version_number,
            )
            .first()
        )
        if version and getattr(version, "content_text", None):
            return version.content_text
        if version and getattr(version, "change_summary", None):
            return version.change_summary
    content = (
        db.query(DocumentContent)
        .filter(DocumentContent.document_id == document_id)
        .first()
    )
    return (content.extracted_text if content else None) or ""


def diff_versions(text_a: str, text_b: str) -> dict:
    """Deterministic line-level diff between two document texts.

    Returns added/removed/unchanged lines plus a change classification.
    """
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    set_a = set(lines_a)
    set_b = set(lines_b)

    added = [l for l in lines_b if l not in set_a and l.strip()]
    removed = [l for l in lines_a if l not in set_b and l.strip()]
    common = [l for l in lines_b if l in set_a and l.strip()]

    # Structural change: very different line counts
    structural = abs(len(lines_a) - len(lines_b)) > max(len(lines_a), len(lines_b)) * 0.5 if lines_a or lines_b else False

    classification = "CONTENT_CHANGE"
    if structural:
        classification = "STRUCTURAL_CHANGE"
    else:
        changed_text = " ".join(added + removed)
        if DATE_RE.search(changed_text):
            classification = "DATE_CHANGE"
        elif any(kw in changed_text.lower() for kw in ("policy", "procedur", "shall", "must", "require")):
            classification = "POLICY_CHANGE"
        elif any(kw in changed_text.lower() for kw in ENTITY_HINTS):
            classification = "ENTITY_CHANGE"
        elif NUMBER_RE.search(changed_text):
            classification = "NUMERIC_CHANGE"

    importance = "HIGH" if classification in ("POLICY_CHANGE", "DATE_CHANGE", "NUMERIC_CHANGE", "ENTITY_CHANGE") else "MEDIUM"

    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "unchanged_count": len(common),
        "added_preview": added[:5],
        "removed_preview": removed[:5],
        "classification": classification,
        "importance": importance,
    }


def analyze_change(
    db: Session,
    document_id: int,
    workspace_id: int,
    old_version: Optional[int],
    new_version: int,
    user_id: Optional[int] = None,
    organization_id: Optional[int] = None,
) -> dict:
    """Full change analysis: diff, classification, and impact summary."""
    text_a = _text_for_version(db, document_id, old_version)
    text_b = _text_for_version(db, document_id, new_version)
    diff = diff_versions(text_a, text_b)

    impact = {
        "related_documents": _related_documents(db, workspace_id, document_id, limit=5),
        "collections": _document_collections(db, document_id),
        "potential_conflicts": diff["classification"] in ("DATE_CHANGE", "NUMERIC_CHANGE", "POLICY_CHANGE"),
    }

    if diff["importance"] == "HIGH":
        _record_insight(
            db,
            workspace_id,
            organization_id,
            user_id,
            document_id,
            diff,
        )
    return {**diff, "impact": impact}


def _related_documents(db: Session, workspace_id: int, document_id: int, limit: int = 5) -> list[dict]:
    """Documents sharing tags/entities with the changed document (conservative)."""
    related = []
    try:
        from ..models.document_tag import document_tags
        from ..models.knowledge_graph import Entity
        tags = (
            db.query(document_tags.c.tag_id)
            .filter(document_tags.c.document_id == document_id)
            .all()
        )
        tag_ids = [r[0] for r in tags]
        if tag_ids:
            shared = (
                db.query(document_tags.c.document_id)
                .filter(document_tags.c.tag_id.in_(tag_ids), document_tags.c.document_id != document_id)
                .all()
            )
            for r in shared[:limit]:
                related.append({"document_id": r[0], "reason": "shared tag"})
    except Exception:
        pass
    return related


def _document_collections(db: Session, document_id: int) -> list[int]:
    from ..models.collection import collection_documents
    rows = (
        db.query(collection_documents.c.collection_id)
        .filter(collection_documents.c.document_id == document_id)
        .all()
    )
    return [r[0] for r in rows]


def _record_insight(db, workspace_id, organization_id, user_id, document_id, diff):
    title = f"High-importance change: {diff['classification']}"
    existing = (
        db.query(KnowledgeInsight)
        .filter(
            KnowledgeInsight.workspace_id == workspace_id,
            KnowledgeInsight.insight_type == "change",
            KnowledgeInsight.title == title,
        )
        .order_by(KnowledgeInsight.id.desc())
        .first()
    )
    if existing and diff.get("importance") == "HIGH":
        pass  # dedup: only record once per classification
    else:
        db.add(KnowledgeInsight(
            workspace_id=workspace_id,
            organization_id=organization_id,
            owner_id=user_id,
            insight_type="change",
            title=title,
            detail=f"Document {document_id}: {diff['added_count']} lines added, {diff['removed_count']} removed.",
            importance=diff["importance"],
        ))
        db.flush()