"""Deterministic AI suggestion engine.

Suggestions originate from observable events (document ready, conflicts,
deadlines, duplicates, health) and are always explainable. The engine never
generates dangerous autonomous actions — suggestions require human action.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.document_version import DocumentVersion
from ..models.knowledge import Deadline
from ..models.ai_action import AISuggestion
from ..services.ai_action_service import create_suggestion

SUGGESTION_SOURCES = ("document", "conflict", "deadline", "duplicate", "collection", "health", "metadata")


def suggest_for_document_ready(
    db: Session,
    document: Document,
    workspace_id: int,
    owner_id: int,
    organization_id: Optional[int] = None,
) -> list[AISuggestion]:
    """Generate suggestions when a document becomes READY."""
    suggestions = []
    from ..services.health_service import document_metadata, document_name
    meta = document_metadata(db, document)

    # Missing classification
    if not meta.get("classification"):
        suggestions.append(create_suggestion(
            db, workspace_id, owner_id,
            title=f"Classify '{document_name(db, document)}'",
            suggestion_type="classification",
            description="This document has no classification yet.",
            reason="Document metadata is missing a classification, which limits retrieval quality.",
            source_type="document", source_id=document.id,
            organization_id=organization_id,
            evidence={"document_id": document.id},
        ))

    # Missing summary
    if not meta.get("summary"):
        suggestions.append(create_suggestion(
            db, workspace_id, owner_id,
            title=f"Summarize '{document_name(db, document)}'",
            suggestion_type="summary",
            description="Generate a summary for this document.",
            reason="A summary improves search and collection health.",
            source_type="document", source_id=document.id,
            organization_id=organization_id,
            evidence={"document_id": document.id},
        ))

    # Recommend a collection when the document has none
    from ..models.collection import Collection, collection_documents
    assigned = (
        db.query(collection_documents)
        .filter(collection_documents.c.document_id == document.id)
        .count()
    )
    if assigned == 0:
        collections = db.query(Collection).filter(Collection.workspace_id == workspace_id).count()
        if collections > 0:
            suggestions.append(create_suggestion(
                db, workspace_id, owner_id,
                title=f"Add '{document_name(db, document)}' to a collection",
                suggestion_type="collection",
                description="This document is not in any collection.",
                reason="Documents outside collections are harder to retrieve contextually.",
                source_type="document", source_id=document.id,
                organization_id=organization_id,
                evidence={"document_id": document.id},
            ))

    return suggestions


def suggest_deadline_approaching(db: Session, workspace_id: int, owner_id: int, organization_id: Optional[int] = None) -> list[AISuggestion]:
    """Suggest deadlines approaching within 30 days."""
    from ..models.knowledge import Deadline
    from ..services.deadline_service import _as_utc
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=30)
    due = (
        db.query(Deadline)
        .filter(
            Deadline.workspace_id == workspace_id,
            Deadline.status == "UPCOMING",
        )
        .all()
    )
    suggestions = []
    for d in due:
        due_date = _as_utc(d.due_date)
        if due_date < now or due_date > horizon:
            continue
        days_left = (due_date - now).days
        existing = (
            db.query(AISuggestion)
            .filter(
                AISuggestion.workspace_id == workspace_id,
                AISuggestion.suggestion_type == "deadline",
                AISuggestion.source_id == d.id,
                AISuggestion.status == "OPEN",
            )
            .first()
        )
        if existing:
            continue
        suggestions.append(create_suggestion(
            db, workspace_id, owner_id,
            title=f"Deadline approaching: {d.title} ({days_left} days)",
            suggestion_type="deadline",
            description=d.description,
            reason="Deadline is within the 30-day horizon.",
            source_type="deadline", source_id=d.id,
            priority="HIGH" if days_left <= 7 else "NORMAL",
            organization_id=organization_id,
            evidence={"deadline_id": d.id, "due_date": d.due_date.isoformat(), "days_left": days_left},
        ))
    return suggestions


def suggest_duplicate_candidates(db: Session, workspace_id: int, owner_id: int, organization_id: Optional[int] = None) -> list[AISuggestion]:
    """Suggest near-duplicate document pairs based on normalized title similarity."""
    from ..services.health_service import document_name
    docs = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
        .limit(300)
        .all()
    )
    seen = set()
    suggestions = []
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            if _title_similarity(db, a, b) >= 0.9:
                pair = tuple(sorted((a.id, b.id)))
                if pair in seen:
                    continue
                seen.add(pair)
                existing = (
                    db.query(AISuggestion)
                    .filter(
                        AISuggestion.workspace_id == workspace_id,
                        AISuggestion.suggestion_type == "duplicate",
                        AISuggestion.source_type == "document",
                        AISuggestion.source_id == a.id,
                    )
                    .first()
                )
                if existing:
                    continue
                suggestions.append(create_suggestion(
                    db, workspace_id, owner_id,
                    title=f"'{document_name(db, a)}' may be a duplicate of '{document_name(db, b)}'",
                    suggestion_type="duplicate",
                    description="Two documents appear to describe the same content.",
                    reason="Normalized titles match closely; review before any action.",
                    source_type="document", source_id=a.id,
                    organization_id=organization_id,
                    evidence={"document_a": a.id, "document_b": b.id},
                ))
    return suggestions


def _title_similarity(db: Session, a: Document, b: Document) -> float:
    """Simple normalized token-overlap similarity (deterministic)."""
    from ..services.health_service import document_name
    import re
    ta = set(re.findall(r"[a-z0-9]+", document_name(db, a).lower()))
    tb = set(re.findall(r"[a-z0-9]+", document_name(db, b).lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)