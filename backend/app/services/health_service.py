"""Knowledge health service — document, collection, and workspace health.

Health is a heuristic (0–100) with human-readable reasons, never presented
as mathematically precise.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.document_content import DocumentContent
from ..models.collection import Collection, collection_documents
from ..models.knowledge import DocumentHealth, KnowledgeInsight
from ..models.ai_execution import AIExecution


def _safe_meta(doc: Document) -> dict:
    try:
        raw = getattr(doc, "metadata_json", None)
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (ValueError, TypeError):
        return {}


def _as_utc(value) -> datetime:
    """Normalize naive datetimes (e.g. SQLite storage) to aware UTC."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def document_metadata(db: Session, document: Document) -> dict:
    """Build the effective metadata dict for a document from the real schema.

    Reads classification, summary, and version-level metadata (title, author,
    created, effective date). Returns {} when nothing is present.
    """
    meta = {}

    from ..models.document_tag import DocumentClassification, DocumentSummary
    classification = (
        db.query(DocumentClassification)
        .filter(DocumentClassification.document_id == document.id)
        .order_by(DocumentClassification.id.desc())
        .first()
    )
    if classification:
        meta["classification"] = classification.document_type

    summary = (
        db.query(DocumentSummary)
        .filter(DocumentSummary.document_id == document.id)
        .order_by(DocumentSummary.id.desc())
        .first()
    )
    if summary and summary.content:
        meta["summary"] = summary.content

    from ..models.document_version import DocumentVersion
    version = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.document_id == document.id)
        .order_by(DocumentVersion.version_number.desc())
        .first()
    )
    if version and version.metadata_json:
        try:
            version_meta = json.loads(version.metadata_json) if isinstance(version.metadata_json, str) else version.metadata_json
            if isinstance(version_meta, dict):
                for key, value in version_meta.items():
                    meta.setdefault(key, value)
        except (ValueError, TypeError):
            pass

    return meta


def compute_document_health(db: Session, document: Document) -> DocumentHealth:
    """Compute (or refresh) the health score for one document.

    Factors (each 0–100):
      - metadata completeness
      - extraction completeness (text available)
      - processing state (READY is best)
      - classification/summary presence
      - version freshness (a READY version exists)
    """
    factors = {}
    reasons = []

    # Processing state (weight 30)
    status_map = {
        "READY": 100,
        "UPLOADED": 40,
        "PROCESSING": 30,
        "QUEUED": 20,
        "FAILED": 0,
        "ARCHIVED": 20,
        "DELETED": 0,
    }
    proc = status_map.get(getattr(document, "status", ""), 20)
    factors["processing_state"] = proc
    if proc == 100:
        reasons.append("Document is processed and ready")
    elif proc == 0:
        reasons.append("Document failed or was deleted")

    # Metadata completeness (weight 25)
    meta = document_metadata(db, document)
    expected = ("classification", "summary", "author", "created")
    present = sum(1 for f in expected if meta.get(f))
    metadata_score = round((present / len(expected)) * 100)
    factors["metadata_completeness"] = metadata_score
    if present < len(expected):
        missing = [f for f in expected if not meta.get(f)]
        reasons.append(f"Missing metadata: {', '.join(missing)}")

    # Extraction completeness (weight 25)
    content = (
        db.query(DocumentContent)
        .filter(DocumentContent.document_id == document.id)
        .first()
    )
    text = (content.extracted_text if content else None) or ""
    extraction = 100 if len(text) > 50 else (50 if len(text) > 0 else 0)
    factors["extraction_completeness"] = extraction
    if extraction < 100:
        reasons.append("Extracted text is missing or very short")

    # Version presence (weight 20)
    from ..models.document_version import DocumentVersion
    version_count = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.document_id == document.id)
        .count()
    )
    version_score = 100 if version_count >= 1 else 40
    factors["version_presence"] = version_score
    if version_count == 0:
        reasons.append("No version history recorded")

    score = round(
        proc * 0.30 + metadata_score * 0.25 + extraction * 0.25 + version_score * 0.20
    )

    # If too many inputs are uncertain, avoid false precision
    if proc == 0 or (extraction == 0 and not reasons):
        score = min(score, 50)

    health = (
        db.query(DocumentHealth)
        .filter(DocumentHealth.document_id == document.id)
        .first()
    )
    if health is None:
        # Legacy user-scoped documents may have NULL workspace_id; resolve the
        # owner's current workspace so the health row stays tenant-scoped.
        workspace_id = document.workspace_id
        if workspace_id is None:
            from ..services.workspace_service import get_current_workspace, WorkspaceResolutionError
            try:
                workspace_id = get_current_workspace(db, document.user_id).id
            except WorkspaceResolutionError:
                workspace_id = None
        health = DocumentHealth(
            workspace_id=workspace_id,
            document_id=document.id,
        )
        db.add(health)
    health.score = score
    health.factors_json = json.dumps(factors)
    health.reasons_json = json.dumps(reasons)
    health.computed_at = datetime.now(timezone.utc)
    db.flush()
    return health


def get_document_health(db: Session, document_id: int) -> Optional[DocumentHealth]:
    return db.query(DocumentHealth).filter(DocumentHealth.document_id == document_id).first()


def workspace_health(db: Session, workspace_id: int) -> dict:
    """Aggregate workspace-level knowledge health metrics (tenant scoped)."""
    docs = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
        .all()
    )
    total = len(docs)
    if total == 0:
        return {
            "total_documents": 0,
            "active_knowledge": 0,
            "stale_knowledge": 0,
            "processing_failures": 0,
            "metadata_completeness": 0.0,
            "duplicate_rate": 0.0,
            "conflict_rate": 0.0,
            "avg_health": None,
        }

    ready = [d for d in docs if d.status == "READY"]
    failed = [d for d in docs if d.status == "FAILED"]
    stale_cutoff = datetime.now(timezone.utc) - timedelta(days=180)
    stale = [
        d for d in docs
        if _as_utc(d.updated_at) < stale_cutoff
    ]

    # Metadata completeness average
    meta_scores = [document_metadata(db, d) for d in docs]
    avg_meta = round(sum(
        sum(1 for f in ("classification", "summary", "author", "created") if m.get(f)) / 4
        for m in meta_scores
    ) / total * 100, 1)

    # Duplicate candidates via title similarity
    duplicate_pairs = 0
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            if _title_sim(db, a, b) >= 0.9:
                duplicate_pairs += 1
    duplicate_rate = round(duplicate_pairs / max(total, 1) * 100, 1)

    # Conflict signals: count deadline conflicts as a proxy (from insights)
    conflict_insights = (
        db.query(KnowledgeInsight)
        .filter(
            KnowledgeInsight.workspace_id == workspace_id,
            KnowledgeInsight.insight_type == "conflict",
        )
        .count()
    )
    conflict_rate = round(conflict_insights / max(total, 1) * 100, 1)

    # Average health score
    healths = db.query(DocumentHealth).filter(DocumentHealth.workspace_id == workspace_id).all()
    avg_health = (
        round(sum(h.score for h in healths) / len(healths)) if healths else None
    )

    return {
        "total_documents": total,
        "active_knowledge": len(ready),
        "stale_knowledge": len(stale),
        "processing_failures": len(failed),
        "metadata_completeness": avg_meta,
        "duplicate_rate": duplicate_rate,
        "conflict_rate": conflict_rate,
        "avg_health": avg_health,
    }


def collection_health(db: Session, collection: Collection) -> dict:
    """Compute health for a collection."""
    rows = (
        db.query(collection_documents)
        .filter(collection_documents.c.collection_id == collection.id)
        .all()
    )
    doc_ids = [r.document_id for r in rows]
    count = len(doc_ids)
    if count == 0:
        return {"collection_id": collection.id, "document_count": 0, "health": None, "recommendations": ["No documents in this collection"]}

    docs = (
        db.query(Document)
        .filter(Document.id.in_(doc_ids))
        .all()
    )
    ready = sum(1 for d in docs if d.status == "READY")
    failed = sum(1 for d in docs if d.status == "FAILED")

    healths = db.query(DocumentHealth).filter(DocumentHealth.document_id.in_(doc_ids)).all()
    avg_health = round(sum(h.score for h in healths) / len(healths)) if healths else None

    recommendations = []
    if ready < count:
        recommendations.append(f"{count - ready} document(s) are not ready for retrieval")
    if failed > 0:
        recommendations.append(f"{failed} document(s) failed processing")
    if avg_health is not None and avg_health < 60:
        recommendations.append("Average document health is below 60 — consider enriching metadata")
    if not recommendations:
        recommendations.append("Collection is healthy")

    return {
        "collection_id": collection.id,
        "document_count": count,
        "ready_documents": ready,
        "failed_documents": failed,
        "health": avg_health,
        "recommendations": recommendations,
    }


def ai_usage_summary(db: Session, workspace_id: int) -> dict:
    """Workspace AI usage summary from persisted executions."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    executions = (
        db.query(AIExecution)
        .filter(
            AIExecution.workspace_id == workspace_id,
            AIExecution.created_at >= cutoff,
        )
        .all()
    )
    total = len(executions)
    if total == 0:
        return {"ai_executions_30d": 0, "ai_cost_30d": 0.0, "success_rate": None}
    succeeded = sum(1 for e in executions if e.status == "completed")
    cost = sum(e.estimated_cost or 0 for e in executions)
    return {
        "ai_executions_30d": total,
        "ai_cost_30d": round(cost, 4),
        "success_rate": round(succeeded / total, 3),
    }


def document_name(db: Session, document: Document) -> str:
    """Deterministic schema-backed display name for a document.

    Prefers a persisted title from version metadata; falls back to the
    original filename. Never relies on transient instance attributes.
    """
    meta = document_metadata(db, document)
    title = meta.get("title")
    if title:
        return str(title)
    return document.original_filename or "Untitled"


def _title_sim(db: Session, a: Document, b: Document) -> float:
    import re
    ta = set(re.findall(r"[a-z0-9]+", document_name(db, a).lower()))
    tb = set(re.findall(r"[a-z0-9]+", document_name(db, b).lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)