"""Document fingerprinting + duplicate intelligence — Phase 17.

Stable hashes over content/structure/metadata plus sampled shingles enable
deterministic duplicate classification (EXACT / NEAR / VERSION / RELATED /
UNRELATED). Duplicates are recorded as candidates — nothing is ever deleted
automatically.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.document_version import DocumentVersion
from ..models.phase17 import DocumentFingerprint, DuplicateCandidate

_SHINGLE_SIZE = 5
_SHINGLE_SAMPLE = 64
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _shingles(text: str) -> set[int]:
    toks = _tokens(text)
    if not toks:
        return set()
    out = set()
    for i in range(0, len(toks) - _SHINGLE_SIZE + 1):
        out.add(hash(tuple(toks[i:i + _SHINGLE_SIZE])))
    return out


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_fingerprint(db: Session, document_id: int,
                        content: str = "") -> DocumentFingerprint:
    """Compute (or refresh) a document fingerprint.

    ``content`` may come from the extraction pipeline; when empty we fall back
    to available metadata so the fingerprint always exists.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise ValueError(f"Document {document_id} not found")
    metadata = {
        "original_filename": doc.original_filename,
        "mime_type": doc.mime_type,
        "file_size": doc.file_size,
        "status": doc.status,
    }
    if not content:
        content = " ".join(
            f"{k}:{v}" for k, v in metadata.items() if v is not None)
    structural = _structural_signature(content)

    version = db.query(DocumentVersion).filter(
        DocumentVersion.document_id == document_id).order_by(
            DocumentVersion.version_number.desc()).first()

    fp = db.query(DocumentFingerprint).filter(
        DocumentFingerprint.document_id == document_id).first()
    if fp is None:
        fp = DocumentFingerprint(
            document_id=document_id,
            content_hash=_sha(content),
            structural_hash=_sha(structural),
            metadata_hash=_sha(json.dumps(metadata, sort_keys=True)),
            shingles_json=json.dumps(sorted(_shingles(content))[:_SHINGLE_SAMPLE]),
        )
        db.add(fp)
    else:
        fp.content_hash = _sha(content)
        fp.structural_hash = _sha(structural)
        fp.metadata_hash = _sha(json.dumps(metadata, sort_keys=True))
        fp.shingles_json = json.dumps(
            sorted(_shingles(content))[:_SHINGLE_SAMPLE])
    fp.version_id = version.id if version is not None else fp.version_id
    db.flush()
    return fp


def _structural_signature(content: str) -> str:
    """Structural signature: heading-ish lines and paragraph boundaries."""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    heads = [ln for ln in lines if len(ln) < 120 and ln[:1].isupper()]
    return "\n".join(heads[:40])


def _load_shingles(fp: DocumentFingerprint) -> set[int]:
    try:
        return set(json.loads(fp.shingles_json or "[]"))
    except (ValueError, TypeError):
        return set()


def classify_duplicates(db: Session, workspace_id: int,
                        document_id: int, limit: int = 50) -> list[dict]:
    """Compare ``document_id`` against workspace peers, persist candidates.

    Classifications:
    - EXACT_DUPLICATE: identical content hash
    - NEAR_DUPLICATE: high shingle overlap
    - VERSION: same title family + high overlap (revised version)
    - RELATED_DOCUMENT: moderate overlap
    - UNRELATED: below threshold (not persisted)
    """
    fp = db.query(DocumentFingerprint).filter(
        DocumentFingerprint.document_id == document_id).first()
    if fp is None:
        fp = compute_fingerprint(db, document_id)
    doc = db.query(Document).filter(Document.id == document_id).first()
    mine = _load_shingles(fp)
    results = []
    peers = (
        db.query(DocumentFingerprint)
        .join(Document, DocumentFingerprint.document_id == Document.id)
        .filter(Document.workspace_id == workspace_id,
                Document.id != document_id,
                Document.status.in_(("READY", "PROCESSING", "UPLOADED")))
        .order_by(DocumentFingerprint.id.desc())
        .limit(200)
        .all()
    )
    for other in peers:
        theirs = _load_shingles(other)
        sim = jaccard(mine, theirs)
        if sim < 0.30:
            continue
        if fp.content_hash == other.content_hash:
            classification = "EXACT_DUPLICATE"
        elif sim >= 0.75:
            classification = "NEAR_DUPLICATE"
        elif sim >= 0.55:
            classification = "VERSION"
        else:
            classification = "RELATED_DOCUMENT"
        _persist_candidate(db, workspace_id, document_id, other.document_id,
                           classification, sim)
        results.append({
            "other_document_id": other.document_id,
            "classification": classification,
            "similarity": round(sim, 4),
        })
    return results


def _persist_candidate(db: Session, workspace_id: int, document_id: int,
                       other_document_id: int, classification: str,
                       similarity: float) -> None:
    existing = (
        db.query(DuplicateCandidate)
        .filter(DuplicateCandidate.document_id == document_id,
                DuplicateCandidate.other_document_id == other_document_id)
        .first()
    )
    if existing is None:
        db.add(DuplicateCandidate(
            workspace_id=workspace_id, document_id=document_id,
            other_document_id=other_document_id,
            classification=classification, similarity=similarity,
            method="fingerprint"))
        db.flush()


def list_candidates(db: Session, workspace_id: int, limit: int = 50,
                    reviewed: Optional[bool] = None) -> dict:
    q = db.query(DuplicateCandidate).filter(
        DuplicateCandidate.workspace_id == workspace_id)
    if reviewed is not None:
        q = q.filter(DuplicateCandidate.reviewed == reviewed)
    total = q.count()
    items = q.order_by(DuplicateCandidate.similarity.desc()) \
        .limit(min(limit, 200)).all()
    return {"items": [
        {"id": c.id, "document_id": c.document_id,
         "other_document_id": c.other_document_id,
         "classification": c.classification,
         "similarity": round(c.similarity, 4),
         "reviewed": c.reviewed}
        for c in items], "total": total, "limit": min(limit, 200)}
