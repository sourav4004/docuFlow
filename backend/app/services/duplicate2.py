"""Duplicate intelligence 2.0 — classify document pairs.

Classifies EXACT_DUPLICATE / NEAR_DUPLICATE / VERSION / RELATED_DOCUMENT /
UNRELATED using deterministic signals. Never deletes documents automatically;
output is advisory only.
"""

import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.document_content import DocumentContent
from ..services.health_service import document_name


@dataclass
class DuplicateAssessment:
    document_a_id: int
    document_b_id: int
    classification: str
    similarity: float
    reasons: list[str]
    same_version_chain: bool = False


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", "", text or "")).strip().lower()


def _text_for(db: Session, document_id: int) -> str:
    content = (
        db.query(DocumentContent)
        .filter(DocumentContent.document_id == document_id)
        .first()
    )
    return (content.extracted_text if content else "") or (content.content if content else "")


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _same_version_chain(db: Session, a: Document, b: Document) -> bool:
    """Version chain: same base name with v1/v2 suffix pattern."""
    name_a = document_name(db, a)
    name_b = document_name(db, b)
    base_a = re.sub(r"\s*v\d+$", "", name_a.lower().strip())
    base_b = re.sub(r"\s*v\d+$", "", name_b.lower().strip())
    if base_a != base_b:
        return False
    return bool(re.search(r"v\d+$", name_a.lower().strip())) or bool(re.search(r"v\d+$", name_b.lower().strip()))


def assess_pair(db: Session, a: Document, b: Document) -> DuplicateAssessment:
    """Classify a document pair deterministically."""
    text_a = _normalize(_text_for(db, a.id))
    text_b = _normalize(_text_for(db, b.id))
    words_a = set(text_a.split())
    words_b = set(text_b.split())

    sim = _jaccard(words_a, words_b)
    reasons = []
    version_chain = _same_version_chain(db, a, b)

    if not text_a or not text_b:
        # No text: fall back to filename similarity
        name_a = document_name(db, a).lower()
        name_b = document_name(db, b).lower()
        name_sim = _jaccard(set(name_a.split()), set(name_b.split()))
        sim = name_sim
        classification = (
            "EXACT_DUPLICATE" if name_a == name_b
            else "RELATED_DOCUMENT" if name_sim >= 0.6
            else "UNRELATED"
        )
        reasons.append("Classified from filenames (no extracted text available)")
        return DuplicateAssessment(a.id, b.id, classification, round(sim, 3), reasons)

    if sim >= 0.98:
        classification = "EXACT_DUPLICATE"
        reasons.append(f"Word overlap {sim:.0%} — effectively identical text")
    elif sim >= 0.75:
        classification = "VERSION" if version_chain else "NEAR_DUPLICATE"
        reasons.append(f"Word overlap {sim:.0%} — {'version chain detected' if version_chain else 'near-duplicate content'}")
    elif sim >= 0.45:
        classification = "RELATED_DOCUMENT"
        reasons.append(f"Word overlap {sim:.0%} — related subject matter")
    else:
        classification = "UNRELATED"
        reasons.append(f"Word overlap {sim:.0%} — unrelated content")

    return DuplicateAssessment(a.id, b.id, classification, round(sim, 3), reasons, version_chain)


def scan_workspace(db: Session, workspace_id: int, limit: int = 200) -> list[dict]:
    """Scan a workspace for duplicate candidates (advisory only)."""
    docs = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id, Document.status == "READY")
        .limit(limit)
        .all()
    )
    results = []
    seen_pairs = set()
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            key = frozenset((a.id, b.id))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            assessment = assess_pair(db, a, b)
            if assessment.classification in ("EXACT_DUPLICATE", "NEAR_DUPLICATE", "VERSION"):
                results.append({
                    "document_a_id": a.id,
                    "document_b_id": b.id,
                    "classification": assessment.classification,
                    "similarity": assessment.similarity,
                    "reasons": assessment.reasons,
                })
                if len(results) >= 50:
                    return results
    return results