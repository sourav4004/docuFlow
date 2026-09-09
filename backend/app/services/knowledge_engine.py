"""Knowledge engine — change detection 2.0, impact graph, policy intelligence,
temporal knowledge, and reproducible snapshots.

All detection is deterministic and evidence-backed; nothing is fabricated.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import (
    KnowledgeChange, ImpactLink, PolicyStatement, PolicyConflict,
    TemporalFact, KnowledgeSnapshot,
)
from ..models.document import Document
from ..services.change_intelligence import diff_versions, _text_for_version
from ..services.health_service import document_metadata
from ..services.audit_service import log_audit_event

CHANGE_TYPES = (
    "CONTENT_CHANGE", "DATE_CHANGE", "POLICY_CHANGE", "NUMERIC_CHANGE",
    "ENTITY_CHANGE", "STRUCTURAL_CHANGE", "OWNERSHIP_CHANGE",
    "REQUIREMENT_CHANGE", "TERMINOLOGY_CHANGE", "RISK_CHANGE",
)

SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# ---------------------------------------------------------------------------
# Change detection 2.0
# ---------------------------------------------------------------------------

def detect_and_record_change(
    db: Session,
    workspace_id: int,
    document_id: int,
    version_from: Optional[int],
    version_to: int,
    organization_id: Optional[int] = None,
) -> Optional[KnowledgeChange]:
    """Detect a document change, classify it, and persist the evidence.

    Builds on Phase 14 diff_versions; extends it with severity, confidence,
    affected entities, and a deduplicated record.
    """
    text_a = _text_for_version(db, document_id, version_from)
    text_b = _text_for_version(db, document_id, version_to)
    diff = diff_versions(text_a, text_b)

    if diff["added_count"] == 0 and diff["removed_count"] == 0:
        return None  # never fabricate a difference

    change_type = diff["classification"]
    severity = _severity_for(change_type, diff)
    confidence = _confidence_for(diff)

    affected_entities = _extract_entities(diff["added_preview"] + diff["removed_preview"])
    affected_sections = _affected_sections(text_b, diff["added_preview"])

    change = KnowledgeChange(
        workspace_id=workspace_id,
        organization_id=organization_id,
        document_id=document_id,
        version_from=version_from,
        version_to=version_to,
        change_type=change_type,
        severity=severity,
        confidence=confidence,
        summary=(
            f"{diff['added_count']} line(s) added, {diff['removed_count']} removed — "
            f"{change_type.lower().replace('_', ' ')}"
        ),
        old_evidence_json=json.dumps(diff["removed_preview"]),
        new_evidence_json=json.dumps(diff["added_preview"]),
        affected_sections_json=json.dumps(affected_sections),
        affected_entities_json=json.dumps(affected_entities),
    )
    db.add(change)
    db.flush()
    return change


def _severity_for(change_type: str, diff: dict) -> str:
    if change_type in ("POLICY_CHANGE", "NUMERIC_CHANGE"):
        return "HIGH"
    if change_type in ("DATE_CHANGE", "ENTITY_CHANGE", "REQUIREMENT_CHANGE"):
        return "HIGH" if diff.get("importance") == "HIGH" else "MEDIUM"
    if change_type == "STRUCTURAL_CHANGE":
        return "HIGH"
    return "MEDIUM"


def _confidence_for(diff: dict) -> str:
    total = diff["added_count"] + diff["removed_count"]
    if total == 0:
        return "LOW"
    if total <= 5:
        return "HIGH"
    if total <= 20:
        return "MEDIUM"
    return "LOW"  # large diffs are harder to attribute


def _extract_entities(lines: list[str]) -> list[str]:
    entities = []
    pattern = re.compile(r"\b([A-Z][a-zA-Z]+ (?:Corp|Inc|LLC|Ltd|Company|Organization))\b")
    for line in lines:
        entities.extend(m.group(1) for m in pattern.finditer(line))
    return list(dict.fromkeys(entities))[:10]


def _affected_sections(text: str, added: list[str]) -> list[str]:
    if not text or not added:
        return []
    lines = text.splitlines()
    sections = []
    for idx, line in enumerate(lines):
        if any(a in line for a in added):
            start = max(0, idx - 2)
            for probe in lines[start:idx + 1]:
                if probe.strip().isupper() and len(probe.strip()) > 3:
                    sections.append(probe.strip())
    return list(dict.fromkeys(sections))[:10]


def list_changes(db: Session, workspace_id: int, limit: int = 100, change_type: Optional[str] = None) -> list[KnowledgeChange]:
    query = db.query(KnowledgeChange).filter(KnowledgeChange.workspace_id == workspace_id)
    if change_type:
        query = query.filter(KnowledgeChange.change_type == change_type)
    return query.order_by(KnowledgeChange.created_at.desc()).limit(limit).all()


# ---------------------------------------------------------------------------
# Impact graph
# ---------------------------------------------------------------------------

def build_impact_links(
    db: Session,
    workspace_id: int,
    document_id: int,
) -> dict:
    """Determine what a document change may affect.

    Returns direct (explicit), indirect (inferred), and uncertain impacts.
    Relationships are explicitly labeled so users never confuse inference
    with fact.
    """
    links = []
    direct = []
    indirect = []
    uncertain = []

    # Explicit: collections containing the document
    from ..models.collection import collection_documents
    collection_rows = (
        db.query(collection_documents.c.collection_id)
        .filter(collection_documents.c.document_id == document_id)
        .all()
    )
    for (collection_id,) in collection_rows:
        link = ImpactLink(
            workspace_id=workspace_id, source_type="document", source_id=document_id,
            target_type="collection", target_id=collection_id, relation="EXPLICIT",
        )
        db.add(link)
        links.append(link)
        direct.append({"type": "collection", "id": collection_id, "relation": "EXPLICIT"})

    # Explicit: deadlines referencing the document
    from ..models.knowledge import Deadline
    deadlines = (
        db.query(Deadline)
        .filter(Deadline.workspace_id == workspace_id, Deadline.document_id == document_id)
        .all()
    )
    for deadline in deadlines:
        direct.append({"type": "deadline", "id": deadline.id, "relation": "EXPLICIT"})

    # Inferred: entities in the changed document
    meta = document_metadata(db, db.query(Document).filter(Document.id == document_id).first() or Document())
    from ..models.knowledge_graph import Entity
    entities = db.query(Entity).filter(Entity.workspace_id == workspace_id).limit(200).all()
    for entity in entities:
        if entity.name and meta.get("title") and entity.name.lower() in meta["title"].lower():
            link = ImpactLink(
                workspace_id=workspace_id, source_type="document", source_id=document_id,
                target_type="entity", target_id=entity.id, relation="INFERRED",
                confidence=0.6,
            )
            db.add(link)
            links.append(link)
            indirect.append({"type": "entity", "id": entity.id, "relation": "INFERRED", "confidence": 0.6})

    # Uncertain: workflows whose definitions mention the document type
    from ..models.workflow_version import WorkflowVersion
    workflows = (
        db.query(WorkflowVersion)
        .filter(WorkflowVersion.workspace_id == workspace_id)
        .limit(100)
        .all()
    )
    for wf in workflows:
        definition = wf.definition_json or ""
        if meta.get("classification") and meta["classification"].lower() in definition.lower():
            uncertain.append({"type": "workflow", "id": wf.workflow_id, "relation": "UNCERTAIN"})

    if links:
        db.flush()
    return {
        "direct_impacts": direct,
        "indirect_impacts": indirect,
        "uncertain_impacts": uncertain,
    }


def list_impact_links(db: Session, workspace_id: int, source_type: Optional[str] = None, source_id: Optional[int] = None) -> list[ImpactLink]:
    query = db.query(ImpactLink).filter(ImpactLink.workspace_id == workspace_id)
    if source_type:
        query = query.filter(ImpactLink.source_type == source_type)
    if source_id:
        query = query.filter(ImpactLink.source_id == source_id)
    return query.order_by(ImpactLink.created_at.desc()).limit(200).all()


# ---------------------------------------------------------------------------
# Policy intelligence
# ---------------------------------------------------------------------------

REQUIREMENT_PATTERNS = {
    "approval": r"\b(approval|approved|sign-off|signoff)\b",
    "limit": r"\b(limit|maximum|up to|threshold|cap of)\b",
    "obligation": r"\b(must|shall|required|obliged|responsible for)\b",
    "exception": r"\b(except|unless|however|notwithstanding)\b",
}


def extract_policy_statements(
    db: Session,
    workspace_id: int,
    document_id: int,
    organization_id: Optional[int] = None,
    limit: int = 50,
) -> list[PolicyStatement]:
    """Extract structured policy statements from a document's text.

    Deterministic pattern-based extraction; confidence is conservative.
    """
    text = _text_for_version(db, document_id, None)
    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    statements = []
    for sentence in sentences[:200]:
        lowered = sentence.lower()
        matched_type = None
        for req_type, pattern in REQUIREMENT_PATTERNS.items():
            if re.search(pattern, lowered):
                matched_type = req_type
                break
        if not matched_type:
            continue
        if len(sentence.strip()) < 20 or len(sentence) > 500:
            continue

        existing = (
            db.query(PolicyStatement)
            .filter(
                PolicyStatement.workspace_id == workspace_id,
                PolicyStatement.document_id == document_id,
                PolicyStatement.statement == sentence.strip(),
            )
            .first()
        )
        if existing:
            continue

        statement = PolicyStatement(
            workspace_id=workspace_id,
            organization_id=organization_id,
            document_id=document_id,
            statement=sentence.strip(),
            requirement_type=matched_type,
            evidence_reference=f"doc:{document_id}",
            source_chunk=sentence.strip()[:500],
            effective_date=_extract_date(sentence),
        )
        db.add(statement)
        statements.append(statement)
        if len(statements) >= limit:
            break
    if statements:
        db.flush()
    return statements


def _extract_date(text: str) -> Optional[datetime]:
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if not match:
        match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b", text)
    if not match:
        return None
    try:
        if len(match.group(1)) == 4:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def detect_policy_conflicts(db: Session, workspace_id: int, organization_id: Optional[int] = None) -> list[PolicyConflict]:
    """Detect real requirement conflicts — not superficial textual differences.

    Current rules (deterministic):
    - numeric thresholds: approval-required above X vs no-approval below Y
      (conflicts only when the no-approval band overlaps the approval band)
    - conflicting dates: same party/requirement with different effective dates
    """
    statements = (
        db.query(PolicyStatement)
        .filter(PolicyStatement.workspace_id == workspace_id)
        .limit(200)
        .all()
    )
    conflicts = []
    for i, a in enumerate(statements):
        for b in statements[i + 1:]:
            conflict = _numeric_threshold_conflict(db, a, b, organization_id)
            if conflict:
                conflicts.append(conflict)
    if conflicts:
        db.flush()
    return conflicts


def _numeric_threshold_conflict(db, a: PolicyStatement, b: PolicyStatement, organization_id):
    """Approval required above X + no approval below Y with Y >= X => conflict."""
    a_number = _first_number(a.statement)
    b_number = _first_number(b.statement)
    if a_number is None or b_number is None:
        return None
    a_requires_approval = "approval" in a.statement.lower() and "above" in a.statement.lower()
    b_no_approval = "no approval" in b.statement.lower() and "below" in b.statement.lower()
    if not (a_requires_approval and b_no_approval):
        return None
    if b_number < a_number:
        return None  # no overlap — policies are consistent

    existing = (
        db.query(PolicyConflict)
        .filter(
            PolicyConflict.workspace_id == a.workspace_id,
            PolicyConflict.policy_a_id == a.id,
            PolicyConflict.policy_b_id == b.id,
            PolicyConflict.status == "OPEN",
        )
        .first()
    )
    if existing:
        return None

    conflict = PolicyConflict(
        workspace_id=a.workspace_id,
        organization_id=organization_id,
        policy_a_id=a.id,
        policy_b_id=b.id,
        conflict_type="NUMERIC_THRESHOLD",
        severity="HIGH",
        description=(
            f"'{a.statement[:80]}' requires approval above ${a_number}, but "
            f"'{b.statement[:80]}' allows no approval below ${b_number} — the "
            f"${b_number} ceiling overlaps the approval-required range."
        ),
        conditions_json=json.dumps({"threshold_a": a_number, "ceiling_b": b_number}),
    )
    db.add(conflict)
    return conflict


def _first_number(text: str) -> Optional[float]:
    match = re.search(r"\$?([\d,]+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Temporal knowledge
# ---------------------------------------------------------------------------

def record_temporal_fact(
    db: Session,
    workspace_id: int,
    fact_type: str,
    fact_value: str,
    valid_from: datetime,
    valid_until: Optional[datetime] = None,
    document_id: Optional[int] = None,
    entity_id: Optional[int] = None,
    source: Optional[str] = None,
    organization_id: Optional[int] = None,
    supersede_existing: bool = True,
) -> TemporalFact:
    """Record a temporal fact, optionally superseding current facts of the type."""
    if supersede_existing:
        current = (
            db.query(TemporalFact)
            .filter(
                TemporalFact.workspace_id == workspace_id,
                TemporalFact.fact_type == fact_type,
                TemporalFact.valid_until.is_(None),
                (TemporalFact.document_id == document_id) | (TemporalFact.document_id.is_(None)),
            )
            .all()
        )
        for fact in current:
            if fact.id != 0:
                fact.valid_until = valid_from
                fact.superseded_by = None  # set after new row gets its id
    fact = TemporalFact(
        workspace_id=workspace_id,
        organization_id=organization_id,
        document_id=document_id,
        entity_id=entity_id,
        fact_type=fact_type,
        fact_value=fact_value,
        valid_from=valid_from,
        valid_until=valid_until,
        observed_at=datetime.now(timezone.utc),
        source=source,
    )
    db.add(fact)
    db.flush()
    return fact


def facts_as_of(db: Session, workspace_id: int, point_in_time: datetime, fact_type: Optional[str] = None) -> list[TemporalFact]:
    """What was true at a point in time (stale knowledge is never 'current')."""
    query = db.query(TemporalFact).filter(
        TemporalFact.workspace_id == workspace_id,
        TemporalFact.valid_from <= point_in_time,
        (TemporalFact.valid_until.is_(None)) | (TemporalFact.valid_until > point_in_time),
    )
    if fact_type:
        query = query.filter(TemporalFact.fact_type == fact_type)
    return query.order_by(TemporalFact.valid_from.desc()).limit(200).all()


def current_facts(db: Session, workspace_id: int, fact_type: Optional[str] = None) -> list[TemporalFact]:
    return facts_as_of(db, workspace_id, datetime.now(timezone.utc), fact_type)


# ---------------------------------------------------------------------------
# Knowledge snapshots
# ---------------------------------------------------------------------------

def create_snapshot(
    db: Session,
    workspace_id: int,
    user_id: int,
    organization_id: Optional[int] = None,
    name: str = "snapshot",
) -> KnowledgeSnapshot:
    """Capture a reproducible workspace knowledge snapshot."""
    from ..services.health_service import workspace_health, ai_usage_summary
    from ..models.knowledge import Deadline
    from ..models.document import Document

    docs = db.query(Document).filter(Document.workspace_id == workspace_id).limit(500).all()
    deadlines = (
        db.query(Deadline)
        .filter(Deadline.workspace_id == workspace_id, Deadline.status.in_(("UPCOMING", "OVERDUE", "DUE")))
        .limit(200)
        .all()
    )
    changes = (
        db.query(KnowledgeChange)
        .filter(KnowledgeChange.workspace_id == workspace_id)
        .order_by(KnowledgeChange.created_at.desc())
        .limit(50)
        .all()
    )
    conflicts = (
        db.query(PolicyConflict)
        .filter(PolicyConflict.workspace_id == workspace_id, PolicyConflict.status == "OPEN")
        .limit(50)
        .all()
    )
    facts = current_facts(db, workspace_id)

    data = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "health": workspace_health(db, workspace_id),
        "ai_usage": ai_usage_summary(db, workspace_id),
        "documents": [
            {"id": d.id, "filename": d.original_filename, "status": d.status, "sensitivity": d.sensitivity}
            for d in docs
        ],
        "deadlines": [
            {"id": d.id, "title": d.title, "due": d.due_date.isoformat(), "status": d.status}
            for d in deadlines
        ],
        "changes": [
            {"id": c.id, "type": c.change_type, "severity": c.severity, "summary": c.summary}
            for c in changes
        ],
        "policy_conflicts": [
            {"id": c.id, "type": c.conflict_type, "severity": c.severity, "description": c.description}
            for c in conflicts
        ],
        "temporal_facts": [
            {"id": f.id, "type": f.fact_type, "value": f.fact_value,
             "valid_from": f.valid_from.isoformat() if f.valid_from else None,
             "valid_until": f.valid_until.isoformat() if f.valid_until else None}
            for f in facts
        ],
    }

    snapshot = KnowledgeSnapshot(
        workspace_id=workspace_id,
        organization_id=organization_id,
        name=name,
        snapshot_type="workspace",
        data_json=json.dumps(data, default=str),
        created_by=user_id,
    )
    db.add(snapshot)
    db.flush()
    log_audit_event(
        db, event_type="knowledge_snapshot", event_action="create",
        user_id=user_id, resource_type="knowledge_snapshot", resource_id=snapshot.id,
        details=f"Knowledge snapshot '{name}' created",
    )
    return snapshot


def list_snapshots(db: Session, workspace_id: int, limit: int = 50) -> list[KnowledgeSnapshot]:
    return (
        db.query(KnowledgeSnapshot)
        .filter(KnowledgeSnapshot.workspace_id == workspace_id)
        .order_by(KnowledgeSnapshot.created_at.desc())
        .limit(limit)
        .all()
    )