"""Copilot 2.0 — document/collection/workspace/organization copilots and
natural-language analytics.

Everything remains permission-scoped: document copilots cannot see unrelated
workspace data; workspace copilots only see their own workspace; the
organization copilot only exposes aggregates (never cross-workspace document
contents) and requires organization-admin permission.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.phase15 import KnowledgeChange, PolicyConflict, KnowledgeGap
from ..models.knowledge import Deadline, DocumentHealth
from ..models.workspace import Workspace
from ..models.organization import OrganizationMember
from ..services.permission_service import require_workspace_membership
from ..services.health_service import workspace_health
from ..services.memory_service import list_memories
from ..services.claim_validator import validate_claims, final_answer_text

COPILOT_CONTEXTS = ("DOCUMENT", "COLLECTION", "WORKSPACE", "CONVERSATION", "ORGANIZATION")


class CopilotScopeError(Exception):
    """Raised when a copilot scope is invalid or unauthorized."""


def _require_member(db: Session, workspace_id: int, user_id: int) -> None:
    """Workspace membership gate that raises CopilotScopeError, not HTTP errors.

    Service-level copilots must fail with a domain error; the API layer maps
    it to the appropriate HTTP status.
    """
    try:
        require_workspace_membership(db, workspace_id, user_id)
    except Exception as exc:  # noqa: BLE001 — any HTTP/permission failure -> scope error
        raise CopilotScopeError(str(exc)) from exc


def _require_org_admin(db: Session, organization_id: int, user_id: int) -> None:
    member = (
        db.query(OrganizationMember)
        .filter(
            OrganizationMember.organization_id == organization_id,
            OrganizationMember.user_id == user_id,
        )
        .first()
    )
    if not member or member.role not in ("OWNER", "ADMIN"):
        raise CopilotScopeError("Organization copilot requires an organization admin")


def document_copilot_context(
    db: Session,
    workspace_id: int,
    user_id: int,
    document_id: int,
) -> dict:
    """Document-scoped intelligence bundle (no cross-document data)."""
    _require_member(db, workspace_id, user_id)
    doc = db.query(Document).filter(
        Document.id == document_id, Document.workspace_id == workspace_id
    ).first()
    if not doc:
        raise CopilotScopeError("Document not found in this workspace")

    health = (
        db.query(DocumentHealth)
        .filter(DocumentHealth.document_id == document_id)
        .order_by(DocumentHealth.computed_at.desc())
        .first()
    )
    changes = (
        db.query(KnowledgeChange)
        .filter(KnowledgeChange.document_id == document_id)
        .order_by(KnowledgeChange.created_at.desc())
        .limit(10)
        .all()
    )
    return {
        "scope": "DOCUMENT",
        "document_id": document_id,
        "filename": doc.original_filename,
        "status": doc.status,
        "sensitivity": doc.sensitivity,
        "health": {
            "score": health.score if health else None,
            "reasons": health.reasons if health else [],
        } if health else None,
        "recent_changes": [
            {"id": c.id, "type": c.change_type, "severity": c.severity, "summary": c.summary}
            for c in changes
        ],
    }


def workspace_copilot_context(
    db: Session,
    workspace_id: int,
    user_id: int,
) -> dict:
    """Workspace-scoped knowledge briefing (health, policies, deadlines, gaps)."""
    _require_member(db, workspace_id, user_id)
    health = workspace_health(db, workspace_id)
    deadlines = (
        db.query(Deadline)
        .filter(
            Deadline.workspace_id == workspace_id,
            Deadline.status.in_(("UPCOMING", "DUE", "OVERDUE")),
        )
        .order_by(Deadline.due_date.asc())
        .limit(20)
        .all()
    )
    conflicts = (
        db.query(PolicyConflict)
        .filter(PolicyConflict.workspace_id == workspace_id, PolicyConflict.status == "OPEN")
        .limit(20)
        .all()
    )
    gaps = (
        db.query(KnowledgeGap)
        .filter(KnowledgeGap.workspace_id == workspace_id, KnowledgeGap.status == "OPEN")
        .limit(20)
        .all()
    )
    return {
        "scope": "WORKSPACE",
        "workspace_id": workspace_id,
        "health": health,
        "deadlines": [
            {"id": d.id, "title": d.title, "due": d.due_date.isoformat(), "status": d.status}
            for d in deadlines
        ],
        "policy_conflicts": [
            {"id": c.id, "type": c.conflict_type, "severity": c.severity}
            for c in conflicts
        ],
        "knowledge_gaps": [
            {"id": g.id, "type": g.gap_type, "severity": g.severity, "title": g.title}
            for g in gaps
        ],
    }


def organization_copilot_context(
    db: Session,
    organization_id: int,
    user_id: int,
) -> dict:
    """Organization aggregate briefing — never document contents."""
    _require_org_admin(db, organization_id, user_id)
    workspaces = (
        db.query(Workspace)
        .filter(Workspace.organization_id == organization_id)
        .limit(200)
        .all()
    )
    per_workspace = []
    for ws in workspaces:
        try:
            h = workspace_health(db, ws.id)
        except Exception:  # noqa: BLE001 — a broken workspace must not break the org view
            h = {}
        per_workspace.append({
            "workspace_id": ws.id,
            "name": ws.name,
            "health_score": h.get("score"),
            "documents": h.get("total_documents", 0),
            "conflicts": h.get("conflict_count", 0),
            "duplicates": h.get("duplicate_count", 0),
        })
    return {
        "scope": "ORGANIZATION",
        "organization_id": organization_id,
        "workspaces": per_workspace,
        "note": "Aggregate metrics only — document contents never cross workspaces",
    }


# ---------------------------------------------------------------------------
# Natural language analytics (deterministic; never fabricates metrics)
# ---------------------------------------------------------------------------

def nl_analytics(
    db: Session,
    workspace_id: int,
    query: str,
) -> dict:
    """Answer structured analytics questions deterministically.

    Supported intents are matched by keyword; anything unrecognized returns
    an explicit 'unsupported' result rather than a made-up answer.
    """
    lower = query.lower()
    now = datetime.now(timezone.utc)

    if "changed this month" in lower or "changed last month" in lower:
        from sqlalchemy import func
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        count = (
            db.query(func.count(Document.id))
            .filter(Document.workspace_id == workspace_id, Document.updated_at >= month_start)
            .scalar()
        )
        return {"intent": "documents_changed", "metric": count, "unit": "documents", "period": "this month", "explanation": f"{count} document(s) changed this month."}

    if "expire soon" in lower or "expir" in lower:
        upcoming = (
            db.query(Deadline)
            .filter(
                Deadline.workspace_id == workspace_id,
                Deadline.status.in_(("UPCOMING", "DUE")),
            )
            .order_by(Deadline.due_date.asc())
            .limit(10)
            .all()
        )
        return {
            "intent": "expiring_deadlines",
            "metric": len(upcoming),
            "unit": "deadlines",
            "items": [{"title": d.title, "due": d.due_date.isoformat()} for d in upcoming],
            "explanation": f"{len(upcoming)} deadline(s) are upcoming.",
        }

    if "unresolved conflicts" in lower:
        count = (
            db.query(PolicyConflict)
            .filter(PolicyConflict.workspace_id == workspace_id, PolicyConflict.status == "OPEN")
            .count()
        )
        return {"intent": "open_conflicts", "metric": count, "unit": "conflicts", "explanation": f"{count} open policy conflict(s)."}

    if "health" in lower or "knowledge health" in lower:
        health = workspace_health(db, workspace_id)
        return {"intent": "workspace_health", "metric": health.get("score"), "unit": "score", "explanation": f"Workspace knowledge health is {health.get('score')}/100.", "health": health}

    return {
        "intent": "unsupported",
        "metric": None,
        "explanation": "This analytics question is not supported. Supported intents: documents changed, expiring deadlines, unresolved conflicts, workspace health.",
    }


def validated_rag_answer(
    db: Session,
    workspace_id: int,
    user_id: int,
    question: str,
    answer: str,
    evidence_chunks: list[str],
    source_document: Optional[int] = None,
) -> dict:
    """Attach claim-level validation to a RAG answer (evidence-first)."""
    validation = validate_claims(answer, evidence_chunks, source_document=source_document)
    return {
        "answer": final_answer_text(validation) if validation.unsupported_count else answer,
        "validation": validation.to_dict(),
    }