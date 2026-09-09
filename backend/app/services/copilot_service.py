"""Unified AI Copilot — context-aware assistant over documents, collections,
workspace knowledge, and conversations.

Scope is enforced by the retrieval layer (user_id authorization) and by
explicit scope parameters. A DOCUMENT copilot can never reach documents
outside the authorized workspace.
"""

from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.collection import Collection
from ..services.rag_service import answer_question_with_history, RAGResponse, RAGError
from ..services.permission_service import require_workspace_membership

COPILOT_SCOPES = ("DOCUMENT", "COLLECTION", "WORKSPACE", "CONVERSATION")

# Command intents the copilot understands deterministically
COMMAND_PATTERNS = {
    "summarize": ["summar", "overview of this", "tl;dr", "tldr"],
    "extract": ["extract", "pull out"],
    "risks": ["risk", "obligation", "liability"],
    "dates": ["date", "deadline", "when", "expir"],
    "compare": ["compare", "difference", "versus"],
    "missing": ["missing", "gap", "incomplete"],
    "question": [],  # default fallback
}


class CopilotScopeError(Exception):
    """Raised when copilot scope is invalid or unauthorized."""


def classify_command(question: str) -> str:
    """Deterministic command classification (no LLM required)."""
    lower = question.lower()
    for command, patterns in COMMAND_PATTERNS.items():
        if any(p in lower for p in patterns):
            return command
    return "question"


def _resolve_scope(
    db: Session,
    user_id: int,
    scope: str,
    document_id: Optional[int] = None,
    collection_id: Optional[int] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Resolve scope into (document_id, collection_id) with authorization."""
    if scope == "DOCUMENT":
        if document_id is None:
            raise CopilotScopeError("DOCUMENT copilot requires a document_id")
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise CopilotScopeError("Document not found")
        from ..services.workspace_service import resolve_document_workspace
        workspace_id = resolve_document_workspace(db, doc, user_id)
        require_workspace_membership(db, workspace_id, user_id)
        return document_id, None
    if scope == "COLLECTION":
        if collection_id is None:
            raise CopilotScopeError("COLLECTION copilot requires a collection_id")
        collection = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            raise CopilotScopeError("Collection not found")
        # Workspace-scoped collections require workspace membership; legacy
        # user collections require ownership. Never grant cross-tenant access.
        if collection.workspace_id is not None:
            require_workspace_membership(db, collection.workspace_id, user_id)
        elif collection.user_id != user_id:
            raise CopilotScopeError("Collection not found")
        return None, collection_id
    if scope == "WORKSPACE":
        return None, None
    if scope == "CONVERSATION":
        return document_id, None
    raise CopilotScopeError(f"Unknown copilot scope: {scope}")


def copilot_ask(
    db: Session,
    user_id: int,
    scope: str,
    question: str,
    document_id: Optional[int] = None,
    collection_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    conversation_history: str = "",
    llm_service=None,
) -> dict:
    """Ask the copilot within a scope.

    Returns a standard response envelope with answer, sources, scope,
    command, and confidence.
    """
    if scope not in COPILOT_SCOPES:
        raise CopilotScopeError(f"Unknown copilot scope: {scope}")

    if workspace_id is not None:
        require_workspace_membership(db, workspace_id, user_id)

    resolved_doc, resolved_coll = _resolve_scope(
        db, user_id, scope, document_id, collection_id
    )

    try:
        response: RAGResponse = answer_question_with_history(
            db=db,
            user_id=user_id,
            question=question,
            conversation_history=conversation_history,
            document_id=resolved_doc,
            collection_id=resolved_coll,
            llm_service=llm_service,
        )
    except RAGError as exc:
        return {
            "scope": scope,
            "command": classify_command(question),
            "answer": f"The copilot could not answer: {exc}",
            "sources": [],
            "grounded": False,
            "confidence": {"level": "UNKNOWN", "score": None, "reasons": ["provider_error"]},
            "matched_documents": 0,
        }

    confidence = response.confidence
    return {
        "scope": scope,
        "command": classify_command(question),
        "answer": response.answer,
        "sources": [
            {
                "document_id": s.document_id,
                "filename": s.filename,
                "chunk_id": s.chunk_id,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "similarity_score": s.similarity_score,
                "match_type": getattr(s, "match_type", "vector"),
            }
            for s in response.sources
        ],
        "grounded": response.grounded,
        "confidence": {
            "level": confidence.level if confidence else "LOW",
            "grounding_score": confidence.grounding_score if confidence else 0.0,
            "supporting_sources": confidence.supporting_sources if confidence else 0,
        },
        "matched_documents": response.retrieval_count,
        "model": response.model,
        "provider": response.provider,
    }