"""AI context builder — authorization-FIRST context assembly with budgeting.

The context builder enforces authorization before adding any information:
it only ever queries resources the caller may see, and it drops low-value
context first when the budget is exceeded. It never constructs a large
context and filters later.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..services.permission_service import require_workspace_membership

PRIORITY_ORDER = {
    "direct_user_request": 1,
    "explicit_cited_evidence": 2,
    "relevant_retrieved_evidence": 3,
    "current_workflow_state": 4,
    "approved_memory": 5,
    "historical_context": 6,
}

MAX_CONTEXT_CHARS = 32_000
CHAR_PER_TOKEN_ESTIMATE = 4


@dataclass
class ContextBlock:
    """A single context item with a priority class."""
    priority_class: str  # one of PRIORITY_ORDER keys
    label: str
    content: str
    source_type: Optional[str] = None
    source_id: Optional[int] = None

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.content) // CHAR_PER_TOKEN_ESTIMATE)


class ContextBudgetExceeded(Exception):
    """Raised when even the highest-priority context cannot fit."""


class ContextBuilder:
    def __init__(self, max_chars: int = MAX_CONTEXT_CHARS):
        self.max_chars = max_chars
        self.blocks: list[ContextBlock] = []

    def add(self, block: ContextBlock) -> None:
        self.blocks.append(block)

    def assemble(self) -> dict:
        """Assemble context: high-priority first, drop low-value overflow."""
        ordered = sorted(
            self.blocks,
            key=lambda b: (PRIORITY_ORDER.get(b.priority_class, 99), b.label),
        )
        total = 0
        kept: list[dict] = []
        dropped = []
        for block in ordered:
            size = len(block.content)
            if total + size > self.max_chars:
                dropped.append(block.label)
                continue
            total += size
            kept.append({
                "priority_class": block.priority_class,
                "label": block.label,
                "content": block.content,
                "source_type": block.source_type,
                "source_id": block.source_id,
            })
        return {
            "blocks": kept,
            "total_chars": total,
            "estimated_tokens": sum(
                max(1, len(b["content"]) // CHAR_PER_TOKEN_ESTIMATE) for b in kept
            ),
            "dropped_blocks": dropped,
            "budget": self.max_chars,
        }


# ---------------------------------------------------------------------------
# Database-backed builders (authorization enforced here, not after)
# ---------------------------------------------------------------------------

def build_document_context(
    db: Session,
    workspace_id: int,
    user_id: int,
    document_id: int,
    include_content: bool = True,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> dict:
    """Build document-scoped context. Only authorized documents are read."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc or doc.workspace_id != workspace_id:
        raise PermissionError("Document not accessible in this workspace")
    require_workspace_membership(db, workspace_id, user_id)

    builder = ContextBuilder(max_chars=max_chars)
    builder.add(ContextBlock(
        priority_class="direct_user_request",
        label="document",
        content=f"Document: {doc.original_filename} (status={doc.status})",
        source_type="document", source_id=doc.id,
    ))
    if include_content:
        from ..models.document_content import DocumentContent
        content = (
            db.query(DocumentContent)
            .filter(DocumentContent.document_id == document_id)
            .first()
        )
        if content:
            builder.add(ContextBlock(
                priority_class="explicit_cited_evidence",
                label="document_content",
                content=(content.content or "")[:max_chars],
                source_type="document", source_id=doc.id,
            ))
    return builder.assemble()


def build_workspace_context(
    db: Session,
    workspace_id: int,
    user_id: int,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> dict:
    """Build workspace-scoped context: only the caller's workspace."""
    require_workspace_membership(db, workspace_id, user_id)

    builder = ContextBuilder(max_chars=max_chars)
    docs = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
        .limit(50)
        .all()
    )
    doc_lines = [
        f"- #{d.id} {d.original_filename} ({d.status})" for d in docs
    ]
    builder.add(ContextBlock(
        priority_class="relevant_retrieved_evidence",
        label="workspace_documents",
        content="\n".join(doc_lines) or "(no documents)",
        source_type="workspace", source_id=workspace_id,
    ))
    return builder.assemble()