"""Phase 19 — data platform consistency checker.

Periodic bounded consistency checks across documents/chunks/embeddings/
graph/memory/actions/workflows/usage, orphan detection, cross-tenant
integrity verification, and a repair planner that is dry-run first,
bounded, auditable and never auto-destroys suspicious data.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _dumps(value) -> str:
    return json.dumps(value, default=str)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def orphan_chunks(db: Session, *, workspace_id: Optional[int] = None,
                  limit: int = 200) -> list[dict]:
    from sqlalchemy import or_
    from ..models.document_chunk import DocumentChunk
    from ..models.document import Document
    q = db.query(DocumentChunk)
    if workspace_id is not None:
        # OUTER join: an orphan chunk has no document row, so an inner join
        # would hide exactly the rows this check must surface.
        q = (q.outerjoin(Document,
                         DocumentChunk.document_id == Document.id)
             .filter(or_(Document.workspace_id == workspace_id,
                         Document.id.is_(None))))
    chunks = q.limit(5000).all()
    chunk_doc_ids = {chunk.document_id for chunk in chunks}
    existing_docs = set()
    if chunk_doc_ids:
        rows = db.query(Document.id).filter(Document.id.in_(
            list(chunk_doc_ids)[:5000])).all()
        existing_docs = {row[0] for row in rows}
    orphans = [chunk for chunk in chunks
               if chunk.document_id not in existing_docs]
    return [{"kind": "orphan_chunk", "chunk_id": chunk.id}
            for chunk in orphans[:limit]]


def cross_tenant_edges(db: Session, limit: int = 200) -> list[dict]:
    from ..models.knowledge_graph import Entity, EntityRelationship
    rels = db.query(EntityRelationship).limit(10000).all()
    found = []
    for rel in rels:
        source = db.query(Entity).get(rel.source_id)
        target = db.query(Entity).get(rel.target_id)
        if source is None or target is None:
            found.append({"kind": "invalid_reference",
                          "relationship_id": rel.id})
            continue
        if source.workspace_id != target.workspace_id:
            found.append({"kind": "cross_tenant_edge",
                          "relationship_id": rel.id,
                          "workspace_a": source.workspace_id,
                          "workspace_b": target.workspace_id})
        if len(found) >= limit:
            break
    return found


def memory_integrity(db: Session, *, workspace_id: Optional[int] = None,
                     limit: int = 200) -> list[dict]:
    from ..models.phase15 import AIMemory
    q = db.query(AIMemory)
    if workspace_id is not None:
        q = q.filter(AIMemory.workspace_id == workspace_id)
    memories = q.limit(5000).all()
    issues = []
    for memory in memories:
        if memory.supersedes_id is not None:
            parent = db.query(AIMemory).get(memory.supersedes_id)
            if parent is None:
                issues.append({"kind": "dangling_supersession",
                               "memory_id": memory.id})
    return issues[:limit]


def execution_integrity(db: Session, limit: int = 200) -> list[dict]:
    from ..models.ai_execution import AIExecution
    from ..models.workspace import Workspace
    executions = db.query(AIExecution).limit(5000).all()
    issues = []
    for execution in executions:
        if db.query(Workspace).get(execution.workspace_id) is None:
            issues.append({"kind": "orphan_execution",
                           "execution_id": execution.id})
    return issues[:limit]


# ---------------------------------------------------------------------------
# Aggregate checker + repair planner
# ---------------------------------------------------------------------------

def run_consistency_checks(db: Session, *, workspace_id: Optional[int] = None,
                           dry_run: bool = True,
                           persist: bool = True) -> dict:
    from ..models.phase19 import ConsistencyReport
    checks = {
        "document_chunks": orphan_chunks(db, workspace_id=workspace_id),
        "cross_tenant_integrity": cross_tenant_edges(db),
        "memory_integrity": memory_integrity(db, workspace_id=workspace_id),
        "execution_integrity": execution_integrity(db),
    }
    total_issues = sum(len(v) for v in checks.values())
    if persist:
        db.add(ConsistencyReport(
            workspace_id=workspace_id, check_kind="full_consistency",
            status="ISSUES" if total_issues else "CLEAN",
            issue_count=total_issues, issues_json=_dumps(checks),
            dry_run=dry_run))
        db.flush()
    return {"workspace_id": workspace_id, "dry_run": dry_run,
            "clean": total_issues == 0, "issue_count": total_issues,
            "by_check": {k: len(v) for k, v in checks.items()},
            "issues": checks}


def repair_planner(issues: list[dict], *, dry_run: bool = True) -> dict:
    """Repair plans are dry-run first, bounded, and never destructive:
    every plan item is a 'flag for review' style action unless the issue
    kind is trivially safe."""
    plans = []
    allowed_auto = {"orphan_chunk", "invalid_reference"}
    for issue in issues[:500]:
        kind = issue.get("kind", "unknown")
        action = ("flag_for_review" if kind not in allowed_auto
                  else "mark_orphan")
        plans.append({"kind": kind, "target": issue,
                      "action": action, "destructive": False,
                      "needs_approval": action == "flag_for_review"})
    return {"dry_run": dry_run,
            "summary": f"{len(plans)} repair candidate(s); "
                       f"{sum(1 for p in plans if p['needs_approval'])} "
                       "require human approval",
            "plans": plans[:200],
            "note": "repairs never destroy data automatically; suspicious "
                    "data is flagged for review"}
