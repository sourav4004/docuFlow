"""Phase 20 — Export / reporting with immutable versioned reports.

AI quality reports (quality, cost, safety, reliability), knowledge health
reports (stale knowledge, gaps, conflicts, graph health), and governance
reports (policies, changes, violations, approvals). Every report is
immutable and versioned.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import ReportVersion

VALID_KINDS = ("ai_quality", "knowledge_health", "governance", "reliability",
               "cost")


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def create_report(db: Session, *, kind: str, scope_type: str,
                  content: dict, scope_id: Optional[int] = None,
                  generated_by: Optional[int] = None) -> ReportVersion:
    if kind not in VALID_KINDS:
        raise ValueError(f"Unknown report kind: {kind}")
    prev = db.query(ReportVersion).filter_by(kind=kind,
                                             scope_type=scope_type,
                                             scope_id=scope_id)\
        .order_by(ReportVersion.version.desc()).first()
    version = (prev.version + 1) if prev else 1
    row = ReportVersion(kind=kind, scope_type=scope_type, scope_id=scope_id,
                        version=version, content_json=_dumps(content),
                        generated_by=generated_by)
    db.add(row)
    db.flush()
    return row


def latest_report(db: Session, *, kind: str, scope_type: str,
                  scope_id: Optional[int] = None) -> Optional[dict]:
    row = db.query(ReportVersion).filter_by(kind=kind, scope_type=scope_type,
                                            scope_id=scope_id)\
        .order_by(ReportVersion.version.desc()).first()
    if row is None:
        return None
    return {"id": row.id, "kind": row.kind, "version": row.version,
            "content": json.loads(row.content_json or "{}"),
            "generated_by": row.generated_by, "created_at": row.created_at}


def list_reports(db: Session, *, kind: Optional[str] = None,
                 scope_type: Optional[str] = None,
                 limit: int = 100) -> list[dict]:
    q = db.query(ReportVersion)
    if kind:
        q = q.filter(ReportVersion.kind == kind)
    if scope_type:
        q = q.filter(ReportVersion.scope_type == scope_type)
    rows = q.order_by(ReportVersion.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "kind": r.kind, "scope_type": r.scope_type,
             "scope_id": r.scope_id, "version": r.version,
             "created_at": r.created_at} for r in rows]


def ai_quality_report(db: Session, *, workspace_id: int,
                      generated_by: Optional[int] = None,
                      quality: Optional[dict] = None,
                      cost: Optional[dict] = None,
                      safety: Optional[dict] = None,
                      reliability: Optional[dict] = None) -> ReportVersion:
    """Assemble an AI quality report from provided quality/cost/safety/
    reliability sections (injected for deterministic composition)."""
    content = {
        "workspace_id": workspace_id,
        "sections": {
            "quality": quality or {},
            "cost": cost or {},
            "safety": safety or {},
            "reliability": reliability or {},
        },
    }
    return create_report(db, kind="ai_quality", scope_type="WORKSPACE",
                         scope_id=workspace_id, content=content,
                         generated_by=generated_by)


def knowledge_health_report(db: Session, *, workspace_id: int,
                            generated_by: Optional[int] = None,
                            freshness: Optional[dict] = None,
                            gaps: Optional[list] = None,
                            conflicts: Optional[list] = None,
                            graph: Optional[dict] = None) -> ReportVersion:
    content = {
        "workspace_id": workspace_id,
        "sections": {
            "freshness": freshness or {},
            "gaps": gaps or [],
            "conflicts": conflicts or [],
            "graph_health": graph or {},
        },
    }
    return create_report(db, kind="knowledge_health", scope_type="WORKSPACE",
                         scope_id=workspace_id, content=content,
                         generated_by=generated_by)


def governance_report(db: Session, *, scope_type: str,
                      scope_id: Optional[int] = None,
                      generated_by: Optional[int] = None,
                      policies: Optional[list] = None,
                      changes: Optional[list] = None,
                      violations: Optional[list] = None,
                      approvals: Optional[list] = None) -> ReportVersion:
    content = {
        "scope_type": scope_type, "scope_id": scope_id,
        "sections": {
            "policies": policies or [],
            "changes": changes or [],
            "violations": violations or [],
            "approvals": approvals or [],
        },
    }
    return create_report(db, kind="governance", scope_type=scope_type,
                         scope_id=scope_id, content=content,
                         generated_by=generated_by)