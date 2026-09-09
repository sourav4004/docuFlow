"""Report generation 2.0 — reports distinguish FACTS, EVIDENCE, INFERENCES,
RECOMMENDATIONS, and UNCERTAINTIES. The executive knowledge brief aggregates
workspace/organization state with evidence-backed factual items.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import KnowledgeChange, PolicyConflict, KnowledgeGap
from ..models.knowledge import Deadline, KnowledgeInsight
from ..models.ai_action import AIAction
from ..services.health_service import workspace_health, ai_usage_summary
from ..services.artifact_service import create_artifact

REPORT_TEMPLATES = (
    "executive_summary", "weekly_knowledge_report", "risk_report",
    "document_change_report", "collection_health_report", "ai_usage_report",
    "compliance_readiness_report",
)


class ReportError(Exception):
    """Raised on invalid report requests."""


def build_report(
    db: Session,
    workspace_id: int,
    template: str,
    user_id: int,
    date_range_days: int = 30,
    document_ids: Optional[list[int]] = None,
) -> dict:
    """Build a report with explicit FACTS / EVIDENCE / INFERENCES /
    RECOMMENDATIONS / UNCERTAINTIES sections."""
    if template not in REPORT_TEMPLATES:
        raise ReportError(f"Unknown report template: {template}")

    if template == "executive_summary":
        return _executive_brief(db, workspace_id, user_id, date_range_days)
    if template == "weekly_knowledge_report":
        return _weekly_report(db, workspace_id, user_id, date_range_days)
    if template == "risk_report":
        return _risk_report(db, workspace_id)
    if template == "document_change_report":
        return _change_report(db, workspace_id, date_range_days)
    if template == "ai_usage_report":
        return _ai_usage_report(db, workspace_id, date_range_days)
    if template == "compliance_readiness_report":
        return _compliance_report(db, workspace_id)
    # collection health handled via health service
    return _collection_health_report(db, workspace_id, document_ids)


def _executive_brief(db, workspace_id, user_id, days) -> dict:
    health = workspace_health(db, workspace_id)
    deadlines = (
        db.query(Deadline)
        .filter(Deadline.workspace_id == workspace_id, Deadline.status.in_(("UPCOMING", "DUE", "OVERDUE")))
        .order_by(Deadline.due_date.asc())
        .limit(10)
        .all()
    )
    conflicts = (
        db.query(PolicyConflict)
        .filter(PolicyConflict.workspace_id == workspace_id, PolicyConflict.status == "OPEN")
        .limit(10)
        .all()
    )
    gaps = (
        db.query(KnowledgeGap)
        .filter(KnowledgeGap.workspace_id == workspace_id, KnowledgeGap.status == "OPEN")
        .limit(10)
        .all()
    )
    changes = (
        db.query(KnowledgeChange)
        .filter(KnowledgeChange.workspace_id == workspace_id)
        .order_by(KnowledgeChange.created_at.desc())
        .limit(10)
        .all()
    )
    usage = ai_usage_summary(db, workspace_id)

    facts = [
        {"fact": f"{health.get('total_documents', 0)} documents in workspace, {health.get('active_knowledge', 0)} ready",
         "evidence": "workspace health aggregation"},
    ]
    if deadlines:
        facts.append({"fact": f"{len(deadlines)} upcoming deadline(s)", "evidence": "deadline registry"})
    if conflicts:
        facts.append({"fact": f"{len(conflicts)} open policy conflict(s)", "evidence": "policy conflict detector"})
    if changes:
        facts.append({"fact": f"{len(changes)} knowledge change(s) recorded recently", "evidence": "change detector"})

    return {
        "template": "executive_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "FACTS": facts,
        "EVIDENCE": [
            {"type": "deadline", "items": [{"id": d.id, "title": d.title, "due": d.due_date.isoformat()} for d in deadlines]},
            {"type": "conflict", "items": [{"id": c.id, "severity": c.severity, "description": c.description} for c in conflicts]},
            {"type": "change", "items": [{"id": c.id, "type": c.change_type, "summary": c.summary} for c in changes]},
        ],
        "INFERENCES": [
            "Workspace knowledge health is {} — {}".format(
                health.get("avg_health", "n/a"),
                "above average" if (health.get("avg_health") or 0) >= 60 else "below average",
            )
        ],
        "RECOMMENDATIONS": [
            {"action": "Review overdue deadlines", "items": [d.title for d in deadlines if d.status == "OVERDUE"]},
            {"action": "Resolve open policy conflicts", "items": [f"conflict {c.id}" for c in conflicts]},
            {"action": "Address knowledge gaps", "items": [g.title for g in gaps]},
        ],
        "UNCERTAINTIES": [
            "AI usage is estimated; provider invoices may differ.",
            "Health scores are heuristics, not exact measurements.",
        ],
        "ai_usage_30d": usage,
        "metadata": {
            "model": "deterministic",
            "generated_by": user_id,
            "date_range_days": days,
        },
    }


def _weekly_report(db, workspace_id, user_id, days) -> dict:
    brief = _executive_brief(db, workspace_id, user_id, days)
    brief["template"] = "weekly_knowledge_report"
    return brief


def _risk_report(db, workspace_id) -> dict:
    conflicts = (
        db.query(PolicyConflict)
        .filter(PolicyConflict.workspace_id == workspace_id, PolicyConflict.status == "OPEN")
        .limit(50)
        .all()
    )
    gaps = (
        db.query(KnowledgeGap)
        .filter(KnowledgeGap.workspace_id == workspace_id, KnowledgeGap.status == "OPEN")
        .limit(50)
        .all()
    )
    health = workspace_health(db, workspace_id)
    return {
        "template": "risk_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "FACTS": [
            {"fact": f"{len(conflicts)} open policy conflict(s)", "evidence": "policy conflict detector"},
            {"fact": f"{len(gaps)} open knowledge gap(s)", "evidence": "knowledge gap engine"},
            {"fact": f"{health.get('stale_knowledge', 0)} stale document(s)", "evidence": "workspace health"},
        ],
        "EVIDENCE": [
            {"type": "policy_conflict", "items": [{"id": c.id, "severity": c.severity, "description": c.description} for c in conflicts]},
            {"type": "knowledge_gap", "items": [{"id": g.id, "severity": g.severity, "title": g.title} for g in gaps]},
        ],
        "INFERENCES": [
            "Conflicts and gaps concentrated in these areas may indicate policy drift."
        ],
        "RECOMMENDATIONS": [
            {"action": "Prioritize CRITICAL/HIGH conflicts", "items": [f"conflict {c.id}" for c in conflicts if c.severity in ("CRITICAL", "HIGH")]},
            {"action": "Refresh stale documents", "items": [f"{health.get('stale_knowledge', 0)} stale document(s)"]},
        ],
        "UNCERTAINTIES": [
            "Severity classification is rule-based and may misjudge edge cases.",
        ],
    }


def _change_report(db, workspace_id, days) -> dict:
    from datetime import timedelta
    since = datetime.now(timezone.utc) - timedelta(days=days)
    changes = (
        db.query(KnowledgeChange)
        .filter(KnowledgeChange.workspace_id == workspace_id, KnowledgeChange.created_at >= since)
        .order_by(KnowledgeChange.created_at.desc())
        .limit(100)
        .all()
    )
    return {
        "template": "document_change_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "FACTS": [{"fact": f"{len(changes)} knowledge change(s) in the last {days} days", "evidence": "change detector"}],
        "EVIDENCE": [
            {"type": "change", "items": [
                {"id": c.id, "type": c.change_type, "severity": c.severity, "summary": c.summary,
                 "document_id": c.document_id, "created_at": c.created_at.isoformat()}
                for c in changes
            ]},
        ],
        "INFERENCES": [],
        "RECOMMENDATIONS": [
            {"action": "Review HIGH/CRITICAL changes", "items": [f"change {c.id}" for c in changes if c.severity in ("HIGH", "CRITICAL")]}
        ],
        "UNCERTAINTIES": ["Change classification confidence varies with diff size."],
    }


def _ai_usage_report(db, workspace_id, days) -> dict:
    from datetime import timedelta
    from ..services.cost_engine import cost_summary, forecast_spend
    since = datetime.now(timezone.utc) - timedelta(days=days)
    summary = cost_summary(db, workspace_id=workspace_id, since=since)
    forecast = forecast_spend(db, workspace_id=workspace_id, lookback_days=days)
    return {
        "template": "ai_usage_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "FACTS": [
            {"fact": f"{summary['execution_count']} AI execution(s), ${summary['total_cost_usd']:.4f} in the last {days} days", "evidence": "cost engine"},
        ],
        "EVIDENCE": [{"type": "cost_summary", "items": [summary]}],
        "INFERENCES": [f"Projected monthly spend ≈ ${forecast['monthly_estimate_usd']:.4f} (estimate)"],
        "RECOMMENDATIONS": [
            {"action": "Review model mix", "items": list(summary.get("by_model", {}).keys())}
        ],
        "UNCERTAINTIES": [
            "Forecasts are estimates; provider pricing may change.",
        ],
        "forecast": forecast,
    }


def _compliance_report(db, workspace_id) -> dict:
    return {
        "template": "compliance_readiness_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "FACTS": [
            {"fact": "Compliance readiness reflects implemented controls, not certification", "evidence": "platform configuration"},
        ],
        "EVIDENCE": [
            {"type": "control", "items": [
                {"control": "tenant isolation", "status": "implemented"},
                {"control": "audit logging", "status": "implemented"},
                {"control": "data export", "status": "implemented"},
                {"control": "data retention", "status": "implemented"},
                {"control": "access control", "status": "implemented"},
                {"control": "encryption at rest", "status": "deployment dependent"},
                {"control": "SOC 2 / ISO certification", "status": "NOT certified — readiness only"},
            ]},
        ],
        "INFERENCES": [],
        "RECOMMENDATIONS": [
            {"action": "Verify deployment-level encryption and subprocessor documentation"}
        ],
        "UNCERTAINTIES": [
            "Certification status depends on the operator's audit program.",
        ],
    }


def _collection_health_report(db, workspace_id, document_ids) -> dict:
    from ..models.collection import Collection, collection_documents
    if document_ids:
        rows = (
            db.query(collection_documents)
            .join(Collection, Collection.id == collection_documents.c.collection_id)
            .filter(collection_documents.c.document_id.in_(document_ids))
            .all()
        )
        collection_ids = [r.collection_id for r in rows]
    else:
        collection_ids = [c.id for c in db.query(Collection).filter(Collection.workspace_id == workspace_id).limit(100).all()]
    from ..services.health_service import collection_health
    healths = [
        {"collection_id": cid, **collection_health(db, db.query(Collection).filter(Collection.id == cid).first())}
        for cid in collection_ids
    ]
    return {
        "template": "collection_health_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "FACTS": [{"fact": f"{len(healths)} collection(s) assessed", "evidence": "collection health"}],
        "EVIDENCE": [{"type": "collection_health", "items": healths}],
        "INFERENCES": [],
        "RECOMMENDATIONS": [
            {"action": "Review collections below health 60", "items": [h for h in healths if (h.get("health") or 0) < 60]}
        ],
        "UNCERTAINTIES": [],
    }


def save_report_artifact(
    db: Session,
    workspace_id: int,
    user_id: int,
    report: dict,
    name: Optional[str] = None,
) -> dict:
    """Persist a report as a versioned AI artifact."""
    template = report.get("template", "report")
    artifact = create_artifact(
        db, workspace_id=workspace_id, user_id=user_id,
        artifact_type="report",
        name=name or f"{template.replace('_', ' ').title()} — {datetime.now(timezone.utc).date().isoformat()}",
        content=report,
    )
    return {"artifact_id": artifact.id, "version": artifact.version, "name": artifact.name}