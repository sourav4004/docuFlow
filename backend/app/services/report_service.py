"""AI report builder — reusable report generation with templates.

Reports are saved as versioned AI artifacts with sources and generation
metadata. Recommendations are clearly separated from factual findings.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.ai_execution import AIArtifact
from ..models.document import Document
from ..services.health_service import workspace_health, ai_usage_summary
from ..services.audit_service import log_audit_event

REPORT_TEMPLATES = {
    "executive_summary": {
        "name": "Executive Summary",
        "sections": ["overview", "findings", "recommendations"],
    },
    "weekly_knowledge": {
        "name": "Weekly Knowledge Report",
        "sections": ["knowledge_changes", "document_activity", "ai_usage", "deadlines", "conflicts", "health"],
    },
    "risk_report": {
        "name": "Risk Report",
        "sections": ["risks", "evidence", "recommendations"],
    },
    "document_change": {
        "name": "Document Change Report",
        "sections": ["changes", "impact"],
    },
    "collection_health": {
        "name": "Collection Health Report",
        "sections": ["health", "recommendations"],
    },
    "ai_usage": {
        "name": "AI Usage Report",
        "sections": ["usage", "cost"],
    },
    "compliance_readiness": {
        "name": "Compliance Readiness Report",
        "sections": ["controls", "gaps", "recommendations"],
    },
}


class UnknownTemplateError(Exception):
    """Raised for unknown report templates."""


def list_templates() -> list[dict]:
    return [
        {"template": key, **value}
        for key, value in REPORT_TEMPLATES.items()
    ]


def generate_report(
    db: Session,
    workspace_id: int,
    user_id: int,
    template: str,
    organization_id: Optional[int] = None,
    document_ids: Optional[list[int]] = None,
) -> dict:
    """Generate a report from a template (deterministic, evidence-backed)."""
    if template not in REPORT_TEMPLATES:
        raise UnknownTemplateError(f"Unknown report template: {template}")

    health = workspace_health(db, workspace_id)
    usage = ai_usage_summary(db, workspace_id)
    now = datetime.now(timezone.utc)

    findings: list[dict] = []
    recommendations: list[dict] = []

    if template == "weekly_knowledge":
        findings.append({"type": "knowledge", "detail": f"{health['total_documents']} documents, {health['active_knowledge']} ready", "source": "workspace_health"})
        findings.append({"type": "ai_usage", "detail": f"{usage['ai_executions_30d']} AI executions in the last 30 days", "source": "usage"})
        findings.append({"type": "deadlines", "detail": _deadline_summary(db, workspace_id), "source": "deadlines"})
    elif template == "ai_usage":
        findings.append({"type": "usage", "detail": f"{usage['ai_executions_30d']} executions, ${usage['ai_cost_30d']} estimated cost (30d)", "source": "usage"})
        findings.append({"type": "success_rate", "detail": f"Success rate: {usage['success_rate'] if usage['success_rate'] is not None else 'n/a'}", "source": "usage"})
    elif template == "collection_health":
        from ..models.collection import Collection
        collections = db.query(Collection).filter(Collection.workspace_id == workspace_id).limit(50).all()
        from ..services.health_service import collection_health
        for c in collections:
            info = collection_health(db, c)
            findings.append({"type": "collection", "detail": f"'{c.name}': {info['document_count']} documents", "source": "collection_health"})
            for rec in info["recommendations"]:
                recommendations.append({"type": "recommendation", "detail": f"[{c.name}] {rec}", "source": "collection_health"})
    elif template in ("risk_report", "compliance_readiness"):
        gaps = _knowledge_gaps(db, workspace_id)
        for gap in gaps:
            findings.append({"type": "gap", "detail": gap["detail"], "source": gap["type"]})
        if not gaps:
            findings.append({"type": "gaps", "detail": "No known knowledge gaps detected", "source": "knowledge_gap"})
    else:  # executive_summary, document_change
        findings.append({"type": "overview", "detail": f"Workspace contains {health['total_documents']} documents", "source": "workspace_health"})
        findings.append({"type": "health", "detail": f"Average health: {health['avg_health'] if health['avg_health'] is not None else 'n/a'}", "source": "workspace_health"})

    if health["processing_failures"] > 0:
        findings.append({"type": "failures", "detail": f"{health['processing_failures']} processing failures", "source": "workspace_health"})
        recommendations.append({"type": "recommendation", "detail": "Investigate failed document processing", "source": "workspace_health"})
    if health["duplicate_rate"] and health["duplicate_rate"] > 5:
        recommendations.append({"type": "recommendation", "detail": "Review potential duplicate documents", "source": "workspace_health"})

    report = {
        "title": REPORT_TEMPLATES[template]["name"],
        "template": template,
        "executive_summary": _executive_summary(findings),
        "findings": findings,
        "recommendations": recommendations,  # clearly separated from facts
        "generated_at": now.isoformat(),
        "workspace_id": workspace_id,
        "source_documents": document_ids or [],
        "schema_version": "1.0",
    }

    _save_report_artifact(db, workspace_id, user_id, organization_id, template, report)
    return report


def _executive_summary(findings: list[dict]) -> str:
    if not findings:
        return "No findings to report."
    return " | ".join(f["detail"] for f in findings[:3])


def _deadline_summary(db: Session, workspace_id: int) -> str:
    from ..models.knowledge import Deadline
    upcoming = (
        db.query(Deadline)
        .filter(Deadline.workspace_id == workspace_id, Deadline.status == "UPCOMING")
        .count()
    )
    overdue = (
        db.query(Deadline)
        .filter(Deadline.workspace_id == workspace_id, Deadline.status == "OVERDUE")
        .count()
    )
    return f"{upcoming} upcoming, {overdue} overdue"


def _knowledge_gaps(db: Session, workspace_id: int) -> list[dict]:
    from ..services.knowledge_gap import detect_policy_gaps, detect_entity_gaps
    try:
        return detect_policy_gaps(db, workspace_id) + detect_entity_gaps(db, workspace_id)
    except Exception:
        return []


def _save_report_artifact(
    db: Session,
    workspace_id: int,
    user_id: int,
    organization_id: Optional[int],
    template: str,
    report: dict,
) -> AIArtifact:
    artifact = AIArtifact(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        execution_id=None,
        user_id=user_id,
        artifact_type="report",
        name=report["title"],
        content_json=report,
        version=1,
        source_document_ids_json=json.dumps(report["source_documents"]),
        status="ai_generated",
    )
    db.add(artifact)
    db.flush()
    log_audit_event(
        db,
        event_type="report",
        event_action="generate",
        user_id=user_id,
        resource_type="ai_artifact",
        resource_id=artifact.id,
        details=f"Report '{template}' generated",
    )
    return artifact