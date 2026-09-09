"""AI action risk + approval platform — risk classification 2.0, approval
policy, expiry enforcement, and previews.

Consequential actions ALWAYS require explicit human approval; approvals
expire and an expired approval can never authorize execution. Every decision
is audited with requester, reviewer, decision, timestamp, policy, and an
action hash.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Deterministic risk rules: (condition description, level, predicate)
_RISK_RULES = [
    ("external side effect", "HIGH",
     lambda side_effects: bool(side_effects & {"email", "webhook", "http",
                                                "sms", "external_api"})),
    ("financial impact", "HIGH",
     lambda side_effects: bool(side_effects & {"payment", "invoice",
                                               "refund", "pricing"})),
    ("deletion", "CRITICAL",
     lambda side_effects: bool(side_effects & {"delete", "bulk_delete"})),
    ("bulk scope", "CRITICAL",
     lambda scope_size: scope_size >= 1000),
    ("restricted data", "HIGH",
     lambda sensitivity: sensitivity == "RESTRICTED"),
    ("confidential data", "MEDIUM",
     lambda sensitivity: sensitivity == "CONFIDENTIAL"),
    ("autonomous workflow", "MEDIUM",
     lambda side_effects: bool(side_effects & {"create_workflow",
                                               "activate_workflow"})),
    ("write to shared knowledge", "MEDIUM",
     lambda side_effects: bool(side_effects & {"metadata_update",
                                               "tag_update", "share"})),
]


class ApprovalExpiredError(Exception):
    """Raised when an approval is used after its expiry."""


def classify_action_risk(
    action_type: str,
    data_sensitivity: str = "INTERNAL",
    side_effects: Optional[list] = None,
    scope_size: int = 1,
) -> dict:
    """Deterministic risk classification from configurable rules."""
    effects = set(side_effects or [])
    # Baseline by action type (documented, low-consequence defaults).
    baseline = {
        "summarize": "LOW", "extract": "LOW", "classify": "LOW",
        "tag": "LOW", "search": "LOW", "ask": "LOW", "compare": "LOW",
        "detect_conflict": "LOW", "detect_missing_info": "LOW",
        "generate_report": "LOW", "research": "LOW",
        "metadata_update": "MEDIUM",
        "send_notification": "MEDIUM",
        "request_review": "LOW",
        "create_workflow": "MEDIUM",
        "run_agent": "MEDIUM",
        "share_document": "HIGH",
        "delete_document": "CRITICAL",
        "bulk_delete": "CRITICAL",
    }
    level_idx = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    level = baseline.get(action_type, "MEDIUM")
    hits = []

    def escalate(rule_level: str, reason: str):
        nonlocal level
        if level_idx[rule_level] > level_idx[level]:
            level = rule_level
            hits.append(reason)

    if effects & {"email", "webhook", "http", "sms", "external_api"}:
        escalate("HIGH", "external side effect")
    if effects & {"payment", "invoice", "refund", "pricing"}:
        escalate("HIGH", "financial impact")
    if effects & {"delete"}:
        escalate("CRITICAL", "deletion")
    if effects & {"bulk_delete"}:
        escalate("CRITICAL", "bulk deletion")
    if scope_size >= 1000:
        escalate("CRITICAL", "bulk scope")
    if data_sensitivity == "RESTRICTED":
        escalate("HIGH", "restricted data")
    elif data_sensitivity == "CONFIDENTIAL":
        escalate("MEDIUM", "confidential data")
    if effects & {"create_workflow", "activate_workflow"}:
        escalate("MEDIUM", "autonomous workflow")
    if effects & {"metadata_update", "tag_update", "share"}:
        escalate("MEDIUM", "write to shared knowledge")
    return {"risk_level": level, "rules_applied": hits,
            "action_type": action_type}


def approval_required(risk_level: str, sensitivity: str = "INTERNAL",
                      organization_policy: Optional[str] = None,
                      user_role: Optional[str] = None) -> dict:
    """Policy decision: is human approval required for this action?

    Default policy (deterministic, safe-by-default):
    - LOW               → auto-approve
    - MEDIUM            → approval when data is CONFIDENTIAL/RESTRICTED or an
                          org policy demands it
    - HIGH / CRITICAL   → approval always required
    """
    if risk_level in ("HIGH", "CRITICAL"):
        return {"required": True,
                "policy": "high-risk-actions-require-approval",
                "reasons": [f"action risk level {risk_level}"]}
    if risk_level == "MEDIUM":
        reasons = []
        if sensitivity in ("CONFIDENTIAL", "RESTRICTED"):
            reasons.append(f"data sensitivity {sensitivity}")
        if organization_policy == "strict":
            reasons.append("organization policy is strict")
        if user_role in ("VIEWER",):
            reasons.append("caller role lacks execution rights")
        return {"required": bool(reasons), "policy": "medium-risk-gated",
                "reasons": reasons}
    return {"required": False, "policy": "low-risk-auto",
            "reasons": ["action risk level LOW"]}


def approval_valid(expires_at, now: Optional[datetime] = None) -> bool:
    if expires_at is None:
        return False
    now = now or datetime.now(timezone.utc)
    if isinstance(expires_at, datetime):
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at > now
    return False


def enforce_approval(approval, now: Optional[datetime] = None) -> None:
    """Refuse execution when the approval has expired."""
    if not approval_valid(approval.expires_at, now):
        raise ApprovalExpiredError(
            f"Approval {getattr(approval, 'id', '?')} has expired and can no "
            f"longer authorize execution")


def action_hash(action_type: str, payload: Optional[dict],
                risk_level: str) -> str:
    canonical = json.dumps({"type": action_type, "payload": payload or {},
                            "risk": risk_level}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def audit_decision(db: Session, workspace_id: int, requester_id: int,
                   reviewer_id: int, decision: str, action_type: str,
                   payload: Optional[dict], policy: str, risk_level: str,
                   resource_id: Optional[str] = None) -> None:
    """Audit an approval decision (requester, reviewer, decision, policy,
    action hash, timestamp)."""
    from ..services.audit_service import log_audit_event
    log_audit_event(
        db, event_type="approval", event_action=f"decision:{decision.lower()}",
        user_id=reviewer_id,
        resource_type="ai_approval", resource_id=resource_id,
        details=(f"requester={requester_id} policy={policy} "
                 f"risk={risk_level} action={action_type} "
                 f"action_hash={action_hash(action_type, payload, risk_level)}"),
    )


def action_preview(action_type: str, payload: Optional[dict],
                   data_scope: Optional[list] = None,
                   tools: Optional[list] = None,
                   external_systems: Optional[list] = None,
                   estimated_cost_usd: Optional[float] = None,
                   risk_level: Optional[str] = None,
                   sensitivity: str = "INTERNAL") -> dict:
    """Structured, user-facing action preview — what happens, what data is
    used, tools, external systems, cost, impact. No hidden reasoning."""
    risk = risk_level or classify_action_risk(
        action_type, sensitivity)["risk_level"]
    return {
        "action_type": action_type,
        "what_will_happen": _describe(action_type, payload),
        "data_used": data_scope or [],
        "sensitivity": sensitivity,
        "tools": tools or [],
        "external_systems": external_systems or [],
        "estimated_cost_usd": estimated_cost_usd,
        "risk_level": risk,
        "approval_required": approval_required(
            risk, sensitivity)["required"],
        "action_hash": action_hash(action_type, payload, risk),
    }


def _describe(action_type: str, payload: Optional[dict]) -> str:
    descriptions = {
        "send_notification": "Sends a notification to the listed recipients.",
        "metadata_update": "Updates document metadata fields.",
        "delete_document": "Deletes the specified document (irreversible).",
        "bulk_delete": "Deletes a batch of documents (irreversible).",
        "share_document": "Shares the document with the listed users/teams.",
        "create_workflow": "Creates a workflow draft (remains DRAFT).",
        "activate_workflow": "Activates a validated workflow.",
        "run_agent": "Runs an agent plan against the stated data scope.",
        "generate_report": "Generates a report artifact from the stated data.",
        "send_email": "Sends an email to the listed recipients.",
    }
    return descriptions.get(
        action_type,
        f"Executes an AI '{action_type}' action with the provided inputs.")
