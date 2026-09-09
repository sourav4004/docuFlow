"""Phase 19 — AI action platform 3.0.

Risk engine 2.0 (sensitivity, external side effects, destructiveness,
affected-resource count, reversibility, tenant scope), approval policy
engine (auto-allow low risk, approval for medium/high, critical blocked —
never granted merely because an AI requested it), action previews, approval
expiration enforcement, and complete action audit outcomes.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SENSITIVITY_WEIGHT = {"PUBLIC": 0.0, "INTERNAL": 0.2, "CONFIDENTIAL": 0.5,
                      "RESTRICTED": 1.0}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    """SQLite returns naive datetimes even for timezone-aware columns."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# ---------------------------------------------------------------------------
# Risk engine 2.0
# ---------------------------------------------------------------------------

def risk_score(*, sensitivity: str = "INTERNAL",
               external_side_effect: bool = False,
               destructive: bool = False,
               affected_resources: int = 1,
               reversible: bool = True,
               cross_tenant: bool = False) -> dict:
    """Deterministic risk classification (LOW/MEDIUM/HIGH/CRITICAL)."""
    score = 0.0
    score += SENSITIVITY_WEIGHT.get(sensitivity.upper(), 0.2) * 0.3
    if external_side_effect:
        score += 0.25
    if destructive:
        score += 0.3
    if affected_resources >= 50:
        score += 0.15
    elif affected_resources >= 10:
        score += 0.08
    if not reversible:
        score += 0.1
    if cross_tenant:
        score += 0.2
    score = round(min(1.0, score), 3)
    if score >= 0.8:
        level = "CRITICAL"
    elif score >= 0.5:
        level = "HIGH"
    elif score >= 0.25:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"score": score, "level": level,
            "factors": {"sensitivity": sensitivity,
                        "external_side_effect": external_side_effect,
                        "destructive": destructive,
                        "affected_resources": affected_resources,
                        "reversible": reversible,
                        "cross_tenant": cross_tenant}}


def approval_policy(risk: dict) -> dict:
    """Auto-allow LOW; approval for MEDIUM/HIGH; CRITICAL is blocked."""
    level = risk["level"]
    if level == "CRITICAL":
        return {"action": "BLOCK",
                "reason": "critical operations are never granted even at AI "
                          "request"}
    if level in ("MEDIUM", "HIGH"):
        return {"action": "REQUIRE_APPROVAL",
                "reason": f"{level} risk requires explicit human approval"}
    return {"action": "ALLOW", "reason": "low-risk action auto-allowed"}


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

def action_preview(*, action: str, target: str,
                   affected_resources: int = 1,
                   sensitivity: str = "INTERNAL",
                   destructive: bool = False,
                   external_side_effect: bool = False,
                   estimated_cost: Optional[float] = None,
                   reversible: bool = True) -> dict:
    """User-facing preview: target, operation, resources, risk, expected
    effect, required approval. Never hidden reasoning."""
    risk = risk_score(sensitivity=sensitivity, destructive=destructive,
                      external_side_effect=external_side_effect,
                      affected_resources=affected_resources,
                      reversible=reversible)
    policy = approval_policy(risk)
    return {
        "action": action, "target": target,
        "affected_resources": affected_resources,
        "risk": risk, "expected_effect": (
            f"{action} on {target} affecting {affected_resources} "
            "resource(s)"),
        "estimated_cost": estimated_cost,
        "required_approval": policy["action"] == "REQUIRE_APPROVAL",
        "blocked": policy["action"] == "BLOCK",
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Approval lifecycle
# ---------------------------------------------------------------------------

def request_approval(db: Session, *, execution_id: str, workspace_id: int,
                     user_id: int, action: str, risk_level: str,
                     reason: Optional[str] = None,
                     affected_resources: Optional[list] = None,
                     proposed_parameters: Optional[dict] = None,
                     expires_in_minutes: int = 60) -> dict:
    """Create a durable approval request (never auto-approved)."""
    from ..models.ai_execution import AIApproval
    row = AIApproval(
        id=str(uuid.uuid4())[:32], execution_id=execution_id,
        workspace_id=workspace_id, user_id=user_id, action=action,
        reason=reason, risk_level=risk_level,
        affected_resources_json=json.dumps(affected_resources or [],
                                            default=str),
        proposed_parameters_json=json.dumps(proposed_parameters or {},
                                             default=str),
        status="pending",
        expires_at=_utcnow() + timedelta(minutes=max(1,
                                                     expires_in_minutes)))
    db.add(row)
    db.flush()
    return {"approval_id": row.id, "status": "pending",
            "expires_at": row.expires_at}


def approve_action(db: Session, approval_id: str, *, approved_by: int,
                   rejection_reason: Optional[str] = None) -> dict:
    from ..models.ai_execution import AIApproval
    row = db.query(AIApproval).get(approval_id)
    if row is None:
        raise KeyError("approval not found")
    if row.status != "pending":
        return {"status": "ALREADY_DECIDED", "approval_id": row.id,
                "decision": row.status}
    now = _utcnow()
    expires_at = _as_utc(row.expires_at)
    if expires_at is not None and expires_at < now:
        row.status = "expired"
        db.flush()
        return {"status": "EXPIRED",
                "reason": "approval expired — cannot execute"}
    if rejection_reason:
        row.status = "rejected"
        row.approved_by = approved_by
        row.rejection_reason = rejection_reason
        row.approved_at = now
    else:
        row.status = "approved"
        row.approved_by = approved_by
        row.approved_at = now
    db.flush()
    return {"status": row.status, "approval_id": row.id,
            "approved_by": approved_by}


def approval_valid(db: Session, approval_id: str) -> dict:
    """Expired approvals cannot execute. Idempotent, race-safe read."""
    from ..models.ai_execution import AIApproval
    row = db.query(AIApproval).get(approval_id)
    if row is None:
        return {"valid": False, "reason": "approval not found"}
    now = _utcnow()
    expires_at = _as_utc(row.expires_at)
    if expires_at is not None and expires_at < now:
        if row.status == "pending":
            row.status = "expired"
            db.flush()
        return {"valid": False, "reason": "approval expired"}
    if row.status != "approved":
        return {"valid": False, "reason": f"approval {row.status}"}
    return {"valid": True, "approval_id": row.id,
            "approved_by": row.approved_by, "approved_at": row.approved_at}


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def action_audit(*, requester: Optional[int], approver: Optional[int],
                 action: str, target: str, policy: str,
                 outcome: str, request_id: Optional[str] = None) -> dict:
    """Structured action audit record (requester/approver/action/target/
    policy/timestamp/outcome)."""
    return {"requester": requester, "approver": approver, "action": action,
            "target": target, "policy": policy, "outcome": outcome,
            "timestamp": _utcnow().isoformat(),
            "request_id": request_id,
            "note": "audit events are written by the caller through the "
                    "audit service"}
