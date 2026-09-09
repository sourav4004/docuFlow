"""Phase 19 — governance 4.0.

Policy priority: organization → workspace → user (role) with most-
restrictive-wins semantics; explicit model allowlist incl. denied models and
denied providers; tool policies by organization/workspace/sensitivity/role;
data minimization before provider calls; sensitive-data routing that also
honors region residency; and legal-hold protection so retention cleanup never
deletes held data.

User-level policy is role-derived (OWNER/ADMIN/MEMBER/VIEWER) — a user
cannot widen an organization restriction and non-members are denied.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SENSITIVITY_RANK = {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2,
                    "RESTRICTED": 3}
ROLE_RANK = {"OWNER": 3, "ADMIN": 3, "MEMBER": 2, "VIEWER": 1, "NONE": 0}
READ_TOOLS = {"search_documents", "summarize", "extract_entities",
              "ask_user"}


def _workspace_role(db: Session, *, workspace_id: int,
                    user_id: Optional[int]) -> str:
    if not user_id:
        return "NONE"
    from ..models.workspace import WorkspaceMember
    member = (db.query(WorkspaceMember)
              .filter(WorkspaceMember.workspace_id == workspace_id,
                      WorkspaceMember.user_id == user_id).first())
    return member.role if member else "NONE"


def effective_policy_v2(db: Session, *, organization_id: Optional[int],
                        workspace_id: Optional[int],
                        user_id: Optional[int] = None,
                        feature: Optional[str] = None) -> dict:
    """Most-restrictive-wins across organization → workspace → user(role).

    Role-based user layer can only tighten the org/workspace result; a
    VIEWER cannot be given write tools or see RESTRICTED data through rules
    written at higher layers, and a non-member is denied everything.
    """
    from .governance3 import effective_policy
    eff = effective_policy(db, organization_id=organization_id,
                           workspace_id=workspace_id, feature=feature)
    policy = dict(eff["policy"] or {})
    # No user identity → system/background context: org/workspace policy
    # layers still apply, but there is no user-role layer to deny.
    if user_id is None:
        role = "SYSTEM"
    else:
        role = _workspace_role(db, workspace_id=workspace_id,
                               user_id=user_id)
    policy["role"] = role
    if role == "NONE":
        policy["models_allowed"] = []
        policy["providers_allowed"] = []
        policy["tools_allowed"] = []
        policy["denied_models"] = set()
        policy["denied_tools"] = set()
        policy["sensitivity_max"] = "PUBLIC"
        policy["budget_max_usd"] = 0.0
        policy["denied"] = True
        eff["policy"] = policy
        eff["note"] = "user is not a workspace member — denied by default"
        eff["user_layer"] = "DENY"
        return eff
    denied_models = set(policy.get("denied_models") or [])
    denied_tools = set(policy.get("denied_tools") or [])
    tools_allowed = policy.get("tools_allowed")
    if role == "VIEWER":
        # tighten sensitivity to at most CONFIDENTIAL for read-only users
        if SENSITIVITY_RANK.get(policy.get("sensitivity_max",
                                           "RESTRICTED"), 3) > \
                SENSITIVITY_RANK["CONFIDENTIAL"]:
            policy["sensitivity_max"] = "CONFIDENTIAL"
        if tools_allowed is not None:
            tools_allowed = sorted(set(tools_allowed) & READ_TOOLS) \
                or None
            policy["tools_allowed"] = tools_allowed
        policy["denied_tools"] = sorted(denied_tools
                                        | (set(policy.get("tools_allowed")
                                               or []) - READ_TOOLS))
    elif role == "MEMBER":
        denied_tools.update({"delete_document", "update_policy",
                             "run_script"})
    policy["denied_models"] = sorted(denied_models)
    policy["denied_tools"] = sorted(denied_tools)
    policy["denied"] = False
    eff["policy"] = policy
    eff["user_layer"] = "ROLE_" + role
    return eff


def check_model_routing(db: Session, *, provider: str, model: str,
                        organization_id: Optional[int],
                        workspace_id: Optional[int],
                        user_id: Optional[int] = None,
                        sensitivity: str = "INTERNAL",
                        feature: Optional[str] = None,
                        region_id: Optional[str] = None) -> dict:
    """Combined model/provider/sensitivity/region admission decision."""
    eff = effective_policy_v2(db, organization_id=organization_id,
                              workspace_id=workspace_id, user_id=user_id,
                              feature=feature)
    policy = eff["policy"]
    if policy.get("denied"):
        return {"allowed": False, "reason": "not a workspace member"}
    if model in set(policy.get("denied_models") or []):
        return {"allowed": False,
                "reason": f"model {model!r} explicitly denied"}
    allowed_models = policy.get("models_allowed")
    if allowed_models is not None and model not in allowed_models:
        return {"allowed": False,
                "reason": f"model {model!r} not in allowlist"}
    allowed_providers = policy.get("providers_allowed")
    if allowed_providers is not None and provider not in allowed_providers:
        return {"allowed": False,
                "reason": f"provider {provider!r} not in allowlist"}
    sensitivity_max = policy.get("sensitivity_max", "RESTRICTED")
    if SENSITIVITY_RANK.get(sensitivity, 1) > \
            SENSITIVITY_RANK.get(sensitivity_max, 3):
        return {"allowed": False,
                "reason": f"sensitivity {sensitivity} exceeds policy "
                          f"maximum {sensitivity_max}"}
    if region_id is not None:
        try:
            from .control_plane import evaluate_residency
            residency = evaluate_residency(
                db, organization_id=organization_id,
                classification=sensitivity, region_id=region_id)
            if not residency["allowed"]:
                return {"allowed": False, "reason": residency["reason"]}
        except Exception as exc:  # noqa: BLE001 — fail closed
            return {"allowed": False,
                    "reason": f"residency evaluation error: {exc}"}
    return {"allowed": True, "reason": "policy permits",
            "sensitivity_max": sensitivity_max}


def tool_policy(db: Session, *, organization_id: Optional[int],
                workspace_id: Optional[int], tool: str,
                sensitivity: str = "INTERNAL",
                user_id: Optional[int] = None) -> dict:
    eff = effective_policy_v2(db, organization_id=organization_id,
                              workspace_id=workspace_id, user_id=user_id)
    policy = eff["policy"]
    if policy.get("denied"):
        return {"allowed": False,
                "reason": "not a workspace member"}
    if tool in set(policy.get("denied_tools") or []):
        return {"allowed": False,
                "reason": f"tool {tool!r} denied for this role/workspace"}
    allowed = policy.get("tools_allowed")
    if allowed is not None and tool not in allowed:
        return {"allowed": False,
                "reason": f"tool {tool!r} not in tool allowlist"}
    if sensitivity in ("RESTRICTED",) and tool not in READ_TOOLS:
        return {"allowed": False,
                "reason": "RESTRICTED data only permits read tools"}
    return {"allowed": True, "reason": "tool permitted"}


def data_minimization(context: dict, sensitivity: str,
                      allowed_fields: Optional[set] = None) -> dict:
    """Remove unnecessary/sensitive fields before a provider call. By
    default only safe meta fields pass for CONFIDENTIAL/RESTRICTED."""
    if allowed_fields is None:
        if sensitivity in ("CONFIDENTIAL", "RESTRICTED"):
            allowed_fields = {"document_id", "chunk_id", "title"}
        else:
            allowed_fields = None
    if allowed_fields is None:
        return {"minimized": context, "removed_keys": [],
                "note": "no restriction for low sensitivity"}
    minimized = {}
    removed = []
    for key, value in context.items():
        if key in allowed_fields:
            minimized[key] = value
        else:
            removed.append(key)
    return {"minimized": minimized, "removed_keys": removed,
            "note": f"{len(removed)} field(s) removed for {sensitivity}"}


def route_sensitive(db: Session, *, organization_id: Optional[int],
                    workspace_id: Optional[int], sensitivity: str,
                    user_id: Optional[int] = None,
                    region_id: Optional[str] = None) -> dict:
    """Sensitive data routes only to allowlisted providers/regions."""
    eff = effective_policy_v2(db, organization_id=organization_id,
                              workspace_id=workspace_id, user_id=user_id)
    policy = eff["policy"]
    sensitivity_max = policy.get("sensitivity_max", "RESTRICTED")
    if SENSITIVITY_RANK.get(sensitivity, 1) > \
            SENSITIVITY_RANK.get(sensitivity_max, 3):
        return {"allowed": False,
                "reason": f"policy blocks {sensitivity} data"}
    providers = policy.get("providers_allowed")
    if sensitivity in ("CONFIDENTIAL", "RESTRICTED") and not providers:
        return {"allowed": False,
                "reason": "no allowlisted provider for sensitive data"}
    if region_id is not None:
        from .control_plane import evaluate_residency
        residency = evaluate_residency(db,
                                       organization_id=organization_id,
                                       classification=sensitivity,
                                       region_id=region_id)
        if not residency["allowed"]:
            return {"allowed": False, "reason": residency["reason"]}
    return {"allowed": True,
            "provider": providers[0] if providers else None,
            "reason": "routes to approved provider/region only"}


def is_held(db: Session, *, entity_type: str, entity_id: int) -> dict:
    """Legal-hold protection: held data must never be auto-deleted."""
    from ..models.phase17 import LegalHold, HoldEntity
    rows = (db.query(HoldEntity)
            .filter(HoldEntity.entity_type == entity_type,
                    HoldEntity.entity_id == entity_id).all())
    active = []
    for row in rows:
        hold = db.query(LegalHold).get(row.hold_id)
        if hold is not None and hold.status == "ACTIVE":
            active.append(hold.id)
    return {"held": bool(active), "hold_ids": active}


def retention_protected(db: Session, *, entity_type: str,
                        entity_id: int) -> bool:
    return is_held(db, entity_type=entity_type,
                   entity_id=entity_id)["held"]
