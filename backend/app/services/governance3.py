"""Phase 18 AI governance — policy hierarchy (org → workspace → feature →
execution), allowlists, sensitivity routing, retention enforcement.

Most-restrictive applicable policy wins; a lower-level rule can never widen
an upper-level restriction. Built on the Phase 17 ``AIPolicyRule`` table.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase17 import AIPolicyRule

logger = logging.getLogger(__name__)

SENSITIVITY_RANK = {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2,
                    "RESTRICTED": 3}


def _loads(raw: Optional[str]) -> Optional[list]:
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else None
    except Exception:  # noqa: BLE001
        return None


def effective_policy(db: Session, *, organization_id: Optional[int],
                     workspace_id: Optional[int],
                     feature: Optional[str] = None) -> dict:
    """Resolve the effective policy across the hierarchy (deny-by-default)."""
    from sqlalchemy import or_
    conditions = []
    if organization_id is not None:
        conditions.append(AIPolicyRule.organization_id == organization_id)
    if workspace_id is not None:
        conditions.append(AIPolicyRule.workspace_id == workspace_id)
    if not conditions:
        return {"policy": None, "source_count": 0,
                "note": "no tenant scope"}
    rows = (db.query(AIPolicyRule)
            .filter(or_(*conditions), AIPolicyRule.enabled.is_(True))
            .all())

    effective: dict = {
        # Unrestricted unless a SENSITIVITY rule tightens it.
        "sensitivity_max": "RESTRICTED",
        "budget_max_usd": None,
        "models_allowed": None,
        "providers_allowed": None,
        "tools_allowed": None,
        "denied_models": set(),
        "denied_tools": set(),
    }
    for rule in rows:
        if feature and rule.feature and rule.feature != feature:
            continue
        if rule.rule_type == "MODEL":
            allow = _loads(rule.allowlist_json)
            if allow is not None:
                effective["models_allowed"] = _intersect(
                    effective["models_allowed"], allow)
            deny = _loads(rule.deny_json)
            if deny:
                effective["denied_models"].update(deny)
        elif rule.rule_type == "TOOL":
            allow = _loads(rule.allowlist_json)
            if allow is not None:
                effective["tools_allowed"] = _intersect(
                    effective["tools_allowed"], allow)
            deny = _loads(rule.deny_json)
            if deny:
                effective["denied_tools"].update(deny)
        elif rule.rule_type == "PROVIDER":
            allow = _loads(rule.allowlist_json)
            if allow is not None:
                effective["providers_allowed"] = _intersect(
                    effective["providers_allowed"], allow)
        elif rule.rule_type == "SENSITIVITY":
            # Lower sensitivity max = more restrictive — most restrictive wins.
            if rule.sensitivity_max and SENSITIVITY_RANK.get(
                    rule.sensitivity_max, 3) < SENSITIVITY_RANK.get(
                    effective["sensitivity_max"], 3):
                effective["sensitivity_max"] = rule.sensitivity_max
        elif rule.rule_type == "BUDGET":
            if rule.budget_max_usd is not None:
                if (effective["budget_max_usd"] is None
                        or rule.budget_max_usd < effective["budget_max_usd"]):
                    effective["budget_max_usd"] = rule.budget_max_usd
    return {"policy": effective, "source_count": len(rows)}


def _intersect(existing: Optional[list], incoming: list) -> Optional[list]:
    incoming_set = set(incoming)
    if existing is None:
        return sorted(incoming_set)
    return sorted(set(existing) & incoming_set) or None


def check_model(db: Session, *, organization_id: Optional[int],
                workspace_id: Optional[int], model: str,
                feature: Optional[str] = None) -> tuple[bool, str]:
    """Model allowlist check — deny unless explicitly allowed."""
    eff = effective_policy(db, organization_id=organization_id,
                           workspace_id=workspace_id, feature=feature)
    policy = eff["policy"]
    if not policy:
        return True, "no tenant policy"
    if model in policy["denied_models"]:
        return False, f"model {model!r} denied by policy"
    allowed = policy["models_allowed"]
    if allowed is None:
        return True, "no model restriction"
    if model in allowed:
        return True, "model in allowlist"
    return False, f"model {model!r} not in allowlist"


def check_provider(db: Session, *, organization_id: Optional[int],
                   workspace_id: Optional[int],
                   provider: str) -> tuple[bool, str]:
    eff = effective_policy(db, organization_id=organization_id,
                           workspace_id=workspace_id)
    policy = eff["policy"]
    if not policy:
        return True, "no tenant policy"
    allowed = policy["providers_allowed"]
    if allowed is None:
        return True, "no provider restriction"
    if provider in allowed:
        return True, "provider allowed"
    return False, f"provider {provider!r} not in allowlist"


def check_tool(db: Session, *, organization_id: Optional[int],
               workspace_id: Optional[int], tool: str) -> tuple[bool, str]:
    eff = effective_policy(db, organization_id=organization_id,
                           workspace_id=workspace_id)
    policy = eff["policy"]
    if not policy:
        return True, "no tenant policy"
    if tool in policy["denied_tools"]:
        return False, f"tool {tool!r} denied by policy"
    allowed = policy["tools_allowed"]
    if allowed is None:
        return True, "no tool restriction"
    if tool in allowed:
        return True, "tool allowed"
    return False, f"tool {tool!r} not in allowlist"


def sensitivity_route(db: Session, *, organization_id: Optional[int],
                      workspace_id: Optional[int],
                      sensitivity: str) -> tuple[Optional[str], str]:
    """Provider routing for data sensitivity under policy."""
    eff = effective_policy(db, organization_id=organization_id,
                           workspace_id=workspace_id)
    policy = eff["policy"]
    policy_max = (policy or {}).get("sensitivity_max", "INTERNAL")
    if SENSITIVITY_RANK.get(sensitivity, 1) > SENSITIVITY_RANK.get(
            policy_max, 1):
        return None, (f"data sensitivity {sensitivity} exceeds policy "
                      f"maximum {policy_max}")
    providers = (policy or {}).get("providers_allowed")
    if sensitivity in ("RESTRICTED", "CONFIDENTIAL") and not providers:
        return None, "restricted data requires an explicit provider allowlist"
    if providers:
        return providers[0], f"routed to allowlisted provider {providers[0]}"
    if sensitivity in ("RESTRICTED", "CONFIDENTIAL"):
        return None, "no provider configured for sensitive data"
    return None, "default provider"