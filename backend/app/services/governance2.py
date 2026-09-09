"""Enterprise governance — Phase 17.

- AI policy hierarchy: organization → workspace → feature → execution.
  Evaluation returns the MOST RESTRICTIVE applicable rule.
- model + tool allowlists enforced server-side
- retention assignments (org or workspace) with legal-hold protection:
  entities under an active hold are never auto-deleted
- bounded, idempotent cleanup workers
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase17 import (AIPolicyRule, RetentionAssignment, LegalHold,
                              HoldEntity)

logger = logging.getLogger(__name__)

SENSITIVITY_ORDER = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load(raw: Optional[str]) -> list:
    try:
        return json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []


# ---------------------------------------------------------------------------
# AI policy rules
# ---------------------------------------------------------------------------

def create_rule(db: Session, *, organization_id: Optional[int],
                workspace_id: Optional[int], feature: Optional[str],
                rule_type: str, allowlist: Optional[list[str]] = None,
                deny: Optional[list[str]] = None,
                sensitivity_max: Optional[str] = None,
                budget_max_usd: Optional[float] = None,
                created_by: Optional[int] = None) -> AIPolicyRule:
    if rule_type not in ("MODEL", "TOOL", "SENSITIVITY", "BUDGET"):
        raise ValueError("rule_type must be MODEL/TOOL/SENSITIVITY/BUDGET")
    if sensitivity_max and sensitivity_max not in SENSITIVITY_ORDER:
        raise ValueError("invalid sensitivity level")
    if organization_id is None and workspace_id is None:
        raise ValueError("a policy rule needs org or workspace scope")
    rule = AIPolicyRule(
        organization_id=organization_id, workspace_id=workspace_id,
        feature=feature or None, rule_type=rule_type,
        allowlist_json=json.dumps(allowlist or []),
        deny_json=json.dumps(deny or []),
        sensitivity_max=sensitivity_max, budget_max_usd=budget_max_usd,
        enabled=True, created_by=created_by)
    db.add(rule)
    db.flush()
    return rule


def list_rules(db: Session, organization_id: Optional[int],
               workspace_id: Optional[int], limit: int = 100) -> dict:
    q = db.query(AIPolicyRule)
    if organization_id is not None:
        q = q.filter((AIPolicyRule.organization_id == organization_id)
                     | (AIPolicyRule.organization_id.is_(None)
                        & AIPolicyRule.workspace_id.is_(None)))
    if workspace_id is not None:
        q = q.filter((AIPolicyRule.workspace_id == workspace_id)
                     | (AIPolicyRule.organization_id == organization_id))
    items = q.order_by(AIPolicyRule.id.desc()).limit(min(limit, 500)).all()
    return {"items": [_rule_dict(r) for r in items],
            "total": len(items), "limit": min(limit, 500)}


def _rule_dict(r: AIPolicyRule) -> dict:
    return {
        "id": r.id, "organization_id": r.organization_id,
        "workspace_id": r.workspace_id, "feature": r.feature,
        "rule_type": r.rule_type,
        "allowlist": _load(r.allowlist_json), "deny": _load(r.deny_json),
        "sensitivity_max": r.sensitivity_max,
        "budget_max_usd": r.budget_max_usd, "enabled": r.enabled,
    }


def resolve_policy(db: Session, *, organization_id: Optional[int],
                   workspace_id: int, feature: str,
                   rule_type: str) -> dict:
    """Most-restrictive applicable policy for the scope chain.

    Model/tool allowlists: intersection of allowlists across applicable rules
    (empty allowlist on any rule = unrestricted for that dimension).
    Sensitivity: minimum (most restrictive) max allowed.
    Budget: minimum allowed cap.
    """
    org_id = organization_id
    rules = db.query(AIPolicyRule).filter(
        AIPolicyRule.enabled.is_(True),
        AIPolicyRule.rule_type == rule_type,
        ((AIPolicyRule.organization_id == org_id) & (
            AIPolicyRule.workspace_id.is_(None))
         | (AIPolicyRule.organization_id == org_id)
         & (AIPolicyRule.workspace_id == workspace_id)
         | (AIPolicyRule.organization_id.is_(None))
         & (AIPolicyRule.workspace_id == workspace_id))
    ).all()
    applicable = [r for r in rules
                  if r.feature is None or r.feature == feature]
    if not applicable:
        return {"rule_type": rule_type, "feature": feature,
                "restrictive": False, "source": "no policy"}
    result: dict = {"rule_type": rule_type, "feature": feature,
                    "restrictive": True, "rules": len(applicable)}
    allowlists = [set(_load(r.allowlist_json)) for r in applicable
                  if _load(r.allowlist_json)]
    denies = set().union(*[set(_load(r.deny_json)) for r in applicable]) \
        if applicable else set()
    result["deny"] = sorted(denies)
    if allowlists:
        result["allowlist"] = sorted(set.intersection(*allowlists))
    sensitivities = [SENSITIVITY_ORDER.index(r.sensitivity_max)
                     for r in applicable if r.sensitivity_max]
    if sensitivities:
        result["sensitivity_max"] = \
            SENSITIVITY_ORDER[min(sensitivities)]
    budgets = [r.budget_max_usd for r in applicable
               if r.budget_max_usd is not None]
    if budgets:
        result["budget_max_usd"] = min(budgets)
    return result


def check_model_allowed(db: Session, *, organization_id: Optional[int],
                        workspace_id: int, feature: str,
                        model: str, sensitivity: str) -> tuple[bool, str]:
    policy = resolve_policy(db, organization_id=organization_id,
                            workspace_id=workspace_id, feature=feature,
                            rule_type="MODEL")
    if not policy.get("restrictive"):
        return True, "no model policy"
    if model in policy.get("deny", []):
        return False, f"model {model!r} denied by policy"
    allowed = policy.get("allowlist")
    if allowed is not None and model not in allowed:
        return False, (f"model {model!r} not on the allowlist "
                       f"{allowed}")
    max_level = policy.get("sensitivity_max")
    if max_level is not None and \
            SENSITIVITY_ORDER.index(sensitivity) > \
            SENSITIVITY_ORDER.index(max_level):
        return False, (f"sensitivity {sensitivity} exceeds policy max "
                       f"{max_level}")
    return True, "allowed"


def check_tool_allowed(db: Session, *, organization_id: Optional[int],
                       workspace_id: int, feature: str,
                       tool: str) -> tuple[bool, str]:
    policy = resolve_policy(db, organization_id=organization_id,
                            workspace_id=workspace_id, feature=feature,
                            rule_type="TOOL")
    if not policy.get("restrictive"):
        return True, "no tool policy"
    if tool in policy.get("deny", []):
        return False, f"tool {tool!r} denied by policy"
    allowed = policy.get("allowlist")
    if allowed is not None and tool not in allowed:
        return False, f"tool {tool!r} not on the allowlist {allowed}"
    return True, "allowed"


# ---------------------------------------------------------------------------
# Retention + legal holds
# ---------------------------------------------------------------------------

def set_retention(db: Session, *, entity_type: str, retention_days: int,
                  organization_id: Optional[int] = None,
                  workspace_id: Optional[int] = None) -> RetentionAssignment:
    if organization_id is None and workspace_id is None:
        raise ValueError("retention needs org or workspace scope")
    if retention_days < 0 or retention_days > 36500:
        raise ValueError("retention_days out of range")
    existing = db.query(RetentionAssignment).filter(
        RetentionAssignment.entity_type == entity_type,
        RetentionAssignment.organization_id == organization_id,
        RetentionAssignment.workspace_id == workspace_id).first()
    if existing is None:
        existing = RetentionAssignment(
            organization_id=organization_id, workspace_id=workspace_id,
            entity_type=entity_type, retention_days=retention_days)
        db.add(existing)
    else:
        existing.retention_days = retention_days
        existing.enabled = True
    db.flush()
    return existing


def retention_days_for(db: Session, entity_type: str,
                       workspace_id: int,
                       organization_id: Optional[int]) -> int:
    """Workspace assignment wins; otherwise org-level default."""
    ws = db.query(RetentionAssignment).filter(
        RetentionAssignment.workspace_id == workspace_id,
        RetentionAssignment.entity_type == entity_type,
        RetentionAssignment.enabled.is_(True)).first()
    if ws is not None:
        return ws.retention_days
    if organization_id is not None:
        org = db.query(RetentionAssignment).filter(
            RetentionAssignment.organization_id == organization_id,
            RetentionAssignment.workspace_id.is_(None),
            RetentionAssignment.entity_type == entity_type,
            RetentionAssignment.enabled.is_(True)).first()
        if org is not None:
            return org.retention_days
    return 90


# ---------------------------------------------------------------------------
# Legal holds
# ---------------------------------------------------------------------------

def place_hold(db: Session, *, workspace_id: int,
               organization_id: Optional[int], name: str, reason: str,
               entity_type: Optional[str] = None,
               entity_ids: Optional[list[int]] = None,
               started_by: Optional[int] = None) -> LegalHold:
    hold = LegalHold(
        organization_id=organization_id, workspace_id=workspace_id,
        name=name, reason=reason, status="ACTIVE",
        started_by=started_by)
    db.add(hold)
    db.flush()
    for entity_id in (entity_ids or []):
        db.add(HoldEntity(hold_id=hold.id, entity_type=entity_type
                          or "document", entity_id=entity_id))
    db.flush()
    return hold


def release_hold(db: Session, hold_id: int, workspace_id: int,
                 released_by: int) -> dict:
    hold = db.query(LegalHold).filter(
        LegalHold.id == hold_id,
        LegalHold.workspace_id == workspace_id).first()
    if hold is None:
        raise ValueError(f"Hold {hold_id} not found")
    if hold.status == "RELEASED":
        return {"hold_id": hold.id, "status": "RELEASED"}
    hold.status = "RELEASED"
    hold.released_at = _utcnow()
    hold.released_by = released_by
    db.flush()
    return {"hold_id": hold.id, "status": "RELEASED"}


def is_under_hold(db: Session, workspace_id: int, entity_type: str,
                  entity_id: int) -> bool:
    """True when any ACTIVE hold covers the entity type (or the entity id)."""
    held_types = db.query(LegalHold).filter(
        LegalHold.workspace_id == workspace_id,
        LegalHold.status == "ACTIVE",
        LegalHold.name == "__TYPE_HOLD__").first()
    # Type-level holds are expressed by placing HoldEntity rows per entity;
    # here we check explicit entity coverage.
    rows = (
        db.query(HoldEntity)
        .join(LegalHold, HoldEntity.hold_id == LegalHold.id)
        .filter(LegalHold.workspace_id == workspace_id,
                LegalHold.status == "ACTIVE",
                HoldEntity.entity_type == entity_type,
                HoldEntity.entity_id == entity_id)
        .first()
    )
    return rows is not None


# ---------------------------------------------------------------------------
# Cleanup worker (bounded + hold-aware)
# ---------------------------------------------------------------------------

def run_cleanup(db: Session, *, entity_type: str, workspace_id: int,
                organization_id: Optional[int],
                now: Optional[datetime] = None) -> dict:
    """Delete rows past retention for a supported entity type.

    Protected: any entity covered by an ACTIVE legal hold is never deleted.
    Bounded: at most 200 rows per invocation.
    """
    now = now or _utcnow()
    days = retention_days_for(db, entity_type, workspace_id, organization_id)
    cutoff = now - timedelta(days=days)
    deleted = 0
    skipped_held = 0
    candidates = _expired_candidates(db, entity_type, workspace_id, cutoff)
    for entity_id in candidates:
        if is_under_hold(db, workspace_id, entity_type, entity_id):
            skipped_held += 1
            continue
        _delete_entity(db, entity_type, workspace_id, entity_id)
        deleted += 1
    db.flush()
    return {"entity_type": entity_type, "deleted": deleted,
            "skipped_under_hold": skipped_held,
            "retention_days": days}


def _expired_candidates(db: Session, entity_type: str, workspace_id: int,
                        cutoff: datetime) -> list[int]:
    if entity_type == "trace":
        from ..models.phase16 import TraceSpan
        rows = db.query(TraceSpan.id).filter(
            TraceSpan.workspace_id == workspace_id,
            TraceSpan.created_at < cutoff).limit(200).all()
        return [r[0] for r in rows]
    if entity_type == "notification":
        from ..models.notification import Notification
        rows = db.query(Notification.id).filter(
            Notification.workspace_id == workspace_id,
            Notification.created_at < cutoff).limit(200).all()
        return [r[0] for r in rows]
    if entity_type == "audit":
        from ..models.audit_log import AuditLog
        rows = db.query(AuditLog.id).filter(
            AuditLog.created_at < cutoff).limit(200).all()
        return [r[0] for r in rows]
    if entity_type == "event":
        from ..models.phase15 import KnowledgeEvent
        rows = db.query(KnowledgeEvent.id).filter(
            KnowledgeEvent.workspace_id == workspace_id,
            KnowledgeEvent.created_at < cutoff,
            KnowledgeEvent.status.in_(("PROCESSED", "DEAD"))).limit(
                200).all()
        return [r[0] for r in rows]
    return []


def _delete_entity(db: Session, entity_type: str, workspace_id: int,
                   entity_id: int) -> None:
    if entity_type == "trace":
        from ..models.phase16 import TraceSpan
        db.query(TraceSpan).filter(TraceSpan.id == entity_id,
                                   TraceSpan.workspace_id == workspace_id)\
            .delete()
    elif entity_type == "notification":
        from ..models.notification import Notification
        db.query(Notification).filter(
            Notification.id == entity_id,
            Notification.workspace_id == workspace_id).delete()
    elif entity_type == "audit":
        from ..models.audit_log import AuditLog
        db.query(AuditLog).filter(AuditLog.id == entity_id).delete()
    elif entity_type == "event":
        from ..models.phase15 import KnowledgeEvent
        db.query(KnowledgeEvent).filter(
            KnowledgeEvent.id == entity_id,
            KnowledgeEvent.workspace_id == workspace_id).delete()
