"""Phase 20 — AI governance 5.0.

Every production AI configuration change must have owner, reason,
evaluation, approval, timestamp, and rollback. Model/tool/data policy
versioning, human-readable governance diffs, and a no-side-effect
governance simulation ("would this AI operation be allowed?").
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import PolicyVersion


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def record_change(db: Session, *, owner_user_id: int, policy_type: str,
                  scope_type: str, scope_id: Optional[int],
                  policy: dict, reason: str,
                  evaluation_ref: Optional[str] = None,
                  approval_ref: Optional[str] = None) -> PolicyVersion:
    """Versioned, governed AI configuration change. Requires an owner and a
    reason; evaluation and approval references are recorded when present."""
    if not owner_user_id:
        raise ValueError("AI change governance requires an owner")
    if not reason.strip():
        raise ValueError("AI change governance requires a reason")
    prev = db.query(PolicyVersion).filter_by(scope_type=scope_type,
                                             scope_id=scope_id)\
        .order_by(PolicyVersion.version.desc()).first()
    version = (prev.version + 1) if prev else 1
    from ..services.policy_intel import policy_diff
    diff = None
    if prev is not None:
        diff = policy_diff(json.loads(prev.policy_json or "{}"), policy)
    row = PolicyVersion(
        scope_type=scope_type, scope_id=scope_id, version=version,
        policy_json=_dumps(policy),
        diff_json=_dumps({"diff": diff,
                          "policy_type": policy_type,
                          "owner_user_id": owner_user_id,
                          "evaluation_ref": evaluation_ref,
                          "approval_ref": approval_ref}),
        actor_user_id=owner_user_id, reason=reason)
    db.add(row)
    db.flush()
    return row


def governance_diff(before_version: int, after_version: int,
                    db: Session, *, scope_type: str,
                    scope_id: Optional[int] = None) -> dict:
    """Human-readable difference between two policy versions."""
    before = db.query(PolicyVersion).filter_by(
        scope_type=scope_type, scope_id=scope_id,
        version=before_version).first()
    after = db.query(PolicyVersion).filter_by(
        scope_type=scope_type, scope_id=scope_id,
        version=after_version).first()
    if before is None or after is None:
        raise KeyError("policy version not found")
    b = json.loads(before.policy_json or "{}")
    a = json.loads(after.policy_json or "{}")
    from ..services.policy_intel import policy_diff
    return {"before_version": before_version,
            "after_version": after_version,
            "diff": policy_diff(b, a)}


def simulate(db: Session, *, operation: dict, policy: dict) -> dict:
    """No-side-effect governance simulation: allowed? approval required?
    blocked? — without mutating anything."""
    from ..services.policy_intel import simulate as policy_simulate
    result = policy_simulate(db, operation=operation, policy=policy)
    return {
        "allowed": result["allowed"],
        "approval_required": result.get("approval_required", False),
        "blocked_reasons": result["denied_reasons"],
        "policy_version": policy.get("_version"),
        "side_effects": "none",
    }


def policy_types() -> dict:
    """Supported governance policy types with their key fields."""
    return {
        "model_policy": ["allowed_models", "denied_models"],
        "tool_policy": ["allowed_tools", "denied_tools"],
        "provider_policy": ["allowed_providers", "denied_providers"],
        "data_policy": ["allowed_regions", "prohibited_regions",
                        "sensitivity_routing", "retention_days"],
        "workflow_policy": ["max_nodes", "allowed_actions",
                            "auto_approval_risk"],
    }


def effective_policy(policies: list[dict]) -> dict:
    """Most-restrictive-wins merge of org → workspace → user policies."""
    merged: dict = {}
    for p in policies:
        for field, value in p.items():
            if isinstance(value, list):
                if field.startswith("allowed_") or field.startswith("denied_"):
                    # intersect allowlists, union denylists
                    if field.startswith("allowed_"):
                        cur = merged.get(field)
                        merged[field] = (set(value) if cur is None
                                         else (set(cur) & set(value)))
                    else:
                        merged[field] = set(merged.get(field, set())) | \
                            set(value)
                else:
                    merged[field] = value
            elif isinstance(value, (int, float)) and \
                    field in ("retention_days", "max_nodes", "max_cost"):
                # most restrictive: min for caps, max for retention
                if field == "retention_days":
                    cur = merged.get(field)
                    merged[field] = (value if cur is None
                                     else max(cur, value))
                else:
                    cur = merged.get(field)
                    merged[field] = (value if cur is None
                                     else min(cur, value))
            else:
                merged[field] = value
    out: dict = {}
    for k, v in merged.items():
        out[k] = sorted(v) if isinstance(v, set) else v
    return out


def change_audit(db: Session, *, scope_type: str,
                 scope_id: Optional[int] = None,
                 limit: int = 50) -> list[dict]:
    """Recent governed changes with owner, reason, and version."""
    rows = db.query(PolicyVersion).filter_by(scope_type=scope_type,
                                             scope_id=scope_id)\
        .order_by(PolicyVersion.created_at.desc()).limit(limit).all()
    out = []
    for r in rows:
        meta = json.loads(r.diff_json or "{}")
        diff = meta.get("diff") or {}
        out.append({
            "version": r.version, "reason": r.reason,
            "owner_user_id": r.actor_user_id,
            "policy_type": meta.get("policy_type"),
            "evaluation_ref": meta.get("evaluation_ref"),
            "approval_ref": meta.get("approval_ref"),
            "changed_keys": [c["key"] for c in diff.get("changes", [])],
            "created_at": r.created_at,
        })
    return out