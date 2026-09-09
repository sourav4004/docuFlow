"""Phase 20 — Policy intelligence.

Immutable versioned policies with human-readable diffs, drift detection
(AI models/tools/data/regions/retention/workflows), impact analysis against
live objects, deterministic conflict detection (uncertain conflicts go to
review), and a side-effect-free simulation: "would this AI operation be
allowed under this policy?".
"""

from __future__ import annotations

import difflib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import PolicyVersion, PolicyImpact

VALID_SCOPES = ("GLOBAL", "ORGANIZATION", "WORKSPACE")


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _policy_text(policy: dict) -> str:
    return _dumps(policy)


def version_policy(db: Session, *, scope_type: str,
                   scope_id: Optional[int], policy: dict,
                   actor_user_id: Optional[int] = None,
                   reason: Optional[str] = None) -> PolicyVersion:
    """Persist a new immutable policy version with a computed diff."""
    if scope_type not in VALID_SCOPES:
        raise ValueError(f"Unknown scope: {scope_type}")
    prev = db.query(PolicyVersion).filter_by(scope_type=scope_type,
                                             scope_id=scope_id)\
        .order_by(PolicyVersion.version.desc()).first()
    version = (prev.version + 1) if prev else 1
    diff = None
    if prev is not None:
        before = json.loads(prev.policy_json or "{}")
        diff = policy_diff(before, policy)
    row = PolicyVersion(
        scope_type=scope_type, scope_id=scope_id, version=version,
        policy_json=_dumps(policy), diff_json=_dumps(diff),
        actor_user_id=actor_user_id, reason=reason)
    db.add(row)
    db.flush()
    return row


def policy_diff(before: dict, after: dict) -> list[dict]:
    """Human-readable structural diff between two policy dicts."""
    changes = []
    keys = sorted(set(before) | set(after))
    for k in keys:
        b = before.get(k)
        a = after.get(k)
        if b == a:
            continue
        changes.append({"key": k, "before": b, "after": a})
    # text-level diff of the full JSON for readability
    text_before = _policy_text(before).splitlines()
    text_after = _policy_text(after).splitlines()
    lines = list(difflib.unified_diff(text_before, text_after, lineterm="",
                                      n=1))
    return {"changes": changes, "unified_diff": lines[:200]}


def latest_policy(db: Session, *, scope_type: str,
                  scope_id: Optional[int] = None) -> Optional[dict]:
    row = db.query(PolicyVersion).filter_by(scope_type=scope_type,
                                            scope_id=scope_id)\
        .order_by(PolicyVersion.version.desc()).first()
    if row is None:
        return None
    return {"id": row.id, "version": row.version,
            "policy": json.loads(row.policy_json or "{}"),
            "diff": json.loads(row.diff_json or "null"),
            "reason": row.reason, "created_at": row.created_at}


def list_versions(db: Session, *, scope_type: str,
                  scope_id: Optional[int] = None,
                  limit: int = 50) -> list[dict]:
    rows = db.query(PolicyVersion).filter_by(scope_type=scope_type,
                                             scope_id=scope_id)\
        .order_by(PolicyVersion.version.desc()).limit(limit).all()
    return [{"id": r.id, "version": r.version, "reason": r.reason,
             "created_at": r.created_at} for r in rows]


def drift_report(db: Session, *, scope_type: str,
                 scope_id: Optional[int] = None) -> dict:
    """Detect which policy dimensions changed between the two most recent
    versions (drift)."""
    rows = db.query(PolicyVersion).filter_by(scope_type=scope_type,
                                             scope_id=scope_id)\
        .order_by(PolicyVersion.version.desc()).limit(2).all()
    if len(rows) < 2:
        return {"drift": False, "reason": "fewer than two versions"}
    latest_row, prev_row = rows[0], rows[1]
    diff = json.loads(latest_row.diff_json or "{}")
    changed_keys = {c["key"] for c in diff.get("changes", [])}
    dimension_map = {
        "allowed_models": "models", "denied_models": "models",
        "allowed_providers": "providers", "tools": "tools",
        "allowed_regions": "regions", "prohibited_regions": "regions",
        "retention_days": "retention", "workflow_policy": "workflows",
        "sensitivity_routing": "data",
    }
    affected = sorted({dimension_map.get(k, "other")
                       for k in changed_keys})
    return {"drift": bool(affected), "affected_dimensions": affected,
            "changed_keys": sorted(changed_keys),
            "from_version": prev_row.version,
            "to_version": latest_row.version}


def impact_analysis(db: Session, *, policy: dict,
                    workspace_id: Optional[int] = None) -> PolicyImpact:
    """Identify objects affected by a policy: providers, models, tools,
    regions, retention, workflows, agents, documents, workspaces."""
    from ..models.document import Document
    allowed_providers = set(policy.get("allowed_providers", []) or [])
    allowed_models = set(policy.get("allowed_models", []) or [])
    allowed_regions = set(policy.get("allowed_regions", []) or [])
    q = db.query(Document)
    if workspace_id is not None:
        q = q.filter(Document.workspace_id == workspace_id)
    doc_count = q.count()
    impact = {
        "documents_in_scope": doc_count,
        "providers_restricted": sorted(allowed_providers) if allowed_providers
        else [],
        "models_restricted": sorted(allowed_models) if allowed_models else [],
        "regions_restricted": sorted(allowed_regions) if allowed_regions
        else [],
        "retention_days": policy.get("retention_days"),
    }
    row = PolicyImpact(policy_version_id=0, impact_json=_dumps(impact))
    db.add(row)
    db.flush()
    return row


def detect_conflicts(policies: list[dict]) -> list[dict]:
    """Deterministic conflict detection across policies. Uncertain conflicts
    are flagged for review rather than auto-resolved."""
    conflicts = []
    allow_models = set()
    deny_models = set()
    for i, p in enumerate(policies):
        for m in (p.get("allowed_models") or []):
            allow_models.add(m)
        for m in (p.get("denied_models") or []):
            deny_models.add(m)
    for m in sorted(allow_models & deny_models):
        conflicts.append({
            "type": "model_conflict", "value": m,
            "review_required": True,
            "detail": "Model is both allowed and denied across policies"})
    # retention conflicts
    retentions = [p.get("retention_days") for p in policies
                  if p.get("retention_days") is not None]
    if len(set(retentions)) > 1:
        conflicts.append({
            "type": "retention_conflict",
            "value": retentions, "review_required": True,
            "detail": "Conflicting retention_days across policies"})
    return conflicts


def simulate(db: Session, *, operation: dict, policy: dict) -> dict:
    """No-side-effect simulation: would this AI operation be allowed under
    the policy?"""
    model = operation.get("model")
    provider = operation.get("provider")
    tool = operation.get("tool")
    region = operation.get("region")
    sensitivity = operation.get("sensitivity", "INTERNAL")
    reasons = []
    denied = []
    allowed_providers = set(policy.get("allowed_providers") or [])
    denied_providers = set(policy.get("denied_providers") or [])
    allowed_models = set(policy.get("allowed_models") or [])
    denied_models = set(policy.get("denied_models") or [])
    allowed_tools = set(policy.get("allowed_tools") or [])
    denied_tools = set(policy.get("denied_tools") or [])
    prohibited_regions = set(policy.get("prohibited_regions") or [])

    if provider:
        if denied_providers and provider in denied_providers:
            denied.append(f"provider {provider} is denied")
        if allowed_providers and provider not in allowed_providers:
            denied.append(f"provider {provider} is not allowed")
    if model:
        if denied_models and model in denied_models:
            denied.append(f"model {model} is denied")
        if allowed_models and model not in allowed_models:
            denied.append(f"model {model} is not allowed")
    if tool:
        if denied_tools and tool in denied_tools:
            denied.append(f"tool {tool} is denied")
        if allowed_tools and tool not in allowed_tools:
            denied.append(f"tool {tool} is not allowed")
    if region and region in prohibited_regions:
        denied.append(f"region {region} is prohibited")
    if sensitivity == "RESTRICTED" and not policy.get(
            "allow_restricted", True):
        denied.append("RESTRICTED operations are disabled by policy")
    allowed = not denied
    if not allowed:
        reasons.append("; ".join(denied))
    return {"allowed": allowed, "denied_reasons": denied,
            "approval_required": policy.get(
                "approval_required_sensitivity", []).count(
                    sensitivity) > 0 and allowed,
            "policy_version": policy.get("_version")}