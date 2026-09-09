"""Entitlement service — plans, feature flags, and quota checks.

Deterministic evaluation; no hard-coded pricing in business logic.
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.billing import Plan, Subscription
from ..models.feature_flag import FeatureFlag

# Default limits when no subscription exists (safe minimal defaults)
DEFAULT_LIMITS = {
    "max_workspaces": 1,
    "max_members": 5,
    "max_documents": 100,
    "max_storage_bytes": 1024 * 1024 * 1024,  # 1 GB
    "max_ai_requests": 100,
    "max_agent_runs": 10,
    "max_workflow_runs": 10,
    "max_api_requests": 1000,
    "max_api_keys": 2,
    "max_webhooks": 2,
}

# Feature flags with sensible defaults
DEFAULT_FEATURES = {
    "advanced_search": True,
    "agents": True,
    "workflows": True,
    "enterprise_sso": False,
    "audit_logs": True,
    "advanced_analytics": False,
    "external_integrations": False,
    "custom_models": False,
}


def ensure_default_plan(db: Session) -> Plan:
    """Create the default FREE plan if none exists."""
    plan = db.query(Plan).filter(Plan.is_default == 1).first()
    if plan:
        return plan
    plan = Plan(
        code="FREE",
        name="Free",
        description="Default plan for new organizations",
        limits_json=json.dumps(DEFAULT_LIMITS),
        features_json=json.dumps(DEFAULT_FEATURES),
        is_default=1,
    )
    db.add(plan)
    db.flush()
    return plan


def get_plan_for_organization(db: Session, organization_id: Optional[int]) -> Plan:
    """Resolve the effective plan for an organization (or the default)."""
    if organization_id is not None:
        subscription = (
            db.query(Subscription)
            .filter(Subscription.organization_id == organization_id)
            .order_by(Subscription.id.desc())
            .first()
        )
        if subscription and subscription.plan:
            return subscription.plan
    return ensure_default_plan(db)


def get_limits(db: Session, organization_id: Optional[int] = None) -> dict:
    """Return effective quota limits for an organization."""
    plan = get_plan_for_organization(db, organization_id)
    limits = dict(DEFAULT_LIMITS)
    limits.update(plan.limits)
    return limits


def get_features(db: Session, organization_id: Optional[int] = None) -> dict:
    """Return effective feature set for an organization."""
    plan = get_plan_for_organization(db, organization_id)
    features = dict(DEFAULT_FEATURES)
    features.update(plan.features)
    return features


def has_feature(
    db: Session,
    feature: str,
    organization_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> bool:
    """Check whether a feature is enabled, considering plan and feature flags.

    Resolution order (most specific wins):
      1. Explicit feature flag rows (WORKSPACE > ORGANIZATION > GLOBAL)
      2. Plan features
    """
    # Workspace-level flag
    flag = _find_flag(db, feature, "WORKSPACE", workspace_id)
    if flag is not None:
        return _flag_enabled(flag, user_id)
    # Organization-level flag
    flag = _find_flag(db, feature, "ORGANIZATION", organization_id)
    if flag is not None:
        return _flag_enabled(flag, user_id)
    # Global flag
    flag = _find_flag(db, feature, "GLOBAL", None)
    if flag is not None:
        return _flag_enabled(flag, user_id)
    return get_features(db, organization_id).get(feature, False)


def check_limit(
    db: Session,
    metric: str,
    current_usage: int,
    organization_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
) -> bool:
    """Check whether current_usage is within the quota for a metric."""
    limits = get_limits(db, organization_id)
    limit = limits.get(metric)
    if limit is None or limit < 0:
        return True  # unlimited
    return current_usage < limit


def remaining_quota(
    db: Session,
    metric: str,
    current_usage: int,
    organization_id: Optional[int] = None,
) -> dict:
    """Return quota usage information for a metric."""
    limits = get_limits(db, organization_id)
    limit = limits.get(metric)
    if limit is None or limit < 0:
        return {"metric": metric, "limit": None, "used": current_usage, "remaining": None, "percent": 0.0}
    percent = round((current_usage / limit) * 100, 1) if limit else 0.0
    return {
        "metric": metric,
        "limit": limit,
        "used": current_usage,
        "remaining": max(limit - current_usage, 0),
        "percent": percent,
    }


def _find_flag(db: Session, name: str, scope_type: str, scope_id: Optional[int]):
    return (
        db.query(FeatureFlag)
        .filter(
            FeatureFlag.name == name,
            FeatureFlag.scope_type == scope_type,
            FeatureFlag.scope_id == scope_id,
        )
        .first()
    )


def _flag_enabled(flag: FeatureFlag, user_id: Optional[int]) -> bool:
    """Deterministic rollout: enabled plus percentage targeting."""
    if not flag.enabled:
        return False
    if flag.rollout_percentage >= 100:
        return True
    if flag.rollout_percentage <= 0:
        return False
    if user_id is None:
        return True  # no targeting info — treat as enabled
    bucket = int(hashlib.md5(f"{flag.name}:{user_id}".encode()).hexdigest(), 16) % 100
    return bucket < flag.rollout_percentage