"""Worker priority and fair-scheduling helpers.

Provides deterministic priority ordering and tenant-aware concurrency
limits so no single workspace can monopolize worker capacity.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

PRIORITIES = ("CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND")
PRIORITY_RANK = {"CRITICAL": 4, "HIGH": 3, "NORMAL": 2, "LOW": 1, "BACKGROUND": 0}

# Default tenant-aware concurrency limits (configurable per deployment)
DEFAULT_LIMITS = {
    "concurrent_ai_calls": 10,
    "concurrent_agents": 4,
    "concurrent_workflows": 6,
    "concurrent_documents": 6,
    "per_workspace_cap": 3,
    "per_organization_cap": 6,
}


@dataclass
class ScheduledJob:
    """A scheduled background job with priority and tenant context."""
    job_id: str
    queue: str  # DOCUMENT_PROCESSING, AI_TASKS, AGENTS, WORKFLOWS, NOTIFICATIONS
    priority: str = "NORMAL"
    workspace_id: Optional[int] = None
    organization_id: Optional[int] = None
    created_at: datetime = None  # type: ignore

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    @property
    def rank(self) -> int:
        return PRIORITY_RANK.get(self.priority, 2)


def validate_priority(priority: str) -> str:
    """Validate and normalize a priority value."""
    p = priority.upper()
    if p not in PRIORITIES:
        raise ValueError(f"Invalid priority: {priority}")
    return p


def sort_by_priority(jobs: list[ScheduledJob]) -> list[ScheduledJob]:
    """Sort jobs by priority (CRITICAL first), then creation time."""
    return sorted(jobs, key=lambda j: (-j.rank, j.created_at))


def can_admit(
    active_counts: dict,
    workspace_id: Optional[int],
    organization_id: Optional[int],
    limits: Optional[dict] = None,
) -> tuple[bool, str]:
    """Check whether a new job can be admitted under tenant-aware limits.

    Args:
        active_counts: {'ai_tasks': n, 'agents': n, 'workflows': n, per_workspace: n, per_org: n}
        workspace_id / organization_id: tenant context
        limits: optional override of DEFAULT_LIMITS

    Returns:
        (admitted, reason)
    """
    cfg = {**DEFAULT_LIMITS, **(limits or {})}
    total = active_counts.get("total", 0)
    ai = active_counts.get("ai_tasks", 0)
    agents = active_counts.get("agents", 0)
    workflows = active_counts.get("workflows", 0)
    docs = active_counts.get("documents", 0)
    per_ws = active_counts.get("per_workspace", 0)
    per_org = active_counts.get("per_organization", 0)

    if ai >= cfg["concurrent_ai_calls"]:
        return False, "concurrent AI call limit reached"
    if agents >= cfg["concurrent_agents"]:
        return False, "concurrent agent limit reached"
    if workflows >= cfg["concurrent_workflows"]:
        return False, "concurrent workflow limit reached"
    if docs >= cfg["concurrent_documents"]:
        return False, "concurrent document limit reached"
    if per_ws >= cfg["per_workspace_cap"]:
        return False, "workspace concurrency cap reached"
    if per_org >= cfg["per_organization_cap"]:
        return False, "organization concurrency cap reached"
    return True, "admitted"