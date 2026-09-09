"""Workflow intelligence — DAG validation, versioned definitions, activation,
dry-run planning, and natural-language draft generation (Phase 14).

Workflows are stored as immutable WorkflowVersion rows keyed by a stable
workflow_id; each edit creates a new version. Definitions are always
validated before persistence or execution:

- trigger must be a known event
- nodes must be non-empty with known types and unique ids
- the DAG must be acyclic (nodes may depend only on existing nodes)
- notify nodes require an approval gate reachable through their inputs

Validation errors raise WorkflowValidationError. Draft generation is
deterministic keyword-driven output that stays a DRAFT until explicitly
created and activated — AI-generated drafts are never active on their own.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from ..models.workflow_version import WorkflowVersion

KNOWN_TRIGGERS = {
    "document.ready", "document.uploaded", "document.updated",
    "document.deadline_approaching", "schedule.cron", "manual",
}

AI_NODE_TYPES = {"summarize", "extract", "classify", "compare", "research"}
KNOWN_NODE_TYPES = AI_NODE_TYPES | {"notify", "approval"}

TOKENS_PER_AI_CALL = 2000
COST_PER_M_TOKENS = 0.002  # clearly labeled heuristic estimate


class WorkflowValidationError(Exception):
    """Raised when a workflow definition fails validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowValidationError(message)


def _has_cycle(nodes: list[dict]) -> bool:
    """Detect a cycle in the node dependency DAG (edges node -> inputs)."""
    incoming: dict[str, list[str]] = {}
    for node in nodes:
        nid = node["id"]
        for dep in node.get("inputs", []):
            incoming.setdefault(nid, []).append(dep)
    # Kahn's algorithm
    remaining = {n["id"] for n in nodes}
    while remaining:
        ready = [nid for nid in remaining
                 if not set(incoming.get(nid, [])) & remaining]
        if not ready:
            return True  # cycle
        for nid in ready:
            remaining.discard(nid)
    return False


def _approval_gate_present(nodes: list[dict], notify_id: str) -> bool:
    """An approval node must be reachable from the notify node through its
    input dependency closure."""
    by_id = {n["id"]: n for n in nodes}
    seen: set[str] = set()
    stack = list(by_id[notify_id].get("inputs", []))
    while stack:
        dep = stack.pop()
        if dep in seen or dep not in by_id:
            continue
        seen.add(dep)
        node = by_id[dep]
        if node["type"] == "approval":
            return True
        stack.extend(node.get("inputs", []))
    return False


def validate_definition(definition: dict) -> dict:
    """Validate a workflow definition in place and return it. Raises
    WorkflowValidationError on any violation (unknown trigger/node type,
    empty nodes, duplicate ids, unknown dependencies, cycles, notify without
    an approval gate)."""
    trigger = definition.get("trigger")
    _require(trigger in KNOWN_TRIGGERS, f"Unknown trigger: {trigger}")
    nodes = definition.get("nodes")
    _require(isinstance(nodes, list) and len(nodes) > 0,
             "Workflow must contain at least one node")
    ids = [n.get("id") for n in nodes]
    _require(len(ids) == len(set(ids)), "Duplicate node ids are not allowed")
    known_ids = set(ids)
    for node in nodes:
        _require(node.get("type") in KNOWN_NODE_TYPES,
                 f"Unknown node type: {node.get('type')}")
        for dep in node.get("inputs", []):
            _require(dep in known_ids,
                     f"Node {node.get('id')} depends on unknown node {dep}")
    _require(not _has_cycle(nodes), "Workflow contains a cycle")
    for node in nodes:
        if node.get("type") == "notify":
            _require(_approval_gate_present(nodes, node["id"]),
                     "notify node requires an approval gate")
    return definition


def dry_run(definition: dict) -> dict:
    """Side-effect-free planning for a definition: estimated AI calls, cost,
    approval gates, and node counts. Never persists anything."""
    validate_definition(definition)
    nodes = definition["nodes"]
    ai_calls = sum(1 for n in nodes if n["type"] in AI_NODE_TYPES)
    approvals = [n["name"] for n in nodes if n["type"] == "approval"]
    estimated_tokens = ai_calls * TOKENS_PER_AI_CALL
    return {
        "mode": "DRY_RUN",
        "trigger": definition.get("trigger"),
        "node_count": len(nodes),
        "estimated_ai_calls": ai_calls,
        "estimated_tokens": estimated_tokens,
        "estimated_cost": round(
            estimated_tokens / 1_000_000 * COST_PER_M_TOKENS, 6),
        "cost_estimate_is_exact": False,
        "approval_gates": approvals,
        "approval_required": bool(approvals),
    }


def create_workflow_version(db: Session, workspace_id: int, user_id: int,
                            name: str, definition: dict,
                            workflow_id: Optional[str] = None,
                            description: Optional[str] = None
                            ) -> WorkflowVersion:
    """Persist a new immutable DRAFT version. New workflow_id when omitted;
    otherwise the version counter is incremented from the latest version."""
    validate_definition(definition)
    if workflow_id is None:
        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        version_num = 1
    else:
        latest = db.query(WorkflowVersion).filter_by(
            workflow_id=workflow_id).order_by(
            WorkflowVersion.version.desc()).first()
        version_num = (latest.version + 1) if latest else 1
    row = WorkflowVersion(
        workflow_id=workflow_id, workspace_id=workspace_id,
        created_by=user_id, version=version_num, name=name,
        description=description,
        definition_json=json.dumps(definition, default=str),
        status="DRAFT", is_active=False)
    db.add(row)
    db.flush()
    return row


def activate_workflow(db: Session, version: WorkflowVersion) -> WorkflowVersion:
    """Activate a DRAFT version: only one version of a workflow is active."""
    if version.status != "DRAFT":
        raise WorkflowValidationError(
            f"Only DRAFT workflows can be activated (status={version.status})")
    db.query(WorkflowVersion).filter_by(
        workflow_id=version.workflow_id, is_active=True).update(
        {"is_active": False, "status": "INACTIVE"})
    version.status = "ACTIVE"
    version.is_active = True
    db.flush()
    return version


def list_versions(db: Session, workflow_id: str) -> list[WorkflowVersion]:
    """All versions of a workflow, newest first."""
    return db.query(WorkflowVersion).filter_by(workflow_id=workflow_id)\
        .order_by(WorkflowVersion.version.desc()).all()


def get_active_version(db: Session, workflow_id: str
                       ) -> Optional[WorkflowVersion]:
    return db.query(WorkflowVersion).filter_by(workflow_id=workflow_id,
                                               is_active=True).first()


def generate_workflow_draft(nl_text: str, user_id: int,
                            workspace_id: int) -> dict:
    """Deterministic keyword-driven draft from natural language. The result
    is a definition only (no persisted status) — it stays a draft until
    explicitly created and activated through create_workflow_version."""
    text = nl_text.lower()
    if "deadline" in text:
        trigger = "document.deadline_approaching"
    else:
        trigger = "document.ready"
    wanted: list[str] = []
    if "summar" in text:
        wanted.append("summarize")
    if "extract" in text:
        wanted.append("extract")
    if "classif" in text:
        wanted.append("classify")
    if "compar" in text:
        wanted.append("compare")
    if "research" in text or "investigat" in text:
        wanted.append("research")
    if "notify" in text or "alert" in text:
        wanted.append("notify")
    if not wanted:
        wanted = ["summarize"]
    nodes: list[dict] = []
    index = 1
    type_name = {
        "summarize": "Summarize", "extract": "Extract",
        "classify": "Classify", "compare": "Compare",
        "research": "Research", "notify": "Notify", "approval": "Approval",
    }
    for ntype in wanted:
        if ntype == "notify":
            approval_id = f"n{index}"
            nodes.append({"id": approval_id, "type": "approval",
                          "name": "Approval", "inputs": []})
            index += 1
            nodes.append({"id": f"n{index}", "type": "notify",
                          "name": "Notify", "inputs": [approval_id]})
            index += 1
        else:
            nodes.append({"id": f"n{index}", "type": ntype,
                          "name": type_name.get(ntype, ntype.capitalize()),
                          "inputs": []})
            index += 1
    return {"trigger": trigger, "nodes": nodes}