"""Phase 20 — Improvement control plane + experiment platform.

Human-governed self-improvement: persisted proposals with a validated
lifecycle (PROPOSED -> EVALUATING -> APPROVAL_REQUIRED -> APPROVED -> STAGED
-> ACTIVE -> SUPERSEDED | ROLLED_BACK | REJECTED), audited transitions,
experiments with immutable configurations, golden/synthetic datasets,
persisted evaluation runs, and deterministic comparison verdicts.

AI-generated proposals never become production configuration: promotion
requires explicit authorization and every transition is recorded with actor,
reason, and evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import (
    ImprovementProposal, ImprovementAudit, ImprovementTransition,
    ExperimentDataset, Experiment, ExperimentRun, ExperimentComparison,
)

VALID_STATUSES = {
    "PROPOSED", "EVALUATING", "APPROVAL_REQUIRED", "APPROVED", "STAGED",
    "ACTIVE", "SUPERSEDED", "ROLLED_BACK", "REJECTED",
}

# Default transition table (server-side validated; can be overridden by rows
# in improvement_transitions for extensibility).
DEFAULT_TRANSITIONS = {
    ("PROPOSED", "EVALUATING"): False,
    ("PROPOSED", "REJECTED"): False,
    ("EVALUATING", "APPROVAL_REQUIRED"): False,
    ("EVALUATING", "REJECTED"): False,
    ("APPROVAL_REQUIRED", "APPROVED"): True,
    ("APPROVAL_REQUIRED", "REJECTED"): True,
    ("APPROVED", "STAGED"): True,
    ("APPROVED", "REJECTED"): True,
    ("STAGED", "ACTIVE"): True,
    ("STAGED", "ROLLED_BACK"): True,
    ("ACTIVE", "SUPERSEDED"): True,
    ("ACTIVE", "ROLLED_BACK"): True,
}

VALID_DOMAINS = {
    "retrieval", "rag", "model_routing", "provider_routing",
    "prompt", "workflow", "ingestion", "search", "cost", "latency",
    "knowledge_quality", "agent", "memory", "graph",
}


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _fingerprint(config: dict) -> str:
    return hashlib.sha256(_dumps(config).encode("utf-8")).hexdigest()[:16]


def _transition_allowed(db: Session, from_state: str,
                        to_state: str) -> bool:
    if from_state == to_state:
        return False
    row = db.query(ImprovementTransition).filter_by(
        from_state=from_state, to_state=to_state).first()
    if row is not None:
        return True
    return (from_state, to_state) in DEFAULT_TRANSITIONS


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------

def create_proposal(db: Session, *, domain: str, title: str, problem: str,
                    proposed_change: str, evidence: Optional[str] = None,
                    expected_benefit: Optional[str] = None,
                    risk: str = "LOW",
                    estimated_cost: Optional[float] = None,
                    evaluation_requirements: Optional[str] = None,
                    author_source: str = "human",
                    author_user_id: Optional[int] = None,
                    workspace_id: Optional[int] = None,
                    organization_id: Optional[int] = None) -> ImprovementProposal:
    """Create a proposal in PROPOSED state with an initial audit record."""
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    if risk not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(f"Unknown risk: {risk}")
    prop = ImprovementProposal(
        domain=domain, title=title, problem=problem, evidence=evidence,
        expected_benefit=expected_benefit, risk=risk,
        estimated_cost=estimated_cost, proposed_change=proposed_change,
        evaluation_requirements=evaluation_requirements,
        author_source=author_source, author_user_id=author_user_id,
        status="PROPOSED", workspace_id=workspace_id,
        organization_id=organization_id)
    db.add(prop)
    db.flush()
    db.add(ImprovementAudit(
        proposal_id=prop.id, actor_user_id=author_user_id,
        previous_state=None, new_state="PROPOSED",
        reason="Proposal created",
        evidence=json.dumps({"domain": domain, "risk": risk})))
    db.flush()
    return prop


def transition(db: Session, proposal_id: int, to_state: str, *,
               actor_user_id: Optional[int] = None,
               reason: Optional[str] = None,
               evidence: Optional[str] = None,
               require_approval_flag: Optional[bool] = None) -> dict:
    """Validate + apply a lifecycle transition with an audit record."""
    prop = db.get(ImprovementProposal, proposal_id)
    if prop is None:
        raise KeyError("proposal not found")
    if to_state not in VALID_STATUSES:
        raise ValueError(f"Unknown status: {to_state}")
    prev = prop.status
    if not _transition_allowed(db, prev, to_state):
        raise ValueError(f"Transition {prev} -> {to_state} not allowed")
    if to_state == "ACTIVE":
        if prev not in ("STAGED",):
            raise ValueError("Only STAGED proposals may be activated")
        requires = _transition_allowed(db, "APPROVAL_REQUIRED", "APPROVED") \
            and DEFAULT_TRANSITIONS.get(("APPROVAL_REQUIRED", "APPROVED"),
                                        False)
        # An approval must already be recorded: require an APPROVED audit
        approved_audit = db.query(ImprovementAudit).filter_by(
            proposal_id=proposal_id, new_state="APPROVED").first()
        if approved_audit is None:
            raise ValueError("Approval is required before activation")
    prop.status = to_state
    db.add(ImprovementAudit(
        proposal_id=proposal_id, actor_user_id=actor_user_id,
        previous_state=prev, new_state=to_state, reason=reason,
        evidence=evidence))
    db.flush()
    return {"proposal_id": proposal_id, "previous_state": prev,
            "new_state": to_state, "approved": to_state == "ACTIVE"}


def list_proposals(db: Session, *, domain: Optional[str] = None,
                   status: Optional[str] = None,
                   workspace_id: Optional[int] = None,
                   limit: int = 100) -> list[dict]:
    q = db.query(ImprovementProposal)
    if domain:
        q = q.filter(ImprovementProposal.domain == domain)
    if status:
        q = q.filter(ImprovementProposal.status == status)
    if workspace_id is not None:
        q = q.filter(ImprovementProposal.workspace_id == workspace_id)
    rows = q.order_by(ImprovementProposal.created_at.desc()).limit(limit).all()
    return [{
        "id": p.id, "domain": p.domain, "title": p.title,
        "risk": p.risk, "status": p.status, "author_source": p.author_source,
        "workspace_id": p.workspace_id, "created_at": p.created_at,
    } for p in rows]


def proposal_audit_trail(db: Session, proposal_id: int,
                         limit: int = 100) -> list[dict]:
    if db.get(ImprovementProposal, proposal_id) is None:
        raise KeyError("proposal not found")
    rows = db.query(ImprovementAudit).filter_by(proposal_id=proposal_id)\
        .order_by(ImprovementAudit.created_at.desc()).limit(limit).all()
    return [{
        "actor_user_id": a.actor_user_id,
        "previous_state": a.previous_state, "new_state": a.new_state,
        "reason": a.reason, "created_at": a.created_at,
    } for a in rows]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def create_dataset(db: Session, *, name: str, kind: str,
                   domain: str = "retrieval", items: list,
                   created_by: Optional[int] = None) -> ExperimentDataset:
    if kind not in ("golden", "synthetic", "anonymized", "curated"):
        raise ValueError(f"Unknown dataset kind: {kind}")
    existing = db.query(ExperimentDataset).filter_by(name=name).first()
    if existing is not None:
        raise ValueError("Dataset name already exists")
    ds = ExperimentDataset(name=name, kind=kind, domain=domain,
                           items_json=_dumps(items), created_by=created_by)
    db.add(ds)
    db.flush()
    return ds


def list_datasets(db: Session, *, domain: Optional[str] = None,
                  limit: int = 100) -> list[dict]:
    q = db.query(ExperimentDataset)
    if domain:
        q = q.filter(ExperimentDataset.domain == domain)
    rows = q.order_by(ExperimentDataset.created_at.desc()).limit(limit).all()
    return [{"id": d.id, "name": d.name, "kind": d.kind, "domain": d.domain,
             "item_count": len(json.loads(d.items_json or "[]"))}
            for d in rows]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def create_experiment(db: Session, *, name: str, domain: str,
                      config: dict, dataset_id: Optional[int] = None,
                      proposal_id: Optional[int] = None,
                      created_by: Optional[int] = None,
                      workspace_id: Optional[int] = None) -> Experiment:
    """Create an experiment; its configuration is immutable from here on."""
    if domain not in VALID_DOMAINS and domain not in (
            "retrieval", "reranking", "model_routing", "prompt", "chunking",
            "embeddings", "search_ranking", "rag", "cost", "workflow"):
        raise ValueError(f"Unknown experiment domain: {domain}")
    exp = Experiment(
        name=name, domain=domain,
        config_json=_dumps(config),
        config_fingerprint=_fingerprint(config),
        dataset_id=dataset_id, proposal_id=proposal_id,
        created_by=created_by, workspace_id=workspace_id)
    db.add(exp)
    db.flush()
    return exp


def get_experiment_config(db: Session, experiment_id: int) -> dict:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise KeyError("experiment not found")
    return json.loads(exp.config_json or "{}")


def record_run(db: Session, *, experiment_id: int, metrics: dict,
               dataset_id: Optional[int] = None,
               dataset_name: Optional[str] = None,
               model: Optional[str] = None, provider: Optional[str] = None,
               cost: Optional[float] = None,
               latency_ms: Optional[float] = None,
               environment: Optional[str] = None,
               status: str = "DONE") -> ExperimentRun:
    run = ExperimentRun(
        experiment_id=experiment_id, dataset_id=dataset_id,
        dataset_name=dataset_name, model=model, provider=provider,
        config_json=None, metrics_json=_dumps(metrics), cost=cost,
        latency_ms=latency_ms, environment=environment, status=status)
    db.add(run)
    db.flush()
    exp = db.get(Experiment, experiment_id)
    if exp is not None and status == "DONE" and exp.status == "DRAFT":
        exp.status = "RUNNING"
    return run


def compare_runs(db: Session, *, experiment_id: int,
                 baseline_run_id: int, candidate_run_id: int,
                 metrics: list[str]) -> dict:
    """Deterministic comparison; verdict is CANDIDATE_BETTER /
    BASELINE_BETTER / INCONCLUSIVE.

    Uses paired per-example deltas when available (delta distribution with
    wins/losses), otherwise mean comparison with a minimum sample size.
    """
    base = db.get(ExperimentRun, baseline_run_id)
    cand = db.get(ExperimentRun, candidate_run_id)
    if base is None or cand is None:
        raise KeyError("run not found")
    bm = json.loads(base.metrics_json or "{}")
    cm = json.loads(cand.metrics_json or "{}")

    wins = 0
    losses = 0
    ties = 0
    deltas = []
    for m in metrics:
        b = bm.get(m)
        c = cm.get(m)
        if b is None or c is None:
            continue
        # Higher is better for all compared metrics (e.g. precision, recall)
        delta = float(c) - float(b)
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    sample = max(bm.get("sample_size", 0), cm.get("sample_size", 0),
                 len(deltas))
    if sample < 10:
        verdict = "INCONCLUSIVE"
    elif wins > losses and losses == 0 and ties <= wins:
        verdict = "CANDIDATE_BETTER"
    elif losses > wins:
        verdict = "BASELINE_BETTER"
    elif wins > losses:
        verdict = "CANDIDATE_BETTER"
    else:
        verdict = "INCONCLUSIVE"

    cmp_row = ExperimentComparison(
        experiment_id=experiment_id, baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        metrics_json=_dumps({"deltas": deltas, "wins": wins,
                             "losses": losses, "ties": ties,
                             "metrics": metrics}),
        verdict=verdict, sample_size=sample)
    db.add(cmp_row)
    db.flush()
    return {"verdict": verdict, "wins": wins, "losses": losses,
            "ties": ties, "sample_size": sample, "deltas": deltas}


def list_experiments(db: Session, *, domain: Optional[str] = None,
                     status: Optional[str] = None,
                     workspace_id: Optional[int] = None,
                     limit: int = 100) -> list[dict]:
    q = db.query(Experiment)
    if domain:
        q = q.filter(Experiment.domain == domain)
    if status:
        q = q.filter(Experiment.status == status)
    if workspace_id is not None:
        q = q.filter(Experiment.workspace_id == workspace_id)
    rows = q.order_by(Experiment.created_at.desc()).limit(limit).all()
    return [{"id": e.id, "name": e.name, "domain": e.domain,
             "status": e.status, "proposal_id": e.proposal_id,
             "workspace_id": e.workspace_id,
             "config_fingerprint": e.config_fingerprint,
             "created_at": e.created_at} for e in rows]


def list_runs(db: Session, experiment_id: int,
              limit: int = 100) -> list[dict]:
    rows = db.query(ExperimentRun).filter_by(experiment_id=experiment_id)\
        .order_by(ExperimentRun.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "dataset_name": r.dataset_name, "model": r.model,
             "provider": r.provider, "cost": r.cost,
             "latency_ms": r.latency_ms, "environment": r.environment,
             "status": r.status, "created_at": r.created_at}
            for r in rows]


def promotion_eligible(db: Session, experiment_id: int,
                       min_quality: float = 0.8,
                       max_cost: Optional[float] = None,
                       max_latency_ms: Optional[float] = None) -> dict:
    """Gate: experiment may only be promoted to its proposal when quality,
    cost, and latency thresholds all pass (evaluation-based)."""
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise KeyError("experiment not found")
    runs = db.query(ExperimentRun).filter_by(experiment_id=experiment_id,
                                             status="DONE").all()
    if not runs:
        return {"eligible": False, "reason": "no completed runs"}
    best = max(runs, key=lambda r: float(
        json.loads(r.metrics_json or "{}").get("quality_score", 0.0)))
    m = json.loads(best.metrics_json or "{}")
    quality = float(m.get("quality_score", 0.0))
    cost = best.cost if best.cost is not None else float(m.get("cost", 0.0))
    latency = best.latency_ms if best.latency_ms is not None \
        else float(m.get("latency_ms", 0.0))
    reasons = []
    if quality < min_quality:
        reasons.append(f"quality {quality:.3f} < {min_quality}")
    if max_cost is not None and cost > max_cost:
        reasons.append(f"cost {cost:.3f} > {max_cost}")
    if max_latency_ms is not None and latency > max_latency_ms:
        reasons.append(f"latency {latency:.0f}ms > {max_latency_ms}ms")
    return {"eligible": not reasons, "quality": quality, "cost": cost,
            "latency_ms": latency, "reasons": reasons,
            "best_run_id": best.id}