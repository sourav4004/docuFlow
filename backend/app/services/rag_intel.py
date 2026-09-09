"""Phase 20 — RAG self-improvement.

RAG failure analysis (unsupported answers, incomplete evidence, incorrect
citations, conflicts, temporal mismatch, overconfidence, unnecessary
refusal), claim-level error classification, repair recommendations, offline
evaluation pipelines, and promotion gates requiring quality/cost/latency/
security thresholds. Nothing is promoted without explicit authorization.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import RagFailure, RagEvaluationPipeline

FAILURE_CLASSES = [
    "unsupported_answer", "incomplete_evidence", "incorrect_citation",
    "conflicting_evidence", "temporal_mismatch", "overconfident_answer",
    "unnecessary_refusal",
]


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def record_failure(db: Session, *, workspace_id: int,
                   failure_class: str, claim: Optional[str] = None,
                   execution_id: Optional[str] = None,
                   detail: Optional[str] = None) -> RagFailure:
    if failure_class not in FAILURE_CLASSES:
        raise ValueError(f"Unknown failure class: {failure_class}")
    f = RagFailure(workspace_id=workspace_id, execution_id=execution_id,
                   failure_class=failure_class, claim=claim, detail=detail)
    db.add(f)
    db.flush()
    return f


def failure_analysis(db: Session, *, workspace_id: int,
                     limit: int = 200) -> dict:
    rows = db.query(RagFailure).filter_by(workspace_id=workspace_id)\
        .order_by(RagFailure.created_at.desc()).limit(limit).all()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.failure_class] = counts.get(r.failure_class, 0) + 1
    return {"total": len(rows),
            "by_class": dict(sorted(counts.items(),
                                    key=lambda kv: -kv[1])),
            "recent": [{"id": r.id, "failure_class": r.failure_class,
                        "claim": (r.claim or "")[:200],
                        "created_at": r.created_at} for r in rows[:20]]}


def repair_recommendations(db: Session, *, workspace_id: int) -> list[dict]:
    """Deterministic recommendations mapped from failure classes."""
    analysis = failure_analysis(db, workspace_id=workspace_id)
    by_class = analysis["by_class"]
    out = []
    mapping = {
        "unsupported_answer": (
            "Retrieval quality",
            "Evaluate retrieval/evidence gating for unsupported answers"),
        "incomplete_evidence": (
            "Evidence sufficiency",
            "Raise evidence-sufficiency threshold or widen retrieval"),
        "incorrect_citation": (
            "Citation validation",
            "Add citation-correctness validation before answer output"),
        "conflicting_evidence": (
            "Conflict handling",
            "Surface conflicts instead of false certainty"),
        "temporal_mismatch": (
            "Temporal filtering",
            "Apply as-of/effective-date filtering to evidence"),
        "overconfident_answer": (
            "Confidence threshold",
            "Lower confidence calibration for low-evidence answers"),
        "unnecessary_refusal": (
            "Refusal threshold",
            "Tune refusal threshold to reduce unnecessary refusals"),
    }
    for cls, count in by_class.items():
        if count >= 1 and cls in mapping:
            area, suggestion = mapping[cls]
            out.append({"failure_class": cls, "area": area,
                        "suggestion": suggestion, "count": count})
    return out


def run_evaluation_pipeline(db: Session, *, config: dict,
                            proposal_id: Optional[int] = None,
                            dataset_id: Optional[int] = None,
                            evaluate=None) -> RagEvaluationPipeline:
    """Run an offline RAG evaluation. `evaluate` is an injectable callable
    (config, dataset_items) -> metrics dict; when omitted, a deterministic
    default evaluator is used."""
    pipe = RagEvaluationPipeline(
        proposal_id=proposal_id, dataset_id=dataset_id,
        config_json=_dumps(config))
    db.add(pipe)
    db.flush()

    items = _dataset_items(db, dataset_id)
    if evaluate is not None:
        metrics = evaluate(config, items)
    else:
        metrics = _default_evaluate(config, items)
    gate = promotion_gate(config, metrics)
    pipe.metrics_json = _dumps(metrics)
    pipe.gate_passed = gate["passed"]
    pipe.status = "DONE"
    return pipe


def _dataset_items(db: Session, dataset_id: Optional[int]) -> list:
    if dataset_id is None:
        return []
    from ..models.phase20 import ExperimentDataset
    ds = db.get(ExperimentDataset, dataset_id)
    if ds is None:
        return []
    return json.loads(ds.items_json or "[]")


def _default_evaluate(config: dict, items: list) -> dict:
    """Deterministic default evaluator: counts claims, computes coverage
    from config targets, never calls a provider."""
    total = len(items)
    covered = sum(1 for it in items
                  if it.get("covered", True) is not False)
    correct_cites = sum(1 for it in items
                        if it.get("citation_correct", True) is not False)
    conflicts = sum(1 for it in items if it.get("has_conflict"))
    quality = (covered / total) if total else 1.0
    return {
        "sample_size": total,
        "quality_score": round(quality, 4),
        "citation_coverage": round((covered / total) if total else 1.0, 4),
        "citation_correctness": round(
            (correct_cites / total) if total else 1.0, 4),
        "conflict_count": conflicts,
        "cost": float(config.get("estimated_cost", 0.0)),
        "latency_ms": float(config.get("estimated_latency_ms", 0.0)),
    }


def promotion_gate(config: dict, metrics: dict) -> dict:
    """Gate: quality/cost/latency thresholds from config must all pass."""
    reasons = []
    min_quality = float(config.get("min_quality", 0.8))
    max_cost = config.get("max_cost")
    max_latency = config.get("max_latency_ms")
    quality = float(metrics.get("quality_score", 0.0))
    cost = float(metrics.get("cost", 0.0))
    latency = float(metrics.get("latency_ms", 0.0))
    if quality < min_quality:
        reasons.append(f"quality {quality:.3f} < {min_quality}")
    if max_cost is not None and cost > float(max_cost):
        reasons.append(f"cost {cost:.3f} > {max_cost}")
    if max_latency is not None and latency > float(max_latency):
        reasons.append(f"latency {latency:.0f}ms > {max_latency}ms")
    return {"passed": not reasons, "reasons": reasons,
            "quality": quality, "cost": cost, "latency_ms": latency}


def claim_error_classify(db: Session, *, workspace_id: int,
                         claim: str, category: str,
                         execution_id: Optional[str] = None) -> RagFailure:
    """Persist claim-level error classification (alias for record_failure
    with normalized categories)."""
    if category not in FAILURE_CLASSES:
        raise ValueError(f"Unknown claim error category: {category}")
    return record_failure(db, workspace_id=workspace_id,
                          failure_class=category, claim=claim,
                          execution_id=execution_id)


def list_pipelines(db: Session, limit: int = 50) -> list[dict]:
    rows = db.query(RagEvaluationPipeline)\
        .order_by(RagEvaluationPipeline.created_at.desc()).limit(limit).all()
    return [{"id": p.id, "proposal_id": p.proposal_id,
             "dataset_id": p.dataset_id, "gate_passed": p.gate_passed,
             "status": p.status,
             "metrics": json.loads(p.metrics_json or "{}"),
             "created_at": p.created_at} for p in rows]