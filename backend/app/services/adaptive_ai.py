"""Phase 21 — Adaptive retrieval + adaptive RAG.

Continuous retrieval evaluation (golden datasets, approved feedback, synthetic
cases, zero-result queries), ranking drift detection, bounded candidate tuning
(vector weight, keyword weight, reranking, freshness, diversity) with
promotion gates, RAG quality monitoring (groundedness, citations, refusals,
conflicts, temporal correctness), RAG drift, failure clustering, recovery
recommendations, controlled config candidates, and evaluation-gated promotion.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.phase20 import RagFailure, RetrievalFailure
from ..models.phase21 import (
    AdaptiveCandidate, RagDriftSnapshot, RagFailureCluster,
    RetrievalDriftSnapshot,
)
from .knowledge_heal import evaluate_candidate, promote_candidate

RETRIEVAL_TUNABLES = ["vector_weight", "keyword_weight", "rerank",
                      "freshness", "diversity"]

_RAG_CAUSES = {
    "missing_context": "retrieval_miss",
    "no_grounding": "weak_groundedness",
    "citation_missing": "citation_gap",
    "wrong_citation": "citation_incorrect",
    "stale_answer": "temporal_mismatch",
    "conflict": "conflicting_evidence",
    "refusal": "over_refusal",
}


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


# ---------------------------------------------------------------------------
# Adaptive retrieval
# ---------------------------------------------------------------------------

def record_retrieval_evaluation(db: Session, workspace_id: int,
                                cases: list[dict]) -> dict:
    """Evaluate a batch of retrieval cases (golden/synthetic/feedback).

    Each case: {query, expected_doc_id?, hit_rank?|hit?, zero_result?}.
    Produces deterministic retrieval metrics (hit rate, MRR, zero-result rate).
    """
    if not cases or len(cases) > 500:
        raise ValueError("cases must be between 1 and 500")
    hits, ranks, zeros = 0, [], 0
    for case in cases:
        if case.get("zero_result"):
            zeros += 1
            continue
        if case.get("hit"):
            hits += 1
            rank = case.get("hit_rank")
            if isinstance(rank, int) and rank > 0:
                ranks.append(1.0 / rank)
    evaluated = len(cases) - zeros
    metrics = {
        "hit_rate": round(hits / evaluated, 4) if evaluated else 0.0,
        "mrr": round(sum(ranks) / evaluated, 4) if evaluated else 0.0,
        "zero_result_rate": round(zeros / len(cases), 4),
        "case_count": len(cases),
    }
    return metrics


def record_retrieval_drift(db: Session, workspace_id: int,
                           baseline_value: float, current_value: float,
                           metric: str = "ndcg_proxy",
                           degrade_threshold_percent: float = 10.0) \
        -> RetrievalDriftSnapshot:
    """Measure and persist ranking drift; flags degradation past threshold."""
    baseline = float(baseline_value)
    current = float(current_value)
    drift = 0.0 if baseline <= 0 else (baseline - current) / baseline * 100.0
    row = RetrievalDriftSnapshot(
        workspace_id=workspace_id, metric=metric,
        baseline_value=round(baseline, 4), current_value=round(float(current), 4),
        drift_percent=round(drift, 1), degraded=drift >= degrade_threshold_percent)
    db.add(row)
    db.commit()
    return row


def propose_retrieval_candidate(db: Session, workspace_id: int,
                                change_kind: str, proposed_value: Any,
                                rationale: str) -> AdaptiveCandidate:
    """Create a bounded retrieval tuning candidate (never auto-applied)."""
    if change_kind not in RETRIEVAL_TUNABLES:
        raise ValueError(f"unsupported retrieval tunable: {change_kind}")
    candidate = AdaptiveCandidate(
        workspace_id=workspace_id, domain="retrieval",
        change_kind=change_kind, proposed_value=_bounded_json(proposed_value),
        rationale=rationale[:1000],
        idempotency_key=f"retrieval:{change_kind}:{workspace_id}")
    db.add(candidate)
    db.commit()
    return candidate


# ---------------------------------------------------------------------------
# Adaptive RAG
# ---------------------------------------------------------------------------

def record_rag_quality(db: Session, workspace_id: int, answers: list[dict]) \
        -> dict:
    """Score RAG answer batches across quality dimensions.

    Each answer: {grounded?, citation_count?, citations_correct?,
    complete?, refused?, refusal_correct?, conflict?, temporal_ok?}.
    """
    if not answers or len(answers) > 500:
        raise ValueError("answers must be between 1 and 500")
    n = len(answers)
    grounded = sum(1 for a in answers if a.get("grounded"))
    cited = sum(1 for a in answers if (a.get("citation_count") or 0) > 0)
    cited_correct = sum(1 for a in answers
                        if a.get("citation_count") and a.get("citations_correct"))
    complete = sum(1 for a in answers if a.get("complete"))
    refused = sum(1 for a in answers if a.get("refused"))
    refusals_correct = sum(1 for a in answers
                           if a.get("refused") and a.get("refusal_correct"))
    conflicts = sum(1 for a in answers if a.get("conflict"))
    temporal = sum(1 for a in answers
                   if a.get("temporal_ok") is not False)
    metrics = {
        "groundedness": round(grounded / n, 4),
        "citation_coverage": round(cited / n, 4),
        "citation_correctness": round(cited_correct / cited, 4) if cited else 1.0,
        "completeness": round(complete / n, 4),
        "refusal_rate": round(refused / n, 4),
        "refusal_accuracy": round(refusals_correct / refused, 4) if refused else 1.0,
        "conflict_rate": round(conflicts / n, 4),
        "temporal_correctness": round(temporal / n, 4),
        "answer_count": n,
    }
    return metrics


def record_rag_drift(db: Session, workspace_id: int,
                     baseline_value: float, current_value: float,
                     metric: str = "groundedness",
                     degrade_threshold_percent: float = 10.0) -> RagDriftSnapshot:
    baseline = float(baseline_value)
    current = float(current_value)
    drift = 0.0 if baseline <= 0 else (baseline - current) / baseline * 100.0
    row = RagDriftSnapshot(
        workspace_id=workspace_id, metric=metric,
        baseline_value=round(baseline, 4),
        current_value=round(float(current), 4),
        drift_percent=round(drift, 1),
        degraded=drift >= degrade_threshold_percent)
    db.add(row)
    db.commit()
    return row


def cluster_rag_failures(db: Session, workspace_id: int,
                         limit: int = 200) -> list[RagFailureCluster]:
    """Cluster persisted RAG failures by cause; persist clusters + advice."""
    rows = (db.query(RagFailure)
            .filter_by(workspace_id=workspace_id)
            .order_by(RagFailure.id.desc()).limit(limit).all())
    by_cause: dict[str, list[RagFailure]] = {}
    for row in rows:
        by_cause.setdefault(row.failure_class or "unknown", []).append(row)
    clusters = []
    for cause, items in sorted(by_cause.items(),
                               key=lambda kv: -len(kv[1])):
        rec = _RAG_CAUSES.get(cause, "review_pipeline")
        cluster = RagFailureCluster(
            workspace_id=workspace_id, cause=cause,
            sample_count=len(items),
            representative_question=(items[0].claim or "")[:500],
            recommendation=_rag_recommendation(rec))
        db.add(cluster)
        clusters.append(cluster)
    if clusters:
        db.commit()
    return clusters


def _rag_recommendation(kind: str) -> str:
    return {
        "retrieval_miss": "adjust retrieval weights (candidate)",
        "weak_groundedness": "raise evidence threshold (candidate)",
        "citation_gap": "enforce citation coverage policy",
        "citation_incorrect": "improve citation validation",
        "temporal_mismatch": "prefer fresher sources for temporal queries",
        "conflicting_evidence": "surface conflict handling config",
        "over_refusal": "tune refusal threshold (candidate)",
        "review_pipeline": "review RAG pipeline configuration",
    }.get(kind, "review pipeline")


def propose_rag_candidate(db: Session, workspace_id: int, change_kind: str,
                          proposed_value: Any, rationale: str) \
        -> AdaptiveCandidate:
    """Create a controlled RAG config candidate."""
    allowed = {"context_size", "evidence_threshold", "refusal_threshold",
               "citation_strategy", "rerank"}
    if change_kind not in allowed:
        raise ValueError(f"unsupported RAG tunable: {change_kind}")
    candidate = AdaptiveCandidate(
        workspace_id=workspace_id, domain="rag", change_kind=change_kind,
        proposed_value=_bounded_json(proposed_value),
        rationale=rationale[:1000],
        idempotency_key=f"rag:{change_kind}:{workspace_id}")
    db.add(candidate)
    db.commit()
    return candidate


# Re-export governed evaluation/promotion for the router.
__all__ = [
    "record_retrieval_evaluation", "record_retrieval_drift",
    "propose_retrieval_candidate", "record_rag_quality", "record_rag_drift",
    "cluster_rag_failures", "propose_rag_candidate",
    "evaluate_candidate", "promote_candidate",
    "RETRIEVAL_TUNABLES",
]
