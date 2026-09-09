"""Phase 19 — vector platform 2.0.

Embedding-model lifecycle (active/deprecated/migration-required/retired),
compatibility + dimension enforcement, rebuild planning over the durable
backfill abstraction, coverage monitoring snapshots, drift detection when a
large share of data uses outdated models, and a retrieval benchmarking
harness (keyword / vector / hybrid) computing precision, recall, MRR and
latency deterministically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

LIFECYCLES = ("ACTIVE", "DEPRECATED", "MIGRATION_REQUIRED", "RETIRED")
TRANSITIONS = {
    "ACTIVE": ("DEPRECATED", "MIGRATION_REQUIRED", "RETIRED"),
    "DEPRECATED": ("ACTIVE", "MIGRATION_REQUIRED", "RETIRED"),
    "MIGRATION_REQUIRED": ("ACTIVE", "DEPRECATED", "RETIRED"),
    "RETIRED": ("ACTIVE",),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _loads(raw: Optional[str]):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def set_lifecycle(db: Session, *, model_id: int, lifecycle: str,
                  reason: Optional[str] = None,
                  decided_by: Optional[int] = None) -> dict:
    from ..models.phase18 import EmbeddingModel
    from ..models.phase19 import EmbeddingLifecycleEvent
    if lifecycle not in LIFECYCLES:
        raise ValueError("invalid lifecycle")
    model = db.query(EmbeddingModel).get(model_id)
    if model is None:
        raise ValueError("embedding model not found")
    current = model_lifecycle(db, model_id)
    if lifecycle == current:
        return {"status": "NOOP", "lifecycle": lifecycle}
    if lifecycle not in TRANSITIONS[current]:
        return {"status": "BLOCKED",
                "reason": f"cannot transition {current} -> {lifecycle}"}
    db.add(EmbeddingLifecycleEvent(embedding_model_id=model_id,
                                   lifecycle=lifecycle, reason=reason,
                                   decided_by=decided_by))
    if lifecycle == "RETIRED":
        model.active = False
    elif lifecycle == "ACTIVE":
        model.active = True
    db.flush()
    return {"status": "TRANSITIONED", "from": current, "to": lifecycle}


def model_lifecycle(db: Session, model_id: int) -> str:
    from ..models.phase19 import EmbeddingLifecycleEvent
    event = (db.query(EmbeddingLifecycleEvent)
             .filter(EmbeddingLifecycleEvent.embedding_model_id == model_id)
             .order_by(EmbeddingLifecycleEvent.created_at.desc()).first())
    return event.lifecycle if event else "ACTIVE"


def lifecycle_summary(db: Session) -> dict:
    from ..models.phase18 import EmbeddingModel
    from ..models.phase19 import EmbeddingLifecycleEvent
    summary = {}
    for model in db.query(EmbeddingModel).limit(500).all():
        summary.setdefault(model_lifecycle(db, model.id), 0)
        summary[model_lifecycle(db, model.id)] += 1
    return {"by_lifecycle": summary,
            "deprecated_or_worse": sum(v for k, v in summary.items()
                                       if k in ("DEPRECATED",
                                                "MIGRATION_REQUIRED",
                                                "RETIRED"))}


# ---------------------------------------------------------------------------
# Compatibility + dimension enforcement
# ---------------------------------------------------------------------------

def validate_compatible(db: Session, *, provider: str, model: str,
                        dimensions: int, version: str = "v1",
                        distance: str = "cosine",
                        allow_register: bool = False) -> dict:
    """Dimension + version + lifecycle enforcement.

    Incompatible embeddings (wrong dimensions, retired models, distance
    mismatches) are rejected — never silently mixed.
    """
    from ..models.phase18 import EmbeddingModel
    # Look the model up by (provider, model, version) first so a dimension
    # mismatch against a registered model is reported precisely instead of
    # masquerading as "not registered".
    row = (db.query(EmbeddingModel)
           .filter(EmbeddingModel.provider == provider,
                   EmbeddingModel.model == model,
                   EmbeddingModel.version == version).first())
    if row is None:
        if allow_register:
            from ..services.vector_registry import register_model
            row = register_model(db, provider=provider, model=model,
                                 dimensions=dimensions, version=version)
        else:
            return {"compatible": False,
                    "reason": "embedding model not registered"}
    if dimensions != row.dimensions:
        return {"compatible": False,
                "reason": "dimension mismatch with registered model"}
    if distance != "cosine":
        return {"compatible": False,
                "reason": "only cosine distance is supported by this backend"}
    lifecycle = model_lifecycle(db, row.id)
    if lifecycle == "RETIRED":
        return {"compatible": False,
                "reason": "embedding model is retired"}
    return {"compatible": True, "embedding_model_id": row.id,
            "lifecycle": lifecycle, "dimensions": row.dimensions}


# ---------------------------------------------------------------------------
# Coverage + drift
# ---------------------------------------------------------------------------

def compute_coverage(db: Session, *, workspace_id: Optional[int] = None,
                     persist: bool = True) -> dict:
    """Chunk-level embedding coverage + cached-model distribution."""
    from ..models.document import Document
    from ..models.document_chunk import DocumentChunk
    from ..models.phase16 import EmbeddingCache
    from ..models.phase19 import VectorCoverageSnapshot
    total_q = db.query(DocumentChunk)
    embedded_q = db.query(DocumentChunk).filter(
        DocumentChunk.embedding.isnot(None))
    if workspace_id is not None:
        total_q = total_q.join(
            Document, DocumentChunk.document_id == Document.id).filter(
            Document.workspace_id == workspace_id)
        embedded_q = embedded_q.join(
            Document, DocumentChunk.document_id == Document.id).filter(
            Document.workspace_id == workspace_id)
    total = total_q.count()
    embedded = embedded_q.count()
    cache_q = db.query(EmbeddingCache.provider, EmbeddingCache.model,
                       EmbeddingCache.dimensions)
    if workspace_id is not None:
        cache_q = cache_q.filter(EmbeddingCache.workspace_id == workspace_id)
    model_dist = {}
    for provider, model, dims in cache_q.all():
        key = f"{provider}:{model}"
        model_dist[key] = model_dist.get(key, 0) + 1
    coverage = (embedded / total) if total else 0.0
    stale = _stale_cache_fraction(db, model_dist)
    if persist:
        snap = VectorCoverageSnapshot(
            workspace_id=workspace_id, total_chunks=total,
            embedded_chunks=embedded, stale_chunks=int(stale * (embedded
                                                                or 0)),
            failed_chunks=0,
            model_distribution_json=_dumps(model_dist))
        db.add(snap)
        db.flush()
    return {"workspace_id": workspace_id, "total_chunks": total,
            "embedded_chunks": embedded,
            "coverage": round(coverage, 4),
            "stale_fraction": round(stale, 4),
            "model_distribution": model_dist}


def _stale_cache_fraction(db: Session, model_dist: dict) -> float:
    """Share of cached embeddings produced by non-active models."""
    from ..models.phase18 import EmbeddingModel
    active_keys = {f"{m.provider}:{m.model}"
                   for m in db.query(EmbeddingModel)
                   .filter(EmbeddingModel.active.is_(True)).all()}
    if not model_dist:
        return 0.0
    total = sum(model_dist.values())
    stale = sum(v for k, v in model_dist.items() if k not in active_keys)
    return stale / total if total else 0.0


def embedding_drift(db: Session, *, stale_threshold_pct: float = 25.0,
                    coverage_min: float = 0.5) -> dict:
    """Detect drift: outdated embeddings or low coverage."""
    from ..models.phase19 import VectorCoverageSnapshot
    latest = (db.query(VectorCoverageSnapshot)
              .order_by(VectorCoverageSnapshot.computed_at.desc()).first())
    if latest is None:
        report = compute_coverage(db)
    else:
        report = {"workspace_id": latest.workspace_id,
                  "total_chunks": latest.total_chunks,
                  "embedded_chunks": latest.embedded_chunks,
                  "coverage": (latest.embedded_chunks / latest.total_chunks
                               if latest.total_chunks else 0.0),
                  "stale_fraction": (latest.stale_chunks /
                                     latest.embedded_chunks
                                     if latest.embedded_chunks else 0.0)}
    drift = (report["stale_fraction"] * 100 > stale_threshold_pct) or (
        report["coverage"] < coverage_min)
    return {"drift_detected": bool(drift),
            "stale_pct": round(report["stale_fraction"] * 100, 2),
            "coverage": round(report["coverage"], 4),
            "thresholds": {"stale_pct": stale_threshold_pct,
                           "coverage_min": coverage_min},
            "total_chunks": report["total_chunks"],
            "embedded_chunks": report["embedded_chunks"],
            "recommendation": ("rebuild vectors with the active model"
                               if drift else "no action")}


# ---------------------------------------------------------------------------
# Rebuild planner
# ---------------------------------------------------------------------------

def plan_rebuild(db: Session, *, model: str, dimensions: int,
                 workspace_id: Optional[int] = None,
                 dry_run: bool = True) -> dict:
    """Create a durable, resumable rebuild run (PREVIEW by default)."""
    from ..models.phase17 import VectorBackfillRun
    from ..models.document import Document
    from ..models.document_chunk import DocumentChunk
    total_q = db.query(DocumentChunk).filter(
        DocumentChunk.embedding.isnot(None))
    if workspace_id is not None:
        total_q = total_q.join(
            Document, DocumentChunk.document_id == Document.id).filter(
            Document.workspace_id == workspace_id)
    total = total_q.count()
    run = VectorBackfillRun(model=model, dimensions=dimensions,
                            workspace_id=workspace_id, dry_run=dry_run,
                            status="PREVIEW", total=total)
    db.add(run)
    db.flush()
    return {"run_id": run.id, "status": run.status, "total": total,
            "dry_run": dry_run, "estimate": {
                "chunks": total,
                "note": "estimate only; actual work is bounded per batch"}}


def start_rebuild(db: Session, run_id: int) -> dict:
    from ..models.phase17 import VectorBackfillRun
    run = db.query(VectorBackfillRun).get(run_id)
    if run is None:
        raise ValueError("rebuild run not found")
    run.status = "RUNNING"
    db.flush()
    return {"run_id": run.id, "status": run.status}


def pause_rebuild(db: Session, run_id: int) -> dict:
    from ..models.phase17 import VectorBackfillRun
    run = db.query(VectorBackfillRun).get(run_id)
    if run is None:
        raise ValueError("rebuild run not found")
    if run.status == "COMPLETED":
        return {"run_id": run.id, "status": "ALREADY_COMPLETED"}
    run.status = "PAUSED"
    db.flush()
    return {"run_id": run.id, "status": run.status}


def resume_rebuild(db: Session, run_id: int) -> dict:
    from ..models.phase17 import VectorBackfillRun
    run = db.query(VectorBackfillRun).get(run_id)
    if run is None:
        raise ValueError("rebuild run not found")
    if run.status not in ("PAUSED", "FAILED", "PREVIEW"):
        return {"run_id": run.id, "status": "NOT_RESUMABLE",
                "detail": run.status}
    run.status = "RUNNING"
    db.flush()
    return {"run_id": run.id, "status": "RUNNING",
            "resume_from": run.processed}


def rebuild_batch(db: Session, run_id: int, *, batch_size: int = 50,
                  progress: Optional[int] = None) -> dict:
    """Bounded batch progress update with retry/failure reporting."""
    from ..models.phase17 import VectorBackfillRun
    run = db.query(VectorBackfillRun).get(run_id)
    if run is None:
        raise ValueError("rebuild run not found")
    if run.dry_run:
        return {"run_id": run.id, "dry_run": True, "detail":
                "dry-run runs do not mutate embeddings", "processed":
                run.processed, "total": run.total}
    advanced = progress if progress is not None else (
        min(run.processed + batch_size, run.total))
    run.processed = advanced
    if run.processed >= run.total and run.total > 0:
        run.status = "COMPLETED"
        run.completed_at = _utcnow()
    db.flush()
    return {"run_id": run.id, "processed": run.processed, "total": run.total,
            "status": run.status,
            "pct": round((run.processed / run.total) * 100, 1)
            if run.total else 100.0}


def rebuild_failure(db: Session, run_id: int, *, error: str) -> dict:
    from ..models.phase17 import VectorBackfillRun
    run = db.query(VectorBackfillRun).get(run_id)
    if run is None:
        raise ValueError("rebuild run not found")
    run.failed += 1
    run.status = "FAILED"
    errors = _loads(run.errors_json) or []
    errors.append(str(error)[:500])
    run.errors_json = _dumps(errors)
    db.flush()
    return {"run_id": run.id, "failed": run.failed,
            "last_error": str(error)[:500]}


def rebuild_report(db: Session, run_id: int) -> dict:
    from ..models.phase17 import VectorBackfillRun
    run = db.query(VectorBackfillRun).get(run_id)
    if run is None:
        raise ValueError("rebuild run not found")
    return {"run_id": run.id, "model": run.model,
            "dimensions": run.dimensions, "status": run.status,
            "processed": run.processed, "total": run.total,
            "failed": run.failed,
            "errors": _loads(run.errors_json) or [],
            "pct": round((run.processed / run.total) * 100, 1)
            if run.total else 100.0}


# ---------------------------------------------------------------------------
# Retrieval benchmarking (deterministic)
# ---------------------------------------------------------------------------

def retrieval_benchmark(*, mode: str, retrieved: list[list],
                        relevant: list[set], latencies_ms: Optional[list]
                        = None) -> dict:
    """Benchmark harness: precision@k, recall@k, MRR, latency.

    ``retrieved``: per query ordered id list; ``relevant``: per query ground
    truth; ``latencies_ms`` optional samples for p50/p95/p99.
    """
    if len(retrieved) != len(relevant):
        raise ValueError("retrieved/relevant length mismatch")
    precisions, recalls, rr = [], [], []
    for retrieved_ids, truth in zip(retrieved, relevant):
        if not truth:
            continue
        hits = [i for i, rid in enumerate(retrieved_ids) if rid in truth]
        precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(hits) / len(truth)
        precisions.append(precision)
        recalls.append(recall)
        if hits:
            rr.append(1.0 / (hits[0] + 1))
        else:
            rr.append(0.0)
    latency = latencies_ms or []
    latency_sorted = sorted(latency)
    def pct(p):
        if not latency_sorted:
            return None
        idx = min(len(latency_sorted) - 1,
                  int(round((p / 100.0) * (len(latency_sorted) - 1))))
        return round(latency_sorted[idx], 1)
    return {
        "mode": mode,
        "queries": len(retrieved),
        "precision": round(sum(precisions) / len(precisions), 4)
        if precisions else None,
        "recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mrr": round(sum(rr) / len(rr), 4) if rr else None,
        "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99)},
        "evidence_coverage": round(len([r for r in recalls if r > 0])
                                   / len(recalls), 4) if recalls else None,
    }
