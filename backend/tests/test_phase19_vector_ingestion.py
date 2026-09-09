"""Phase 19 tests — vector platform 2.0 + ingestion platform 3.0.

Vector: model lifecycle, compatibility + dimension enforcement, coverage
snapshots, drift detection, rebuild planner ops, retrieval benchmarks.
Ingestion: resource governor, stage DAG validation, stage checkpoints and
resume, quality scores, safe failure explanations. Performance 2.0:
dedupe/single-flight + cache stampede protection.
"""

from datetime import datetime, timezone

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.document_chunk import DocumentChunk  # noqa: E402
from app.models.phase16 import EmbeddingCache  # noqa: E402
from app.models.phase17 import (  # noqa: E402
    IngestionRun, IngestionStage, VectorBackfillRun,
)
from app.models.phase18 import EmbeddingModel  # noqa: E402
from app.models.phase19 import (  # noqa: E402
    EmbeddingLifecycleEvent, VectorCoverageSnapshot,
    IngestionQualityReport,
)
from app.services import vector_ops2 as vo  # noqa: E402
from app.services import ingestion_ops3 as io3  # noqa: E402
from app.services import perf2  # noqa: E402

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(IngestionQualityReport).delete()
    db_session.query(VectorCoverageSnapshot).delete()
    db_session.query(EmbeddingLifecycleEvent).delete()
    db_session.query(VectorBackfillRun).delete()
    db_session.query(EmbeddingCache).delete()
    db_session.query(IngestionStage).delete()
    db_session.query(IngestionRun).delete()
    db_session.query(DocumentChunk).delete()
    db_session.query(Document).delete()
    db_session.query(EmbeddingModel).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p19vi"):
    _counter[0] += 1
    user = User(name=f"P19 VI {_counter[0]}",
                email=f"{tag}{_counter[0]}@p19-vi.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p19 vi ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def doc(db, ws, user, name="doc.pdf"):
    _counter[0] += 1
    row = Document(user_id=user.id, workspace_id=ws.id,
                   original_filename=name,
                   storage_key=f"vi-{ws.id}-{_counter[0]}-{name}",
                   mime_type="application/pdf", file_size=100,
                   status="READY")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def chunk(db, document, *, embedding=None):
    _counter[0] += 1
    vec = [float(embedding)] * 384 if embedding is not None else None
    row = DocumentChunk(document_id=document.id,
                        chunk_index=_counter[0], text="sample text",
                        char_start=0, char_end=10, embedding=vec)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def model(db, *, provider="openai", name="text-embed-3", dims=768,
          version="v1", active=True):
    row = EmbeddingModel(provider=provider, model=name, dimensions=dims,
                         version=version, active=active)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def run(db, ws):
    row = IngestionRun(workspace_id=ws.id, status="RUNNING")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ===========================================================================
# Vector lifecycle + compatibility
# ===========================================================================

class TestVectorLifecycle:
    def test_default_lifecycle_active(self, db_session):
        row = model(db_session)
        assert vo.model_lifecycle(db_session, row.id) == "ACTIVE"

    def test_transition_to_deprecated(self, db_session):
        row = model(db_session)
        result = vo.set_lifecycle(db_session, model_id=row.id,
                                  lifecycle="DEPRECATED", reason="new model")
        assert result["status"] == "TRANSITIONED"
        assert vo.model_lifecycle(db_session, row.id) == "DEPRECATED"

    def test_invalid_lifecycle_rejected(self, db_session):
        row = model(db_session)
        with pytest.raises(ValueError):
            vo.set_lifecycle(db_session, model_id=row.id,
                             lifecycle="BOGUS")

    def test_noop_when_same_lifecycle(self, db_session):
        row = model(db_session)
        result = vo.set_lifecycle(db_session, model_id=row.id,
                                  lifecycle="ACTIVE")
        assert result["status"] == "NOOP"

    def test_full_chain_to_retired(self, db_session):
        row = model(db_session)
        vo.set_lifecycle(db_session, model_id=row.id,
                         lifecycle="DEPRECATED")
        vo.set_lifecycle(db_session, model_id=row.id,
                         lifecycle="MIGRATION_REQUIRED")
        result = vo.set_lifecycle(db_session, model_id=row.id,
                                  lifecycle="RETIRED")
        assert result["status"] == "TRANSITIONED"
        db_session.flush()
        updated = db_session.query(EmbeddingModel).get(row.id)
        assert updated.active is False

    def test_retired_model_cannot_go_deprecated(self, db_session):
        row = model(db_session)
        vo.set_lifecycle(db_session, model_id=row.id, lifecycle="RETIRED")
        result = vo.set_lifecycle(db_session, model_id=row.id,
                                  lifecycle="DEPRECATED")
        assert result["status"] == "BLOCKED"

    def test_lifecycle_summary_counts(self, db_session):
        active = model(db_session)
        deprecated = model(db_session, name="old-model", active=False)
        vo.set_lifecycle(db_session, model_id=deprecated.id,
                         lifecycle="DEPRECATED")
        summary = vo.lifecycle_summary(db_session)
        assert summary["by_lifecycle"].get("ACTIVE", 0) >= 1
        assert summary["by_lifecycle"].get("DEPRECATED", 0) >= 1
        assert summary["deprecated_or_worse"] >= 1
        _ = active

    def test_lifecycle_event_recorded(self, db_session):
        row = model(db_session)
        vo.set_lifecycle(db_session, model_id=row.id, lifecycle="RETIRED",
                         decided_by=3, reason="retire")
        events = db_session.query(EmbeddingLifecycleEvent).all()
        assert len(events) == 1
        assert events[0].lifecycle == "RETIRED"
        assert events[0].decided_by == 3


class TestVectorCompatibility:
    def test_unknown_model_rejected_without_register(self, db_session):
        result = vo.validate_compatible(db_session, provider="p",
                                        model="m", dimensions=384)
        assert result["compatible"] is False

    def test_allow_register_creates_model(self, db_session):
        result = vo.validate_compatible(db_session, provider="p",
                                        model="m", dimensions=384,
                                        allow_register=True)
        assert result["compatible"] is True
        assert db_session.query(EmbeddingModel).count() == 1

    def test_dimension_mismatch_rejected(self, db_session):
        row = model(db_session, dims=768)
        result = vo.validate_compatible(db_session, provider=row.provider,
                                        model=row.model, dimensions=300)
        assert result["compatible"] is False
        assert "dimension" in result["reason"]

    def test_retired_model_rejected(self, db_session):
        row = model(db_session)
        vo.set_lifecycle(db_session, model_id=row.id, lifecycle="RETIRED")
        result = vo.validate_compatible(db_session, provider=row.provider,
                                        model=row.model, dimensions=row.
                                        dimensions)
        assert result["compatible"] is False
        assert "retired" in result["reason"]

    def test_non_cosine_distance_rejected(self, db_session):
        row = model(db_session)
        result = vo.validate_compatible(db_session, provider=row.provider,
                                        model=row.model,
                                        dimensions=row.dimensions,
                                        distance="l2")
        assert result["compatible"] is False

    def test_matching_model_compatible(self, db_session):
        row = model(db_session)
        result = vo.validate_compatible(db_session, provider=row.provider,
                                        model=row.model,
                                        dimensions=row.dimensions)
        assert result["compatible"] is True
        assert result["embedding_model_id"] == row.id


class TestCoverageDrift:
    def test_coverage_zero_when_no_chunks(self, db_session):
        report = vo.compute_coverage(db_session)
        assert report["total_chunks"] == 0
        assert report["coverage"] == 0.0

    def test_coverage_embedded_ratio(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        document = doc(db_session, ws, user)
        chunk(db_session, document, embedding=None)
        chunk(db_session, document, embedding=0.1)
        report = vo.compute_coverage(db_session, workspace_id=ws.id)
        assert report["total_chunks"] == 2
        assert report["embedded_chunks"] == 1
        assert report["coverage"] == 0.5

    def test_coverage_snapshot_persisted(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        document = doc(db_session, ws, user)
        chunk(db_session, document, embedding=0.1)
        vo.compute_coverage(db_session, workspace_id=ws.id)
        assert db_session.query(VectorCoverageSnapshot).count() >= 1

    def test_drift_detected_on_low_coverage(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        document = doc(db_session, ws, user)
        chunk(db_session, document, embedding=None)
        vo.compute_coverage(db_session, workspace_id=ws.id)
        drift = vo.embedding_drift(db_session)
        assert drift["drift_detected"] is True

    def test_stale_cache_fraction_drives_drift(self, db_session):
        # active model only — old cached embeddings count as stale
        active = model(db_session)
        db_session.add_all([
            EmbeddingCache(cache_key="k1", provider="openai",
                           model=active.model,
                           dimensions=active.dimensions,
                           content_hash="h1", embedding_json="[0.1]"),
            EmbeddingCache(cache_key="k2", provider="openai",
                           model="old-embed", dimensions=384,
                           content_hash="h2", embedding_json="[0.2]"),
        ])
        db_session.commit()
        report = vo.compute_coverage(db_session)
        assert report["stale_fraction"] == 0.5

    def test_model_distribution_grouping(self, db_session):
        active = model(db_session)
        db_session.add_all([
            EmbeddingCache(cache_key="a1", provider="openai",
                           model=active.model, dimensions=384,
                           content_hash="h1", embedding_json="[]"),
            EmbeddingCache(cache_key="a2", provider="openai",
                           model=active.model, dimensions=384,
                           content_hash="h2", embedding_json="[]"),
        ])
        db_session.commit()
        report = vo.compute_coverage(db_session)
        assert report["model_distribution"][
            f"openai:{active.model}"] == 2


class TestRebuildPlanner:
    def test_plan_preview_counts_chunks(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        document = doc(db_session, ws, user)
        chunk(db_session, document, embedding=0.1)
        plan = vo.plan_rebuild(db_session, model="m", dimensions=3,
                               workspace_id=ws.id)
        assert plan["status"] == "PREVIEW"
        assert plan["total"] == 1

    def test_dry_run_batch_does_not_mutate(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        document = doc(db_session, ws, user)
        chunk(db_session, document, embedding=0.1)
        plan = vo.plan_rebuild(db_session, model="m", dimensions=3,
                               workspace_id=ws.id, dry_run=True)
        result = vo.rebuild_batch(db_session, plan["run_id"], batch_size=10)
        assert result["dry_run"] is True
        assert result["processed"] == 0

    def test_start_pause_resume(self, db_session):
        plan = vo.plan_rebuild(db_session, model="m", dimensions=3,
                               dry_run=False)
        vo.start_rebuild(db_session, plan["run_id"])
        vo.pause_rebuild(db_session, plan["run_id"])
        resume = vo.resume_rebuild(db_session, plan["run_id"])
        assert resume["status"] == "RUNNING"
        assert "resume_from" in resume

    def test_resume_after_completed_not_resumable(self, db_session):
        plan = vo.plan_rebuild(db_session, model="m", dimensions=3,
                               dry_run=False, workspace_id=None)
        # complete by forcing totals
        row = db_session.query(VectorBackfillRun).get(plan["run_id"])
        row.total = 0
        db_session.commit()
        vo.start_rebuild(db_session, plan["run_id"])
        vo.rebuild_batch(db_session, plan["run_id"])
        result = vo.resume_rebuild(db_session, plan["run_id"])
        assert result["status"] == "NOT_RESUMABLE"

    def test_batch_progress_and_completion(self, db_session):
        plan = vo.plan_rebuild(db_session, model="m", dimensions=3,
                               dry_run=False)
        row = db_session.query(VectorBackfillRun).get(plan["run_id"])
        row.total = 10
        db_session.commit()
        vo.start_rebuild(db_session, plan["run_id"])
        vo.rebuild_batch(db_session, plan["run_id"], progress=10)
        assert db_session.query(VectorBackfillRun).get(
            plan["run_id"]).status == "COMPLETED"

    def test_failure_reporting(self, db_session):
        plan = vo.plan_rebuild(db_session, model="m", dimensions=3,
                               dry_run=False)
        result = vo.rebuild_failure(db_session, plan["run_id"],
                                    error="embed failed")
        assert result["failed"] == 1
        report = vo.rebuild_report(db_session, plan["run_id"])
        assert report["status"] == "FAILED"
        assert report["errors"]

    def test_rebuild_report_shape(self, db_session):
        plan = vo.plan_rebuild(db_session, model="m", dimensions=3,
                               dry_run=False)
        report = vo.rebuild_report(db_session, plan["run_id"])
        assert report["pct"] == 100.0


class TestBenchmark:
    def test_benchmark_metrics(self):
        result = vo.retrieval_benchmark(
            mode="hybrid",
            retrieved=[["d1", "d2"], ["d3", "d4"]],
            relevant=[{"d1"}, {"d3"}],
            latencies_ms=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # hits=1 of 2 retrieved per query → precision 0.5, recall 1.0
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0
        assert result["mrr"] == 1.0
        assert result["latency_ms"]["p50"] == 50.0
        assert result["latency_ms"]["p99"] == 100.0

    def test_benchmark_mrr_rank_two(self):
        result = vo.retrieval_benchmark(
            mode="vector",
            retrieved=[["d9", "d1"]],
            relevant=[{"d1"}])
        assert result["mrr"] == 0.5

    def test_benchmark_missing_hit_recall_zero(self):
        result = vo.retrieval_benchmark(
            mode="keyword",
            retrieved=[["d5"]],
            relevant=[{"d1"}])
        assert result["recall"] == 0.0
        assert result["mrr"] == 0.0

    def test_benchmark_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            vo.retrieval_benchmark(mode="hybrid", retrieved=[["a"]],
                                   relevant=[{"a"}, {"b"}])


# ===========================================================================
# Ingestion 3.0
# ===========================================================================

class TestResourceGovernor:
    def test_default_limits_bounded(self):
        limits = io3.resource_governor()
        assert limits["max_file_size_mb"] >= 1
        assert limits["max_page_count"] >= 1

    def test_oversized_file_rejected(self):
        decision = io3.governor_decision(file_size_mb=500,
                                         limits={"max_file_size_mb": 100,
                                                 "max_page_count": 1000,
                                                 "max_extraction_seconds":
                                                 600,
                                                 "max_memory_mb": 1024,
                                                 "max_ocr_pages": 500})
        assert decision["allowed"] is False
        assert decision["code"] == "INGESTION_RESOURCE_LIMIT"

    def test_within_limits_allowed(self):
        decision = io3.governor_decision(
            file_size_mb=10, page_count=50,
            limits=io3.resource_governor())
        assert decision["allowed"] is True

    def test_page_count_violation(self):
        decision = io3.governor_decision(page_count=2000,
                                         limits=io3.resource_governor())
        assert decision["allowed"] is False

    def test_ocr_pages_violation(self):
        decision = io3.governor_decision(ocr_pages=900,
                                         limits=io3.resource_governor())
        assert decision["allowed"] is False

    def test_stage_priorities_known(self):
        assert io3.stage_priority("OCR") == "LOW"
        assert io3.stage_priority("UPLOAD") == "HIGH"
        assert io3.stage_priority("UNKNOWN") == "NORMAL"


class TestStageDag:
    def test_chain_valid_and_ordered(self):
        nodes = [{"id": stage, "depends_on": [] if i == 0
                  else [io3.STAGES[i - 1]]}
                 for i, stage in enumerate(io3.STAGES)]
        result = io3.validate_stage_dag(nodes)
        assert result["valid"] is True
        assert result["order"][0] == "UPLOAD"

    def test_unknown_stage_invalid(self):
        nodes = [{"id": "UPLOAD", "depends_on": []},
                 {"id": "NOPE", "depends_on": ["UPLOAD"]}]
        result = io3.validate_stage_dag(nodes)
        assert result["valid"] is False

    def test_cycle_detected(self):
        nodes = [{"id": "UPLOAD", "depends_on": ["CHUNK"]},
                 {"id": "CHUNK", "depends_on": ["UPLOAD"]}]
        result = io3.validate_stage_dag(nodes)
        assert result["valid"] is False
        assert "cycle" in result["reason"]


class TestCheckpointsResume:
    def test_checkpoint_progress_advances(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = run(db_session, ws)
        io3.checkpoint_stage(db_session, run_id=row.id, stage="UPLOAD")
        io3.checkpoint_stage(db_session, run_id=row.id,
                             stage="VALIDATE")
        db_session.refresh(row)
        assert row.progress_pct == 20
        assert row.current_stage == "VALIDATE"

    def test_checkpoint_failure_records_error(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = run(db_session, ws)
        io3.checkpoint_stage(db_session, run_id=row.id, stage="OCR",
                             status="FAILED", error="provider down")
        db_session.refresh(row)
        assert row.error == "provider down"

    def test_unfinished_stages(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = run(db_session, ws)
        io3.checkpoint_stage(db_session, run_id=row.id, stage="UPLOAD")
        pending = io3.unfinished_stages(db_session, row.id)
        assert "UPLOAD" not in pending
        assert len(pending) == len(io3.STAGES) - 1

    def test_resume_restarts_only_unfinished(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = run(db_session, ws)
        io3.checkpoint_stage(db_session, run_id=row.id, stage="UPLOAD")
        io3.checkpoint_stage(db_session, run_id=row.id, stage="VALIDATE")
        resume = io3.resume_run(db_session, row.id)
        assert resume["status"] == "RUNNING"
        assert "UPLOAD" not in resume["restarted_stages"]
        assert "OCR" in resume["restarted_stages"]

    def test_resume_completes_empty_run(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = run(db_session, ws)
        for stage in io3.STAGES:
            io3.checkpoint_stage(db_session, run_id=row.id, stage=stage)
        resume = io3.resume_run(db_session, row.id)
        assert resume["status"] == "COMPLETED"

    def test_resume_failed_run_increments_retries(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = run(db_session, ws)
        io3.resume_run(db_session, row.id)
        db_session.refresh(row)
        assert row.retry_count == 1


class TestQuality:
    def test_quality_score_excellent(self):
        result = io3.quality_score(extraction_completeness=1.0,
                                   ocr_quality=0.95,
                                   metadata_completeness=1.0,
                                   chunk_quality=1.0,
                                   embedding_coverage=1.0)
        assert result["score"] >= 0.9
        assert result["grade"] == "EXCELLENT"

    def test_quality_score_poor(self):
        result = io3.quality_score(extraction_completeness=0.2,
                                   ocr_quality=0.1,
                                   metadata_completeness=0.1,
                                   chunk_quality=0.2,
                                   embedding_coverage=0.1)
        assert result["grade"] in ("POOR", "FAIR")

    def test_quality_explanations(self):
        result = io3.quality_score(extraction_completeness=1.0,
                                   ocr_quality=0.3,
                                   metadata_completeness=0.2,
                                   chunk_quality=1.0,
                                   embedding_coverage=0.5)
        assert any("OCR" in text for text in result["explanation"])
        assert any("embed" in text.lower()
                   for text in result["explanation"])

    def test_record_quality_persists(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = io3.record_quality(db_session, workspace_id=ws.id,
                                    extraction_completeness=1.0,
                                    ocr_quality=0.9,
                                    metadata_completeness=1.0,
                                    chunk_quality=1.0,
                                    embedding_coverage=1.0)
        assert result["report_id"]
        assert db_session.query(IngestionQualityReport).count() == 1


class TestFailureExplanations:
    def test_known_error_mapped(self):
        result = io3.explain_failure("provider_unavailable", stage="EMBED")
        assert "[EMBED]" in result["message"]
        assert result["retryable"] is True

    def test_file_too_large_not_retryable(self):
        result = io3.explain_failure("file_too_large")
        assert result["retryable"] is False

    def test_unknown_error_safe_message(self):
        result = io3.explain_failure("weird_internal")
        assert "stack" not in result["message"].lower()

    def test_detail_truncated(self):
        result = io3.explain_failure("unknown", detail="x" * 5000)
        assert result["detail"] is None or len(result["detail"]) <= 200

    def test_quarantine_explanation(self):
        result = io3.explain_failure("quarantined")
        assert "quarantine" in result["message"]


# ===========================================================================
# Performance 2.0
# ===========================================================================

class TestSingleFlight:
    def test_first_claim_is_leader(self):
        registry = perf2.SingleFlightRegistry()
        claim = registry.try_claim("k", now=1.0)
        assert claim["decision"] == "LEADER"

    def test_second_claim_joins(self):
        registry = perf2.SingleFlightRegistry()
        registry.try_claim("k", now=1.0)
        claim = registry.try_claim("k", now=2.0)
        assert claim["decision"] == "RUNNING"

    def test_claim_expires(self):
        registry = perf2.SingleFlightRegistry()
        registry.try_claim("k", now=1.0)
        claim = registry.try_claim("k", now=100.0, ttl_s=30.0)
        assert claim["decision"] == "LEADER"

    def test_complete_releases(self):
        registry = perf2.SingleFlightRegistry()
        registry.try_claim("k", now=1.0)
        registry.complete("k")
        assert registry.try_claim("k", now=2.0)["decision"] == "LEADER"

    def test_dedupe_policy_rejects_non_idempotent(self):
        assert perf2.dedupe_policy(idempotent=False)["dedupe_allowed"] \
            is False
        assert perf2.dedupe_policy(idempotent=True)["dedupe_allowed"] \
            is True


class TestStampede:
    def test_miss_regenerates(self):
        cache = {}
        result = perf2.stampede_protect(cache, "k", ttl_s=10, now=1.0)
        assert result["decision"] == "REGENERATE"

    def test_hit_returns_value(self):
        cache = {}
        perf2.complete_regeneration(cache, "k", "v", now=1.0)
        result = perf2.stampede_protect(cache, "k", ttl_s=100, now=2.0)
        assert result["decision"] == "HIT"
        assert result["value"] == "v"

    def test_stale_serves_while_regen(self):
        cache = {}
        perf2.complete_regeneration(cache, "k", "old", now=1.0)
        result = perf2.stampede_protect(cache, "k", ttl_s=10, now=50.0)
        assert result["decision"] == "REGENERATE_WITH_STALE"
        assert result["value"] == "old"

    def test_wait_while_regenerating(self):
        cache = {}
        perf2.complete_regeneration(cache, "k", "v", now=1.0)
        perf2.stampede_protect(cache, "k", ttl_s=10, now=50.0)
        second = perf2.stampede_protect(cache, "k", ttl_s=10, now=51.0)
        assert second["decision"] == "WAIT_FOR_REGEN"

    def test_cache_audit_issues(self):
        audit = perf2.cache_audit(ttl_s=None, tenant_isolated=False,
                                  invalidated_on_write=False,
                                  max_entries=None)
        assert audit["healthy"] is False
        assert len(audit["issues"]) >= 3

    def test_cache_stats(self):
        cache = {}
        perf2.complete_regeneration(cache, "a", 1, now=1.0)
        assert perf2.cache_stats(cache)["entries"] == 1
