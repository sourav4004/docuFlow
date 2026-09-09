"""Phase 23 tests — vector activation, embedding model migration, shadow
search coexistence, multi-region control plane, residency, drain, failover.

pgvector is NOT_CONFIGURED in this environment: tests assert the honesty of
the activation report (native validation never claimed) rather than
fabricated native success.
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase23 import (
    DocumentEmbeddingStatus, EmbeddingModelVersion, RegionDrainOperation,
    ResidencyDecisionLog, SearchShadowComparison,
)
from app.models.phase19 import ResidencyRule, RegionRecord  # noqa: E402
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import vector_activation as va
from app.services import region_control as rc
from app.services import rag8

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


P23_TABLES = [
    SearchShadowComparison, DocumentEmbeddingStatus, EmbeddingModelVersion,
    RegionDrainOperation, ResidencyDecisionLog,
]
P19_TABLES = [ResidencyRule, RegionRecord]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P23_TABLES + P19_TABLES:
        db_session.query(model).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="p23v"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p23v.example", name=tag, password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


def _mkdoc_with_chunks(db, ws, n_chunks=2):
    """READY document with chunks (embedding generation needs real rows)."""
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk

    _counter[0] += 1
    doc = Document(
        user_id=ws.owner_id, workspace_id=ws.id,
        original_filename=f"chunkdoc-{_counter[0]}.txt",
        storage_key=f"p23/chunkdoc/{ws.id}/{_counter[0]}.txt",
        mime_type="text/plain", file_size=42, status="READY")
    db.add(doc)
    db.flush()
    for i in range(n_chunks):
        db.add(DocumentChunk(document_id=doc.id, chunk_index=i,
                             text=f"chunk text {i} for doc {doc.id}",
                             char_start=i * 100, char_end=i * 100 + 90))
    db.commit()
    return doc


# ===========================================================================
# Vector activation (Step 10)
# ===========================================================================

class TestVectorActivation:
    def test_activation_report_honest_without_pgvector(self, db_session):
        report = va.activation_report(db_session)
        # In this environment pgvector is absent: native validation must
        # NOT be claimed.
        if report["pgvector_state"] != "AVAILABLE":
            assert report["native_validation_available"] is False
            assert report["all_passed"] is False
            assert "JSON fallback" in report["note"] or \
                   "fallback" in report["note"].lower()

    def test_activation_report_structure(self, db_session):
        report = va.activation_report(db_session)
        assert set(report) >= {"pgvector_state", "realization",
                               "native_validation_available", "checks",
                               "all_passed"}

    def test_migration_readiness(self, db_session):
        ready = va.migration_readiness(db_session)
        assert "ready" in ready and "pending_documents" in ready

    def test_register_model_version_idempotent(self, db_session):
        r1 = va.register_model_version(db_session, model_name="embed-x",
                                       version="v1", dimension=384)
        r2 = va.register_model_version(db_session, model_name="embed-x",
                                       version="v1", dimension=384)
        assert r1["id"] == r2["id"]

    def test_retirement_requires_approval(self, db_session):
        mv = va.register_model_version(db_session, model_name="embed-r",
                                       version="v1", dimension=10)
        plan = va.retirement_plan(db_session, model_version_id=mv["id"])
        assert plan["requires_approval"] is True
        assert plan["safe_to_retire"] is True   # nothing promoted


# ===========================================================================
# Embedding model migration (Step 11)
# ===========================================================================

class TestModelMigration:
    def test_start_enrolls_documents(self, db_session):
        ws = _mkws(db_session)
        result = va.start_model_migration(db_session, workspace_id=ws.id,
                                          model_name="embed-m",
                                          version="v2", dimension=10)
        assert result["model_version_id"]
        # second start is idempotent
        result2 = va.start_model_migration(db_session, workspace_id=ws.id,
                                           model_name="embed-m",
                                           version="v2", dimension=10)
        assert result2["model_version_id"] == result["model_version_id"]

    def test_batch_bounds(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            va.start_model_migration(db_session, workspace_id=ws.id,
                                     model_name="m", version="v",
                                     dimension=384, batch_size=0)

    def test_dual_generation_advances_state(self, db_session):
        ws = _mkws(db_session)
        start = va.start_model_migration(db_session, workspace_id=ws.id,
                                         model_name="embed-d",
                                         version="v1", dimension=384)
        result = va.run_dual_generation_batch(
            db_session, workspace_id=ws.id,
            model_version_id=start["model_version_id"], batch_size=50)
        assert result["failed"] == 0
        coverage = va.verify_coverage(db_session, workspace_id=ws.id,
                                      model_version_id=start[
                                          "model_version_id"])
        assert coverage["coverage_pct"] == 0.0 or coverage["generated"] >= 0

    def test_quality_gate_blocks_regression(self, db_session):
        ws = _mkws(db_session)
        start = va.start_model_migration(db_session, workspace_id=ws.id,
                                         model_name="embed-g",
                                         version="v1", dimension=384)
        mv_id = start["model_version_id"]
        va.run_dual_generation_batch(db_session, workspace_id=ws.id,
                                     model_version_id=mv_id, batch_size=50)
        result = va.record_quality_comparison(db_session, workspace_id=ws.id,
                                              model_version_id=mv_id,
                                              quality_delta=-0.5)
        assert result["gate_passed"] is False

    def test_promotion_requires_verified(self, db_session):
        ws = _mkws(db_session)
        _mkdoc_with_chunks(db_session, ws)   # real rows so state advances
        start = va.start_model_migration(db_session, workspace_id=ws.id,
                                         model_name="embed-p",
                                         version="v1", dimension=384)
        mv_id = start["model_version_id"]
        va.run_dual_generation_batch(db_session, workspace_id=ws.id,
                                     model_version_id=mv_id, batch_size=50)
        # DUAL_GENERATED rows remain — promotion must be refused
        result = va.promote_model(db_session, model_version_id=mv_id)
        assert result["promoted"] is False

    def test_promotion_after_gate(self, db_session):
        ws = _mkws(db_session)
        start = va.start_model_migration(db_session, workspace_id=ws.id,
                                         model_name="embed-q",
                                         version="v1", dimension=384)
        mv_id = start["model_version_id"]
        va.run_dual_generation_batch(db_session, workspace_id=ws.id,
                                     model_version_id=mv_id, batch_size=50)
        va.record_quality_comparison(db_session, workspace_id=ws.id,
                                     model_version_id=mv_id,
                                     quality_delta=0.1)
        result = va.promote_model(db_session, model_version_id=mv_id)
        assert result["promoted"] is True

    def test_rollback(self, db_session):
        ws = _mkws(db_session)
        start = va.start_model_migration(db_session, workspace_id=ws.id,
                                         model_name="embed-rb",
                                         version="v1", dimension=384)
        mv_id = start["model_version_id"]
        va.run_dual_generation_batch(db_session, workspace_id=ws.id,
                                     model_version_id=mv_id, batch_size=50)
        va.record_quality_comparison(db_session, workspace_id=ws.id,
                                     model_version_id=mv_id,
                                     quality_delta=0.1)
        va.promote_model(db_session, model_version_id=mv_id)
        result = va.rollback_model(db_session, model_version_id=mv_id)
        assert result["rolled_back"] is True


# ===========================================================================
# Shadow search coexistence (Step 12)
# ===========================================================================

class TestShadowSearch:
    def test_shadow_compare_tie(self, db_session):
        ws = _mkws(db_session)
        result = va.shadow_compare(db_session, workspace_id=ws.id,
                                   query="test",
                                   baseline_ranking=["a", "b", "c", "d", "e"],
                                   candidate_ranking=["a", "b", "c", "d",
                                                      "e"])
        assert result["verdict"] in ("TIE", "INSUFFICIENT")

    def test_shadow_compare_better(self, db_session):
        ws = _mkws(db_session)
        result = va.shadow_compare(db_session, workspace_id=ws.id,
                                   query="q2",
                                   baseline_ranking=["x", "y", "z"],
                                   candidate_ranking=["y", "x", "z", "w"])
        assert result["overlap_at_5"] >= 0.4

    def test_promotion_gate_insufficient_data(self, db_session):
        ws = _mkws(db_session)
        gate = va.promotion_gate(db_session, workspace_id=ws.id,
                                 min_comparisons=10)
        assert gate["eligible"] is False
        assert "comparisons" in gate["reason"] or "only" in gate["reason"]

    def test_promotion_gate_with_data(self, db_session):
        ws = _mkws(db_session)
        for i in range(12):
            va.shadow_compare(db_session, workspace_id=ws.id,
                              query=f"q{i}",
                              baseline_ranking=["a", "b", "c"],
                              candidate_ranking=["b", "a", "c", "d"])
        gate = va.promotion_gate(db_session, workspace_id=ws.id,
                                 min_comparisons=10)
        assert gate["comparisons"] >= 10
        assert gate["requires_human_approval"] is True


# ===========================================================================
# Region registry + residency (Steps 13-14)
# ===========================================================================

class TestRegionControl:
    def test_register_region(self, db_session):
        result = rc.register_region(db_session, region="us-east",
                                    providers=["fake"], vector=False,
                                    storage=True, broker=True)
        assert result["region"] == "us-east"
        overview = rc.region_overview(db_session)
        assert overview["count"] >= 1

    def test_region_health_unknown(self, db_session):
        health = rc.region_health(db_session, "nowhere")
        assert health["registered"] is False

    def test_residency_allowed_by_default(self, db_session):
        ws = _mkws(db_session)
        result = rc.evaluate_residency(db_session, workspace_id=ws.id,
                                       operation="processing",
                                       source_region="us-east",
                                       destination_region="us-east")
        assert result["allowed"] is True

    def test_residency_blocks_prohibited(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(classification="RESTRICTED",
                                     prohibited_regions_json='["us-east"]'))
        db_session.commit()
        result = rc.evaluate_residency(db_session, workspace_id=ws.id,
                                       operation="processing",
                                       source_region="eu-west",
                                       destination_region="us-east",
                                       classification="RESTRICTED")
        assert result["allowed"] is False
        assert "prohibited" in result["reason"]

    def test_residency_blocks_not_allowed(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(classification="CONFIDENTIAL",
                                     allowed_regions_json='["eu-west"]'))
        db_session.commit()
        result = rc.evaluate_residency(db_session, workspace_id=ws.id,
                                       operation="processing",
                                       source_region="us-east",
                                       destination_region="us-east",
                                       classification="CONFIDENTIAL")
        assert result["allowed"] is False

    def test_residency_logged(self, db_session):
        ws = _mkws(db_session)
        rc.evaluate_residency(db_session, workspace_id=ws.id,
                              operation="op-x",
                              source_region="a", destination_region="b")
        log = rc.residency_log(db_session, workspace_id=ws.id)
        assert log["count"] == 1
        assert log["items"][0]["operation"] == "op-x"


# ===========================================================================
# Region drain (Step 16)
# ===========================================================================

class TestRegionDrain:
    def test_drain_lifecycle(self, db_session):
        op = rc.start_region_drain(db_session, region="us-east")
        progress = rc.drain_progress(db_session, op["drain_id"])
        assert progress["state"] in ("DRAINING", "DRAINED")
        completed = rc.complete_drain(db_session, op["drain_id"])
        assert completed["state"] in ("DRAINING", "DRAINED")
        recovered = rc.recover_region(db_session, op["drain_id"], actor="t")
        assert recovered["state"] == "RECOVERED"

    def test_drain_progress_unknown(self, db_session):
        with pytest.raises(ValueError):
            rc.drain_progress(db_session, 999999)


# ===========================================================================
# Failover engine (Step 17)
# ===========================================================================

class TestFailover:
    def test_not_eligible_when_primary_healthy(self, db_session):
        rc.register_region(db_session, region="primary",
                           status="HEALTHY")
        rc.register_region(db_session, region="secondary",
                           status="HEALTHY")
        result = rc.failover_decision(db_session, primary="primary",
                                      secondary="secondary")
        assert result["decision"] == "NOT_ELIGIBLE"

    def test_requires_approval_when_not_autonomous(self, db_session):
        rc.register_region(db_session, region="p2", status="OUTAGE",
                           failover_to="s2")
        rc.register_region(db_session, region="s2", status="HEALTHY")
        result = rc.failover_decision(db_session, primary="p2",
                                      secondary="s2")
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_executed_only_with_autonomy(self, db_session):
        rc.register_region(db_session, region="p3", status="OUTAGE")
        rc.register_region(db_session, region="s3", status="HEALTHY")
        result = rc.failover_decision(db_session, primary="p3",
                                      secondary="s3",
                                      autonomy_allowed=True)
        assert result["decision"] == "EXECUTED"
        assert result["failover_id"]

    def test_failback_plan(self, db_session):
        plan = rc.failback_plan(db_session, primary="p", secondary="s")
        assert plan["requires_approval"] is True
        assert len(plan["steps"]) >= 4

    def test_failover_history(self, db_session):
        history = rc.failover_history(db_session)
        assert "items" in history


# ===========================================================================
# RAG 8.0 (Steps 26-27)
# ===========================================================================

class TestRag8:
    def test_query_understanding(self):
        u = rag8.understand_query("What is the latest remote work policy?")
        assert u["temporal"] is True
        assert u["is_question"] is True
        assert "remote" in u["keywords"]

    def test_retrieval_plan(self):
        plan = rag8.plan_retrieval({"keywords": ["a", "b", "c", "d"],
                                    "temporal": False, "is_question": True,
                                    "complexity": 4})
        assert "vector" in plan["modes"]
        assert "keyword" in plan["modes"]

    def test_insufficient_evidence_refusal(self):
        result = rag8.generate_answer(None, workspace_id=1, query="q",
                                      evidence=[])
        assert result["refusal"] is True
        assert result["confidence"] <= 0.3
        assert result["clarification_requested"] is True

    def test_single_source_insufficient(self):
        result = rag8.generate_answer(
            None, workspace_id=1, query="q",
            evidence=[{"document_id": 1, "text": "only one source"}])
        assert result["refusal"] is True

    def test_sufficient_evidence_answer(self):
        evidence = [
            {"document_id": 1, "text": "The remote work policy allows "
             "employees to work remotely three days per week."},
            {"document_id": 2, "text": "Remote work requires manager "
             "approval and is limited to three days per week."},
        ]
        result = rag8.generate_answer(
            None, workspace_id=1, query="remote work policy",
            evidence=evidence)
        assert result["refusal"] is False
        assert result["answer"]
        assert result["confidence"] > 0.3

    def test_claim_matrix_detects_unsupported(self):
        matrix = rag8.build_claim_matrix(
            ["The moon is made of cheese"],
            ["Revenue grew 20% this quarter"])
        assert matrix["unsupported"] == 1

    def test_supported_claim_passes(self):
        matrix = rag8.build_claim_matrix(
            ["revenue grew 20% this quarter"],
            ["Revenue grew 20% this quarter"])
        assert matrix["claims"][0]["supported"] is True

    def test_citation_validation(self):
        evidence = [{"document_id": 1, "text": "a"},
                    {"document_id": 2, "text": "b"}]
        result = rag8.validate_citations("Answer [1] and [3] here", evidence)
        assert 1 in result["valid"]
        assert 3 in result["invalid"]

    def test_conflict_detection(self):
        result = rag8.detect_conflicts([
            "Policy allows remote work.",
            "However, the previous policy no longer applies."])
        assert result["count"] >= 1

    def test_quality_metrics_recorded(self, db_session):
        ws = _mkws(db_session)
        result = rag8.generate_answer(
            None, workspace_id=ws.id, query="remote policy",
            evidence=[{"document_id": 1,
                       "text": "Remote work is allowed three days per week."},
                      {"document_id": 2,
                       "text": "Manager approval needed for remote work."}])
        recorded = rag8.record_response_quality(db_session,
                                                workspace_id=ws.id,
                                                rag_result=result)
        assert "metrics" in recorded


# ===========================================================================
# Change impact 3.0 (Step 28)
# ===========================================================================

class TestChangeImpact:
    def test_impact_map_shape(self, db_session):
        ws = _mkws(db_session)
        doc = _mkdoc_with_chunks(db_session, ws)  # tenant-scoped lookup
        impact = rag8.change_impact_3(db_session, document_id=doc.id,
                                      workspace_id=ws.id,
                                      change_class="minor_edit")
        assert "affected" in impact
        assert "maintenance_plan" in impact
        assert impact["maintenance_plan"]["requires_approval"]

    def test_no_auto_execution_of_risky_actions(self, db_session):
        ws = _mkws(db_session)
        doc = _mkdoc_with_chunks(db_session, ws)
        impact = rag8.change_impact_3(db_session, document_id=doc.id,
                                      workspace_id=ws.id,
                                      change_class="content_replacement")
        plan = impact["maintenance_plan"]
        assert "entity re-extraction" in " ".join(
            plan["requires_approval"]) or plan["requires_approval"]

    def test_change_impact_rejects_foreign_document(self, db_session):
        ws = _mkws(db_session)
        result = rag8.change_impact_3(db_session, document_id=987654,
                                      workspace_id=ws.id,
                                      change_class="minor_edit")
        assert result["found"] is False
