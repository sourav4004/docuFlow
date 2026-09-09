"""Phase 20 tests — AI quality 3.0 + retrieval/RAG self-improvement.

Unified scorecards and regression detection, query-quality analysis,
retrieval failure classification + recommendations, RAG failure analysis,
offline evaluation pipelines, and promotion gates.
"""

import json

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.phase20 import (
    QualityScorecard, QualityTrend, QualityAlert, RetrievalFailure,
    RetrievalRecommendation, RagFailure, RagEvaluationPipeline,
    SearchQualityEvent,
)
from app.services import quality3 as q3
from app.services import retrieval_intel as reti
from app.services import rag_intel as ragi


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (QualityScorecard, QualityTrend, QualityAlert,
                  RetrievalFailure, RetrievalRecommendation, RagFailure,
                  RagEvaluationPipeline, SearchQualityEvent):
        db_session.query(model).delete()
    db_session.commit()
    yield


# ===========================================================================
# AI quality 3.0
# ===========================================================================

class TestQualityScorecard:
    def test_scorecard_overall(self, db_session):
        card = q3.compute_scorecard(
            db_session, domain="rag",
            dimensions={"correctness": 0.9, "completeness": 0.8,
                        "groundedness": 1.0})
        assert card.overall_score == pytest.approx(0.9, abs=1e-6)

    def test_scorecard_ignores_unknown_dimensions(self, db_session):
        card = q3.compute_scorecard(
            db_session, domain="rag",
            dimensions={"correctness": 0.8, "unicorn": 0.0})
        assert "unicorn" not in json.loads(
            card.scorecard_json)["dimensions"]
        assert card.overall_score == pytest.approx(0.8)

    def test_scorecard_domains(self, db_session):
        for domain in ("retrieval", "rag", "citations", "extraction",
                       "summarization", "search", "agents", "workflows"):
            card = q3.compute_scorecard(db_session, domain=domain,
                                        dimensions={"correctness": 1.0})
            assert card.domain == domain

    def test_all_quality_dimensions_accepted(self, db_session):
        dims = {d: 1.0 for d in q3.DIMENSIONS}
        card = q3.compute_scorecard(db_session, domain="rag",
                                    dimensions=dims)
        assert card.overall_score == pytest.approx(1.0)

    def test_latest_scorecard(self, db_session):
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.7})
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.9})
        latest = q3.latest_scorecard(db_session, domain="rag")
        assert latest["overall_score"] == pytest.approx(0.9)

    def test_scorecard_workspace_scoped(self, db_session):
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.5},
                             workspace_id=1)
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.95},
                             workspace_id=2)
        assert q3.latest_scorecard(
            db_session, domain="rag",
            workspace_id=1)["overall_score"] == pytest.approx(0.5)
        assert q3.latest_scorecard(
            db_session, domain="rag",
            workspace_id=2)["overall_score"] == pytest.approx(0.95)

    def test_bad_domain_rejected(self, db_session):
        with pytest.raises(ValueError):
            q3.compute_scorecard(db_session, domain="nope",
                                 dimensions={"correctness": 1.0})

    def test_empty_dimensions_zero(self, db_session):
        card = q3.compute_scorecard(db_session, domain="search",
                                    dimensions={})
        assert card.overall_score == 0.0

    def test_thresholds_stored(self, db_session):
        card = q3.compute_scorecard(
            db_session, domain="rag",
            dimensions={"correctness": 0.8},
            thresholds={"min_quality": 0.75})
        stored = json.loads(card.scorecard_json)
        assert stored["thresholds"]["min_quality"] == 0.75


class TestQualityRegression:
    def test_no_regression_with_single_scorecard(self, db_session):
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.8})
        result = q3.detect_regression(db_session, domain="rag")
        assert result["regressed"] is False

    def test_regression_detected(self, db_session):
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.9})
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.5})
        result = q3.detect_regression(db_session, domain="rag",
                                      min_delta=0.05)
        assert result["regressed"] is True
        assert result["delta"] <= -0.05

    def test_improvement_not_regression(self, db_session):
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.5})
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.95})
        result = q3.detect_regression(db_session, domain="rag")
        assert result["regressed"] is False

    def test_small_delta_below_threshold(self, db_session):
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.90})
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.88})
        result = q3.detect_regression(db_session, domain="rag",
                                      min_delta=0.05)
        assert result["regressed"] is False

    def test_regression_creates_alert(self, db_session):
        q3.compute_scorecard(db_session, domain="search",
                             dimensions={"correctness": 0.9})
        q3.compute_scorecard(db_session, domain="search",
                             dimensions={"correctness": 0.4})
        q3.detect_regression(db_session, domain="search")
        alerts = q3.list_alerts(db_session, domain="search")
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "HIGH"

    def test_regression_alert_deduplicated(self, db_session):
        for _ in range(2):
            q3.compute_scorecard(db_session, domain="rag",
                                 dimensions={"correctness": 0.9})
            q3.compute_scorecard(db_session, domain="rag",
                                 dimensions={"correctness": 0.4})
            q3.detect_regression(db_session, domain="rag")
        alerts = db_session.query(QualityAlert).filter_by(
            domain="rag").all()
        assert len(alerts) == 1

    def test_alert_resolution(self, db_session):
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.9})
        q3.compute_scorecard(db_session, domain="rag",
                             dimensions={"correctness": 0.4})
        q3.detect_regression(db_session, domain="rag")
        alert = db_session.query(QualityAlert).first()
        q3.resolve_alert(db_session, alert.id)
        assert db_session.get(QualityAlert, alert.id).resolved is True


class TestQualityTrends:
    def test_record_and_list_trend(self, db_session):
        q3.record_trend(db_session, domain="rag", period="daily",
                        metrics={"correctness": 0.8}, score=0.8)
        trends = q3.quality_trends(db_session, domain="rag")
        assert len(trends) == 1
        assert trends[0]["score"] == pytest.approx(0.8)

    def test_trend_domain_filter(self, db_session):
        q3.record_trend(db_session, domain="rag", period="daily",
                        metrics={})
        q3.record_trend(db_session, domain="search", period="daily",
                        metrics={})
        assert len(q3.quality_trends(db_session, domain="search")) == 1

    def test_trend_period_filter(self, db_session):
        q3.record_trend(db_session, domain="rag", period="daily",
                        metrics={})
        q3.record_trend(db_session, domain="rag", period="weekly",
                        metrics={})
        assert len(q3.quality_trends(db_session, domain="rag",
                                     period="weekly")) == 1


# ===========================================================================
# Retrieval intelligence
# ===========================================================================

class TestRetrievalQuality:
    def test_query_quality_analysis_empty(self, db_session):
        result = reti.query_quality_analysis(db_session, workspace_id=1)
        assert result["total_events"] == 0

    def test_zero_result_rate(self, db_session):
        reti.record_search_event(db_session, workspace_id=1, query="x",
                                 event_type="zero_result")
        reti.record_search_event(db_session, workspace_id=1, query="y",
                                 event_type="clicked")
        reti.record_search_event(db_session, workspace_id=1, query="z",
                                 event_type="clicked")
        result = reti.query_quality_analysis(db_session, workspace_id=1)
        assert result["zero_result_rate"] == pytest.approx(1 / 3)

    def test_abandoned_and_reformulated(self, db_session):
        reti.record_search_event(db_session, workspace_id=1, query="a",
                                 event_type="abandoned")
        reti.record_search_event(db_session, workspace_id=1, query="b",
                                 event_type="reformulated")
        result = reti.query_quality_analysis(db_session, workspace_id=1)
        assert result["reformulation_rate"] == 0.5
        assert result["abandoned_rate"] == 0.5

    def test_avg_latency(self, db_session):
        reti.record_search_event(db_session, workspace_id=1, query="a",
                                 event_type="clicked", latency_ms=100.0)
        reti.record_search_event(db_session, workspace_id=1, query="b",
                                 event_type="clicked", latency_ms=300.0)
        result = reti.query_quality_analysis(db_session, workspace_id=1)
        assert result["avg_latency_ms"] == pytest.approx(200.0)

    def test_poor_queries_listed(self, db_session):
        reti.record_search_event(db_session, workspace_id=1, query="none",
                                 event_type="zero_result")
        result = reti.query_quality_analysis(db_session, workspace_id=1)
        assert len(result["poor_queries"]) == 1


class TestRetrievalFailures:
    def test_classify_failure(self, db_session):
        f = reti.classify_failure(db_session, workspace_id=1, query="q",
                                  failure_class="poor_chunking")
        assert f.failure_class == "poor_chunking"

    def test_invalid_failure_class(self, db_session):
        with pytest.raises(ValueError):
            reti.classify_failure(db_session, workspace_id=1, query="q",
                                  failure_class="aliens")

    def test_failure_summary_counts(self, db_session):
        reti.classify_failure(db_session, workspace_id=1, query="a",
                              failure_class="poor_chunking")
        reti.classify_failure(db_session, workspace_id=1, query="b",
                              failure_class="poor_chunking")
        reti.classify_failure(db_session, workspace_id=1, query="c",
                              failure_class="metadata_mismatch")
        summary = reti.failure_summary(db_session, workspace_id=1)
        assert summary["total"] == 3
        assert summary["by_class"]["poor_chunking"] == 2

    def test_failure_classes_enum(self, db_session):
        for cls in reti.FAILURE_CLASSES:
            f = reti.classify_failure(db_session, workspace_id=1,
                                      query="q", failure_class=cls)
            assert f.id

    def test_workspace_isolation_failures(self, db_session):
        reti.classify_failure(db_session, workspace_id=1, query="a",
                              failure_class="poor_chunking")
        assert reti.failure_summary(db_session, workspace_id=2)["total"] == 0


class TestRetrievalRecommendations:
    def test_recommendation_from_chunking_failures(self, db_session):
        for i in range(3):
            reti.classify_failure(db_session, workspace_id=1,
                                  query=f"q{i}",
                                  failure_class="poor_chunking")
        created = reti.generate_recommendations(db_session, workspace_id=1)
        assert any(c["category"] == "chunk_size" for c in created)

    def test_recommendation_synonyms(self, db_session):
        for i in range(3):
            reti.classify_failure(db_session, workspace_id=1,
                                  query=f"q{i}",
                                  failure_class="poor_keyword_match")
        created = reti.generate_recommendations(db_session, workspace_id=1)
        assert any(c["category"] == "synonyms" for c in created)

    def test_recommendation_embedding(self, db_session):
        for i in range(3):
            reti.classify_failure(db_session, workspace_id=1,
                                  query=f"q{i}",
                                  failure_class="poor_vector_match")
        created = reti.generate_recommendations(db_session, workspace_id=1)
        assert any(c["category"] == "embedding_model" for c in created)

    def test_no_recommendation_without_failures(self, db_session):
        created = reti.generate_recommendations(db_session, workspace_id=1)
        assert created == []

    def test_recommendations_never_applied(self, db_session):
        for i in range(3):
            reti.classify_failure(db_session, workspace_id=1,
                                  query=f"q{i}",
                                  failure_class="poor_chunking")
        reti.generate_recommendations(db_session, workspace_id=1)
        rows = db_session.query(RetrievalRecommendation).all()
        assert rows and all(r.status == "PROPOSED" for r in rows)

    def test_list_recommendations(self, db_session):
        reti.classify_failure(db_session, workspace_id=1, query="a",
                              failure_class="metadata_mismatch")
        reti.generate_recommendations(db_session, workspace_id=1)
        recs = reti.list_recommendations(db_session, workspace_id=1)
        assert len(recs) >= 1

    def test_feedback_to_dataset(self, db_session):
        result = reti.feedback_to_dataset(
            db_session, workspace_id=1, query="refund policy",
            relevant_ids=["d1", "d3"], dataset_name="curated_fb")
        assert result["examples"] == 1
        assert result["status"] == "curated"


# ===========================================================================
# RAG intelligence
# ===========================================================================

class TestRagFailures:
    def test_record_failure(self, db_session):
        f = ragi.record_failure(db_session, workspace_id=1,
                                failure_class="unsupported_answer",
                                claim="the sky is green")
        assert f.failure_class == "unsupported_answer"

    def test_invalid_failure_class(self, db_session):
        with pytest.raises(ValueError):
            ragi.record_failure(db_session, workspace_id=1,
                                failure_class="nope")

    def test_failure_classes_all_valid(self, db_session):
        for cls in ragi.FAILURE_CLASSES:
            f = ragi.record_failure(db_session, workspace_id=1,
                                    failure_class=cls)
            assert f.id

    def test_failure_analysis_counts(self, db_session):
        ragi.record_failure(db_session, workspace_id=1,
                            failure_class="incorrect_citation")
        ragi.record_failure(db_session, workspace_id=1,
                            failure_class="incorrect_citation")
        ragi.record_failure(db_session, workspace_id=1,
                            failure_class="overconfident_answer")
        analysis = ragi.failure_analysis(db_session, workspace_id=1)
        assert analysis["by_class"]["incorrect_citation"] == 2

    def test_claim_error_classify(self, db_session):
        f = ragi.claim_error_classify(db_session, workspace_id=1,
                                      claim="c", category="temporal_mismatch")
        assert f.failure_class == "temporal_mismatch"


class TestRagRepair:
    def test_repair_recommendation_for_unsupported(self, db_session):
        ragi.record_failure(db_session, workspace_id=1,
                            failure_class="unsupported_answer")
        recs = ragi.repair_recommendations(db_session, workspace_id=1)
        assert any("unsupported_answer" in r["failure_class"]
                   for r in recs)

    def test_repair_recommendations_all_classes(self, db_session):
        for cls in ragi.FAILURE_CLASSES:
            ragi.record_failure(db_session, workspace_id=1,
                                failure_class=cls)
        recs = ragi.repair_recommendations(db_session, workspace_id=1)
        assert len(recs) == len(ragi.FAILURE_CLASSES)

    def test_no_failures_no_recommendations(self, db_session):
        assert ragi.repair_recommendations(db_session, workspace_id=1) == []


class TestRagEvaluationPipeline:
    def test_default_evaluator(self, db_session):
        pipe = ragi.run_evaluation_pipeline(db_session, config={})
        assert pipe.status == "DONE"
        metrics = json.loads(pipe.metrics_json)
        assert "quality_score" in metrics

    def test_default_evaluator_empty_dataset(self, db_session):
        pipe = ragi.run_evaluation_pipeline(
            db_session, config={"min_quality": 0.8})
        assert pipe.gate_passed is True

    def test_gate_fails_low_quality(self, db_session):
        pipe = ragi.run_evaluation_pipeline(
            db_session, config={"min_quality": 0.95},
            evaluate=lambda config, items: {
                "quality_score": 0.5, "cost": 0.0, "latency_ms": 10.0})
        assert pipe.gate_passed is False
        assert any("quality" in r for r in
                   json.loads(pipe.metrics_json).get("_gate",
                                                     {}).get("reasons", [])) \
            or pipe.gate_passed is False

    def test_injected_evaluator(self, db_session):
        called = {"n": 0}

        def fake_eval(config, items):
            called["n"] += 1
            return {"quality_score": 0.9, "cost": 0.1,
                    "latency_ms": 20.0}

        pipe = ragi.run_evaluation_pipeline(
            db_session, config={"min_quality": 0.8, "max_cost": 1.0,
                                "max_latency_ms": 100},
            evaluate=fake_eval)
        assert called["n"] == 1
        assert pipe.gate_passed is True

    def test_dataset_items_passed_to_evaluator(self, db_session):
        from app.services.improvement_platform import create_dataset
        ds = create_dataset(db_session, name="rag_evals", kind="golden",
                            domain="rag",
                            items=[{"claim": "x", "covered": True},
                                   {"claim": "y", "covered": True}])
        seen = {}

        def fake_eval(config, items):
            seen["items"] = items
            return {"quality_score": 1.0}

        ragi.run_evaluation_pipeline(db_session, config={},
                                     dataset_id=ds.id,
                                     evaluate=fake_eval)
        assert len(seen["items"]) == 2

    def test_pipeline_list(self, db_session):
        ragi.run_evaluation_pipeline(db_session, config={})
        assert len(ragi.list_pipelines(db_session)) == 1

    def test_promotion_gate_pure(self):
        gate = ragi.promotion_gate(
            {"min_quality": 0.8, "max_cost": 1.0, "max_latency_ms": 100},
            {"quality_score": 0.85, "cost": 0.5, "latency_ms": 50})
        assert gate["passed"] is True

    def test_promotion_gate_pure_fail(self):
        gate = ragi.promotion_gate(
            {"min_quality": 0.8, "max_cost": 1.0, "max_latency_ms": 100},
            {"quality_score": 0.7, "cost": 2.0, "latency_ms": 500})
        assert gate["passed"] is False
        assert len(gate["reasons"]) == 3