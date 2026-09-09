"""Phase 24 tests — RAG correctness + knowledge consistency (Steps 19, 27,
28, 68, 69, 77).

Evidence sufficiency must gate confident claims, citations must resolve to
provided evidence, conflicts must be detected, freshness must classify
deterministically, and drift detection must produce explainable findings.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.main import app  # noqa: F401
from tests.shared_db import TestingSessionLocal, override_get_db

from app.core.database import get_db  # noqa: E402
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import rag8
from app.services import knowledge_maintenance2 as km2

app.dependency_overrides[get_db] = override_get_db

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
    from app.models.phase23 import KnowledgeFreshnessState, ConsistencyCheckRun
    db_session.query(KnowledgeFreshnessState).delete()
    db_session.query(ConsistencyCheckRun).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db):
    _counter[0] += 1
    n = _counter[0]
    u = User(email=f"p24k{n}@example", name="k", password_hash="x")
    db.add(u)
    db.commit()
    ws = Workspace(name=f"p24k-{n}", owner_id=u.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Step 19 — RAG evidence & citation validation
# ===========================================================================


class TestRAGEvidence:
    def test_sufficient_evidence_passes(self):
        evidence = [
            {"text": "The Eiffel Tower is located in Paris, France.",
             "document_id": 1, "chunk_id": 1},
            {"text": "Paris is the capital city of France.",
             "document_id": 2, "chunk_id": 2},
        ]
        result = rag8.evaluate_evidence_sufficiency(evidence)
        assert result["sufficient"] is True
        assert result["distinct_sources"] == 2
        assert result["usable_sources"] == 2

    def test_no_evidence_is_insufficient(self):
        result = rag8.evaluate_evidence_sufficiency([])
        assert result["sufficient"] is False
        assert "insufficient" in result["reason"]

    def test_single_document_is_insufficient(self):
        """Two chunks from ONE document are not independent evidence."""
        result = rag8.evaluate_evidence_sufficiency(
            [{"text": "Paris.", "document_id": 1, "chunk_id": 1},
             {"text": "France.", "document_id": 1, "chunk_id": 2}])
        assert result["sufficient"] is False

    def test_citation_validation_detects_missing(self):
        answer = "The tower is in Paris [1]. Revenue grew [2]."
        evidence = [{"text": "The Eiffel Tower is in Paris.",
                     "document_id": 1, "chunk_id": 1}]
        result = rag8.validate_citations(answer, evidence)
        assert 2 in result["invalid"], "[2] has no matching evidence"
        assert 1 in result["valid"]

    def test_valid_citations_resolve(self):
        answer = "Paris is in France [1]."
        evidence = [{"text": "Paris is in France.",
                     "document_id": 1, "chunk_id": 1}]
        result = rag8.validate_citations(answer, evidence)
        assert result["invalid"] == []
        assert result["coverage"] == 1.0

    def test_conflict_detection_cues(self):
        result = rag8.detect_conflicts(
            ["The meeting is on Monday.",
             "On the other hand, it is on Friday."])
        assert result["count"] >= 1
        assert result["conflict_signals"][0]["cue"] == "on the other hand"

    def test_agreeing_evidence_no_conflict(self):
        result = rag8.detect_conflicts(
            ["The meeting is on Monday.", "It is scheduled for Monday."])
        assert result["count"] == 0

    def test_claim_matrix_flags_unsupported(self):
        matrix = rag8.build_claim_matrix(
            ["The tower is located in Paris.",
             "The tower is 330 meters tall."],
            ["The Eiffel Tower is located in Paris."],
        )
        assert matrix["total"] == 2
        assert matrix["unsupported"] >= 1
        supported_claims = [c for c in matrix["claims"] if c["supported"]]
        assert supported_claims, "the Paris claim has lexical support"

    def test_query_understanding_shape(self):
        u = rag8.understand_query("What changed in the latest budget report?")
        assert "keywords" in u and "temporal" in u and "is_question" in u
        assert u["temporal"] is True, "'latest' signals a temporal query"
        assert u["is_question"] is True
        assert "budget" in u["keywords"] and "report" in u["keywords"]

    def test_retrieval_plan_bounds_sources(self):
        plan = rag8.plan_retrieval(
            {"keywords": ["x"], "temporal": False, "is_question": True,
             "complexity": 1}, max_sources=4)
        assert plan["max_sources"] == 4
        assert plan["modes"] == ["vector"]

    def test_complex_temporal_query_gets_hybrid_plan(self):
        plan = rag8.plan_retrieval(
            {"keywords": ["a", "b", "c"], "temporal": True,
             "is_question": True, "complexity": 3}, max_sources=8)
        assert "keyword" in plan["modes"]
        assert "freshness_sorted" in plan["modes"]


# ===========================================================================
# Steps 27-28 — knowledge freshness, drift, consistency
# ===========================================================================


class TestKnowledgeConsistency:
    def test_freshness_classification_fresh(self):
        result = km2.classify_freshness(
            updated_at=datetime.now(timezone.utc) - timedelta(days=1))
        assert result["state"] == "FRESH"
        assert result["age_days"] < 7

    def test_freshness_classification_aging(self):
        result = km2.classify_freshness(
            updated_at=datetime.now(timezone.utc) - timedelta(days=20))
        assert result["state"] == "AGING"
        assert result["reasons"], "classification must explain itself"

    def test_freshness_classification_expired(self):
        result = km2.classify_freshness(
            updated_at=datetime.now(timezone.utc) - timedelta(days=400))
        assert result["state"] in ("STALE", "EXPIRED")

    def test_freshness_expiration_policy_wins(self):
        result = km2.classify_freshness(
            updated_at=datetime.now(timezone.utc) - timedelta(days=10),
            expiration_days=5)
        assert result["state"] == "EXPIRED"
        assert any("expiration" in r for r in result["reasons"])

    def test_freshness_unknown(self):
        result = km2.classify_freshness(updated_at=None)
        assert result["state"] == "UNKNOWN"

    def test_low_reliability_accelerates_aging(self):
        reliable = km2.classify_freshness(
            updated_at=datetime.now(timezone.utc) - timedelta(days=20),
            source_reliability=1.0)
        unreliable = km2.classify_freshness(
            updated_at=datetime.now(timezone.utc) - timedelta(days=20),
            source_reliability=0.4)
        assert unreliable["effective_age_days"] > reliable["effective_age_days"]

    def test_drift_detection_runs_and_explains(self, db_session):
        ws = _mkws(db_session)
        result = km2.detect_drift(db_session, workspace_id=ws.id)
        assert "findings" in result
        for finding in result["findings"]:
            assert finding.get("kind"), "drift findings must be explainable"

    def test_maintenance_scan_is_bounded_and_honest(self, db_session):
        ws = _mkws(db_session)
        result = km2.run_maintenance_scan(db_session, workspace_id=ws.id,
                                          batch_size=100)
        assert result["dimensions_scanned"], "scan must cover all dimensions"
        plan = result["plan"]
        assert plan["safe_actions_done"] == [], \
            "scan alone must not execute actions"
        assert plan["proposed_actions"], "findings must become proposals"
        for dim, info in plan["dimensions"].items():
            assert "action" in info, f"{dim} must state its action"

    def test_freshness_overview(self, db_session):
        ws = _mkws(db_session)
        overview = km2.freshness_overview(db_session, workspace_id=ws.id)
        assert isinstance(overview, dict)
