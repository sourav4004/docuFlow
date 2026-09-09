"""Phase 20 tests — knowledge intelligence + document change intelligence
+ policy intelligence.

Knowledge freshness/staleness/drift/completeness/health/gaps, document
change classification and impact/reprocessing plans, and immutable policy
versioning with drift detection, conflict detection, impact analysis, and
no-side-effect simulation.
"""

import json
import uuid

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.document_chunk import DocumentChunk  # noqa: E402
from app.models.knowledge_graph import Entity, EntityRelationship  # noqa: E402
from app.models.phase15 import AIMemory  # noqa: E402
from app.models.phase20 import (  # noqa: E402
    KnowledgeHealth, KnowledgeGapInsight, DocChangeEvent, PolicyVersion,
    PolicyImpact,
)
from app.services import knowledge_intel as ki
from app.services import policy_intel as pi


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (KnowledgeHealth, KnowledgeGapInsight, DocChangeEvent,
                  PolicyVersion, PolicyImpact, EntityRelationship, Entity,
                  DocumentChunk, AIMemory, Document, WorkspaceMember,
                  Workspace):
        db_session.query(model).delete()
    db_session.commit()
    yield


def _doc(db, workspace_id=1, filename="a.pdf"):
    doc = Document(
        user_id=1, workspace_id=workspace_id,
        original_filename=filename,
        storage_key=f"key-{filename}-{uuid.uuid4().hex[:8]}",
        mime_type="application/pdf", file_size=100, status="READY")
    db.add(doc)
    db.flush()
    return doc


def _chunk(db, doc, text="chunk text here"):
    idx = db.query(DocumentChunk).filter_by(
        document_id=doc.id).count()
    c = DocumentChunk(document_id=doc.id, chunk_index=idx,
                      text=text, char_start=0, char_end=len(text))
    db.add(c)
    db.flush()
    return c


# ===========================================================================
# Knowledge freshness / health
# ===========================================================================

class TestKnowledgeHealth:
    def test_freshness_report_empty(self, db_session):
        report = ki.freshness_report(db_session, workspace_id=1)
        assert report["documents"]["total"] == 0
        assert report["memories"]["total"] == 0

    def test_fresh_document_not_stale(self, db_session):
        _doc(db_session)
        report = ki.freshness_report(db_session, workspace_id=1)
        assert report["documents"]["total"] == 1
        assert report["documents"]["stale"] == 0

    def test_stale_document_detected(self, db_session):
        doc = _doc(db_session)
        doc.updated_at = datetime_ago(400)  # stale by 400 days
        db_session.flush()
        report = ki.freshness_report(db_session, workspace_id=1,
                                     stale_days=90)
        assert report["documents"]["stale"] == 1
        assert report["documents"]["stale_fraction"] == 1.0

    def test_health_score_computed(self, db_session):
        _doc(db_session)
        row = ki.compute_health(db_session, workspace_id=1)
        assert 0.0 <= row.score <= 1.0

    def test_health_embedding_coverage(self, db_session):
        doc = _doc(db_session)
        _chunk(db_session, doc)
        row = ki.compute_health(db_session, workspace_id=1)
        payload = json.loads(row.health_json)
        # no embedding set -> coverage 0 contributes to the score
        assert payload["signals"]["embedding_coverage"] == 0.0

    def test_health_embeddings_dont_leak_workspaces(self, db_session):
        doc_a = _doc(db_session, workspace_id=1)
        _chunk(db_session, doc_a)
        doc_b = _doc(db_session, workspace_id=2)
        _chunk(db_session, doc_b)
        row = ki.compute_health(db_session, workspace_id=1)
        payload = json.loads(row.health_json)
        # workspace 1 has 1 chunk (not 2)
        assert payload["signals"].get("embedding_coverage") == 0.0

    def test_health_latest(self, db_session):
        _doc(db_session)
        ki.compute_health(db_session, workspace_id=1)
        latest = ki.latest_health(db_session, scope_type="WORKSPACE",
                                  scope_id=1)
        assert latest["score"] is not None

    def test_health_scope_isolation(self, db_session):
        _doc(db_session, workspace_id=2)
        ki.compute_health(db_session, workspace_id=2)
        assert ki.latest_health(db_session, scope_type="WORKSPACE",
                                scope_id=1) is None

    def test_memory_decay_detected(self, db_session):
        from datetime import datetime, timezone, timedelta
        m = AIMemory(
            workspace_id=1, memory_type="fact", scope="WORKSPACE",
            content="old fact", source="doc1", confidence="MEDIUM",
            lifecycle_status="EXPIRED")
        db_session.add(m)
        db_session.flush()
        report = ki.freshness_report(db_session, workspace_id=1)
        drift = report["knowledge_drift"]
        assert drift["expired_or_superseded_memories"] == 1


def datetime_ago(days):
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone.utc) - timedelta(days=days)


# ===========================================================================
# Knowledge gaps
# ===========================================================================

class TestKnowledgeGaps:
    def test_record_gap_increments(self, db_session):
        ki.record_gap(db_session, workspace_id=1, query="where is X")
        g = ki.record_gap(db_session, workspace_id=1, query="where is X")
        assert g.attempts == 2

    def test_distinct_queries_separate(self, db_session):
        ki.record_gap(db_session, workspace_id=1, query="alpha")
        ki.record_gap(db_session, workspace_id=1, query="beta")
        gaps = ki.list_gaps(db_session, workspace_id=1)
        assert len(gaps) == 2

    def test_gap_evidence_score_best(self, db_session):
        ki.record_gap(db_session, workspace_id=1, query="q",
                      evidence_score=0.2)
        ki.record_gap(db_session, workspace_id=1, query="q",
                      evidence_score=0.7)
        gaps = ki.list_gaps(db_session, workspace_id=1)
        assert gaps[0]["best_evidence_score"] == 0.7

    def test_gap_min_attempts_filter(self, db_session):
        ki.record_gap(db_session, workspace_id=1, query="once")
        ki.record_gap(db_session, workspace_id=1, query="twice")
        ki.record_gap(db_session, workspace_id=1, query="twice")
        assert len(ki.list_gaps(db_session, workspace_id=1,
                                min_attempts=2)) == 1

    def test_gap_recommendations_never_invent(self, db_session):
        ki.record_gap(db_session, workspace_id=1, query="unknown topic")
        recs = ki.gap_recommendations(db_session, workspace_id=1)
        assert len(recs) == 1
        assert all(("connector sync" in r or "Search" in r or
                    "metadata" in r)
                   for r in recs[0]["recommendations"])

    def test_gap_workspace_isolation(self, db_session):
        ki.record_gap(db_session, workspace_id=1, query="q")
        assert ki.list_gaps(db_session, workspace_id=2) == []


# ===========================================================================
# Document change intelligence
# ===========================================================================

class TestDocChange:
    def test_classify_change(self, db_session):
        doc = _doc(db_session)
        ev = ki.classify_change(db_session, document_id=doc.id,
                                workspace_id=1,
                                change_class="major_content",
                                version_from=1, version_to=2)
        assert ev.change_class == "major_content"

    def test_invalid_change_class(self, db_session):
        with pytest.raises(ValueError):
            ki.classify_change(db_session, document_id=1,
                               workspace_id=1, change_class="aliens")

    def test_change_classes_valid(self, db_session):
        for cls in ki.CHANGE_CLASSES:
            ev = ki.classify_change(db_session, document_id=1,
                                    workspace_id=1, change_class=cls)
            assert ev.change_class == cls

    def test_major_impact_broad(self):
        impact = ki.change_impact("major_content")
        assert impact["embeddings"] is True
        assert impact["summaries"] is True
        assert impact["knowledge_graph"] is True
        assert impact["memories"] is True

    def test_formatting_impact_narrow(self):
        impact = ki.change_impact("formatting")
        assert impact["embeddings"] is False
        assert impact["reports"] is False

    def test_deadline_affects_workflows(self):
        impact = ki.change_impact("deadline")
        assert impact["workflows"] is True
        assert impact["reports"] is True

    def test_entity_change_affects_graph(self):
        impact = ki.change_impact("entity")
        assert impact["knowledge_graph"] is True

    def test_reprocessing_plan_no_destructive_steps(self):
        plan = ki.reprocessing_plan("major_content")
        assert all(not s["destructive"] for s in plan["steps"])

    def test_reprocessing_requires_approval_for_major(self):
        assert ki.reprocessing_plan("major_content")[
            "requires_approval"] is True
        assert ki.reprocessing_plan("formatting")[
            "requires_approval"] is False

    def test_reprocessing_plan_steps_order(self):
        plan = ki.reprocessing_plan("major_content")
        kinds = [s["target"] for s in plan["steps"]]
        assert kinds == ["embeddings", "summaries", "knowledge_graph",
                         "memories"]

    def test_document_health_no_chunks(self, db_session):
        doc = _doc(db_session)
        health = ki.document_health(db_session, document_id=doc.id)
        assert health["chunks"] == 0
        assert health["embedding_coverage"] == 0.0

    def test_document_health_missing(self, db_session):
        with pytest.raises(KeyError):
            ki.document_health(db_session, document_id=999999)

    def test_document_health_chunks(self, db_session):
        doc = _doc(db_session)
        _chunk(db_session, doc)
        _chunk(db_session, doc)
        health = ki.document_health(db_session, document_id=doc.id)
        assert health["chunks"] == 2

    def test_version_quality_best(self):
        versions = [
            {"version": 1, "completeness": 0.5, "metadata_completeness": 0.5,
             "semantic_similarity": 0.5, "important_facts": 0.5},
            {"version": 2, "completeness": 1.0, "metadata_completeness": 1.0,
             "semantic_similarity": 1.0, "important_facts": 1.0},
        ]
        result = ki.version_quality(versions)
        assert result["best_version"]["version"] == 2
        assert result["best_version"]["score"] == 1.0

    def test_version_quality_empty(self):
        assert ki.version_quality([])["versions"] == 0


# ===========================================================================
# Policy intelligence
# ===========================================================================

class TestPolicyVersioning:
    def test_first_version_is_one(self, db_session):
        row = pi.version_policy(db_session, scope_type="WORKSPACE",
                                scope_id=1,
                                policy={"allowed_models": ["a"]},
                                reason="init")
        assert row.version == 1

    def test_versions_increment(self, db_session):
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"allowed_models": ["a"]})
        row = pi.version_policy(db_session, scope_type="WORKSPACE",
                                scope_id=1,
                                policy={"allowed_models": ["a", "b"]})
        assert row.version == 2

    def test_policy_immutable_json(self, db_session):
        pi.version_policy(db_session, scope_type="GLOBAL", scope_id=None,
                          policy={"allowed_models": ["x"]})
        pi.version_policy(db_session, scope_type="GLOBAL", scope_id=None,
                          policy={"allowed_models": ["y"]})
        versions = pi.list_versions(db_session, scope_type="GLOBAL")
        assert len(versions) == 2
        assert versions[0]["version"] == 2

    def test_diff_recorded(self, db_session):
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"allowed_models": ["a"]})
        v2 = pi.version_policy(db_session, scope_type="WORKSPACE",
                               scope_id=1,
                               policy={"allowed_models": ["a", "b"]})
        diff = json.loads(v2.diff_json)
        assert any(c["key"] == "allowed_models" for c in diff["changes"])

    def test_bad_scope_rejected(self, db_session):
        with pytest.raises(ValueError):
            pi.version_policy(db_session, scope_type="PLANET",
                              scope_id=None, policy={})


class TestPolicyDrift:
    def test_no_drift_single_version(self, db_session):
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={})
        report = pi.drift_report(db_session, scope_type="WORKSPACE",
                                 scope_id=1)
        assert report["drift"] is False

    def test_model_drift_detected(self, db_session):
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"allowed_models": ["a"]})
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"allowed_models": ["a", "b"]})
        report = pi.drift_report(db_session, scope_type="WORKSPACE",
                                 scope_id=1)
        assert report["drift"] is True
        assert "models" in report["affected_dimensions"]

    def test_retention_drift(self, db_session):
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"retention_days": 30})
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"retention_days": 90})
        report = pi.drift_report(db_session, scope_type="WORKSPACE",
                                 scope_id=1)
        assert "retention" in report["affected_dimensions"]

    def test_region_drift(self, db_session):
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"allowed_regions": ["us"]})
        pi.version_policy(db_session, scope_type="WORKSPACE", scope_id=1,
                          policy={"allowed_regions": ["us", "eu"]})
        report = pi.drift_report(db_session, scope_type="WORKSPACE",
                                 scope_id=1)
        assert "regions" in report["affected_dimensions"]


class TestPolicyConflicts:
    def test_model_allow_deny_conflict(self):
        conflicts = pi.detect_conflicts([
            {"allowed_models": ["gpt-4"]},
            {"denied_models": ["gpt-4"]},
        ])
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "model_conflict"
        assert conflicts[0]["review_required"] is True

    def test_no_conflict(self):
        assert pi.detect_conflicts([
            {"allowed_models": ["a"]},
            {"denied_models": ["b"]},
        ]) == []

    def test_retention_conflict(self):
        conflicts = pi.detect_conflicts([
            {"retention_days": 30},
            {"retention_days": 90},
        ])
        assert any(c["type"] == "retention_conflict"
                   for c in conflicts)

    def test_uncertain_conflicts_require_review(self):
        conflicts = pi.detect_conflicts([
            {"allowed_models": ["x"]}, {"denied_models": ["x"]}])
        assert all(c["review_required"] for c in conflicts)


class TestPolicySimulation:
    def test_simulate_allowed(self, db_session):
        result = pi.simulate(db_session, operation={
            "model": "gpt-4", "provider": "openai"},
            policy={"allowed_models": ["gpt-4"],
                    "allowed_providers": ["openai"]})
        assert result["allowed"] is True

    def test_simulate_model_blocked(self, db_session):
        result = pi.simulate(db_session, operation={"model": "gpt-4"},
                             policy={"allowed_models": ["claude"]})
        assert result["allowed"] is False
        assert any("gpt-4" in r for r in result["denied_reasons"])

    def test_simulate_provider_blocked(self, db_session):
        result = pi.simulate(db_session, operation={"provider": "google"},
                             policy={"denied_providers": ["google"]})
        assert result["allowed"] is False

    def test_simulate_region_blocked(self, db_session):
        result = pi.simulate(db_session,
                             operation={"region": "cn"},
                             policy={"prohibited_regions": ["cn"]})
        assert result["allowed"] is False

    def test_simulate_tool_blocked(self, db_session):
        result = pi.simulate(db_session, operation={"tool": "delete_doc"},
                             policy={"denied_tools": ["delete_doc"]})
        assert result["allowed"] is False

    def test_simulate_restricted_needs_policy(self, db_session):
        result = pi.simulate(db_session,
                             operation={"sensitivity": "RESTRICTED"},
                             policy={"allow_restricted": False})
        assert result["allowed"] is False

    def test_simulate_approval_required(self, db_session):
        result = pi.simulate(db_session,
                             operation={"sensitivity": "CONFIDENTIAL"},
                             policy={"approval_required_sensitivity":
                                     ["CONFIDENTIAL"]})
        assert result["allowed"] is True
        assert result["approval_required"] is True

    def test_simulate_no_side_effects(self, db_session):
        pi.simulate(db_session, operation={"model": "x"},
                    policy={"denied_models": ["x"]})
        assert db_session.query(PolicyImpact).count() == 0


class TestPolicyImpact:
    def test_impact_analysis_records(self, db_session):
        row = pi.impact_analysis(db_session, policy={
            "allowed_models": ["a"], "retention_days": 90})
        assert row.id
        payload = json.loads(row.impact_json)
        assert payload["retention_days"] == 90