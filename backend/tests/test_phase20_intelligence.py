"""Phase 20 tests — agent/workflow/memory/graph/search intelligence.

Agent success metrics + deterministic failure classification + plan-quality
scoring + safety score; workflow success analytics, bottlenecks, failure
hotspots, optimization recommendations and side-effect-free simulation;
memory quality/decay/consolidation/conflicts/user controls; graph health,
drift, recommendations and quality evaluation; search quality scoring,
ranking experiments, safe explanations and personalization safety.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models.phase15 import AIMemory  # noqa: E402
from app.models.knowledge_graph import Entity, EntityRelationship  # noqa: E402
from app.models.phase20 import (  # noqa: E402
    AgentIntelligence, WorkflowIntelligence, MemoryIntelligence,
    GraphHealth, GraphRecommendation, SearchQualityEvent,
)
from app.services import agent_intel as ai2  # noqa: E402
from app.services import workflow_intel2 as wi  # noqa: E402
from app.services import memory_intel as mi  # noqa: E402
from app.services import graph_intel as gi  # noqa: E402
from app.services import search_intel as si  # noqa: E402

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


PH20_TABLES = [
    AgentIntelligence, WorkflowIntelligence, MemoryIntelligence,
    GraphHealth, GraphRecommendation, SearchQualityEvent,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in PH20_TABLES:
        db_session.query(model).delete()
    db_session.query(EntityRelationship).delete()
    db_session.query(Entity).delete()
    db_session.query(AIMemory).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p20i"):
    _counter[0] += 1
    user = User(name=f"P20I {_counter[0]}",
                email=f"{tag}{_counter[0]}@p20-i.example",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p20 intel ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def entity(db, ws, name, etype="person", confidence=1.0):
    row = Entity(workspace_id=ws.id, name=name, entity_type=etype,
                 confidence=confidence)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def memory(db, ws, content, memory_type="fact", confidence="MEDIUM",
           source="test", lifecycle_status="ACTIVE", updated_days_ago=0,
           expires_days=None, user_id=None):
    row = AIMemory(workspace_id=ws.id, memory_type=memory_type,
                   scope="WORKSPACE", content=content, source=source,
                   confidence=confidence, lifecycle_status=lifecycle_status,
                   user_id=user_id,
                   updated_at=datetime.now(timezone.utc)
                   - timedelta(days=updated_days_ago))
    if expires_days is not None:
        row.expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def search_event(db, ws, event_type, count=1):
    for _ in range(count):
        db.add(SearchQualityEvent(
            workspace_id=ws.id, query_hash=uuid.uuid4().hex,
            event_type=event_type))
    db.commit()


# ===========================================================================
# Agent intelligence
# ===========================================================================

class TestAgentMetrics:
    def test_record_metrics_persists(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = ai2.record_metrics(
            db_session, execution_id="exec-1", workspace_id=ws.id,
            metrics={"success": True, "cost": 0.1, "duration_s": 5.0})
        assert row.id is not None
        got = db_session.get(AgentIntelligence, row.id)
        assert got.workspace_id == ws.id

    def test_unknown_failure_class_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            ai2.record_metrics(db_session, execution_id="e",
                               workspace_id=ws.id, metrics={},
                               failure_class="teleport")

    def test_success_metrics_rate(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ai2.record_metrics(db_session, execution_id="a", workspace_id=ws.id,
                           metrics={"success": True, "cost": 1.0,
                                    "duration_s": 2.0})
        ai2.record_metrics(db_session, execution_id="b", workspace_id=ws.id,
                           metrics={"success": False, "cost": 2.0,
                                    "duration_s": 4.0})
        out = ai2.success_metrics(db_session, workspace_id=ws.id)
        assert out["total"] == 2
        assert out["success_rate"] == 0.5
        assert out["total_cost"] == 3.0
        assert out["avg_duration_s"] == 3.0

    def test_success_metrics_handoffs_cancelled(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ai2.record_metrics(db_session, execution_id="a", workspace_id=ws.id,
                           metrics={"success": True, "human_handoff": True})
        ai2.record_metrics(db_session, execution_id="b", workspace_id=ws.id,
                           metrics={"success": False, "cancelled": True})
        out = ai2.success_metrics(db_session, workspace_id=ws.id)
        assert out["human_handoffs"] == 1
        assert out["cancelled"] == 1

    def test_workspace_isolation(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        ai2.record_metrics(db_session, execution_id="a", workspace_id=ws1.id,
                           metrics={"success": False})
        out = ai2.success_metrics(db_session, workspace_id=ws2.id)
        assert out["total"] == 0

    def test_failure_analysis_by_class(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ai2.record_metrics(db_session, execution_id="a", workspace_id=ws.id,
                           metrics={"success": False}, failure_class="tool")
        ai2.record_metrics(db_session, execution_id="b", workspace_id=ws.id,
                           metrics={"success": False}, failure_class="tool")
        ai2.record_metrics(db_session, execution_id="c", workspace_id=ws.id,
                           metrics={"success": False}, failure_class="timeout")
        out = ai2.failure_analysis(db_session, workspace_id=ws.id)
        assert out["total_failures"] == 3
        assert out["by_class"]["tool"] == 2
        assert out["by_class"]["timeout"] == 1

    def test_plan_quality_clean_plan(self):
        out = ai2.plan_quality({
            "steps": [{"id": "s1"}, {"id": "s2"}],
            "tool_calls": [{"name": "search"}],
            "cost": 0.01, "duration_s": 10, "failed_dependencies": 0,
        })
        assert out["score"] == 1.0
        assert not any(out["signals"].values())

    def test_plan_quality_flags_excess(self):
        out = ai2.plan_quality({
            "steps": [{"id": f"s{i}"} for i in range(15)],
            "tool_calls": [{"name": "t1"}, {"name": "t1"}, {"name": "t1"},
                           {"name": "t1"}],
            "cost": 5.0, "cost_budget": 1.0, "duration_s": 400,
            "latency_budget_s": 300, "failed_dependencies": 2,
        })
        assert out["signals"]["unnecessary_steps"]
        assert out["signals"]["redundant_tools"]
        assert out["signals"]["excessive_cost"]
        assert out["signals"]["excessive_latency"]
        assert out["signals"]["failed_dependencies"]
        assert out["score"] < 0.5

    def test_plan_improvements_generate_candidates(self):
        recs = ai2.plan_improvements({
            "steps": [{"id": f"s{i}"} for i in range(15)],
            "tool_calls": [], "cost": 5.0, "cost_budget": 1.0,
            "duration_s": 400, "latency_budget_s": 300,
        })
        kinds = {r["kind"] for r in recs}
        assert "step_reduction" in kinds
        assert "model_downgrade" in kinds
        assert "parallelize" in kinds

    def test_safety_score_clean(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ai2.record_metrics(db_session, execution_id="a", workspace_id=ws.id,
                           metrics={"success": True})
        out = ai2.safety_score(db_session, workspace_id=ws.id)
        assert out["safety_score"] == 1.0

    def test_safety_score_penalized(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ai2.record_metrics(db_session, execution_id="a", workspace_id=ws.id,
                           metrics={"success": True,
                                    "policy_violations": 1,
                                    "rejected_tools": 1,
                                    "unsafe_attempts": 1})
        out = ai2.safety_score(db_session, workspace_id=ws.id)
        assert out["policy_violations"] == 1
        assert out["safety_score"] < 1.0

    def test_safety_score_approval_rate(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ai2.record_metrics(db_session, execution_id="a", workspace_id=ws.id,
                           metrics={"success": True,
                                    "approvals_required": 2})
        out = ai2.safety_score(db_session, workspace_id=ws.id)
        assert out["approval_rate"] == 1.0


# ===========================================================================
# Workflow intelligence
# ===========================================================================

class TestWorkflowAnalytics:
    def test_record_node_persists(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = wi.record_node(db_session, run_id=1, node_id="n1",
                             workspace_id=ws.id,
                             metrics={"duration_s": 5.0, "status": "ok"})
        assert row.id is not None

    def test_success_analytics_completion(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        wi.record_node(db_session, run_id=1, node_id="n1",
                       workspace_id=ws.id, metrics={"completed": True})
        wi.record_node(db_session, run_id=1, node_id="n2",
                       workspace_id=ws.id, metrics={"completed": True})
        wi.record_node(db_session, run_id=2, node_id="n1",
                       workspace_id=ws.id, metrics={"failed": True})
        out = wi.success_analytics(db_session, workspace_id=ws.id)
        assert out["runs"] == 2
        assert out["completed"] == 1
        assert out["failed"] == 1
        assert out["completion_rate"] == 0.5

    def test_success_analytics_retries_timeouts(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        wi.record_node(db_session, run_id=1, node_id="n1",
                       workspace_id=ws.id, metrics={"retries": 3})
        wi.record_node(db_session, run_id=1, node_id="n2",
                       workspace_id=ws.id, metrics={"timed_out": True})
        wi.record_node(db_session, run_id=1, node_id="n3",
                       workspace_id=ws.id, metrics={"compensated": True})
        out = wi.success_analytics(db_session, workspace_id=ws.id)
        assert out["retried_nodes"] == 1
        assert out["timed_out_nodes"] == 1
        assert out["compensated_nodes"] == 1

    def test_bottlenecks_sorted_by_duration(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        wi.record_node(db_session, run_id=1, node_id="slow",
                       workspace_id=ws.id, metrics={"duration_s": 120.0,
                                                    "status": "ok"})
        wi.record_node(db_session, run_id=1, node_id="fast",
                       workspace_id=ws.id, metrics={"duration_s": 1.0,
                                                    "status": "ok"})
        out = wi.bottlenecks(db_session, workspace_id=ws.id)
        assert out[0]["node_id"] == "slow"
        assert out[0]["avg_duration_s"] == 120.0

    def test_failure_hotspots_high_failure_rate(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for _ in range(4):
            wi.record_node(db_session, run_id=1, node_id="flaky",
                           workspace_id=ws.id,
                           metrics={"duration_s": 1.0, "status": "failed"})
        wi.record_node(db_session, run_id=1, node_id="flaky",
                       workspace_id=ws.id,
                       metrics={"duration_s": 1.0, "status": "ok"})
        hotspots = wi.failure_hotspots(db_session, workspace_id=ws.id)
        assert hotspots[0]["node_id"] == "flaky"
        assert hotspots[0]["failure_rate"] > 0.2

    def test_optimization_recommendations(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        wi.record_node(db_session, run_id=1, node_id="heavy",
                       workspace_id=ws.id, metrics={"duration_s": 90.0,
                                                    "status": "failed"})
        recs = wi.optimization_recommendations(db_session,
                                               workspace_id=ws.id)
        kinds = {r["kind"] for r in recs}
        assert "parallelize" in kinds
        assert "retry_policy" in kinds

    def test_optimization_none_when_healthy(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        wi.record_node(db_session, run_id=1, node_id="n1",
                       workspace_id=ws.id, metrics={"duration_s": 1.0,
                                                    "status": "ok"})
        recs = wi.optimization_recommendations(db_session,
                                               workspace_id=ws.id)
        assert recs[0]["kind"] == "none"

    def test_simulate_serial_and_parallel(self):
        workflow = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "parallelizable": True,
        }
        node_metrics = {"a": {"avg_duration_s": 10, "avg_cost": 0.1},
                        "b": {"avg_duration_s": 20, "avg_cost": 0.2},
                        "c": {"avg_duration_s": 30, "avg_cost": 0.3}}
        out = wi.simulate(workflow, node_metrics)
        assert out["serial_duration_s"] == 60.0
        assert out["parallel_duration_s"] < out["serial_duration_s"]
        assert out["estimated_cost"] == 0.6
        assert len(out["critical_path"]) == 3

    def test_simulate_no_side_effects(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        wi.simulate({"nodes": [{"id": "x"}]},
                    {"x": {"avg_duration_s": 5.0}})
        assert db_session.query(WorkflowIntelligence).filter_by(
            workspace_id=ws.id).count() == 0


# ===========================================================================
# Memory intelligence
# ===========================================================================

class TestMemoryIntelligence:
    def test_record_quality_persists(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = memory(db_session, ws, "remember this")
        row = mi.record_quality(db_session, memory_id=m.id,
                                workspace_id=ws.id,
                                metrics={"used": 3})
        assert db_session.get(MemoryIntelligence, row.id) is not None

    def test_memory_quality_aggregates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, "active memory")
        memory(db_session, ws, "expired memory",
               lifecycle_status="EXPIRED")
        memory(db_session, ws, "no source memory", source=None)
        memory(db_session, ws, "low confidence memory", confidence="LOW")
        out = mi.memory_quality(db_session, workspace_id=ws.id)
        assert out["total"] == 4
        assert out["active"] == 3  # 1 active + no-source + low-confidence
        assert out["expired_or_superseded"] == 1
        assert out["missing_provenance"] == 1
        assert out["low_confidence"] == 1

    def test_decay_report_stale(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, "stale memory", updated_days_ago=200)
        memory(db_session, ws, "fresh memory", updated_days_ago=1)
        report = mi.decay_report(db_session, workspace_id=ws.id)
        assert len(report) == 1
        assert report[0]["content"] == "stale memory"

    def test_decay_report_expiring_soon(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, "expiring", expires_days=3)
        report = mi.decay_report(db_session, workspace_id=ws.id)
        assert any(r["memory_id"] for r in report)

    def test_consolidation_candidates_similar(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, "quarterly revenue grew ten percent",
               memory_type="fact")
        memory(db_session, ws, "quarterly revenue grew ten percent",
               memory_type="fact")
        cands = mi.consolidation_candidates(db_session, workspace_id=ws.id)
        assert len(cands) == 1
        assert cands[0]["requires_review"] is True

    def test_consolidation_skips_low_confidence(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, "identical content here",
               confidence="LOW")
        memory(db_session, ws, "identical content here",
               confidence="LOW")
        cands = mi.consolidation_candidates(db_session, workspace_id=ws.id)
        assert cands == []

    def test_consolidation_skips_different_types(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, "same words same words", memory_type="fact")
        memory(db_session, ws, "same words same words",
               memory_type="preference")
        cands = mi.consolidation_candidates(db_session, workspace_id=ws.id)
        assert cands == []

    def test_conflict_queue_uncertain_memories(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, "uncertain", confidence="LOW")
        out = mi.conflict_queue(db_session, workspace_id=ws.id)
        assert any(m["memory_id"] for m in out["uncertain_memories"])

    def test_user_controls_inspect(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = memory(db_session, ws, "inspect me", source="doc-1")
        out = mi.user_controls(db_session, memory_id=m.id,
                               workspace_id=ws.id, action="inspect")
        assert out["content"] == "inspect me"
        assert out["source"] == "doc-1"

    def test_user_controls_suppress(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = memory(db_session, ws, "suppress me")
        out = mi.user_controls(db_session, memory_id=m.id,
                               workspace_id=ws.id, action="suppress")
        assert out["lifecycle_status"] == "SUPPRESSED"
        assert db_session.get(AIMemory, m.id).lifecycle_status == "SUPPRESSED"

    def test_user_controls_delete_own(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = memory(db_session, ws, "delete me", user_id=user.id)
        out = mi.user_controls(db_session, memory_id=m.id,
                               workspace_id=ws.id, action="delete",
                               user_id=user.id)
        assert out["deleted"] is True
        assert db_session.get(AIMemory, m.id) is None

    def test_user_controls_cannot_delete_others(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = memory(db_session, ws, "not mine", user_id=999)
        with pytest.raises(PermissionError):
            mi.user_controls(db_session, memory_id=m.id,
                             workspace_id=ws.id, action="delete",
                             user_id=user.id)

    def test_user_controls_cross_workspace_blocked(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        m = memory(db_session, ws1, "secret")
        with pytest.raises(KeyError):
            mi.user_controls(db_session, memory_id=m.id,
                             workspace_id=ws2.id, action="inspect")

    def test_user_controls_unknown_action(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = memory(db_session, ws, "x")
        with pytest.raises(ValueError):
            mi.user_controls(db_session, memory_id=m.id,
                             workspace_id=ws.id, action="teleport")


# ===========================================================================
# Graph intelligence
# ===========================================================================

class TestGraphIntelligence:
    def test_graph_health_healthy_workspace(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = entity(db_session, ws, "Alpha")
        b = entity(db_session, ws, "Beta")
        db_session.add(EntityRelationship(
            workspace_id=ws.id, source_id=a.id, target_id=b.id,
            relationship_type="works_with"))
        db_session.commit()
        row = gi.graph_health(db_session, workspace_id=ws.id)
        metrics = __import__("json").loads(row.metrics_json)
        assert metrics["orphan_entities"] == 0
        assert metrics["orphan_relationships"] == 0
        assert row.score == 1.0

    def test_graph_health_orphan_entity(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity(db_session, ws, "Lonely")
        row = gi.graph_health(db_session, workspace_id=ws.id)
        metrics = __import__("json").loads(row.metrics_json)
        assert metrics["orphan_entities"] == 1
        assert row.score < 1.0

    def test_graph_health_stale_relationship(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = entity(db_session, ws, "Alpha")
        b = entity(db_session, ws, "Beta")
        db_session.add(EntityRelationship(
            workspace_id=ws.id, source_id=a.id, target_id=b.id,
            relationship_type="works_with", is_current=True,
            observed_at=datetime.now(timezone.utc) - timedelta(days=400),
            valid_until=datetime.now(timezone.utc) - timedelta(days=10)))
        db_session.commit()
        row = gi.graph_health(db_session, workspace_id=ws.id, stale_days=180)
        metrics = __import__("json").loads(row.metrics_json)
        assert metrics["stale_relationships"] == 1
        assert metrics["expired_relationships"] == 1

    def test_graph_health_low_confidence(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity(db_session, ws, "Shaky", confidence=0.1)
        row = gi.graph_health(db_session, workspace_id=ws.id)
        metrics = __import__("json").loads(row.metrics_json)
        assert metrics["low_confidence_entities"] == 1

    def test_health_summary_latest(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity(db_session, ws, "E1")
        gi.graph_health(db_session, workspace_id=ws.id)
        summary = gi.health_summary(db_session, workspace_id=ws.id)
        assert summary is not None
        assert "score" in summary
        assert "metrics" in summary

    def test_health_summary_empty(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        assert gi.health_summary(db_session, workspace_id=ws.id) is None

    def test_graph_drift_detected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity(db_session, ws, "A")
        entity(db_session, ws, "B")
        gi.graph_health(db_session, workspace_id=ws.id)
        # remove one entity -> orphan appears on the next snapshot
        db_session.query(Entity).filter(Entity.name == "B").delete()
        db_session.commit()
        gi.graph_health(db_session, workspace_id=ws.id)
        out = gi.graph_drift(db_session, workspace_id=ws.id)
        assert out["drift"] is True
        assert "orphan_entities" in out["changed_metrics"]

    def test_graph_drift_no_history(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        out = gi.graph_drift(db_session, workspace_id=ws.id)
        assert out["drift"] is False
        assert out["reason"] == "insufficient history"

    def test_recommend_kind_validation(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            gi.recommend(db_session, workspace_id=ws.id, kind="teleport",
                         candidate="x")

    def test_recommend_creates_pending(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        out = gi.recommend(db_session, workspace_id=ws.id, kind="alias",
                           candidate="Normalize ACME")
        assert out["status"] == "PROPOSED"  # never auto-applied
        row = db_session.get(GraphRecommendation, out["id"])
        assert row.kind == "alias"

    def test_generate_recommendations_repeated_names(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity(db_session, ws, "ACME")
        entity(db_session, ws, "ACME")
        recs = gi.generate_recommendations(db_session, workspace_id=ws.id)
        assert len(recs) == 1
        assert recs[0]["kind"] == "alias"

    def test_quality_evaluation_empty(self):
        out = gi.quality_evaluation([])
        assert out["score"] == 0.0

    def test_quality_evaluation_scoring(self):
        entities = [
            {"name": "A", "confidence": 1.0, "relationships": ["r1"]},
            {"name": "B", "confidence": "MEDIUM", "relationships": []},
            {"confidence": 0.5},  # no name
        ]
        out = gi.quality_evaluation(entities)
        assert out["score"] > 0.5
        assert out["signals"]["name_coverage"] == pytest.approx(2 / 3, abs=1e-3)

    def test_graph_recommendations_require_validation(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gi.recommend(db_session, workspace_id=ws.id, kind="entity",
                     candidate="New Entity")
        row = db_session.query(GraphRecommendation).first()
        assert row.status == "PROPOSED"  # never auto-applied


# ===========================================================================
# Search intelligence
# ===========================================================================

class TestSearchIntelligence:
    def test_quality_score_empty(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        out = si.quality_score(db_session, workspace_id=ws.id)
        assert out["score"] is None
        assert out["total_events"] == 0

    def test_quality_score_zero_results_penalty(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        search_event(db_session, ws, "zero_result", count=4)
        search_event(db_session, ws, "clicked", count=1)
        out = si.quality_score(db_session, workspace_id=ws.id)
        assert out["zero_result_rate"] == 0.8
        assert out["score"] < 1.0

    def test_quality_score_good_behavior(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        search_event(db_session, ws, "clicked", count=3)
        search_event(db_session, ws, "reformulated", count=1)
        out = si.quality_score(db_session, workspace_id=ws.id)
        assert out["click_rate"] == 0.75
        assert out["reformulation_rate"] == 0.25

    def test_quality_score_abandoned(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        search_event(db_session, ws, "abandoned", count=2)
        out = si.quality_score(db_session, workspace_id=ws.id)
        assert out["abandoned_rate"] == 1.0

    def test_ranking_experiment_creates_experiment(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        out = si.ranking_experiment(
            db_session, name="rerank-test",
            config={"rerank": True, "weight": 0.5})
        assert out["experiment_id"] is not None
        assert out["status"] == "DRAFT"

    def test_explain_safe_factors(self):
        out = si.explain({
            "matched_metadata": ["author:smith"],
            "semantic_similarity": 0.87,
            "freshness_days": 3,
            "source_authority": "high",
            "rank": 1,
        })
        factors = {e["factor"] for e in out["explanations"]}
        assert factors == {"matched_metadata", "semantic_similarity",
                           "freshness", "source_authority"}
        assert out["rank"] == 1

    def test_explain_never_exposes_reasoning(self):
        out = si.explain({"semantic_similarity": 0.9})
        assert all("reasoning" not in str(e).lower()
                   for e in out["explanations"])
        assert "chain" not in str(out).lower()

    def test_personalization_safe_matching_scope(self):
        assert si.personalization_safe(
            {"workspace_id": 1, "organization_id": 1}, 1, 1) is True

    def test_personalization_safe_workspace_mismatch(self):
        assert si.personalization_safe(
            {"workspace_id": 2, "organization_id": 1}, 1, 1) is False

    def test_personalization_safe_org_mismatch(self):
        assert si.personalization_safe(
            {"workspace_id": 1, "organization_id": 2}, 1, 1) is False

    def test_personalization_safe_cross_tenant(self):
        assert si.personalization_safe(
            {"workspace_id": 99}, 1, None) is False

    def test_quality_score_workspace_isolation(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        search_event(db_session, ws1, "zero_result", count=5)
        out = si.quality_score(db_session, workspace_id=ws2.id)
        assert out["total_events"] == 0