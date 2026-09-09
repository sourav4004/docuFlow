"""Phase 19 tests — knowledge federation 2.0 + knowledge graph 5.0 +
memory platform 4.0.

Federation: connector contract/validation/checkpoints, idempotent sync,
conflict engine. KG: canonicalization candidates, merge authorization,
relationship confidence/expiry, graph consistency checks. Memory:
candidate provenance gate, validation, suppression, conflict workflows,
expiration policy, explainability, scope-hierarchy retrieval.
"""

import json
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models.organization import Organization  # noqa: E402
from app.models.knowledge_graph import Entity, EntityRelationship  # noqa: E402
from app.models.phase15 import AIMemory  # noqa: E402
from app.models.phase16 import MemoryConflict  # noqa: E402
from app.models.phase17 import (  # noqa: E402
    ConnectorSource, ConnectorSync, ConnectorItem, MemorySupersession,
    EntityCandidate,
)
from app.models.phase18 import EntityMergeRequest  # noqa: E402
from app.models.phase19 import (  # noqa: E402
    ConnectorConflict, MemorySuppression, ConsistencyReport,
)
from app.services import federation_ops2 as fo2  # noqa: E402
from app.services import kg5  # noqa: E402
from app.services import kg_ops  # noqa: E402
from app.services import memory4  # noqa: E402

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
    db_session.query(ConsistencyReport).delete()
    db_session.query(MemorySuppression).delete()
    db_session.query(MemorySupersession).delete()
    db_session.query(MemoryConflict).delete()
    db_session.query(AIMemory).delete()
    db_session.query(EntityMergeRequest).delete()
    db_session.query(EntityCandidate).delete()
    db_session.query(ConnectorConflict).delete()
    db_session.query(ConnectorItem).delete()
    db_session.query(ConnectorSync).delete()
    db_session.query(ConnectorSource).delete()
    db_session.query(EntityRelationship).delete()
    db_session.query(Entity).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p19fg"):
    _counter[0] += 1
    user = User(name=f"P19 FG {_counter[0]}",
                email=f"{tag}{_counter[0]}@p19-fg.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_org(db, user):
    _counter[0] += 1
    org = Organization(name=f"p19 fg org {_counter[0]}",
                       slug=f"p19fg-{_counter[0]}", owner_id=user.id)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


def fresh_workspace(db, user, org=None):
    _counter[0] += 1
    ws = Workspace(name=f"p19 fg ws {_counter[0]}", owner_id=user.id,
                   organization_id=org.id if org else None)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def member(db, ws, user, role="MEMBER"):
    row = WorkspaceMember(workspace_id=ws.id, user_id=user.id, role=role)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def connector(db, ws, org=None):
    _counter[0] += 1
    row = ConnectorSource(workspace_id=ws.id,
                          organization_id=org.id if org else None,
                          name=f"src {_counter[0]}", kind="drive",
                          scopes_json=json.dumps(
                              {"organization": True, "workspace": True,
                               "resource": True}),
                          permissions_json=json.dumps(
                              {"read": True, "write": False}),
                          allowed_domains_json=json.dumps(
                              ["example.com"]),
                          credential_ref=f"secret:src-{_counter[0]}",
                          retention_days=90, enabled=True)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def entity(db, ws, name, entity_type="person", aliases=None):
    _counter[0] += 1
    row = Entity(workspace_id=ws.id, name=name, entity_type=entity_type,
                 aliases=json.dumps(aliases or []),
                 normalized_name=name.lower().strip())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def rel(db, ws, source, target, rtype="manages"):
    _counter[0] += 1
    row = EntityRelationship(workspace_id=ws.id, source_id=source.id,
                             target_id=target.id,
                             relationship_type=rtype)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def memory(db, ws, *, content="remembered fact", scope="WORKSPACE",
           user=None, org=None, source="doc:1", lifecycle="ACTIVE",
           memory_type="fact", created_days_ago=None, expires_at=None):
    _counter[0] += 1
    row = AIMemory(workspace_id=ws.id,
                   organization_id=org.id if org else None,
                   user_id=user.id if user else None,
                   memory_type=memory_type, scope=scope, content=content,
                   source=source, confidence="HIGH",
                   lifecycle_status=lifecycle, expires_at=expires_at)
    if created_days_ago:
        row.created_at = datetime.now(timezone.utc) - timedelta(
            days=created_days_ago)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ===========================================================================
# Federation 2.0
# ===========================================================================

class TestConnectorContract:
    def test_contract_defines_required_methods(self):
        contract = fo2.connector_contract()
        assert "incremental_sync" in contract["methods"]
        assert "delete_or_tombstone" in contract["methods"]

    def test_validate_connector_valid(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        src = connector(db_session, ws, org)
        result = fo2.validate_connector(db_session, src.id)
        assert result["valid"] is True
        assert result["credential_ref"] is True

    def test_validate_connector_missing_scopes(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        src.scopes_json = json.dumps({"workspace": True})
        db_session.commit()
        result = fo2.validate_connector(db_session, src.id)
        assert result["valid"] is False

    def test_validate_connector_missing_credential_ref(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        src.credential_ref = None
        db_session.commit()
        result = fo2.validate_connector(db_session, src.id)
        assert result["valid"] is False

    def test_validate_connector_missing_retention(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        src.retention_days = None
        db_session.commit()
        result = fo2.validate_connector(db_session, src.id)
        assert result["valid"] is False

    def test_validate_unknown_connector(self, db_session):
        result = fo2.validate_connector(db_session, 99999)
        assert result["valid"] is False

    def test_connector_security_scope_never_returns_secrets(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        scope = fo2.connector_security_scope(db_session, src.id)
        # Only a boolean boundary is exposed — never the reference value and
        # never any secret-shaped field.
        assert scope["credential_ref_only"] is True
        assert "api_key" not in scope and "token" not in scope
        assert "password" not in scope and "secret_value" not in scope
        assert src.credential_ref not in json.dumps(scope)


class TestConnectorSync:
    def test_classify_content_conflict(self):
        result = fo2.classify_item(external_id="e1",
                                   external_content_hash="new",
                                   local_content_hash="old")
        assert result["decision"] == "SKIP_CONFLICT"
        assert result["conflict"] == "CONTENT"

    def test_classify_external_deletion_tombstone(self):
        # External deletion applies a reversible tombstone; local payload is
        # preserved and never destroyed.
        result = fo2.classify_item(external_id="e1", external_deleted=True,
                                   local_deleted=False)
        assert result["decision"] == "APPLY"
        assert result["conflict"] is None

    def test_classify_recreated_after_tombstone(self):
        result = fo2.classify_item(external_id="e1", external_deleted=False,
                                   local_deleted=True)
        assert result["decision"] == "APPLY"

    def test_classify_metadata_conflict(self):
        result = fo2.classify_item(external_id="e1", external_title="B",
                                   local_title="A")
        assert result["conflict"] == "METADATA"

    def test_sync_adds_and_checkpoints_cursor(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        result = fo2.sync_connector_batch(
            db_session, source_id=src.id, workspace_id=ws.id,
            items=[{"external_id": "x1", "title": "One",
                    "content_hash": "h1"}],
            cursor={"next": "page2"})
        assert result["added"] == 1
        assert result["cursor"] == {"next": "page2"}
        syncs = db_session.query(ConnectorSync).all()
        assert len(syncs) == 1
        assert "page2" in syncs[0].cursor_json

    def test_sync_idempotent_no_duplicates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        batch = [{"external_id": "x1", "title": "One",
                  "content_hash": "h1"}]
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id, items=batch)
        second = fo2.sync_connector_batch(db_session, source_id=src.id,
                                          workspace_id=ws.id, items=batch)
        assert second["added"] == 0
        assert second["unchanged"] == 1
        assert db_session.query(ConnectorItem).count() == 1

    def test_sync_content_change_applied_on_recreation(self, db_session):
        # A tombstoned item recreated externally is applied as a change
        # (content drift on a live item is a conflict instead).
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h1",
                                         "title": "T"}])
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h1",
                                         "deleted": True}])
        result = fo2.sync_connector_batch(
            db_session, source_id=src.id, workspace_id=ws.id,
            items=[{"external_id": "x1", "content_hash": "h2",
                    "title": "T2"}])
        assert result["changed"] == 1
        row = db_session.query(ConnectorItem).first()
        assert row.deleted is False
        assert row.content_hash == "h2"

    def test_sync_tombstones(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h1"}])
        result = fo2.sync_connector_batch(
            db_session, source_id=src.id, workspace_id=ws.id,
            items=[{"external_id": "x1", "content_hash": "h1",
                    "deleted": True}])
        assert result["deleted"] == 1
        row = db_session.query(ConnectorItem).first()
        assert row.deleted is True

    def test_sync_opens_conflict_on_content_mismatch(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h1"}])
        result = fo2.sync_connector_batch(
            db_session, source_id=src.id, workspace_id=ws.id,
            items=[{"external_id": "x1", "content_hash": "h2"}])
        assert result["conflicts"] == 1
        conflicts = fo2.list_conflicts(db_session, source_id=src.id)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "CONTENT"

    def test_conflict_resolution(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h1"}])
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h2"}])
        conflict = fo2.list_conflicts(db_session, source_id=src.id)[0]
        resolved = fo2.resolve_conflict(db_session, conflict.id,
                                        decision="RESOLVED",
                                        resolver_user_id=user.id)
        assert resolved["status"] == "RESOLVED"
        assert fo2.list_conflicts(db_session, source_id=src.id) == []

    def test_sync_never_overwrites_conflicted_local(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = connector(db_session, ws)
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h1"}])
        fo2.sync_connector_batch(db_session, source_id=src.id,
                                 workspace_id=ws.id,
                                 items=[{"external_id": "x1",
                                         "content_hash": "h2"}])
        row = db_session.query(ConnectorItem).first()
        assert row.content_hash == "h1"


# ===========================================================================
# Knowledge graph 5.0
# ===========================================================================

class TestCanonicalization:
    def test_candidates_for_near_names(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "Acme Corp")
        e2 = entity(db_session, ws, "ACME Corporation")
        result = kg5.canonical_candidates(db_session, workspace_id=ws.id,
                                          entity_ids=[e1.id, e2.id])
        assert result["candidates_created"] >= 1
        # never merged automatically
        assert db_session.query(Entity).count() == 2

    def test_no_candidates_distinct_names(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "Alpha")
        e2 = entity(db_session, ws, "Beta")
        result = kg5.canonical_candidates(db_session, workspace_id=ws.id,
                                          entity_ids=[e1.id, e2.id])
        assert result["candidates_created"] == 0

    def test_aliases_produce_candidates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "X Corp")
        e2 = entity(db_session, ws, "Unknown Ltd", aliases=["X Corp"])
        result = kg5.canonical_candidates(db_session, workspace_id=ws.id,
                                          entity_ids=[e1.id, e2.id])
        assert result["candidates_created"] >= 1


class TestMergeSafety:
    def test_member_cannot_authorize_merge(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        other = fresh_user(db_session)
        member(db_session, ws, other, role="MEMBER")
        assert kg5.can_authorize_merge(db_session, workspace_id=ws.id,
                                       user_id=other.id) is False

    def test_admin_can_authorize_merge(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        other = fresh_user(db_session)
        member(db_session, ws, other, role="ADMIN")
        assert kg5.can_authorize_merge(db_session, workspace_id=ws.id,
                                       user_id=other.id) is True

    def test_merge_safety_cross_workspace_blocked(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws1, "One")
        e2 = entity(db_session, ws2, "Two")
        check = kg5.merge_safety_check(db_session, workspace_id=ws1.id,
                                       source_entity_id=e1.id,
                                       target_entity_id=e2.id)
        assert check["safe"] is False

    def test_merge_safety_same_entity_blocked(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "One")
        check = kg5.merge_safety_check(db_session, workspace_id=ws.id,
                                       source_entity_id=e1.id,
                                       target_entity_id=e1.id)
        assert check["safe"] is False

    def test_decide_merge_requires_admin(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "One")
        e2 = entity(db_session, ws, "Two")
        req = kg_ops.request_merge(db_session, workspace_id=ws.id,
                                   source_entity_id=e1.id,
                                   target_entity_id=e2.id)
        with pytest.raises(PermissionError):
            kg5.decide_merge_safe(db_session, workspace_id=ws.id,
                                  merge_id=req.id, decision="APPROVE",
                                  reviewer_id=user.id)

    def test_decide_merge_admin_approves(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        admin = fresh_user(db_session)
        member(db_session, ws, admin, role="ADMIN")
        e1 = entity(db_session, ws, "One")
        e2 = entity(db_session, ws, "Two")
        req = kg_ops.request_merge(db_session, workspace_id=ws.id,
                                   source_entity_id=e1.id,
                                   target_entity_id=e2.id)
        result = kg5.decide_merge_safe(db_session, workspace_id=ws.id,
                                       merge_id=req.id, decision="APPROVE",
                                       reviewer_id=admin.id)
        assert result["merged"] is True
        assert result["target_entity_id"] == e2.id

    def test_expire_pending_merges(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "One")
        e2 = entity(db_session, ws, "Two")
        req = kg_ops.request_merge(db_session, workspace_id=ws.id,
                                   source_entity_id=e1.id,
                                   target_entity_id=e2.id,
                                   expires_in_days=1)
        req.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        db_session.commit()
        result = kg5.expire_pending_merges(db_session, workspace_id=ws.id)
        assert result["expired"] == 1
        db_session.refresh(req)
        assert req.status == "EXPIRED"


class TestRelationships:
    def test_set_confidence(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        e2 = entity(db_session, ws, "B")
        row = rel(db_session, ws, e1, e2)
        result = kg5.set_relationship_confidence(
            db_session, workspace_id=ws.id, relationship_id=row.id,
            confidence=0.8, evidence="from doc 5")
        assert result["confidence"] == 0.8

    def test_confidence_out_of_range(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        e2 = entity(db_session, ws, "B")
        row = rel(db_session, ws, e1, e2)
        with pytest.raises(ValueError):
            kg5.set_relationship_confidence(db_session,
                                            workspace_id=ws.id,
                                            relationship_id=row.id,
                                            confidence=1.5)

    def test_expire_relationship(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        e2 = entity(db_session, ws, "B")
        row = rel(db_session, ws, e1, e2)
        kg5.expire_relationship(db_session, workspace_id=ws.id,
                                relationship_id=row.id)
        active = kg5.active_relationships(db_session, workspace_id=ws.id)
        assert active == []
        db_session.refresh(row)
        assert row.is_current is False

    def test_active_relationships_filter_entity(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        e2 = entity(db_session, ws, "B")
        e3 = entity(db_session, ws, "C")
        rel(db_session, ws, e1, e2)
        rel(db_session, ws, e3, e2)
        active = kg5.active_relationships(db_session, workspace_id=ws.id,
                                          entity_id=e1.id)
        assert len(active) == 1


class TestGraphConsistency:
    def test_clean_graph(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        e2 = entity(db_session, ws, "B")
        rel(db_session, ws, e1, e2)
        report = kg5.graph_consistency(db_session, workspace_id=ws.id)
        assert report["clean"] is True

    def test_orphan_relationship_detected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        row = EntityRelationship(workspace_id=ws.id, source_id=e1.id,
                                 target_id=424242,
                                 relationship_type="x")
        db_session.add(row)
        db_session.commit()
        report = kg5.graph_consistency(db_session, workspace_id=ws.id)
        kinds = {issue["kind"] for issue in report["issues"]}
        assert "orphan_relationship" in kinds

    def test_cross_tenant_edge_detected(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws1, "A")
        e2 = entity(db_session, ws2, "B")
        db_session.add(EntityRelationship(workspace_id=ws1.id,
                                          source_id=e1.id,
                                          target_id=e2.id,
                                          relationship_type="x"))
        db_session.commit()
        report = kg5.graph_consistency(db_session, workspace_id=ws1.id)
        kinds = {issue["kind"] for issue in report["issues"]}
        assert "cross_tenant_edge" in kinds

    def test_impossible_temporal_detected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        e2 = entity(db_session, ws, "B")
        db_session.add(EntityRelationship(workspace_id=ws.id,
                                          source_id=e1.id,
                                          target_id=e2.id,
                                          relationship_type="x",
                                          valid_from=datetime.now(timezone.utc)
                                          + timedelta(days=5),
                                          valid_until=datetime.now(timezone.utc)))
        db_session.commit()
        report = kg5.graph_consistency(db_session, workspace_id=ws.id)
        kinds = {issue["kind"] for issue in report["issues"]}
        assert "impossible_temporal" in kinds

    def test_contradictory_relationships_detected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = entity(db_session, ws, "A")
        e2 = entity(db_session, ws, "B")
        rel(db_session, ws, e1, e2, rtype="manages")
        rel(db_session, ws, e1, e2, rtype="reports_to")
        report = kg5.graph_consistency(db_session, workspace_id=ws.id)
        kinds = {issue["kind"] for issue in report["issues"]}
        assert "contradictory_relationships" in kinds

    def test_consistency_report_persisted(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        kg5.graph_consistency(db_session, workspace_id=ws.id)
        assert db_session.query(ConsistencyReport).count() == 1


# ===========================================================================
# Memory 4.0
# ===========================================================================

class TestMemoryCandidate:
    def test_candidate_requires_evidence(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = memory4.create_candidate(db_session, workspace_id=ws.id,
                                          content="no evidence fact")
        assert result["created"] is False

    def test_candidate_created_with_evidence(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = memory4.create_candidate(db_session, workspace_id=ws.id,
                                          content="fact",
                                          evidence_ref="doc:9",
                                          ttl_days=30)
        assert result["created"] is True
        assert result["lifecycle_status"] == "CANDIDATE"
        assert result["expires_at"] is not None

    def test_invalid_confidence_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            memory4.create_candidate(db_session, workspace_id=ws.id,
                                     content="x", evidence_ref="doc:1",
                                     confidence="DEFINITELY")

    def test_validate_candidate_activates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = memory4.create_candidate(db_session, workspace_id=ws.id,
                                          content="fact",
                                          evidence_ref="doc:1")
        validated = memory4.validate_candidate(
            db_session, workspace_id=ws.id,
            memory_id=result["memory_id"])
        assert validated["validated"] is True
        assert validated["state"] == "ACTIVE"


class TestSuppression:
    def test_suppress_and_is_suppressed(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = memory(db_session, ws)
        memory4.suppress_memory(db_session, memory_id=row.id,
                                scope_type="USER",
                                suppressed_by_user_id=user.id)
        assert memory4.is_suppressed(db_session, memory_id=row.id,
                                     user_id=user.id) is True
        assert memory4.is_suppressed(db_session, memory_id=row.id,
                                     user_id=999) is False

    def test_suppress_duplicate_is_noop(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = memory(db_session, ws)
        memory4.suppress_memory(db_session, memory_id=row.id,
                                scope_type="WORKSPACE",
                                workspace_id=ws.id)
        second = memory4.suppress_memory(db_session, memory_id=row.id,
                                         scope_type="WORKSPACE",
                                         workspace_id=ws.id)
        assert second["status"] == "ALREADY_SUPPRESSED"

    def test_remove_suppression(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = memory(db_session, ws)
        memory4.suppress_memory(db_session, memory_id=row.id,
                                scope_type="USER",
                                suppressed_by_user_id=user.id)
        memory4.remove_suppression(db_session, memory_id=row.id,
                                   scope_type="USER")
        assert memory4.is_suppressed(db_session, memory_id=row.id,
                                     user_id=user.id) is False

    def test_org_scope_suppression(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        row = memory(db_session, ws)
        memory4.suppress_memory(db_session, memory_id=row.id,
                                scope_type="ORGANIZATION",
                                organization_id=org.id)
        assert memory4.is_suppressed(db_session, memory_id=row.id,
                                     organization_id=org.id) is True


class TestConflictWorkflow:
    def test_resolve_keep_a(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = memory(db_session, ws, content="A")
        b = memory(db_session, ws, content="B")
        conflict = MemoryConflict(workspace_id=ws.id, memory_a_id=a.id,
                                  memory_b_id=b.id, status="OPEN")
        db_session.add(conflict)
        db_session.commit()
        result = memory4.resolve_conflict_workflow(
            db_session, workspace_id=ws.id, conflict_id=conflict.id,
            decision="KEEP_A", reviewer_user_id=user.id)
        assert result["status"] == "RESOLVED"
        assert result["winner_memory_id"] == a.id

    def test_resolve_supersede_b(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = memory(db_session, ws, content="A")
        b = memory(db_session, ws, content="B")
        conflict = MemoryConflict(workspace_id=ws.id, memory_a_id=a.id,
                                  memory_b_id=b.id, status="OPEN")
        db_session.add(conflict)
        db_session.commit()
        result = memory4.resolve_conflict_workflow(
            db_session, workspace_id=ws.id, conflict_id=conflict.id,
            decision="SUPERSEDE_A", reviewer_user_id=user.id)
        # A loses to B
        assert result["winner_memory_id"] == b.id
        db_session.refresh(a)
        assert a.lifecycle_status == "SUPERSEDED"
        assert a.supersedes_id == b.id
        assert db_session.query(MemorySupersession).count() == 1

    def test_conflict_already_resolved(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = memory(db_session, ws)
        b = memory(db_session, ws)
        conflict = MemoryConflict(workspace_id=ws.id, memory_a_id=a.id,
                                  memory_b_id=b.id, status="RESOLVED")
        db_session.add(conflict)
        db_session.commit()
        result = memory4.resolve_conflict_workflow(
            db_session, workspace_id=ws.id, conflict_id=conflict.id,
            decision="KEEP_A", reviewer_user_id=user.id)
        assert result["status"] == "ALREADY_RESOLVED"

    def test_conflict_queue_lists_open(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = memory(db_session, ws)
        b = memory(db_session, ws)
        conflict = MemoryConflict(workspace_id=ws.id, memory_a_id=a.id,
                                  memory_b_id=b.id, status="OPEN")
        db_session.add(conflict)
        db_session.commit()
        from app.services.memory_ops import memory_conflicts
        queue = memory_conflicts(db_session, ws.id)
        assert queue["total"] == 1


class TestExpirationExplain:
    def test_policy_expiration_applies(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        memory(db_session, ws, memory_type="fact", created_days_ago=400)
        result = memory4.apply_policy_expiration(db_session,
                                                 workspace_id=ws.id)
        assert result["policy_expired"] >= 1

    def test_expiration_policy_bounded(self):
        policy = memory4.expiration_policy()
        assert policy["fact"] >= 1
        assert policy["event"] >= 1

    def test_explain_memory_fields(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = memory(db_session, ws, source="doc:3")
        explanation = memory4.explain_memory(db_session,
                                             memory_id=row.id,
                                             workspace_id=ws.id)
        assert explanation["source"] == "doc:3"
        assert explanation["confidence"] == "HIGH"
        assert "memory_id" in explanation
        assert "chain-of-thought" not in explanation["why_used"]

    def test_retrieval_scope_hierarchy(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        user_memory = memory(db_session, ws, content="user fact",
                             scope="USER", user=user)
        ws_memory = memory(db_session, ws, content="ws fact",
                           scope="WORKSPACE")
        org_memory = memory(db_session, ws, content="org fact",
                            scope="ORGANIZATION", org=org)
        result = memory4.retrieve_memories(db_session, workspace_id=ws.id,
                                           user_id=user.id,
                                           organization_id=org.id)
        ids = [r["memory_id"] for r in result["results"]]
        assert ids[0] == user_memory.id
        assert ws_memory.id in ids and org_memory.id in ids

    def test_suppressed_memory_not_retrieved(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = memory(db_session, ws, scope="WORKSPACE")
        memory4.suppress_memory(db_session, memory_id=row.id,
                                scope_type="WORKSPACE",
                                workspace_id=ws.id)
        result = memory4.retrieve_memories(db_session, workspace_id=ws.id)
        assert all(r["memory_id"] != row.id for r in result["results"])

    def test_expired_memory_not_retrieved(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = memory(db_session, ws, scope="WORKSPACE",
                     expires_at=datetime.now(timezone.utc)
                     - timedelta(days=1),
                     lifecycle="EXPIRED")
        result = memory4.retrieve_memories(db_session, workspace_id=ws.id)
        assert all(r["memory_id"] != row.id for r in result["results"])
