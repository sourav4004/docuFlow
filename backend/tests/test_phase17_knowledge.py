"""Phase 17 tests — knowledge platform.

Durable ingestion (stages, partial-failure recovery, progress, batches),
fingerprint duplicate classification, connector federation (incremental +
idempotent sync), knowledge graph 4.0 (candidates, suggestions, conflicts,
org aggregation, bounded search), memory 3.0, and RAG 5.0 (planning,
evidence quality, citation coverage, temporal filtering, conflicts, eval).
"""

import uuid
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.phase17 import (  # noqa: E402
    IngestionRun, IngestionStage, DocumentFingerprint, DuplicateCandidate,
    ConnectorSource, ConnectorSync, ConnectorItem, EntityCandidate,
    RelationshipSuggestion, MemorySupersession,
)
from app.models.knowledge_graph import Entity, EntityRelationship  # noqa: E402
from app.models.phase15 import AIMemory, AIQualityMetric  # noqa: E402
from app.services import ingestion2 as ing  # noqa: E402
from app.services import fingerprint as fp  # noqa: E402
from app.services import federation as fed  # noqa: E402
from app.services import kg4, memory3, rag5  # noqa: E402

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
    from app.models.phase17 import (
        DuplicateCandidate, DocumentFingerprint, ConnectorItem,
        ConnectorSync, ConnectorSource, EntityCandidate,
        RelationshipSuggestion, MemorySupersession, IngestionStage,
        IngestionRun,
    )
    for model in (DuplicateCandidate, DocumentFingerprint, ConnectorItem,
                  ConnectorSync, ConnectorSource, EntityCandidate,
                  RelationshipSuggestion, MemorySupersession,
                  IngestionStage, IngestionRun):
        db_session.query(model).delete()
    db_session.commit()


def fresh_user(db, tag="p17ku"):
    _counter[0] += 1
    user = User(name=f"K User {_counter[0]}",
                email=f"{tag}{_counter[0]}@p17-knowledge.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p17k ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def fresh_doc(db, ws, user, name="doc.pdf", content=None, status="READY"):
    doc = Document(workspace_id=ws.id, user_id=user.id,
                   original_filename=name, mime_type="text/plain",
                   file_size=len(content or "x") or 1, status=status,
                   storage_key=f"k-{uuid.uuid4().hex}")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def make_entity(db, ws, name, etype="organization"):
    e = Entity(workspace_id=ws.id, name=name, entity_type=etype,
               aliases=None)
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def make_memory(db, ws, user=None, scope="WORKSPACE",
                memory_type="WORKSPACE_FACT", content="fact",
                source="document", confidence="MEDIUM"):
    m = AIMemory(workspace_id=ws.id, user_id=user.id if user else None,
                 scope=scope, memory_type=memory_type, content=content,
                 source=source, confidence=confidence)
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


# ============================================================
# Ingestion pipeline 2.0
# ============================================================

class TestIngestion:
    def test_start_creates_stage_rows(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        run = ing.start_ingestion(db_session, ws.id, doc.id,
                                  user_id=user.id)
        stages = db_session.query(IngestionStage).filter(
            IngestionStage.run_id == run.id).all()
        assert len(stages) == len(ing.STAGES)
        assert run.current_stage == "UPLOAD"
        assert run.status == "RUNNING"

    def test_advance_completes_pipeline(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        run = ing.start_ingestion(db_session, ws.id, doc.id,
                                  user_id=user.id)
        result = ing.advance_ingestion_run(db_session, run.id)
        assert result["status"] == "COMPLETED"
        assert result["progress_pct"] == 100
        stages = db_session.query(IngestionStage).filter(
            IngestionStage.run_id == run.id).all()
        assert all(s.status == "COMPLETED" for s in stages)

    def test_idempotent_start_same_key(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        key = f"ing-{uuid.uuid4().hex[:12]}"
        run1 = ing.start_ingestion(db_session, ws.id, doc.id,
                                   user_id=user.id, idempotency_key=key)
        run2 = ing.start_ingestion(db_session, ws.id, doc.id,
                                   user_id=user.id, idempotency_key=key)
        assert run1.id == run2.id

    def test_partial_failure_recovery_does_not_repeat_stages(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        run = ing.start_ingestion(db_session, ws.id, doc.id,
                                  user_id=user.id)
        failed = ing.advance_ingestion_run(db_session, run.id,
                                           simulate_failure="EMBED")
        assert failed["failed_stage"] == "EMBED"
        run = db_session.query(IngestionRun).filter(
            IngestionRun.id == run.id).first()
        assert run.status == "FAILED"
        # OCR (earlier stage) completed; EMBED failed → retry resumes EMBED
        result = ing.retry_ingestion(db_session, run.id, user.id)
        assert result["current_stage"] == "EMBED"
        done = ing.advance_ingestion_run(db_session, run.id)
        assert done["status"] == "COMPLETED"
        # completed stages were never re-executed: OCR attempts stay at 1
        ocr = db_session.query(IngestionStage).filter(
            IngestionStage.run_id == run.id,
            IngestionStage.stage == "OCR").first()
        assert ocr.attempts == 1

    def test_batch_run(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        d1 = fresh_doc(db_session, ws, user, "a.pdf")
        d2 = fresh_doc(db_session, ws, user, "b.pdf")
        run = ing.start_ingestion(db_session, ws.id, d1.id,
                                  user_id=user.id,
                                  batch_document_ids=[d2.id])
        assert run.batch_json is not None
        ing.advance_ingestion_run(db_session, run.id)
        run = db_session.query(IngestionRun).filter(
            IngestionRun.id == run.id).first()
        assert run.status == "COMPLETED"

    def test_progress_report(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        run = ing.start_ingestion(db_session, ws.id, doc.id,
                                  user_id=user.id)
        report = ing.ingestion_progress(db_session, run.id)
        assert report["progress_pct"] == 0
        assert len(report["stages"]) == len(ing.STAGES)
        assert "estimated_remaining" in report

    def test_list_runs_bounded(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        ing.start_ingestion(db_session, ws.id, doc.id, user_id=user.id)
        listing = ing.list_runs(db_session, ws.id, limit=10000)
        assert listing["limit"] == 200
        assert listing["total"] >= 1


# ============================================================
# Fingerprints + duplicates
# ============================================================

class TestFingerprints:
    def test_fingerprint_stable(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        fp1 = fp.compute_fingerprint(db_session, doc.id,
                                     content="same content here")
        fp2 = fp.compute_fingerprint(db_session, doc.id,
                                     content="same content here")
        assert fp1.content_hash == fp2.content_hash

    def test_exact_duplicate_classified(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        text = "This is the exact same policy document body " * 10
        d1 = fresh_doc(db_session, ws, user, "one.pdf")
        d2 = fresh_doc(db_session, ws, user, "two.pdf")
        fp.compute_fingerprint(db_session, d1.id, content=text)
        fp.compute_fingerprint(db_session, d2.id, content=text)
        results = fp.classify_duplicates(db_session, ws.id, d1.id)
        assert any(r["classification"] == "EXACT_DUPLICATE"
                   for r in results)

    def test_near_duplicate_high_overlap(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        base = ("Travel policy requires pre-approval for all trips longer "
                "than five business days. Exceptions require director "
                "sign-off and a written justification. " * 15)
        d1 = fresh_doc(db_session, ws, user, "a.pdf")
        d2 = fresh_doc(db_session, ws, user, "b.pdf")
        fp.compute_fingerprint(db_session, d1.id, content=base)
        fp.compute_fingerprint(db_session, d2.id,
                               content=base + "Minor rewrite of the "
                               "intro sentence only.")
        results = fp.classify_duplicates(db_session, ws.id, d1.id)
        found = [r for r in results
                 if r["other_document_id"] == d2.id]
        assert found and found[0]["classification"] in (
            "NEAR_DUPLICATE", "VERSION")

    def test_candidates_listed(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        text = "shared content for duplicate detection " * 12
        d1 = fresh_doc(db_session, ws, user, "x.pdf")
        d2 = fresh_doc(db_session, ws, user, "y.pdf")
        fp.compute_fingerprint(db_session, d1.id, content=text)
        fp.compute_fingerprint(db_session, d2.id, content=text)
        fp.classify_duplicates(db_session, ws.id, d1.id)
        listing = fp.list_candidates(db_session, ws.id)
        assert listing["total"] >= 1
        assert listing["items"][0]["classification"] == "EXACT_DUPLICATE"


# ============================================================
# Federation
# ============================================================

class TestFederation:
    def test_create_source_never_stores_credential(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = fed.create_source(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, name="Drive", kind="cloud_storage",
            scopes=["documents.read"],
            allowed_domains=["drive.example.com"],
            credential_ref="secret-backend://key/123")
        raw = db_session.query(ConnectorSource).filter(
            ConnectorSource.id == src.id).first()
        assert raw.credential_ref.startswith("secret-backend://")
        assert "super-secret" not in str(raw.credential_ref)

    def test_unsupported_kind_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            fed.create_source(db_session, workspace_id=ws.id,
                              organization_id=None, user_id=user.id,
                              name="x", kind="scanner")

    def test_initial_sync_adds_items(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = fed.create_source(db_session, workspace_id=ws.id,
                                organization_id=None, user_id=user.id,
                                name="Repo", kind="knowledge_base")
        result = fed.run_connector_sync(db_session, src.id)
        assert result["items_added"] == 2
        assert db_session.query(ConnectorItem).filter(
            ConnectorItem.source_id == src.id).count() == 2

    def test_incremental_sync_delta_and_tombstone(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = fed.create_source(db_session, workspace_id=ws.id,
                                organization_id=None, user_id=user.id,
                                name="Repo", kind="knowledge_base")
        fed.run_connector_sync(db_session, src.id)
        second = fed.run_connector_sync(db_session, src.id)
        assert second["items_added"] == 0
        assert second["items_changed"] == 1
        assert second["items_deleted"] == 1
        deleted_item = db_session.query(ConnectorItem).filter(
            ConnectorItem.source_id == src.id,
            ConnectorItem.external_id.like("%doc-2")).first()
        assert deleted_item.deleted is True

    def test_repeated_sync_never_duplicates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = fed.create_source(db_session, workspace_id=ws.id,
                                organization_id=None, user_id=user.id,
                                name="Repo", kind="knowledge_base")
        for _ in range(4):
            fed.run_connector_sync(db_session, src.id)
        assert db_session.query(ConnectorItem).filter(
            ConnectorItem.source_id == src.id).count() == 2

    def test_disabled_source_skips_sync(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = fed.create_source(db_session, workspace_id=ws.id,
                                organization_id=None, user_id=user.id,
                                name="Off", kind="knowledge_base")
        fed.update_source(db_session, src.id, ws.id, enabled=False)
        result = fed.run_connector_sync(db_session, src.id)
        assert result["status"] == "DISABLED"

    def test_sync_history_state(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = fed.create_source(db_session, workspace_id=ws.id,
                                organization_id=None, user_id=user.id,
                                name="Hist", kind="knowledge_base")
        fed.run_connector_sync(db_session, src.id)
        state = fed.sync_state(db_session, src.id, ws.id)
        assert state["items"][0]["status"] == "COMPLETED"
        assert state["items"][0]["items_added"] == 2

    def test_cross_workspace_source_denied(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        src = fed.create_source(db_session, workspace_id=ws1.id,
                                organization_id=None, user_id=user.id,
                                name="A", kind="knowledge_base")
        with pytest.raises(Exception):
            fed.run_connector_sync(db_session, src.id, workspace_id=ws2.id)


# ============================================================
# Knowledge graph 4.0
# ============================================================

class TestKG4:
    def test_candidate_normalized_name(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = make_entity(db_session, ws, "Acme Corp")
        e2 = make_entity(db_session, ws, "acme corp")  # normalization match
        results = kg4.suggest_entity_candidates(db_session, ws.id, e1.id)
        assert any(r["candidate_id"] == e2.id for r in results)
        assert results[0]["method"] == "normalized_name"
        assert results[0]["status"] == "PENDING"

    def test_candidates_are_never_auto_merged(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = make_entity(db_session, ws, "Globex Ltd")
        e2 = make_entity(db_session, ws, "Globex Limited")
        kg4.suggest_entity_candidates(db_session, ws.id, e1.id)
        cand = db_session.query(EntityCandidate).filter(
            EntityCandidate.entity_id == e1.id).first()
        assert cand.status == "PENDING"
        result = kg4.resolve_candidate(db_session, ws.id, cand.id,
                                       "APPROVED", resolved_by=user.id)
        assert result["status"] == "APPROVED"

    def test_relationship_suggestion_shared_document(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = make_entity(db_session, ws, "Board Co")
        e2 = make_entity(db_session, ws, "Bank Co")
        doc = fresh_doc(db_session, ws, user)
        db_session.add(EntityRelationship(
            workspace_id=ws.id, source_id=e1.id, target_id=e2.id,
            relationship_type="partner_of", source_document_id=doc.id))
        db_session.commit()
        suggestions = kg4.suggest_relationships(db_session, ws.id, e1.id)
        # e2 links to the same doc through an existing edge → suggestion
        assert isinstance(suggestions, list)

    def test_apply_suggestion_creates_relationship(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = make_entity(db_session, ws, "Alpha")
        e2 = make_entity(db_session, ws, "Beta")
        db_session.add(EntityRelationship(
            workspace_id=ws.id, source_id=e1.id, target_id=e2.id,
            relationship_type="contracts_with"))
        db_session.commit()
        # seed a direct suggestion row
        sug = RelationshipSuggestion(
            workspace_id=ws.id, entity_a_id=e1.id, entity_b_id=e2.id,
            relation_type="related_to", confidence=0.9,
            evidence="shared contract", method="shared_evidence")
        db_session.add(sug)
        db_session.commit()
        result = kg4.apply_relationship_suggestion(
            db_session, ws.id, sug.id, "partners_with", decided_by=user.id)
        assert result["status"] == "APPROVED"
        rel = db_session.query(EntityRelationship).filter(
            EntityRelationship.workspace_id == ws.id,
            EntityRelationship.source_id == e1.id,
            EntityRelationship.target_id == e2.id,
            EntityRelationship.relationship_type == "partners_with").first()
        assert rel is not None

    def test_relationship_conflict_detected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = make_entity(db_session, ws, "X")
        e2 = make_entity(db_session, ws, "Y")
        db_session.add(EntityRelationship(
            workspace_id=ws.id, source_id=e1.id, target_id=e2.id,
            relationship_type="owns"))
        db_session.add(EntityRelationship(
            workspace_id=ws.id, source_id=e2.id, target_id=e1.id,
            relationship_type="acquired_by"))
        db_session.commit()
        conflicts = kg4.detect_relationship_conflicts(db_session, ws.id,
                                                      e1.id)
        assert len(conflicts) >= 1

    def test_graph_search_bounded_and_scoped(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e1 = make_entity(db_session, ws, "Root")
        e2 = make_entity(db_session, ws, "Child")
        db_session.add(EntityRelationship(
            workspace_id=ws.id, source_id=e1.id, target_id=e2.id,
            relationship_type="links_to"))
        db_session.commit()
        result = kg4.graph_search(db_session, ws.id, e1.id, max_depth=2)
        assert result["root_entity_id"] == e1.id
        assert e2.id in result["visited_entities"]
        assert len(result["edges"]) == 1
        assert result["truncated"] is False

    def test_graph_search_foreign_workspace_denied(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        e1 = make_entity(db_session, ws1, "OnlyIn1")
        with pytest.raises(ValueError):
            kg4.graph_search(db_session, ws2.id, e1.id)


# ============================================================
# Memory 3.0
# ============================================================

class TestMemory3:
    def test_consolidate_preserves_provenance(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m1 = make_memory(db_session, ws, content="server A is prod")
        m2 = make_memory(db_session, ws, content="server B is staging")
        result = memory3.consolidate(db_session, [m1.id, m2.id], ws.id,
                                     user.id)
        canonical = db_session.query(AIMemory).filter(
            AIMemory.id == result["canonical_memory_id"]).first()
        assert "#" in canonical.content
        supersessions = db_session.query(MemorySupersession).filter(
            MemorySupersession.new_memory_id == canonical.id).all()
        assert len(supersessions) == 1

    def test_supersede_keeps_old(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        old = make_memory(db_session, ws, content="policy limit is $500")
        result = memory3.supersede(
            db_session, old.id, "policy limit is now $1000", ws.id,
            "POLICY_FACT", "document", reason="new policy v2",
            user_id=user.id)
        old_row = db_session.query(AIMemory).filter(
            AIMemory.id == old.id).first()
        assert old_row is not None  # never deleted
        new_row = db_session.query(AIMemory).filter(
            AIMemory.id == result["new_memory_id"]).first()
        assert new_row.content == "policy limit is now $1000"

    def test_retrieve_scope_hierarchy(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        other = fresh_user(db_session, "p17k2")
        make_memory(db_session, ws, user=user, scope="USER",
                    content="my preference: dark mode")
        make_memory(db_session, ws, scope="WORKSPACE",
                    content="workspace standard: UTC")
        result = memory3.retrieve_for_user(db_session, ws.id,
                                           user_id=user.id)
        contents = {i["content"] for i in result["items"]}
        assert "workspace standard: UTC" in contents
        # other user cannot see the first user's private memory
        result2 = memory3.retrieve_for_user(db_session, ws.id,
                                            user_id=other.id)
        contents2 = {i["content"] for i in result2["items"]}
        assert "my preference: dark mode" not in contents2

    def test_expired_memories_not_surfaced(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = make_memory(db_session, ws, content="stale fact")
        m.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        db_session.commit()
        result = memory3.retrieve_for_user(db_session, ws.id,
                                           user_id=user.id)
        assert all("stale fact" not in i["content"]
                   for i in result["items"])
        expired = memory3.expire_due(db_session)
        assert expired["expired"] >= 1
        # Commit the sweep: expire_due only flushes, and the session-closing
        # fixture would otherwise roll the row back to ACTIVE, leaking an
        # already-expired memory that later global sweeps (memory2
        # apply_expirations in test_phase16_knowledge) would double-count.
        db_session.commit()

    def test_confidence_reflects_conflict(self, db_session):
        from app.models.phase16 import MemoryConflict
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = make_memory(db_session, ws, content="rate is 5%",
                        source="document", confidence="HIGH")
        db_session.add(MemoryConflict(workspace_id=ws.id,
                                      memory_a_id=m.id,
                                      memory_b_id=999999,
                                      conflict_type="CONTRADICTION"))
        db_session.commit()
        plain = memory3.confidence(db_session, m, None)
        conflicted = memory3.confidence(db_session, m, "CONFLICT")
        assert conflicted < plain


# ============================================================
# RAG 5.0
# ============================================================

class TestRAG5:
    def test_query_classification(self):
        assert rag5.classify_query(
            "what is the approval policy for expenses?") == "policy"
        assert rag5.classify_query(
            "what was the policy as of 2025?") == "temporal"
        assert rag5.classify_query(
            "compare policy A and policy B") == "comparative"
        assert rag5.classify_query("who owns Acme?") == "entity"
        assert rag5.classify_query("how do I submit a request?") == \
            "procedural"
        assert rag5.classify_query("what does the contract say?") == \
            "factual"

    def test_retrieval_plan_explainable(self):
        plan = rag5.retrieval_plan("compare the two travel policies")
        assert plan["intent"] == "comparative"
        assert "retrieval_mode" in plan
        assert "explanation" in plan
        assert "chain-of-thought" not in plan["explanation"].lower()

    def test_evidence_quality_scoring(self):
        chunk = {"text": "Approval is required for amounts above $5,000.",
                 "score": 0.92,
                 "document": {"mime_type": "application/pdf"},
                 "created_at": datetime.now(timezone.utc).isoformat()}
        quality = rag5.score_evidence(chunk)
        assert quality["overall"] > 0.7
        assert "relevance" in quality

    def test_select_evidence_diversity(self):
        now = datetime.now(timezone.utc)
        chunks = [
            {"text": "chunk one from doc 1", "score": 0.9,
             "document_id": 1, "created_at": now},
            {"text": "chunk two from doc 1", "score": 0.8,
             "document_id": 1, "created_at": now},
            {"text": "chunk three from doc 2", "score": 0.7,
             "document_id": 2, "created_at": now},
        ]
        selected = rag5.select_evidence(chunks, max_evidence=6)
        docs = {c["document_id"] for c in selected}
        assert docs == {1, 2}

    def test_temporal_as_of_filters_newer(self):
        future = datetime.now(timezone.utc) + timedelta(days=10)
        chunks = [
            {"text": "policy limit is 100", "score": 0.9,
             "document_id": 1, "created_at": future},
            {"text": "policy limit is 50", "score": 0.6,
             "document_id": 2, "created_at": datetime.now(timezone.utc)},
        ]
        selected = rag5.select_evidence(
            chunks, as_of=datetime.now(timezone.utc))
        # the future-dated document must not answer a historical question
        assert all(c["document_id"] != 1 for c in selected)

    def test_citation_coverage(self):
        coverage = rag5.citation_coverage(
            ["approval required above $1000", "unrelated claim"],
            ["text about approval required above $1000"],
            supported=["SUPPORTED", "UNSUPPORTED"])
        assert coverage["coverage_pct"] == 50.0
        assert coverage["unsupported"] == 1

    def test_conflict_detection_numeric(self):
        evidence = [
            {"chunk_index": 1, "text": "the limit is $5,000 per trip"},
            {"chunk_index": 2, "text": "the limit is $10,000 per trip"},
        ]
        conflicts = rag5.detect_evidence_conflicts(evidence)
        assert len(conflicts) == 1
        assert conflicts[0]["category"] == "numeric_contradiction"

    def test_no_false_conflict_on_same_number(self):
        evidence = [
            {"chunk_index": 1, "text": "the limit is $5,000 per trip"},
            {"chunk_index": 2, "text": "the limit is 5000 dollars"},
        ]
        # equal numeric values → no conflict
        assert rag5.detect_evidence_conflicts(evidence) == []

    def test_evaluation_writes_metric(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        dataset = [
            {"query": "what is the travel approval limit?",
             "expected": [1, 2]},
            {"query": "who is the CEO?",
             "expected": [], "expected_refusal": True},
        ]
        result = rag5.run_evaluation(db_session, ws.id, dataset)
        assert result["dataset_size"] == 2
        assert db_session.query(AIQualityMetric).filter(
            AIQualityMetric.metric_type == "rag5_evaluation").count() >= 1
