"""Phase 16 test suite — knowledge, policy, memory, entities, RAG 4.0,
traces, approval policy, and legacy backfill.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.collection import Collection  # noqa: E402
from app.models.audit_log import AuditLog  # noqa: E402
from app.models.ai_execution import AIApproval  # noqa: E402
from app.models.knowledge_graph import Entity, EntityRelationship  # noqa: E402
from app.models.phase15 import (  # noqa: E402
    PolicyStatement, PolicyConflict, TemporalFact, AIMemory, ReviewItem,
)
from app.models.phase16 import (  # noqa: E402
    DocumentPage, MemoryConflict, EntityChange, TraceSpan, BackfillRun,
)
from app.services import multimodal as mm  # noqa: E402
from app.services import policy2 as p2  # noqa: E402
from app.services import memory2 as m2  # noqa: E402
from app.services import entity3 as e3  # noqa: E402
from app.services import rag4  # noqa: E402
from app.services import trace_service as ts  # noqa: E402
from app.services import approval2 as ap  # noqa: E402
from app.services import backfill_service as bf  # noqa: E402
from app.services.memory_service import MemoryValidationError  # noqa: E402
from app.services import notification2 as n2  # noqa: E402
from app.models.notification import Notification  # noqa: E402

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    return TestClient(app)


def fresh_user(db, tag="ku"):
    _counter[0] += 1
    user = User(name=f"Knowledge User {_counter[0]}",
                email=f"{tag}{_counter[0]}@p16-know.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user, tag="ksw"):
    _counter[0] += 1
    ws = Workspace(name=f"{tag} {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def fresh_document(db, user, ws, owner=None, workspace_scope=True):
    _counter[0] += 1
    doc = Document(
        user_id=owner.id if owner else user.id,
        workspace_id=ws.id if workspace_scope else None,
        original_filename=f"doc-{_counter[0]}.txt",
        storage_key=f"key-{uuid.uuid4().hex}",
        mime_type="text/plain",
        file_size=10,
        status="READY",
        sensitivity="INTERNAL",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def statement(db, ws, text, document_id=None, source_chunk=None):
    st = PolicyStatement(
        workspace_id=ws.id,
        document_id=document_id,
        statement=text,
        source_chunk=source_chunk or text,
    )
    db.add(st)
    db.flush()
    return st


# ============================================================
# Multimodal — layout + tables + pages + OCR
# ============================================================

class TestLayoutAndTables:
    def test_heading_detection(self):
        blocks = mm.classify_layout(
            "3.2 Budget\n\nCapital expenditure is approved by the board.\n")
        kinds = [k for k, _ in blocks]
        assert "heading" in kinds and "paragraph" in kinds

    def test_list_detection(self):
        blocks = mm.classify_layout("Requirements:\n- must encrypt\n- must audit\n")
        assert any(k == "list" for k, _ in blocks)

    def test_table_detection_pipe(self):
        assert mm.is_table_block("Name | Amount\nAlice | 100\nBob | 200")

    def test_table_detection_tab(self):
        assert mm.is_table_block("Col1\tCol2\n1\t2")

    def test_non_table_paragraph(self):
        assert not mm.is_table_block("Just a normal paragraph of prose text.")

    def test_parse_table(self):
        table = mm.parse_table_block("Item | Price\nApples | $1,200.50\n")
        assert table is not None
        assert table["headers"] == ["Item", "Price"]
        assert table["rows"][0][1]["number"] == 1200.5
        assert table["rows"][0][1]["unit"] == "$"

    def test_parse_table_totals(self):
        table = mm.parse_table_block("Total | Sum\nAll | $500\n")
        assert table is not None and table["has_totals"] is True

    def test_normalize_number_variants(self):
        assert mm.normalize_number("$1,234.56") == 1234.56
        assert mm.normalize_number("10%") == 10.0
        assert mm.normalize_number("5k") == 5000.0
        assert mm.normalize_number("n/a") is None

    def test_garbled_block_low_confidence(self):
        conf = mm.LocalTextOCRProvider._block_confidence(
            "\x00\x01\x02\xff broken \xfe", "paragraph")
        assert conf < 0.5


class TestPagesAndOCR:
    def test_ingest_and_list_pages(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws)
        pages = [
            {"page_number": 1, "text": "Executive Summary\n\nRevenue grew 12%.\n",
             "confidence": 0.95},
            {"page_number": 2, "text": "Table:\nRegion | Sales\nEU | 100\n",
             "confidence": 0.88},
        ]
        rows = mm.ingest_pages(db_session, doc.id, 1, pages)
        db_session.commit()
        assert len(rows) == 2
        listed = mm.list_pages(db_session, doc.id, version=1)
        assert len(listed) == 2
        layout = json.loads(listed[0].layout_json)
        assert any(k == "heading" for k, _ in layout)
        tables = json.loads(listed[1].tables_json)
        assert tables and tables[0]["shape"] == [1, 2]

    def test_ingest_upsert_idempotent(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws)
        page = [{"page_number": 1, "text": "same page text"}]
        mm.ingest_pages(db_session, doc.id, 1, page)
        mm.ingest_pages(db_session, doc.id, 1, page)
        db_session.commit()
        assert len(mm.list_pages(db_session, doc.id)) == 1

    def test_local_ocr_provider(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws)
        provider = mm.LocalTextOCRProvider()
        assert provider.name == "local_text"
        result = provider.extract(db_session, doc)  # no content → empty pages
        assert result.provider == "local_text"

    def test_page_region_search_filters_kind(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws)
        mm.ingest_pages(db_session, doc.id, 1, [
            {"page_number": 1, "text": "Overview\n\nPlain body text here.\n",
             "confidence": 0.9},
        ])
        db_session.commit()
        headings = mm.page_region_search(db_session, [doc.id], kind="heading")
        assert len(headings) == 1 and headings[0]["kind"] == "heading"
        tables = mm.page_region_search(db_session, [doc.id], kind="table")
        assert tables == []

    def test_region_search_min_confidence(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws)
        mm.ingest_pages(db_session, doc.id, 1, [
            {"page_number": 1, "text": "Header One", "confidence": 0.2},
        ])
        db_session.commit()
        found = mm.page_region_search(db_session, [doc.id], kind="heading",
                                      min_confidence=0.8)
        assert found == []


# ============================================================
# Policy semantic intelligence + contradictions
# ============================================================

class TestPolicySemantics:
    def test_normalize_threshold(self):
        sem = p2.normalize_statement(
            "Approval required for expenses above $1,000.")
        assert sem["action"] == "require_approval"
        assert sem["threshold"] == 1000.0
        assert sem["unit"] == "USD"

    def test_normalize_negation(self):
        sem = p2.normalize_statement(
            "No approval is required for expenses below $500.")
        assert sem["polarity"] == -1

    def test_normalize_frequency(self):
        sem = p2.normalize_statement("Compliance must review monthly.")
        assert sem["frequency"] == "monthly"

    def test_semanticize_persists(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        st = statement(db_session, ws,
                       "Approval required for expenses above $1,000.")
        sem = p2.semanticize(db_session, st)
        db_session.commit()
        assert st.subject is not None
        assert st.action == "require_approval"
        assert st.threshold_value == 1000.0
        assert sem["unit"] == "USD"

    def test_direct_negation_contradiction(self):
        verdict = p2.policy_conflict_verdict(
            "Approval is required for expenses above $1,000.",
            "Approval is not required for expenses above $1,000.")
        assert verdict["result"] == "CONTRADICTS"
        assert verdict["category"] == "NEGATION"

    def test_numeric_threshold_contradiction(self):
        verdict = p2.policy_conflict_verdict(
            "Approval required for expenses above $1,000.",
            "Expenses below $2,000 require no approval.")
        assert verdict["result"] == "CONTRADICTS"
        assert verdict["category"] == "NUMERIC_THRESHOLD"
        assert "between" in (verdict.get("condition") or "")

    def test_identical_thresholds_no_conflict(self):
        verdict = p2.policy_conflict_verdict(
            "Approval required for expenses above $1,000.",
            "Approval required for any expense above $1,000.")
        assert verdict["result"] != "CONTRADICTS"

    def test_nested_thresholds_do_conflict(self):
        # Approval above $1,000 vs above $2,000 genuinely disagree for the
        # $1,000–$2,000 band — one demands approval, the other does not.
        verdict = p2.policy_conflict_verdict(
            "Approval required for expenses above $1,000.",
            "Approval required for expenses above $2,000.")
        assert verdict["result"] == "CONTRADICTS"
        assert verdict["category"] == "NUMERIC_THRESHOLD"

    def test_frequency_conflict(self):
        verdict = p2.policy_conflict_verdict(
            "Compliance must review access monthly.",
            "Compliance must review access annually.")
        assert verdict["result"] == "CONTRADICTS"
        assert verdict["category"] == "FREQUENCY"

    def test_mutually_exclusive(self):
        verdict = p2.policy_conflict_verdict(
            "Contractors must use the approved vendor.",
            "Contractors are prohibited from using the approved vendor.")
        assert verdict["category"] == "MUTUALLY_EXCLUSIVE"

    def test_unrelated_statements_uncertain(self):
        verdict = p2.policy_conflict_verdict(
            "Pets must be leashed in the lobby.",
            "The cafeteria menu changes weekly.")
        assert verdict["result"] != "CONTRADICTS"

    def test_date_self_conflict(self):
        verdict = p2.policy_conflict_verdict(
            "Policy effective 2024-01-01 expires 2023-06-01.",
            "Anything at all.")
        assert verdict["category"] == "DATE"

    def test_superficial_text_difference_is_not_conflict(self):
        verdict = p2.policy_conflict_verdict(
            "Approval required for expenses above $1,000 per purchase.",
            "Approval required for expenses above $1,000 each purchase.")
        assert verdict["result"] != "CONTRADICTS"

    def test_scan_records_conflicts_and_reviews(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        statement(db_session, ws,
                  "Approval required for expenses above $1,000.")
        statement(db_session, ws,
                  "Expenses below $2,000 require no approval.")
        statement(db_session, ws,
                  "The office is closed on weekends.")  # unrelated
        db_session.commit()
        result = p2.detect_and_record_conflicts(db_session, ws.id)
        assert result["conflicts_recorded"] == 1
        db_session.commit()
        conflicts = db_session.query(PolicyConflict).filter(
            PolicyConflict.workspace_id == ws.id).all()
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "NUMERIC_THRESHOLD"
        reviews = db_session.query(ReviewItem).filter(
            ReviewItem.workspace_id == ws.id,
            ReviewItem.item_type == "POLICY_CONFLICT").all()
        assert len(reviews) == 1  # conflict → review, never auto-resolve

    def test_scan_dedupes(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        statement(db_session, ws,
                  "Approval required for expenses above $1,000.")
        statement(db_session, ws,
                  "Expenses below $2,000 require no approval.")
        db_session.commit()
        p2.detect_and_record_conflicts(db_session, ws.id)
        result = p2.detect_and_record_conflicts(db_session, ws.id)
        assert result["conflicts_recorded"] == 0


# ============================================================
# Memory 2.0
# ============================================================

class TestMemory2:
    def test_provenance_required(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(MemoryValidationError):
            m2.create_memory(db_session, ws.id, user.id, "WORKSPACE_FACT",
                             "WORKSPACE", "approval cap is $1,000",
                             source=None)

    def test_amount_conflict_detected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m2.create_memory(db_session, ws.id, user.id, "POLICY_FACT",
                         "WORKSPACE", "Approval cap is $1,000",
                         source="doc:1")
        mem, conflict = m2.create_memory(
            db_session, ws.id, user.id, "POLICY_FACT", "WORKSPACE",
            "Approval cap is $2,500", source="doc:2")
        db_session.commit()
        assert conflict is not None
        assert conflict.status == "OPEN"
        conflicts = m2.list_conflicts(db_session, ws.id)
        assert len(conflicts) == 1
        # both memories preserved
        both = db_session.query(AIMemory).filter(
            AIMemory.workspace_id == ws.id).count()
        assert both == 2

    def test_no_conflict_for_different_facts(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m2.create_memory(db_session, ws.id, user.id, "POLICY_FACT",
                         "WORKSPACE", "Approval cap is $1,000",
                         source="doc:1")
        mem, conflict = m2.create_memory(
            db_session, ws.id, user.id, "POLICY_FACT", "WORKSPACE",
            "Vacation cap is $1,000", source="doc:2")
        assert conflict is None

    def test_resolve_conflict_supersedes(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a, _ = m2.create_memory(db_session, ws.id, user.id, "POLICY_FACT",
                                "WORKSPACE", "Approval cap is $1,000",
                                source="doc:1")
        b, conflict = m2.create_memory(
            db_session, ws.id, user.id, "POLICY_FACT", "WORKSPACE",
            "Approval cap is $2,500", source="doc:2")
        db_session.commit()
        m2.resolve_conflict(db_session, conflict.id, user.id,
                            keep_memory_id=b.id, note="newer policy wins")
        db_session.commit()
        db_session.refresh(a)
        assert a.lifecycle_status == "SUPERSEDED"
        assert conflict.status == "RESOLVED"

    def test_lifecycle_expire_and_delete(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        mem, _ = m2.create_memory(db_session, ws.id, user.id, "USER_PREFERENCE",
                                  "USER", "Prefers concise answers",
                                  source="settings", expires_at=datetime.now(
                                      timezone.utc) - timedelta(days=1))
        db_session.commit()
        expired = m2.apply_expirations(db_session)
        assert expired == 1
        db_session.refresh(mem)
        assert mem.lifecycle_status == "EXPIRED"
        m2.delete_memory_soft(db_session, mem.id, user.id, workspace_id=ws.id)
        db_session.commit()
        db_session.refresh(mem)
        assert mem.lifecycle_status == "DELETED"

    def test_governance_export_scoped(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        m2.create_memory(db_session, ws.id, user.id, "WORKSPACE_FACT",
                         "WORKSPACE", "fact in ws1", source="d1")
        m2.create_memory(db_session, ws2.id, user.id, "WORKSPACE_FACT",
                         "WORKSPACE", "fact in ws2", source="d1")
        db_session.commit()
        exported = m2.governance_export(db_session, ws.id)
        assert len(exported) == 1
        assert exported[0]["content"] == "fact in ws1"

    def test_governance_summary(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m2.create_memory(db_session, ws.id, user.id, "DECISION", "WORKSPACE",
                         "chose vendor X", source="review")
        db_session.commit()
        summary = m2.governance_summary(db_session, ws.id)
        assert summary["total"] == 1
        assert "conflicts_open" in summary


# ============================================================
# Entity 3.0
# ============================================================

class TestEntity3:
    def test_register_entity_new(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity = e3.register_entity(db_session, ws.id, "Acme Corp", "company",
                                    document_id=1)
        db_session.commit()
        assert entity.normalized_name == e3.canonical_name("Acme Corp")
        changes = db_session.query(EntityChange).filter(
            EntityChange.entity_id == entity.id).all()
        assert any(c.change_type == "NEW_ENTITY" for c in changes)

    def test_register_accumulates_aliases(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        first = e3.register_entity(db_session, ws.id, "Acme Corp", "company")
        second = e3.register_entity(db_session, ws.id, "acme corp", "company",
                                    aliases=["ACME"])
        db_session.commit()
        assert first.id == second.id
        assert second.source_count >= 2
        assert "ACME" in json.loads(second.aliases or "[]")

    def test_canonical_forms(self):
        assert e3.canonical_name("Acme Corp.") == e3.canonical_name("acme corp")
        assert e3.canonical_name("  IBM  ") == "ibm"

    def test_duplicate_candidates_surface(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        e3.register_entity(db_session, ws.id, "Acme Corp", "company")
        e3.register_entity(db_session, ws.id, "ACME Corp", "vendor",
                           aliases=["ACME"])
        db_session.commit()
        candidates = e3.duplicate_candidates(db_session, ws.id)
        assert len(candidates) >= 1
        assert "reason" in candidates[0]
        # no auto-merge happened — both rows still exist in the workspace
        count = db_session.query(Entity).filter(
            Entity.workspace_id == ws.id).count()
        assert count == 2

    def test_relationship_reobservation_closes_prior(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = e3.register_entity(db_session, ws.id, "Alice", "person")
        b = e3.register_entity(db_session, ws.id, "Acme", "company")
        r1 = e3.record_relationship(db_session, ws.id, a.id, b.id, "works_at",
                                    confidence=0.7, source_document_id=1)
        r2 = e3.record_relationship(db_session, ws.id, a.id, b.id, "works_at",
                                    confidence=0.95, source_document_id=2)
        db_session.commit()
        assert r1.is_current is False
        assert r2.is_current is True
        changes = db_session.query(EntityChange).filter(
            EntityChange.entity_id == a.id).all()
        assert any(c.change_type == "RELATIONSHIP_CHANGED" for c in changes)

    def test_attribute_change_detection(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity = e3.register_entity(db_session, ws.id, "OldName", "person")
        event = e3.detect_entity_attribute_change(
            db_session, entity, "name", "OldName", "NewName",
            document_id=3, evidence="version 2")
        db_session.commit()
        assert event is not None and event.change_type == "RENAMED"
        assert e3.detect_entity_attribute_change(
            db_session, entity, "name", "NewName", "NewName") is None

    def test_entity_timeline_ordering(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        entity = e3.register_entity(db_session, ws.id, "Acme", "company")
        db_session.commit()
        timeline = e3.entity_timeline(db_session, entity.id)
        assert timeline  # contains NEW_ENTITY event
        times = [t["at"] for t in timeline]
        assert times == sorted(times, reverse=True)

    def test_entity_summary_scoped(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        entity = e3.register_entity(db_session, ws.id, "Acme", "company")
        e3.register_entity(db_session, ws2.id, "Other", "company")
        db_session.commit()
        summary = e3.entity_summary(db_session, ws.id, entity.id)
        assert summary["entity"]["name"] == "Acme"
        with pytest.raises(ValueError):
            e3.entity_summary(db_session, ws2.id, entity.id)  # cross-ws denied


# ============================================================
# RAG 4.0
# ============================================================

class TestRag4:
    def test_claim_supported(self):
        assert rag4.claim_support_status(
            "Revenue grew 12 percent", ["Revenue grew by 12 percent last year"]
        ) in ("SUPPORTED", "PARTIALLY_SUPPORTED")

    def test_claim_unsupported(self):
        assert rag4.claim_support_status(
            "The moon is made of cheese", ["Revenue grew 12 percent"]
        ) == "UNSUPPORTED"

    def test_no_evidence_unsupported(self):
        assert rag4.claim_support_status("anything factual", []) == "UNSUPPORTED"

    def test_matrix_statuses(self):
        matrix = rag4.claim_evidence_matrix(
            ["Revenue grew 12 percent",
             "The document sets the price to $99"],
            ["Revenue grew by 12 percent", "Price is set to $99 per unit"])
        statuses = [m["status"] for m in matrix]
        assert "UNSUPPORTED" not in statuses

    def test_insufficient_evidence(self):
        verdict = rag4.evidence_sufficiency(
            "What is the renewal deadline and the penalty clause?",
            ["The renewal deadline is March 1."])
        assert verdict["sufficient"] is False

    def test_sufficient_evidence(self):
        verdict = rag4.evidence_sufficiency(
            "What is the renewal deadline?",
            ["The renewal deadline is March 1 and applies to all vendors."])
        assert verdict["sufficient"] is True

    def test_claim_conflict_detection(self):
        claims = [
            {"claim_index": 0, "claim": "Total cost is $500", "status": "SUPPORTED"},
            {"claim_index": 1, "claim": "Total cost is $800", "status": "SUPPORTED"},
        ]
        conflicts = rag4.detect_claim_conflicts(claims)
        assert len(conflicts) == 1
        assert conflicts[0]["category"] == "NUMERIC"

    def test_conflict_aware_answer_surfaces_conflict(self):
        result = rag4.assemble_conflict_aware_answer(
            "What is the cost?",
            [
                {"claim_index": 0, "claim": "Total cost is $500",
                 "status": "SUPPORTED", "evidence_indices": [0]},
                {"claim_index": 1, "claim": "Total cost is $800",
                 "status": "SUPPORTED", "evidence_indices": [1]},
            ],
            ["cost is 500 dollars", "cost is 800 dollars"])
        assert "conflict" in result["answer"].lower()
        assert result["conflicts"]

    def test_insufficient_answer_refuses(self):
        result = rag4.assemble_conflict_aware_answer(
            "What is the penalty?",
            [{"claim_index": 0, "claim": "Penalty is severe",
              "status": "UNSUPPORTED", "evidence_indices": []}],
            ["nothing relevant here"])
        assert "insufficient" in result["answer"].lower()
        assert "not presented as fact" in result["answer"].lower()

    def test_confidence_levels(self):
        high = rag4.answer_confidence(
            [{"claim": "A", "status": "SUPPORTED"},
             {"claim": "B", "status": "SUPPORTED"}],
            ["a chunk", "another"])
        low = rag4.answer_confidence(
            [{"claim": "C", "status": "UNSUPPORTED"}], ["unrelated"])
        assert high["level"] == "HIGH"
        assert low["level"] in ("LOW", "UNKNOWN")
        assert "not a guarantee" in high["note"]

    def test_repair_drops_unsupported(self):
        result = rag4.repair_answer(
            "The cost is exactly $50. The moon is made of cheese.",
            [
                {"claim": "The cost is exactly $50.", "status": "SUPPORTED"},
                {"claim": "The moon is made of cheese.", "status": "UNSUPPORTED"},
            ],
            max_attempts=2)
        assert "moon is made of cheese" not in result["repaired"]
        assert result["attempts"] >= 1
        assert "repaired" in result["repaired"]

    def test_repair_bounded(self):
        claims = [{"claim": f"Unsolved claim number {i}", "status": "UNSUPPORTED"}
                  for i in range(10)]
        text = " ".join(c["claim"] for c in claims)
        result = rag4.repair_answer(text, claims, max_attempts=2)
        assert result["attempts"] <= 2


class TestTemporalRag:
    def _fact(self, db, ws, value, valid_from, valid_until=None):
        fact = TemporalFact(
            workspace_id=ws.id,
            fact_type="policy_expiry",
            fact_value=value,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        db.add(fact)
        db.flush()
        return fact

    def test_facts_as_of_selects_window(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._fact(db_session, ws, "deadline 2025-03-01", t0, t1)
        self._fact(db_session, ws, "deadline 2026-03-01", t1, None)
        db_session.commit()
        as_2024 = rag4.facts_as_of(db_session, ws.id, datetime(2024, 6, 1,
                                   tzinfo=timezone.utc))
        assert len(as_2024) == 1
        assert as_2024[0]["value"] == "deadline 2025-03-01"
        as_2026 = rag4.facts_as_of(db_session, ws.id, datetime(2026, 6, 1,
                                   tzinfo=timezone.utc))
        assert len(as_2026) == 1
        assert as_2026[0]["value"] == "deadline 2026-03-01"

    def test_expired_fact_not_current(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(2021, 1, 1, tzinfo=timezone.utc)
        self._fact(db_session, ws, "old", t0, t1)
        db_session.commit()
        assert rag4.facts_as_of(db_session, ws.id,
                                datetime(2022, 1, 1, tzinfo=timezone.utc)) == []


# ============================================================
# Trace platform
# ============================================================

class TestTraces:
    def test_classify_failures(self):
        assert ts.classify_failure(TimeoutError("slow")) == "timeout"
        assert ts.classify_failure(
            PermissionError("nope")) == "authorization"
        assert ts.classify_failure(ValueError("bad input")) == "validation"
        assert ts.classify_failure(RuntimeError("odd")) == "unknown"
        assert ts.is_retryable("timeout") is True
        assert ts.is_retryable("validation") is False

    def test_span_lifecycle(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        span = ts.start_span(db_session, ws.id, "retrieval",
                             input_summary="q: renewal deadline")
        ts.end_span(db_session, span, status="OK", model="m",
                    input_tokens=10, output_tokens=5)
        db_session.commit()
        assert span.completed_at is not None
        assert span.input_summary == "q: renewal deadline"
        assert span.status == "OK"

    def test_error_span_clears_summary(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        span = ts.start_span(db_session, ws.id, "provider",
                             input_summary="secret content here")
        ts.end_span(db_session, span, error=TimeoutError("slow"))
        db_session.commit()
        assert span.status == "ERROR"
        assert span.error_class == "timeout"
        assert span.input_summary is None  # never keep content on error paths

    def test_invalid_span_type(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            ts.start_span(db_session, ws.id, "garbage")

    def test_correlation_chain(self):
        chain = ts.correlate("req-1", execution_id="exec-1",
                             workflow_id="wf-1", node_execution_id="n-1")
        assert chain["request_id"] == "req-1"
        assert chain["execution_id"] == "exec-1"
        assert chain["workflow_id"] == "wf-1"
        fresh_chain = ts.correlate(None)
        assert fresh_chain["request_id"]

    def test_trace_summary(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for _ in range(3):
            span = ts.start_span(db_session, ws.id, "provider")
            ts.end_span(db_session, span)
        summary = ts.trace_summary(db_session, workspace_id=ws.id)
        assert summary["spans"] == 3
        assert summary["ok"] == 3

    def test_trace_summary_counts_errors(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        span = ts.start_span(db_session, ws.id, "tool")
        ts.end_span(db_session, span, error=ValueError("x"))
        summary = ts.trace_summary(db_session, workspace_id=ws.id)
        assert summary["errors"] == 1


# ============================================================
# Approval policy + risk engine
# ============================================================

class TestApprovalPolicy:
    def test_low_risk_baseline(self):
        result = ap.classify_action_risk("summarize")
        assert result["risk_level"] == "LOW"

    def test_delete_is_critical(self):
        result = ap.classify_action_risk("summarize",
                                         side_effects=["delete"])
        assert result["risk_level"] == "CRITICAL"

    def test_external_side_effect_high(self):
        result = ap.classify_action_risk("summarize",
                                         side_effects=["email"])
        assert result["risk_level"] == "HIGH"

    def test_bulk_scope_critical(self):
        result = ap.classify_action_risk("tag", scope_size=5000)
        assert result["risk_level"] == "CRITICAL"

    def test_restricted_data_high(self):
        result = ap.classify_action_risk("summarize",
                                         data_sensitivity="RESTRICTED")
        assert result["risk_level"] == "HIGH"

    def test_approval_required_high(self):
        decision = ap.approval_required("HIGH")
        assert decision["required"] is True

    def test_approval_not_required_low(self):
        decision = ap.approval_required("LOW")
        assert decision["required"] is False

    def test_medium_gated_by_sensitivity(self):
        assert ap.approval_required("MEDIUM")["required"] is False
        assert ap.approval_required("MEDIUM",
                                    sensitivity="CONFIDENTIAL")["required"]

    def test_approval_expiry(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        assert ap.approval_valid(future) is True
        assert ap.approval_valid(past) is False
        assert ap.approval_valid(None) is False

    def test_enforce_expired_raises(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        approval = AIApproval(
            id=str(uuid.uuid4()),
            execution_id=str(uuid.uuid4()),
            workspace_id=ws.id,
            user_id=user.id,
            action="delete_document",
            risk_level="CRITICAL",
            status="approved",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        db_session.add(approval)
        db_session.commit()
        with pytest.raises(ap.ApprovalExpiredError):
            ap.enforce_approval(approval)

    def test_action_hash_stable_and_scoped(self):
        h1 = ap.action_hash("summarize", {"document_id": 1}, "LOW")
        h2 = ap.action_hash("summarize", {"document_id": 1}, "LOW")
        h3 = ap.action_hash("summarize", {"document_id": 2}, "LOW")
        assert h1 == h2 and h1 != h3

    def test_audit_decision_writes_log(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ap.audit_decision(db_session, ws.id, requester_id=user.id,
                          reviewer_id=user.id, decision="APPROVED",
                          action_type="delete_document", payload={},
                          policy="high-risk-actions-require-approval",
                          risk_level="CRITICAL", resource_id="a1")
        db_session.commit()
        logs = db_session.query(AuditLog).filter(
            AuditLog.event_type == "approval").all()
        assert len(logs) == 1
        assert "action_hash=" in (logs[0].details or "")

    def test_action_preview_fields(self):
        preview = ap.action_preview(
            "delete_document", {"document_id": 42},
            data_scope=["doc 42"], tools=["storage"], estimated_cost_usd=0.0)
        assert preview["risk_level"] == "CRITICAL"
        assert preview["approval_required"] is True
        assert "what_will_happen" in preview
        assert "action_hash" in preview


# ============================================================
# Backfill service
# ============================================================

class TestBackfill:
    def test_audit_legacy_counts(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        fresh_document(db_session, user, ws, workspace_scope=True)
        fresh_document(db_session, user, ws, workspace_scope=False)
        audit = bf.audit_legacy(db_session)
        assert audit["documents_without_workspace"] >= 1

    def test_resolve_single_membership(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ws_id, decision = bf.resolve_workspace_for_user(db_session, user.id)
        assert decision == "assigned" and ws_id == ws.id

    def test_resolve_ambiguous(self, db_session):
        user = fresh_user(db_session)
        fresh_workspace(db_session, user)
        fresh_workspace(db_session, user)
        from app.models.workspace import WorkspaceMember
        db_session.add(WorkspaceMember(user_id=user.id, workspace_id=1,
                                       role="MEMBER"))
        # user owns two workspaces → ambiguous, never guessed
        ws_id, decision = bf.resolve_workspace_for_user(db_session, user.id)
        assert decision == "ambiguous" and ws_id is None

    def test_dry_run_assigns_nothing(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws, workspace_scope=False)
        db_session.commit()
        run = bf.create_run(db_session, created_by=user.id,
                            kind="document_workspace", dry_run=True,
                            batch_size=10, user_id=user.id)
        result = bf.run_complete(db_session, run.id)
        db_session.commit()
        assert result["assigned"] == 1  # would assign
        db_session.refresh(doc)
        assert doc.workspace_id is None  # dry run wrote nothing

    def test_execute_assigns_workspace(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws, workspace_scope=False)
        db_session.commit()
        run = bf.create_run(db_session, created_by=user.id,
                            kind="document_workspace", dry_run=False,
                            batch_size=10, user_id=user.id)
        result = bf.run_complete(db_session, run.id)
        db_session.commit()
        assert result["assigned"] == 1
        db_session.refresh(doc)
        assert doc.workspace_id == ws.id
        remaining = db_session.query(Document).filter(
            Document.id == doc.id,
            Document.workspace_id.is_(None)).count()
        assert remaining == 0

    def test_execute_resumable_batches(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        docs = [fresh_document(db_session, user, ws, workspace_scope=False)
                for _ in range(5)]
        db_session.commit()
        run = bf.create_run(db_session, created_by=user.id,
                            kind="document_workspace", dry_run=False,
                            batch_size=2, user_id=user.id)
        first = bf.run_complete(db_session, run.id)
        db_session.commit()
        assert first["assigned"] == 5
        assert all(d.workspace_id == ws.id for d in docs)

    def test_run_validation(self, db_session):
        user = fresh_user(db_session)
        with pytest.raises(bf.BackfillError):
            bf.create_run(db_session, created_by=user.id, kind="bogus")
        with pytest.raises(bf.BackfillError):
            bf.create_run(db_session, created_by=user.id,
                          kind="document_workspace", batch_size=100000)

    def test_ambiguous_never_assigned(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        doc = fresh_document(db_session, user, ws1, workspace_scope=False)
        db_session.commit()
        run = bf.create_run(db_session, created_by=user.id,
                            kind="document_workspace", dry_run=False,
                            user_id=user.id)
        result = bf.run_complete(db_session, run.id)
        db_session.commit()
        assert result["ambiguous"] == 1
        db_session.refresh(doc)
        assert doc.workspace_id is None  # never guessed


# ============================================================
# Notifications + email abstraction
# ============================================================

class TestNotification2:
    def test_notify_in_app(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        out = n2.notify(db_session, user.id, "Alert", "hello",
                        notification_type="ai_alert", resource_type="doc")
        db_session.commit()
        assert out["created"] is True
        assert out["deliveries"] == [{"channel": "IN_APP", "status": "SENT"}]
        assert n2.list_deliveries(db_session, user.id)

    def test_dedupe_window(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        n2.notify(db_session, user.id, "Same", "x", notification_type="t")
        out = n2.notify(db_session, user.id, "Same", "x", notification_type="t")
        db_session.commit()
        assert out["deduplicated"] is True
        assert db_session.query(Notification).filter(
            Notification.user_id == user.id).count() == 1

    def test_email_channel_uses_provider(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        sent = []

        class FakeEmail(n2.EmailProvider):
            @property
            def name(self):
                return "fake-email"

            def send(self, to, subject, body):
                sent.append((to, subject, len(body)))
                return True, None

        out = n2.notify(db_session, user.id, "Mail", "body here",
                        notification_type="mail",
                        channels=["IN_APP", "EMAIL"], email_to="x@example.com",
                        email_provider=FakeEmail())
        db_session.commit()
        assert sent  # email was "sent" through the abstraction
        channels = [d["channel"] for d in out["deliveries"]]
        assert "EMAIL" in channels

    def test_unknown_channel_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            n2.notify(db_session, user.id, "x", "y", channels=["PAGER"])

    def test_log_email_provider_is_credential_free(self):
        sent = n2.LogEmailProvider().send("someone@example.com", "S", "B")
        assert sent == (True, None)
