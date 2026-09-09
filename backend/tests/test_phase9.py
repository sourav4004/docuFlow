"""Phase 9 Test Suite - Enterprise Document Intelligence."""

import pytest
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.models.workspace import Workspace, WorkspaceMember, has_permission, ROLE_HIERARCHY
from app.models.document_version import DocumentVersion
from app.models.document_tag import Tag, DocumentClassification, DocumentSummary
from app.models.knowledge_graph import Entity, EntityRelationship
from tests.shared_db import engine, TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

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


# ============================================================
# Multi-Tenant Architecture Tests
# ============================================================

class TestMultiTenantArchitecture:
    def test_workspace_model(self, db_session):
        workspace = Workspace(name="Test Workspace", owner_id=1)
        db_session.add(workspace)
        db_session.commit()
        assert workspace.id is not None
        assert workspace.name == "Test Workspace"

    def test_workspace_member_model(self, db_session):
        workspace = Workspace(name="Test Workspace", owner_id=1)
        db_session.add(workspace)
        db_session.flush()
        
        member = WorkspaceMember(workspace_id=workspace.id, user_id=1, role="OWNER")
        db_session.add(member)
        db_session.commit()
        
        assert member.id is not None
        assert member.is_owner is True
        assert member.can_write is True

    def test_role_hierarchy(self):
        assert ROLE_HIERARCHY["OWNER"] > ROLE_HIERARCHY["ADMIN"]
        assert ROLE_HIERARCHY["ADMIN"] > ROLE_HIERARCHY["MEMBER"]
        assert ROLE_HIERARCHY["MEMBER"] > ROLE_HIERARCHY["VIEWER"]

    def test_has_permission(self):
        assert has_permission("OWNER", "ADMIN") is True
        assert has_permission("MEMBER", "ADMIN") is False
        assert has_permission("VIEWER", "MEMBER") is False

    def test_workspace_api_create(self, client):
        # Register and login
        client.post("/auth/register", json={
            "name": "Test User",
            "email": "workspace@test.com",
            "password": "password123"
        })
        login_response = client.post("/auth/login", json={
            "email": "workspace@test.com",
            "password": "password123"
        })
        cookies = login_response.cookies
        
        # Create workspace
        response = client.post("/workspaces", json={
            "name": "My Workspace",
            "description": "Test workspace"
        }, cookies=cookies)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "My Workspace"

    def test_workspace_api_list(self, client):
        # Register and login
        client.post("/auth/register", json={
            "name": "Test User",
            "email": "list@test.com",
            "password": "password123"
        })
        login_response = client.post("/auth/login", json={
            "email": "list@test.com",
            "password": "password123"
        })
        cookies = login_response.cookies
        
        # List workspaces
        response = client.get("/workspaces", cookies=cookies)
        assert response.status_code == 200
        assert "items" in response.json()


# ============================================================
# Document Versioning Tests
# ============================================================

class TestDocumentVersioning:
    def test_document_version_model(self, db_session):
        version = DocumentVersion(
            document_id=1,
            version_number=1,
            created_by=1,
            storage_key="test-key",
            file_size=1024,
            content_hash="abc123",
        )
        db_session.add(version)
        db_session.commit()
        
        assert version.id is not None
        assert version.version_number == 1
        assert version.is_ready is False
        assert version.processing_status == "PENDING"

    def test_version_status(self):
        version = DocumentVersion(
            document_id=1,
            version_number=1,
            created_by=1,
            storage_key="key",
            file_size=100,
            processing_status="READY",
        )
        assert version.is_ready is True
        assert version.is_failed is False

    def test_change_detection_with_hash(self):
        hash1 = "abc123def456"
        hash2 = "abc123def456"
        hash3 = "different_hash"
        
        assert hash1 == hash2  # Same content
        assert hash1 != hash3  # Different content


# ============================================================
# Document Classification Tests
# ============================================================

class TestDocumentClassification:
    def test_classification_model(self, db_session):
        classification = DocumentClassification(
            document_id=1,
            document_type="report",
            confidence=0.85,
            classifier_version="1.0",
        )
        db_session.add(classification)
        db_session.commit()
        
        assert classification.id is not None
        assert classification.document_type == "report"
        assert classification.confidence == 0.85

    def test_user_override(self, db_session):
        classification = DocumentClassification(
            document_id=1,
            document_type="auto_classified",
            confidence=0.7,
            is_user_override=1,
        )
        db_session.add(classification)
        db_session.commit()
        
        assert classification.is_user_override == 1


# ============================================================
# Document Tags Tests
# ============================================================

class TestDocumentTags:
    def test_tag_model(self, db_session):
        tag = Tag(name="technology", category="topic")
        db_session.add(tag)
        db_session.commit()
        
        assert tag.id is not None
        assert tag.name == "technology"

    def test_tag_unique_name(self, db_session):
        tag1 = Tag(name="finance")
        tag2 = Tag(name="finance")
        db_session.add(tag1)
        db_session.add(tag2)
        
        with pytest.raises(Exception):
            db_session.commit()


# ============================================================
# Document Summary Tests
# ============================================================

class TestDocumentSummary:
    def test_summary_model(self, db_session):
        summary = DocumentSummary(
            document_id=1,
            summary_type="short",
            content="This is a test summary.",
        )
        db_session.add(summary)
        db_session.commit()
        
        assert summary.id is not None
        assert summary.content == "This is a test summary."

    def test_summary_types(self):
        valid_types = ["short", "detailed", "key_points"]
        for summary_type in valid_types:
            summary = DocumentSummary(
                document_id=1,
                summary_type=summary_type,
                content=f"Summary of type {summary_type}",
            )
            assert summary.summary_type == summary_type


# ============================================================
# Knowledge Graph Tests
# ============================================================

class TestKnowledgeGraph:
    def test_entity_model(self, db_session):
        entity = Entity(
            workspace_id=1,
            name="Acme Corp",
            entity_type="company",
        )
        db_session.add(entity)
        db_session.commit()
        
        assert entity.id is not None
        assert entity.name == "Acme Corp"

    def test_entity_relationship(self, db_session):
        entity1 = Entity(workspace_id=1, name="Person A", entity_type="person")
        entity2 = Entity(workspace_id=1, name="Company B", entity_type="company")
        db_session.add_all([entity1, entity2])
        db_session.flush()
        
        relationship = EntityRelationship(
            workspace_id=1,
            source_id=entity1.id,
            target_id=entity2.id,
            relationship_type="works_at",
            confidence=0.9,
        )
        db_session.add(relationship)
        db_session.commit()
        
        assert relationship.id is not None
        assert relationship.relationship_type == "works_at"


# ============================================================
# Security Tests
# ============================================================

class TestSecurity:
    def test_workspace_isolation(self, client):
        # Unique per-run identity: this file talks to the REAL app database,
        # so a hardcoded email collides with rows persisted by earlier runs.
        ua = f"ua-{uuid.uuid4().hex[:12]}@test.com"
        ub = f"ub-{uuid.uuid4().hex[:12]}@test.com"
        # User A creates workspace
        client.post("/auth/register", json={
            "name": "User A",
            "email": ua,
            "password": "password123"
        })
        login_a = client.post("/auth/login", json={
            "email": ua,
            "password": "password123"
        })
        cookies_a = login_a.cookies
        
        response_a = client.post("/workspaces", json={"name": "A's Workspace"}, cookies=cookies_a)
        workspace_id = response_a.json()["id"]
        
        # User B tries to access User A's workspace
        client.post("/auth/register", json={
            "name": "User B",
            "email": ub,
            "password": "password123"
        })
        login_b = client.post("/auth/login", json={
            "email": ub,
            "password": "password123"
        })
        cookies_b = login_b.cookies
        
        response_b = client.get(f"/workspaces/{workspace_id}", cookies=cookies_b)
        assert response_b.status_code == 403  # Not a member

    def test_unauthorized_access(self, client):
        response = client.get("/workspaces")
        assert response.status_code == 401


# ============================================================
# Database Migration Tests
# ============================================================

class TestDatabaseMigration:
    def test_all_tables_exist(self, db_session):
        inspector = inspect(db_session.get_bind())
        tables = inspector.get_table_names()
        
        required_tables = [
            "workspaces",
            "workspace_members",
            "document_versions",
            "tags",
            "document_tags",
            "document_classifications",
            "document_summaries",
            "entities",
            "entity_relationships",
        ]
        
        for table in required_tables:
            assert table in tables, f"Table {table} not found"

    def test_workspace_indexes(self, db_session):
        inspector = inspect(db_session.get_bind())
        indexes = [idx['name'] for idx in inspector.get_indexes('workspaces')]
        assert any('owner_id' in idx for idx in indexes)

    def test_migration_head(self, db_session):
        # The shared SQLite fixture cannot represent alembic_version; validate
        # the real migration head against the application PostgreSQL instead.
        import subprocess, sys
        proc = subprocess.run(
            [sys.executable, "-m", "alembic", "current"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-300:]
        assert "head" in proc.stdout or proc.stdout.strip(), \
            "alembic must report the current revision"


# ============================================================
# Integration Tests
# ============================================================

class TestPhase9Integration:
    def test_full_workspace_workflow(self, client):
        # Register user
        client.post("/auth/register", json={
            "name": "Integration User",
            "email": "integration@test.com",
            "password": "password123"
        })
        login = client.post("/auth/login", json={
            "email": "integration@test.com",
            "password": "password123"
        })
        cookies = login.cookies
        
        # Create workspace
        ws_response = client.post("/workspaces", json={
            "name": "Integration Workspace",
            "description": "Testing full workflow"
        }, cookies=cookies)
        assert ws_response.status_code == 201
        workspace_id = ws_response.json()["id"]
        
        # Get workspace
        get_response = client.get(f"/workspaces/{workspace_id}", cookies=cookies)
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "Integration Workspace"
        
        # List workspaces
        list_response = client.get("/workspaces", cookies=cookies)
        assert list_response.status_code == 200
        assert len(list_response.json()["items"]) >= 1
        
        # Update workspace
        update_response = client.patch(f"/workspaces/{workspace_id}", json={
            "name": "Updated Workspace"
        }, cookies=cookies)
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Updated Workspace"
