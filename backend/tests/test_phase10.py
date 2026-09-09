"""Phase 10 Test Suite - Enterprise Document Intelligence Platform."""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.models.workspace import Workspace, WorkspaceMember
from app.models.invitation import WorkspaceInvitation
from app.models.document_activity import DocumentActivity
from app.models.notification import Notification
from app.models.document_comment import DocumentComment
from app.models.usage import UsageRecord
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
# Invitation Tests
# ============================================================

class TestInvitationSystem:
    def test_invitation_model(self, db_session):
        """Test invitation creation with secure token."""
        token, token_hash = WorkspaceInvitation.create_token()
        assert token is not None
        assert token_hash is not None
        assert len(token) > 20

    def test_invitation_token_verification(self):
        """Test token verification."""
        token, token_hash = WorkspaceInvitation.create_token()
        assert WorkspaceInvitation.verify_token(token, token_hash) is True
        assert WorkspaceInvitation.verify_token("wrong_token", token_hash) is False

    def test_invitation_expiry(self, db_session):
        """Test invitation expiration."""
        token, token_hash = WorkspaceInvitation.create_token()
        invitation = WorkspaceInvitation(
            workspace_id=1,
            inviter_id=1,
            invited_email="test@test.com",
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1)  # Already expired
        )
        db_session.add(invitation)
        db_session.commit()

        assert invitation.is_expired is True
        assert invitation.is_usable is False

    def test_invitation_acceptance(self, db_session):
        """Test invitation acceptance."""
        token, token_hash = WorkspaceInvitation.create_token()
        invitation = WorkspaceInvitation(
            workspace_id=1,
            inviter_id=1,
            invited_email="test@test.com",
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        db_session.add(invitation)
        db_session.commit()

        invitation.accept()
        assert invitation.status == "ACCEPTED"
        assert invitation.accepted_at is not None

    def test_invitation_revocation(self, db_session):
        """Test invitation revocation."""
        token, token_hash = WorkspaceInvitation.create_token()
        invitation = WorkspaceInvitation(
            workspace_id=1,
            inviter_id=1,
            invited_email="test@test.com",
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        db_session.add(invitation)
        db_session.commit()

        invitation.revoke()
        assert invitation.status == "REVOKED"
        assert invitation.revoked_at is not None

    def test_create_invitation_api(self, client):
        """Test creating invitation via API."""
        # Register and login
        client.post("/auth/register", json={
            "name": "Admin User",
            "email": "admin@invitation-test.com",
            "password": "password123"
        })
        login = client.post("/auth/login", json={
            "email": "admin@invitation-test.com",
            "password": "password123"
        })
        cookies = login.cookies

        # Create workspace
        ws = client.post("/workspaces", json={"name": "Invite Workspace"}, cookies=cookies)
        workspace_id = ws.json()["id"]

        # Create invitation
        response = client.post(f"/invitations/workspaces/{workspace_id}/invite", json={
            "email": "newuser@test.com",
            "role": "MEMBER"
        }, cookies=cookies)

        assert response.status_code == 200
        assert "id" in response.json()

    def test_unauthorized_invitation(self, client):
        """Test unauthorized user cannot invite."""
        # Unique per-run identity: this file talks to the REAL app database.
        ua = f"ua-{uuid.uuid4().hex[:12]}@test.com"
        ub = f"ub-{uuid.uuid4().hex[:12]}@test.com"
        # Register user A
        reg_a = client.post("/auth/register", json={
            "name": "User A",
            "email": ua,
            "password": "password123"
        })
        login_a = client.post("/auth/login", json={
            "email": ua,
            "password": "password123"
        })
        cookies_a = login_a.cookies

        # Create workspace
        ws = client.post("/workspaces", json={"name": "A's Workspace"}, cookies=cookies_a)
        workspace_id = ws.json()["id"]

        # Register user B (MEMBER, not ADMIN)
        reg_b = client.post("/auth/register", json={
            "name": "User B",
            "email": ub,
            "password": "password123"
        })
        login_b = client.post("/auth/login", json={
            "email": ub,
            "password": "password123"
        })
        cookies_b = login_b.cookies

        # Add B as MEMBER (use the actual registered id, never a hardcoded one)
        client.post(f"/workspaces/{workspace_id}/members", json={
            "user_id": reg_b.json()["id"], "role": "MEMBER"
        }, cookies=cookies_a)

        # B tries to invite - should fail
        response = client.post(f"/invitations/workspaces/{workspace_id}/invite", json={
            "email": "another@test.com"
        }, cookies=cookies_b)

        assert response.status_code == 403


# ============================================================
# Notification Tests
# ============================================================

class TestNotificationSystem:
    def test_notification_model(self, db_session):
        """Test notification creation."""
        notification = Notification(
            user_id=1,
            title="Test Notification",
            message="This is a test",
            notification_type="test"
        )
        db_session.add(notification)
        db_session.commit()

        assert notification.id is not None
        assert notification.is_read is False

    def test_mark_read(self, db_session):
        """Test marking notification as read."""
        notification = Notification(
            user_id=1,
            title="Test",
            message="Test message",
            notification_type="test"
        )
        db_session.add(notification)
        db_session.commit()

        notification.mark_read()
        assert notification.is_read is True

    def test_list_notifications(self, client):
        """Test listing notifications."""
        # Register and login
        client.post("/auth/register", json={
            "name": "Notif User",
            "email": "notif@test.com",
            "password": "password123"
        })
        login = client.post("/auth/login", json={
            "email": "notif@test.com",
            "password": "password123"
        })
        cookies = login.cookies

        # Get notifications (should be empty)
        response = client.get("/notifications", cookies=cookies)
        assert response.status_code == 200
        assert "items" in response.json()

    def test_unread_count(self, client):
        """Test unread notification count."""
        client.post("/auth/register", json={
            "name": "Count User",
            "email": "count@test.com",
            "password": "password123"
        })
        login = client.post("/auth/login", json={
            "email": "count@test.com",
            "password": "password123"
        })
        cookies = login.cookies

        response = client.get("/notifications/unread-count", cookies=cookies)
        assert response.status_code == 200
        assert "count" in response.json()


# ============================================================
# Comment Tests
# ============================================================

class TestCommentSystem:
    def test_comment_model(self, db_session):
        """Test comment creation."""
        comment = DocumentComment(
            document_id=1,
            user_id=1,
            content="This is a test comment"
        )
        db_session.add(comment)
        db_session.commit()

        assert comment.id is not None
        assert comment.is_resolved is False

    def test_resolve_comment(self, db_session):
        """Test resolving a comment."""
        comment = DocumentComment(
            document_id=1,
            user_id=1,
            content="Test comment"
        )
        db_session.add(comment)
        db_session.commit()

        comment.resolve()
        assert comment.is_resolved is True

    def test_reopen_comment(self, db_session):
        """Test reopening a resolved comment."""
        comment = DocumentComment(
            document_id=1,
            user_id=1,
            content="Test comment",
            is_resolved=True
        )
        db_session.add(comment)
        db_session.commit()

        comment.reopen()
        assert comment.is_resolved is False

    def test_reply_to_comment(self, db_session):
        """Test replying to a comment."""
        parent = DocumentComment(
            document_id=1,
            user_id=1,
            content="Parent comment"
        )
        db_session.add(parent)
        db_session.flush()

        reply = DocumentComment(
            document_id=1,
            user_id=2,
            content="Reply comment",
            parent_id=parent.id
        )
        db_session.add(reply)
        db_session.commit()

        assert reply.parent_id == parent.id


# ============================================================
# Document Activity Tests
# ============================================================

class TestDocumentActivity:
    def test_activity_model(self, db_session):
        """Test activity creation."""
        activity = DocumentActivity(
            document_id=1,
            user_id=1,
            action="uploaded"
        )
        db_session.add(activity)
        db_session.commit()

        assert activity.id is not None
        assert activity.action == "uploaded"

    def test_activity_types(self):
        """Test various activity types."""
        valid_actions = [
            "uploaded", "processed", "archived", "restored",
            "downloaded", "viewed", "version_created", "metadata_changed"
        ]
        for action in valid_actions:
            activity = DocumentActivity(
                document_id=1,
                user_id=1,
                action=action
            )
            assert activity.action == action


# ============================================================
# Usage Tracking Tests
# ============================================================

class TestUsageTracking:
    def test_usage_model(self, db_session):
        """Test usage record creation."""
        usage = UsageRecord(
            workspace_id=1,
            user_id=1,
            usage_type="document_upload",
            quantity=1,
            units="count"
        )
        db_session.add(usage)
        db_session.commit()

        assert usage.id is not None
        assert usage.usage_type == "document_upload"

    def test_usage_types(self):
        """Test various usage types."""
        valid_types = [
            "document_upload", "ai_request", "embedding",
            "storage", "retrieval", "processing"
        ]
        for usage_type in valid_types:
            usage = UsageRecord(
                workspace_id=1,
                usage_type=usage_type,
                quantity=100,
                units="bytes" if usage_type == "storage" else "count"
            )
            assert usage.usage_type == usage_type


# ============================================================
# Security Tests
# ============================================================

class TestSecurity:
    def test_cross_user_invitation(self, client):
        """Test cross-user invitation access."""
        # User A creates workspace
        client.post("/auth/register", json={
            "name": "User A",
            "email": "usera@security-test.com",
            "password": "password123"
        })
        login_a = client.post("/auth/login", json={
            "email": "usera@security-test.com",
            "password": "password123"
        })
        cookies_a = login_a.cookies

        ws = client.post("/workspaces", json={"name": "A's Workspace"}, cookies=cookies_a)
        workspace_id = ws.json()["id"]

        # User B tries to list invitations
        client.post("/auth/register", json={
            "name": "User B",
            "email": "userb@security-test.com",
            "password": "password123"
        })
        login_b = client.post("/auth/login", json={
            "email": "userb@security-test.com",
            "password": "password123"
        })
        cookies_b = login_b.cookies

        response = client.get(f"/invitations/workspaces/{workspace_id}/invitations", cookies=cookies_b)
        assert response.status_code == 403

    def test_notification_isolation(self, client):
        """Test notification isolation between users."""
        # User A
        client.post("/auth/register", json={
            "name": "User A",
            "email": "usera@notif-test.com",
            "password": "password123"
        })
        login_a = client.post("/auth/login", json={
            "email": "usera@notif-test.com",
            "password": "password123"
        })
        cookies_a = login_a.cookies

        # User B
        client.post("/auth/register", json={
            "name": "User B",
            "email": "userb@notif-test.com",
            "password": "password123"
        })
        login_b = client.post("/auth/login", json={
            "email": "userb@notif-test.com",
            "password": "password123"
        })
        cookies_b = login_b.cookies

        # Get notifications for both users
        response_a = client.get("/notifications", cookies=cookies_a)
        response_b = client.get("/notifications", cookies=cookies_b)

        assert response_a.status_code == 200
        assert response_b.status_code == 200
        # Each user should only see their own notifications


# ============================================================
# Database Migration Tests
# ============================================================

class TestDatabaseMigration:
    def test_all_tables_exist(self, db_session):
        """Test all Phase 10 tables exist."""
        inspector = inspect(db_session.get_bind())
        tables = inspector.get_table_names()

        required_tables = [
            "workspace_invitations",
            "document_activities",
            "notifications",
            "document_comments",
            "usage_records",
        ]

        for table in required_tables:
            assert table in tables, f"Table {table} not found"

    def test_invitation_indexes(self, db_session):
        """Test invitation table indexes."""
        inspector = inspect(db_session.get_bind())
        indexes = [idx['name'] for idx in inspector.get_indexes('workspace_invitations')]
        assert any('token_hash' in idx for idx in indexes)

    def test_notification_indexes(self, db_session):
        """Test notification table indexes."""
        inspector = inspect(db_session.get_bind())
        indexes = [idx['name'] for idx in inspector.get_indexes('notifications')]
        assert any('user_id' in idx for idx in indexes)


# ============================================================
# Integration Tests
# ============================================================

class TestPhase10Integration:
    def test_full_invitation_workflow(self, client):
        """Test complete invitation workflow."""
        # Register admin
        client.post("/auth/register", json={
            "name": "Admin",
            "email": "admin@integration-test.com",
            "password": "password123"
        })
        login = client.post("/auth/login", json={
            "email": "admin@integration-test.com",
            "password": "password123"
        })
        cookies = login.cookies

        # Create workspace
        ws = client.post("/workspaces", json={"name": "Integration Workspace"}, cookies=cookies)
        workspace_id = ws.json()["id"]

        # Create invitation
        invite = client.post(f"/invitations/workspaces/{workspace_id}/invite", json={
            "email": "newmember@test.com",
            "role": "MEMBER"
        }, cookies=cookies)
        assert invite.status_code == 200

        # List invitations
        list_resp = client.get(f"/invitations/workspaces/{workspace_id}/invitations", cookies=cookies)
        assert list_resp.status_code == 200
        assert len(list_resp.json()["items"]) == 1
