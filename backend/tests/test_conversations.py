"""Phase 5.1 tests: Conversation and Message data models.

Tests cover:
- Model creation
- Relationships
- Cascade deletion
- Message ordering
- Role constraints
- User isolation
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from app.core.database import Base, SessionLocal
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message, VALID_ROLES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Clean up test data before and after each test."""
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'conv_%@example.com'))"))
        db.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'conv_%@example.com')"))
        db.execute(text("DELETE FROM users WHERE email LIKE 'conv_%@example.com'"))
        db.commit()
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'conv_%@example.com'))"))
        db.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'conv_%@example.com')"))
        db.execute(text("DELETE FROM users WHERE email LIKE 'conv_%@example.com'"))
        db.commit()
    finally:
        db.close()


def create_test_user(db, name="TestUser", email="conv_test@example.com"):
    user = User(name=name, email=email, password_hash="h")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ==========================================================================
# 1. CONVERSATION CREATION
# ==========================================================================

class TestConversationCreation:
    def test_create_conversation(self):
        db = SessionLocal()
        try:
            user = create_test_user(db)
            conv = Conversation(user_id=user.id, title="Test Conv")
            db.add(conv)
            db.commit()
            db.refresh(conv)
            assert conv.id is not None
            assert conv.user_id == user.id
            assert conv.title == "Test Conv"
            assert conv.created_at is not None
            assert conv.updated_at is not None
        finally:
            db.close()

    def test_default_title(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_def@example.com")
            conv = Conversation(user_id=user.id)
            db.add(conv)
            db.commit()
            db.refresh(conv)
            assert conv.title == "New Conversation"
        finally:
            db.close()

    def test_belongs_to_user(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_belongs@example.com")
            conv = Conversation(user_id=user.id, title="My Conv")
            db.add(conv)
            db.commit()
            db.refresh(conv)
            assert conv.user_id == user.id
            # Relationship
            assert conv.user.id == user.id
        finally:
            db.close()

    def test_user_has_conversations(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_has@example.com")
            c1 = Conversation(user_id=user.id, title="Conv 1")
            c2 = Conversation(user_id=user.id, title="Conv 2")
            db.add_all([c1, c2])
            db.commit()
            db.refresh(user)
            assert len(user.conversations) == 2
        finally:
            db.close()


# ==========================================================================
# 2. MESSAGE CREATION
# ==========================================================================

class TestMessageCreation:
    def test_create_message(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_msg@example.com")
            conv = Conversation(user_id=user.id, title="Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msg = Message(conversation_id=conv.id, role="user", content="Hello!")
            db.add(msg)
            db.commit()
            db.refresh(msg)

            assert msg.id is not None
            assert msg.conversation_id == conv.id
            assert msg.role == "user"
            assert msg.content == "Hello!"
            assert msg.created_at is not None
        finally:
            db.close()

    def test_belongs_to_conversation(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_belongs2@example.com")
            conv = Conversation(user_id=user.id, title="Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msg = Message(conversation_id=conv.id, role="assistant", content="Hi there!")
            db.add(msg)
            db.commit()
            db.refresh(msg)

            assert msg.conversation.id == conv.id
        finally:
            db.close()

    def test_conversation_has_messages(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_hasmsg@example.com")
            conv = Conversation(user_id=user.id, title="Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            m1 = Message(conversation_id=conv.id, role="user", content="Q1")
            m2 = Message(conversation_id=conv.id, role="assistant", content="A1")
            m3 = Message(conversation_id=conv.id, role="user", content="Q2")
            db.add_all([m1, m2, m3])
            db.commit()
            db.refresh(conv)

            assert len(conv.messages) == 3
        finally:
            db.close()

    def test_multiple_messages_ordering(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_order@example.com")
            conv = Conversation(user_id=user.id, title="Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msgs = []
            for i in range(5):
                role = "user" if i % 2 == 0 else "assistant"
                msg = Message(conversation_id=conv.id, role=role, content=f"Message {i}")
                db.add(msg)
                msgs.append(msg)
            db.commit()

            db.refresh(conv)
            # Messages ordered by id
            for i, msg in enumerate(conv.messages):
                assert msg.content == f"Message {i}"
        finally:
            db.close()


# ==========================================================================
# 3. ROLE VALIDATION
# ==========================================================================

class TestRoleValidation:
    def test_valid_roles(self):
        assert "user" in VALID_ROLES
        assert "assistant" in VALID_ROLES

    def test_user_role_accepted(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_role1@example.com")
            conv = Conversation(user_id=user.id, title="Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msg = Message(conversation_id=conv.id, role="user", content="Q")
            db.add(msg)
            db.commit()
            assert msg.id is not None
        finally:
            db.close()

    def test_assistant_role_accepted(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_role2@example.com")
            conv = Conversation(user_id=user.id, title="Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msg = Message(conversation_id=conv.id, role="assistant", content="A")
            db.add(msg)
            db.commit()
            assert msg.id is not None
        finally:
            db.close()

    def test_invalid_role_rejected(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_role3@example.com")
            conv = Conversation(user_id=user.id, title="Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msg = Message(conversation_id=conv.id, role="system", content="X")
            db.add(msg)
            with pytest.raises(IntegrityError):
                db.commit()
            db.rollback()
        finally:
            db.close()


# ==========================================================================
# 4. CASCADE DELETION
# ==========================================================================

class TestCascadeDeletion:
    def test_delete_conversation_deletes_messages(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_del1@example.com")
            conv = Conversation(user_id=user.id, title="To Delete")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            m1 = Message(conversation_id=conv.id, role="user", content="Q1")
            m2 = Message(conversation_id=conv.id, role="assistant", content="A1")
            db.add_all([m1, m2])
            db.commit()

            conv_id = conv.id
            db.delete(conv)
            db.commit()

            # Verify messages are gone
            remaining = db.query(Message).filter(Message.conversation_id == conv_id).count()
            assert remaining == 0
        finally:
            db.close()

    def test_delete_user_deletes_conversations_and_messages(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_del2@example.com")
            conv = Conversation(user_id=user.id, title="User Conv")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msg = Message(conversation_id=conv.id, role="user", content="Hello")
            db.add(msg)
            db.commit()

            user_id = user.id
            conv_id = conv.id

            # Delete user
            db.delete(user)
            db.commit()

            # Verify conversation is gone
            assert db.query(Conversation).filter(Conversation.id == conv_id).count() == 0
            # Verify messages are gone
            assert db.query(Message).filter(Message.conversation_id == conv_id).count() == 0
        finally:
            db.close()

    def test_no_orphan_records(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_orphan@example.com")
            conv = Conversation(user_id=user.id, title="Orphan Test")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            msg = Message(conversation_id=conv.id, role="user", content="Orphan?")
            db.add(msg)
            db.commit()

            # Save IDs before deletion
            user_id = user.id
            conv_id = conv.id

            # Delete user — everything should cascade
            db.delete(user)
            db.commit()

            # Verify no orphans
            assert db.query(Conversation).filter(Conversation.user_id == user_id).count() == 0
            assert db.query(Message).filter(Message.conversation_id == conv_id).count() == 0
        finally:
            db.close()


# ==========================================================================
# 5. USER ISOLATION
# ==========================================================================

class TestUserIsolation:
    def test_users_have_separate_conversations(self):
        db = SessionLocal()
        try:
            user_a = create_test_user(db, name="UserA", email="conv_iso_a@example.com")
            user_b = create_test_user(db, name="UserB", email="conv_iso_b@example.com")

            conv_a = Conversation(user_id=user_a.id, title="A's Conv")
            conv_b = Conversation(user_id=user_b.id, title="B's Conv")
            db.add_all([conv_a, conv_b])
            db.commit()
            db.refresh(user_a)
            db.refresh(user_b)

            # Each user has only their own conversation
            assert len(user_a.conversations) == 1
            assert user_a.conversations[0].title == "A's Conv"
            assert len(user_b.conversations) == 1
            assert user_b.conversations[0].title == "B's Conv"
        finally:
            db.close()

    def test_cross_user_query_isolation(self):
        db = SessionLocal()
        try:
            user_a = create_test_user(db, name="UserA2", email="conv_iso_a2@example.com")
            user_b = create_test_user(db, name="UserB2", email="conv_iso_b2@example.com")

            conv_a = Conversation(user_id=user_a.id, title="A Private")
            conv_b = Conversation(user_id=user_b.id, title="B Private")
            db.add_all([conv_a, conv_b])
            db.commit()

            # Query as User A — should only see User A's conversations
            a_convs = db.query(Conversation).filter(Conversation.user_id == user_a.id).all()
            assert len(a_convs) == 1
            assert a_convs[0].title == "A Private"

            # Query as User B — should only see User B's conversations
            b_convs = db.query(Conversation).filter(Conversation.user_id == user_b.id).all()
            assert len(b_convs) == 1
            assert b_convs[0].title == "B Private"
        finally:
            db.close()


# ==========================================================================
# 6. EDGE CASES
# ==========================================================================

class TestEdgeCases:
    def test_long_message_content(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_long@example.com")
            conv = Conversation(user_id=user.id, title="Long")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            long_content = "x" * 100_000
            msg = Message(conversation_id=conv.id, role="user", content=long_content)
            db.add(msg)
            db.commit()
            db.refresh(msg)
            assert len(msg.content) == 100_000
        finally:
            db.close()

    def test_long_title(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_longtitle@example.com")
            long_title = "T" * 500
            conv = Conversation(user_id=user.id, title=long_title)
            db.add(conv)
            db.commit()
            db.refresh(conv)
            assert len(conv.title) == 500
        finally:
            db.close()

    def test_empty_messages_list(self):
        db = SessionLocal()
        try:
            user = create_test_user(db, email="conv_empty@example.com")
            conv = Conversation(user_id=user.id, title="Empty")
            db.add(conv)
            db.commit()
            db.refresh(conv)
            assert conv.messages == []
        finally:
            db.close()
