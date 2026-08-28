"""Tests for Document SQLAlchemy database model."""

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.user import User
from app.models.document import Document
from tests.test_auth import TestingSessionLocal


@pytest.fixture
def db():
    """Provide a database session fixture."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_document_model_creation(db):
    """Test creating a Document model and persisting it."""
    user = User(
        name="Doc Owner",
        email="docowner_model@example.com",
        password_hash="hashed_pw"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    doc = Document(
        user_id=user.id,
        original_filename="sample_invoice.pdf",
        storage_key="abc123def456key_model",
        mime_type="application/pdf",
        file_size=1048576,
        status="UPLOADED"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    assert doc.id is not None
    assert doc.user_id == user.id
    assert doc.original_filename == "sample_invoice.pdf"
    assert doc.storage_key == "abc123def456key_model"
    assert doc.mime_type == "application/pdf"
    assert doc.file_size == 1048576
    assert doc.status == "UPLOADED"
    assert doc.created_at is not None
    assert doc.updated_at is not None


def test_document_belongs_to_user(db):
    """Test relationship between User and Document."""
    user = User(
        name="User Relations",
        email="relations_model@example.com",
        password_hash="hashed_pw"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    doc1 = Document(
        user_id=user.id,
        original_filename="file1.pdf",
        storage_key="key_1_model",
        mime_type="application/pdf",
        file_size=100,
        status="UPLOADED"
    )
    doc2 = Document(
        user_id=user.id,
        original_filename="file2.png",
        storage_key="key_2_model",
        mime_type="image/png",
        file_size=200,
        status="UPLOADED"
    )
    db.add_all([doc1, doc2])
    db.commit()
    db.refresh(user)
    db.refresh(doc1)

    # Test document.user backref
    assert doc1.user is not None
    assert doc1.user.id == user.id
    assert doc1.user.email == "relations_model@example.com"

    # Test user.documents relationship
    assert len(user.documents) >= 2
    filenames = [d.original_filename for d in user.documents]
    assert "file1.pdf" in filenames
    assert "file2.png" in filenames


def test_document_unique_storage_key(db):
    """Test that storage_key must be unique."""
    user = User(
        name="Test Unique",
        email="unique_model@example.com",
        password_hash="hashed_pw"
    )
    db.add(user)
    db.commit()

    doc1 = Document(
        user_id=user.id,
        original_filename="first.pdf",
        storage_key="duplicate_key_model",
        mime_type="application/pdf",
        file_size=100
    )
    db.add(doc1)
    db.commit()

    doc2 = Document(
        user_id=user.id,
        original_filename="second.pdf",
        storage_key="duplicate_key_model",
        mime_type="application/pdf",
        file_size=200
    )
    db.add(doc2)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_document_repr():
    """Test string representation of Document."""
    doc = Document(
        id=42,
        user_id=7,
        original_filename="contract.pdf",
        storage_key="k42",
        mime_type="application/pdf",
        file_size=500
    )
    repr_str = repr(doc)
    assert "42" in repr_str
    assert "contract.pdf" in repr_str
    assert "7" in repr_str
