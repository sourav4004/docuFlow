"""Tests for local storage service."""

import io
import pytest
from pathlib import Path
from app.services.storage import StorageService


@pytest.fixture
def storage(tmp_path: Path) -> StorageService:
    """Fixture providing a StorageService instance with an isolated temporary directory."""
    return StorageService(base_dir=tmp_path / "storage_test")


def test_unique_storage_keys_generated(storage: StorageService):
    """Test that generated storage keys are unique and non-empty."""
    keys = {storage.generate_storage_key() for _ in range(100)}
    assert len(keys) == 100
    for key in keys:
        assert isinstance(key, str)
        assert len(key) >= 16


def test_storage_save_and_retrieve_bytes(storage: StorageService):
    """Test saving and retrieving file content using bytes."""
    test_content = b"Hello DocuFlow! Testing storage operations with binary data: \x00\x01\x02\xff"
    
    key = storage.save(test_content)
    assert isinstance(key, str)
    assert storage.exists(key) is True
    
    retrieved = storage.retrieve(key)
    assert retrieved == test_content


def test_storage_save_and_retrieve_stream(storage: StorageService):
    """Test saving file content using a file-like stream."""
    stream_content = b"Streamed document payload content for DocuFlow storage test."
    stream = io.BytesIO(stream_content)
    
    key = storage.save(stream)
    assert storage.exists(key) is True
    
    retrieved = storage.retrieve(key)
    assert retrieved == stream_content


def test_storage_save_with_explicit_key(storage: StorageService):
    """Test saving with a predefined valid storage key."""
    custom_key = "custom-key_12345"
    content = b"Content with custom key"
    
    key = storage.save(content, storage_key=custom_key)
    assert key == custom_key
    assert storage.exists(custom_key) is True
    assert storage.retrieve(custom_key) == content


def test_storage_retrieve_nonexistent_raises_file_not_found(storage: StorageService):
    """Test retrieving a nonexistent file raises FileNotFoundError."""
    nonexistent_key = storage.generate_storage_key()
    assert storage.exists(nonexistent_key) is False
    
    with pytest.raises(FileNotFoundError):
        storage.retrieve(nonexistent_key)


def test_storage_delete_file(storage: StorageService):
    """Test deleting an existing file from storage."""
    content = b"File to be deleted"
    key = storage.save(content)
    assert storage.exists(key) is True
    
    # Delete existing
    deleted = storage.delete(key)
    assert deleted is True
    assert storage.exists(key) is False
    
    # Second delete returns False
    deleted_again = storage.delete(key)
    assert deleted_again is False


def test_storage_get_path(storage: StorageService):
    """Test get_path returns valid Path for existing file and raises for nonexistent."""
    content = b"Path test content"
    key = storage.save(content)
    
    path = storage.get_path(key)
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.read_bytes() == content
    
    nonexistent_key = storage.generate_storage_key()
    with pytest.raises(FileNotFoundError):
        storage.get_path(nonexistent_key)


@pytest.mark.parametrize("malicious_key", [
    "../../etc/passwd",
    "../secret.txt",
    "..\\..\\windows\\system32\\cmd.exe",
    "..\\config.py",
    "/etc/shadow",
    "\\Windows\\System32",
    "subdir/file.txt",
    "subdir\\file.txt",
    "key with spaces",
    "key\x00nullbyte",
    "key;rm -rf /",
    "key$HOME",
    "",
])
def test_path_traversal_rejected_on_save(storage: StorageService, malicious_key: str):
    """Test that path traversal attempts and invalid keys are rejected on save."""
    with pytest.raises(ValueError):
        storage.save(b"malicious content", storage_key=malicious_key)


@pytest.mark.parametrize("malicious_key", [
    "../../etc/passwd",
    "../secret.txt",
    "..\\..\\windows\\system32",
    "/etc/shadow",
    "subdir/file.txt",
    "key\x00nullbyte",
    "",
])
def test_path_traversal_rejected_on_retrieve_delete_exists(storage: StorageService, malicious_key: str):
    """Test that path traversal attempts are rejected on retrieve, delete, and exists."""
    with pytest.raises(ValueError):
        storage.retrieve(malicious_key)
        
    with pytest.raises(ValueError):
        storage.delete(malicious_key)
        
    with pytest.raises(ValueError):
        storage.exists(malicious_key)
