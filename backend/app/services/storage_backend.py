"""Storage backend abstraction for document files.

Provides a clean interface for file storage operations that works with:
A. Local filesystem storage (default)
B. S3-compatible object storage

The backend is selected via STORAGE_BACKEND config:
- "local": Use local filesystem (default)
- "s3": Use S3-compatible storage
"""

import os
import uuid
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional, Union

from ..core.config import settings

# Pattern allowing only alphanumeric characters, hyphens, and underscores for safe keys
SAFE_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, data: Union[bytes, BinaryIO], storage_key: Optional[str] = None) -> str:
        """Save file data to storage.
        
        Args:
            data: Binary data as bytes or a readable binary file-like stream.
            storage_key: Optional storage key. If omitted, a unique key is generated.
            
        Returns:
            str: The storage key under which the file was saved.
        """

    @abstractmethod
    def retrieve(self, storage_key: str) -> bytes:
        """Retrieve file content as bytes.
        
        Args:
            storage_key: The storage key of the file to retrieve.
            
        Returns:
            bytes: The binary content of the file.
        """

    @abstractmethod
    def delete(self, storage_key: str) -> bool:
        """Delete a file from storage.
        
        Args:
            storage_key: The storage key of the file to delete.
            
        Returns:
            bool: True if the file existed and was deleted, False if it did not exist.
        """

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Check if a file exists in storage.
        
        Args:
            storage_key: The storage key to check.
            
        Returns:
            bool: True if the file exists, False otherwise.
        """

    def generate_storage_key(self) -> str:
        """Generate a safe, collision-resistant unique storage key."""
        return uuid.uuid4().hex

    def _validate_key(self, storage_key: str) -> None:
        """Validate storage key against path traversal.
        
        Raises:
            ValueError: If storage key contains path traversal characters.
        """
        if not storage_key or not isinstance(storage_key, str):
            raise ValueError("Storage key must be a non-empty string.")

        # Reject path separators, null bytes, parent dir patterns
        if "/" in storage_key or "\\" in storage_key or "\0" in storage_key or ".." in storage_key:
            raise ValueError("Invalid storage key: path traversal attempt detected.")

        if not SAFE_KEY_PATTERN.match(storage_key):
            raise ValueError("Invalid storage key: contains forbidden characters.")


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            base_dir = settings.storage_dir
        self.base_dir = Path(base_dir).resolve()
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create the storage directory if it does not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, storage_key: str) -> Path:
        """Get validated file path for a storage key."""
        self._validate_key(storage_key)
        return self.base_dir / storage_key

    def save(self, data: Union[bytes, BinaryIO], storage_key: Optional[str] = None) -> str:
        """Save file data to local storage."""
        if storage_key is None:
            storage_key = self.generate_storage_key()

        file_path = self._get_file_path(storage_key)
        self._ensure_storage_dir()

        if isinstance(data, bytes):
            with open(file_path, "wb") as f:
                f.write(data)
        else:
            with open(file_path, "wb") as f:
                while chunk := data.read(1024 * 64):
                    f.write(chunk)

        return storage_key

    def retrieve(self, storage_key: str) -> bytes:
        """Retrieve file content from local storage."""
        file_path = self._get_file_path(storage_key)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found for storage key: {storage_key}")

        with open(file_path, "rb") as f:
            return f.read()

    def delete(self, storage_key: str) -> bool:
        """Delete file from local storage."""
        file_path = self._get_file_path(storage_key)
        if file_path.is_file():
            file_path.unlink()
            return True
        return False

    def exists(self, storage_key: str) -> bool:
        """Check if file exists in local storage."""
        file_path = self._get_file_path(storage_key)
        return file_path.is_file()


class S3StorageBackend(StorageBackend):
    """S3-compatible object storage backend."""

    def __init__(self):
        try:
            import boto3
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.s3_access_key or os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=settings.s3_secret_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=settings.s3_region or os.getenv('AWS_REGION', 'us-east-1'),
                endpoint_url=settings.s3_endpoint_url or None,
            )
            self.bucket = settings.s3_bucket or os.getenv('S3_BUCKET')
            if not self.bucket:
                raise ValueError("S3 bucket name is required")
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

    def save(self, data: Union[bytes, BinaryIO], storage_key: Optional[str] = None) -> str:
        """Save file data to S3."""
        if storage_key is None:
            storage_key = self.generate_storage_key()

        self._validate_key(storage_key)

        if isinstance(data, bytes):
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=storage_key,
                Body=data,
            )
        else:
            self.s3_client.upload_fileobj(
                data,
                self.bucket,
                storage_key,
            )

        return storage_key

    def retrieve(self, storage_key: str) -> bytes:
        """Retrieve file content from S3."""
        self._validate_key(storage_key)
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=storage_key,
            )
            return response['Body'].read()
        except self.s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"File not found for storage key: {storage_key}")

    def delete(self, storage_key: str) -> bool:
        """Delete file from S3."""
        self._validate_key(storage_key)
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=storage_key,
            )
            return True
        except Exception:
            return False

    def exists(self, storage_key: str) -> bool:
        """Check if file exists in S3."""
        self._validate_key(storage_key)
        
        try:
            self.s3_client.head_object(
                Bucket=self.bucket,
                Key=storage_key,
            )
            return True
        except self.s3_client.exceptions.ClientError:
            return False


def get_storage_backend() -> StorageBackend:
    """Factory: return the configured storage backend."""
    backend_name = settings.storage_backend.lower()

    if backend_name == "s3":
        return S3StorageBackend()
    elif backend_name == "local":
        return LocalStorageBackend()
    else:
        raise ValueError(f"Unknown storage backend: {backend_name!r}")


# Singleton instance
_storage_backend: Optional[StorageBackend] = None


def get_storage_backend_singleton() -> StorageBackend:
    """Get or create the singleton storage backend."""
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = get_storage_backend()
    return _storage_backend


def _sanitize_export_key(storage_key: str) -> str:
    """Flatten a storage key to a safe, path-traversal-free key.

    Exports use flat keys derived from server-generated IDs only, so
    slashes and other separators are removed rather than allowed.
    """
    safe = storage_key.replace("/", "_").replace("\\", "_").replace("\0", "")
    if ".." in safe:
        raise ValueError("Invalid storage key: path traversal attempt detected.")
    if not re.match(r"^[a-zA-Z0-9_.-]+$", safe):
        raise ValueError("Invalid storage key: contains forbidden characters.")
    return safe


def save_export(storage_key: str, data: bytes) -> str:
    """Save an export bundle using a safe flat storage key."""
    key = _sanitize_export_key(storage_key)
    backend = get_storage_backend_singleton()
    backend.save(data, storage_key=key)
    return key


def retrieve_export(storage_key: str) -> bytes:
    """Retrieve an export bundle by its safe storage key."""
    key = _sanitize_export_key(storage_key)
    backend = get_storage_backend_singleton()
    return backend.retrieve(key)


def delete_export(storage_key: str) -> bool:
    """Delete an export bundle by its safe storage key."""
    key = _sanitize_export_key(storage_key)
    backend = get_storage_backend_singleton()
    return backend.delete(key)
