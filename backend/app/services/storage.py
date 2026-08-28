"""Storage service for document files."""

import uuid
import re
from pathlib import Path
from typing import BinaryIO, Union, Optional
from ..core.config import settings


# Pattern allowing only alphanumeric characters, hyphens, and underscores for safe keys
SAFE_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class StorageService:
    """Service for handling file storage operations locally."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            base_dir = settings.storage_dir
        self.base_dir = Path(base_dir).resolve()
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create the storage directory if it does not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_storage_key(self) -> str:
        """Generate a safe, collision-resistant unique storage key.
        
        Uses UUID4 hex string which contains only safe alphanumeric characters.
        Original filenames are never used for disk storage.
        """
        return uuid.uuid4().hex

    def _validate_key(self, storage_key: str) -> Path:
        """Validate storage key against path traversal and return resolved path.
        
        Raises:
            ValueError: If storage key contains path traversal characters, separators,
                        or does not match safe key pattern.
        """
        if not storage_key or not isinstance(storage_key, str):
            raise ValueError("Storage key must be a non-empty string.")

        # Reject path separators, null bytes, parent dir patterns
        if "/" in storage_key or "\\" in storage_key or "\0" in storage_key or ".." in storage_key:
            raise ValueError("Invalid storage key: path traversal attempt detected.")

        if not SAFE_KEY_PATTERN.match(storage_key):
            raise ValueError("Invalid storage key: contains forbidden characters.")

        # Resolve path and verify it stays strictly within base_dir
        resolved_path = (self.base_dir / storage_key).resolve()
        try:
            resolved_path.relative_to(self.base_dir)
        except ValueError:
            raise ValueError("Invalid storage key: resolved path is outside storage directory.")

        return resolved_path

    def save(self, data: Union[bytes, BinaryIO], storage_key: Optional[str] = None) -> str:
        """Save file data to local storage.
        
        Args:
            data: Binary data as bytes or a readable binary file-like stream.
            storage_key: Optional storage key. If omitted, a unique key is generated.
            
        Returns:
            str: The storage key under which the file was saved.
            
        Raises:
            ValueError: If storage key validation fails.
            IOError: If writing to disk fails.
        """
        if storage_key is None:
            storage_key = self.generate_storage_key()

        file_path = self._validate_key(storage_key)
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
        """Retrieve file content as bytes.
        
        Args:
            storage_key: The storage key of the file to retrieve.
            
        Returns:
            bytes: The binary content of the file.
            
        Raises:
            ValueError: If storage key validation fails.
            FileNotFoundError: If the file does not exist.
        """
        file_path = self._validate_key(storage_key)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found for storage key: {storage_key}")

        with open(file_path, "rb") as f:
            return f.read()

    def get_path(self, storage_key: str) -> Path:
        """Get the validated filesystem path for a storage key.
        
        Args:
            storage_key: The storage key.
            
        Returns:
            Path: The resolved file Path.
            
        Raises:
            ValueError: If storage key validation fails.
            FileNotFoundError: If the file does not exist.
        """
        file_path = self._validate_key(storage_key)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found for storage key: {storage_key}")
        return file_path

    def delete(self, storage_key: str) -> bool:
        """Delete a file from local storage.
        
        Args:
            storage_key: The storage key of the file to delete.
            
        Returns:
            bool: True if the file existed and was deleted, False if it did not exist.
            
        Raises:
            ValueError: If storage key validation fails.
        """
        file_path = self._validate_key(storage_key)
        if file_path.is_file():
            file_path.unlink()
            return True
        return False

    def exists(self, storage_key: str) -> bool:
        """Check if a file exists in local storage.
        
        Args:
            storage_key: The storage key to check.
            
        Returns:
            bool: True if the file exists, False otherwise.
            
        Raises:
            ValueError: If storage key validation fails.
        """
        file_path = self._validate_key(storage_key)
        return file_path.is_file()


# Default singleton instance
storage_service = StorageService()
