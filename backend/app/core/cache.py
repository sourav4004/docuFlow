"""In-process caching with TTL support.

Provides a simple cache for frequently accessed data.
Architecture allows Redis replacement later.
"""

import time
import threading
from typing import Any, Optional
from functools import wraps

from ..core.config import settings


class CacheEntry:
    """A single cache entry with TTL."""

    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.expires_at = time.monotonic() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class InProcessCache:
    """Simple in-process cache with TTL support."""

    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._cache[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in cache with TTL."""
        if ttl_seconds is None:
            ttl_seconds = settings.cache_ttl_seconds

        with self._lock:
            self._cache[key] = CacheEntry(value, ttl_seconds)

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern.
        
        Args:
            pattern: Simple prefix pattern to match.
            
        Returns:
            Number of keys invalidated.
        """
        count = 0
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
            for key in keys_to_delete:
                del self._cache[key]
                count += 1
        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries.
        
        Returns:
            Number of entries removed.
        """
        count = 0
        with self._lock:
            keys_to_delete = [k for k, v in self._cache.items() if v.is_expired]
            for key in keys_to_delete:
                del self._cache[key]
                count += 1
        return count


# Singleton instance
_cache: Optional[InProcessCache] = None


def get_cache() -> InProcessCache:
    """Get the singleton cache instance."""
    global _cache
    if _cache is None:
        _cache = InProcessCache()
    return _cache


def cached(ttl_seconds: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results.
    
    Args:
        ttl_seconds: Cache TTL in seconds. Uses default if None.
        key_prefix: Prefix for cache key.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not settings.cache_enabled:
                return func(*args, **kwargs)

            # Build cache key from function name and arguments
            cache_key = f"{key_prefix}{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            cache = get_cache()
            result = cache.get(cache_key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds)
            return result
        return wrapper
    return decorator
