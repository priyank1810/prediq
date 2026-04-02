from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class TTLCache:
    """In-memory cache with per-key TTL support."""

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            value, expiry = self._store[key]
            if time.time() < expiry:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl: int):
        self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str):
        self._store.pop(key, None)

    def purge_expired(self):
        """Remove all expired entries to free memory."""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now >= exp]
        for k in expired:
            del self._store[k]
        return len(expired)

    def clear(self):
        self._store.clear()


class RedisCache:
    """Redis-backed cache with the same interface as TTLCache.

    Values are serialised as JSON when possible (simple types),
    falling back to pickle for complex objects like DataFrames.
    """

    _JSON_PREFIX = b"json:"
    _PICKLE_PREFIX = b"pickle:"

    def __init__(self, redis_url: str):
        import redis as _redis

        self._client = _redis.from_url(redis_url, decode_responses=False)
        # Verify connectivity
        self._client.ping()
        logger.info("RedisCache connected to %s", redis_url)

    # -- serialisation helpers ------------------------------------------

    @staticmethod
    def _serialize(value: Any) -> bytes:
        """Try JSON first; fall back to pickle for complex types."""
        try:
            payload = json.dumps(value)
            return RedisCache._JSON_PREFIX + payload.encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            return RedisCache._PICKLE_PREFIX + pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize(raw: bytes) -> Any:
        if raw.startswith(RedisCache._JSON_PREFIX):
            return json.loads(raw[len(RedisCache._JSON_PREFIX):])
        if raw.startswith(RedisCache._PICKLE_PREFIX):
            return pickle.loads(raw[len(RedisCache._PICKLE_PREFIX):])
        # Legacy / unknown format — try JSON, then pickle
        try:
            return json.loads(raw)
        except Exception:
            return pickle.loads(raw)

    # -- public interface (mirrors TTLCache) ----------------------------

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return self._deserialize(raw)
        except Exception:
            logger.warning("RedisCache: failed to deserialise key %r", key)
            return None

    def set(self, key: str, value: Any, ttl: int):
        raw = self._serialize(value)
        self._client.setex(key, ttl, raw)

    def delete(self, key: str):
        self._client.delete(key)

    def clear(self):
        self._client.flushdb()


def _build_cache() -> Union[TTLCache, RedisCache]:
    """Select the cache backend based on the REDIS_URL env var.

    Falls back to the in-memory TTLCache if REDIS_URL is unset or if
    the Redis server is unreachable.
    """
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            return RedisCache(redis_url)
        except Exception as exc:
            logger.warning(
                "Could not connect to Redis at %s (%s). Falling back to in-memory cache.",
                redis_url,
                exc,
            )
    return TTLCache()


cache = _build_cache()
