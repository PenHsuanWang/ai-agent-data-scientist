"""In-memory session stores.

Two separate stores:
  session_store         — for MVP AgentSession (unchanged)
  analysis_session_store — for new AnalysisSession (with TTL-based eviction)

Redis-backed store:
  RedisMemoryManager — async, stateless, 24 h TTL (Phase 2 refactor)
"""
from __future__ import annotations

import time
import logging
from typing import Callable

from app.domain.analysis_models import AnalysisSession
from app.domain.models import AgentSession
from app.domain.state_models import AgentSessionState

logger = logging.getLogger(__name__)


class InMemorySessionStore:
    """Key-value store for AgentSession objects (MVP)."""

    def __init__(self) -> None:
        self._store: dict[str, AgentSession] = {}

    def get_or_create(self, session_id: str) -> AgentSession:
        if session_id not in self._store:
            self._store[session_id] = AgentSession(session_id=session_id)
        return self._store[session_id]

    def save(self, session: AgentSession) -> None:
        self._store[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class AnalysisSessionStore:
    """Key-value store for AnalysisSession objects.

    Tracks last-access time for each session so expired sessions can be
    evicted lazily (on every access) or eagerly (via ``get_expired_ids()``
    called by the background GC task in main.py).

    Args:
        ttl_seconds: Seconds of inactivity before a session is expired.
            Defaults to ``settings.session_ttl_seconds``.
        on_evict: Optional callback invoked with each expired session_id
            before it is removed (use to shut down CodeRunners, etc.).
    """

    def __init__(
        self,
        ttl_seconds: int | None = None,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        from app.core.config import settings as _cfg
        self._store: dict[str, AnalysisSession] = {}
        self._last_active: dict[str, float] = {}
        self._ttl = ttl_seconds if ttl_seconds is not None else _cfg.session_ttl_seconds
        self._on_evict = on_evict

    # ── Internal helpers ──────────────────────────────────────────── #

    def _touch(self, session_id: str) -> None:
        self._last_active[session_id] = time.monotonic()

    def _evict_expired(self) -> list[str]:
        """Remove sessions that have exceeded the TTL. Returns evicted IDs."""
        now = time.monotonic()
        expired = [
            sid for sid, last in list(self._last_active.items())
            if now - last > self._ttl
        ]
        for sid in expired:
            if self._on_evict:
                try:
                    self._on_evict(sid)
                except Exception as exc:
                    logger.warning("on_evict callback failed for session %s: %s", sid, exc)
            self._store.pop(sid, None)
            self._last_active.pop(sid, None)
            logger.debug("Session %s evicted (TTL exceeded)", sid)
        return expired

    # ── Public API ────────────────────────────────────────────────── #

    def get_or_create(self, session_id: str) -> AnalysisSession:
        self._evict_expired()
        if session_id not in self._store:
            self._store[session_id] = AnalysisSession(session_id=session_id)
        self._touch(session_id)
        return self._store[session_id]

    def get(self, session_id: str) -> AnalysisSession | None:
        self._evict_expired()
        session = self._store.get(session_id)
        if session is not None:
            self._touch(session_id)
        return session

    def save(self, session: AnalysisSession) -> None:
        self._store[session.session_id] = session
        self._touch(session.session_id)

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._last_active.pop(session_id, None)

    def get_expired_ids(self) -> list[str]:
        """Return IDs of sessions that have exceeded the TTL (without evicting)."""
        now = time.monotonic()
        return [
            sid for sid, last in list(self._last_active.items())
            if now - last > self._ttl
        ]

    @property
    def active_sessions(self) -> int:
        return len(self._store)


session_store = InMemorySessionStore()
analysis_session_store = AnalysisSessionStore()


# ──────────────────────────────────────────────────────────────────── #
# Redis-backed memory manager (Phase 2 — stateless architecture)       #
# ──────────────────────────────────────────────────────────────────── #


class RedisMemoryManager:
    """Async session store backed by Redis.

    Implements the hydration / dehydration pattern:

    * **Hydration** (``load_session``) — deserialise state from Redis.
      Returns a fresh ``AgentSessionState`` when the key does not exist.
    * **Dehydration** (``save_session``) — serialise state to Redis with a
      rolling 24-hour TTL so idle sessions expire automatically.

    Args:
        redis_client: An async Redis client (``redis.asyncio.Redis``).
        ttl_seconds: Session TTL in seconds.  Defaults to 86 400 (24 h).

    Example::

        from redis.asyncio import Redis
        from app.services.memory import RedisMemoryManager

        redis = Redis.from_url("redis://localhost:6379")
        mgr = RedisMemoryManager(redis)

        state = await mgr.load_session("session-abc")
        state.add_user_message("hello")
        await mgr.save_session(state)
    """

    _KEY_PREFIX = "agent:session:"

    def __init__(self, redis_client: object, ttl_seconds: int = 86_400) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self._KEY_PREFIX}{session_id}"

    async def load_session(self, session_id: str) -> AgentSessionState:
        """Retrieve and deserialise a session from Redis.

        Returns a fresh ``AgentSessionState`` (with auto-populated timestamps)
        when the key is absent or the TTL has expired.
        """
        from datetime import datetime, timezone

        data: bytes | None = await self._redis.get(self._key(session_id))
        if data:
            return AgentSessionState.model_validate_json(data)
        return AgentSessionState(session_id=session_id)

    async def save_session(self, state: AgentSessionState) -> None:
        """Serialise and persist the session with a rolling TTL.

        Updates ``state.last_accessed`` to the current UTC time before
        writing so callers always see an accurate timestamp on reload.
        """
        from datetime import datetime, timezone

        state.last_accessed = datetime.now(timezone.utc)
        await self._redis.setex(
            self._key(state.session_id),
            self._ttl,
            state.model_dump_json(),
        )
        logger.debug(
            "Saved session '%s' to Redis (TTL=%ds)", state.session_id, self._ttl
        )

