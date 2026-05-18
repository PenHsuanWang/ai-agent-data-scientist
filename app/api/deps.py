"""FastAPI dependency providers for the agent API layer.

This module centralises all ``Depends()``-injectable factory functions so
that routers stay thin and dependencies are easily overridden in tests.

Available dependencies
----------------------
get_redis_client()
    Yields an async Redis client.  The connection is closed after the
    request completes.  Returns ``None`` when ``REDIS_URL`` is not set,
    which allows the application to run without Redis in development.

get_memory_manager()
    Yields a ``RedisMemoryManager`` when Redis is configured, otherwise
    returns ``None``.  Callers that require Redis should check for ``None``
    and fall back to the in-memory store.

Example — inject into a router::

    from fastapi import Depends
    from app.api.deps import get_memory_manager
    from app.services.memory import RedisMemoryManager

    @router.post("/chat")
    async def chat(mgr: RedisMemoryManager | None = Depends(get_memory_manager)):
        ...
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from app.core.config import settings

logger = logging.getLogger(__name__)


async def get_redis_client():  # type: ignore[return]
    """Yield an async Redis client if ``REDIS_URL`` is configured.

    Yields ``None`` when Redis is not configured so that routes degrade
    gracefully to the in-memory store.
    """
    redis_url = getattr(settings, "redis_url", None)
    if not redis_url:
        yield None
        return

    try:
        from redis.asyncio import Redis

        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            yield client
        finally:
            await client.aclose()
    except Exception as exc:
        logger.warning("Could not connect to Redis (%s): %s", redis_url, exc)
        yield None


async def get_memory_manager(
    redis_client=None,  # populated by Depends(get_redis_client) in routes
):
    """Return a ``RedisMemoryManager`` when Redis is available, else ``None``.

    Usage in a router::

        from fastapi import Depends
        from app.api.deps import get_memory_manager, get_redis_client

        @router.post("/chat")
        async def chat(
            mgr = Depends(get_memory_manager),
            redis = Depends(get_redis_client),
        ):
            ...

    The simpler pattern is to call ``get_memory_manager`` with an explicit
    ``redis_client`` argument (useful in tests)::

        mgr = await anext(get_memory_manager(redis_client=fake_redis))
    """
    from app.services.memory import RedisMemoryManager

    if redis_client is None:
        return None
    return RedisMemoryManager(redis_client)
