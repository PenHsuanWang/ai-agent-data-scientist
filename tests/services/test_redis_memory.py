"""Tests for app.services.memory.RedisMemoryManager — Redis-backed session store.

TC-MEM-01: load_session on a missing key returns a fresh AgentSessionState.
TC-MEM-02: load_session on an existing key returns the full stored state.
TC-MEM-03: save_session updates last_accessed and persists with 86400s TTL.

Uses fakeredis for a hermetic in-process Redis mock.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
import fakeredis.aioredis

from app.domain.state_models import AgentMessage, AgentSessionState
from app.services.memory import RedisMemoryManager


@pytest_asyncio.fixture
async def redis_client():
    """In-process fake Redis client (no network required)."""
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def mgr(redis_client):
    return RedisMemoryManager(redis_client, ttl_seconds=86_400)


# ── TC-MEM-01: Load non-existent session ──────────────────────────── #


class TestLoadNonExistentSession:
    async def test_returns_agent_session_state(self, mgr):
        state = await mgr.load_session("ghost-session")
        assert isinstance(state, AgentSessionState)

    async def test_fresh_state_has_correct_session_id(self, mgr):
        state = await mgr.load_session("new-123")
        assert state.session_id == "new-123"

    async def test_fresh_state_has_empty_messages(self, mgr):
        state = await mgr.load_session("new-empty")
        assert state.messages == []

    async def test_fresh_state_has_zero_tokens(self, mgr):
        state = await mgr.load_session("new-tokens")
        assert state.total_tokens_used == 0

    async def test_does_not_raise(self, mgr):
        # Must never raise even for completely random IDs
        state = await mgr.load_session("totally-unknown-xyz")
        assert state is not None


# ── TC-MEM-02: Load existing session ──────────────────────────────── #


class TestLoadExistingSession:
    async def test_returns_populated_state(self, redis_client, mgr):
        state = AgentSessionState(session_id="test-123")
        state.add_user_message("Hello, agent")
        state.add_assistant_message("Hello, user")
        state.total_tokens_used = 500

        # Pre-seed Redis
        await redis_client.set(
            f"agent:session:test-123", state.model_dump_json()
        )

        loaded = await mgr.load_session("test-123")
        assert loaded.session_id == "test-123"
        assert len(loaded.messages) == 2
        assert loaded.total_tokens_used == 500

    async def test_messages_content_preserved(self, redis_client, mgr):
        state = AgentSessionState(session_id="content-check")
        state.add_user_message("What is efficiency?")
        await redis_client.set(
            "agent:session:content-check", state.model_dump_json()
        )
        loaded = await mgr.load_session("content-check")
        assert loaded.messages[0].content == "What is efficiency?"

    async def test_tool_use_blocks_preserved(self, redis_client, mgr):
        state = AgentSessionState(session_id="tool-blocks")
        state.add_assistant_message(
            [{"type": "tool_use", "id": "tu_1", "name": "list_datasets", "input": {}}]
        )
        await redis_client.set(
            "agent:session:tool-blocks", state.model_dump_json()
        )
        loaded = await mgr.load_session("tool-blocks")
        content = loaded.messages[0].content
        assert isinstance(content, list)
        assert content[0]["type"] == "tool_use"
        assert content[0]["id"] == "tu_1"


# ── TC-MEM-03: Save session with TTL ──────────────────────────────── #


class TestSaveSessionWithTTL:
    async def test_save_then_load_round_trip(self, mgr):
        state = AgentSessionState(session_id="rt-save")
        state.add_user_message("test message")
        await mgr.save_session(state)
        loaded = await mgr.load_session("rt-save")
        assert loaded.session_id == "rt-save"
        assert len(loaded.messages) == 1

    async def test_last_accessed_is_updated_on_save(self, mgr):
        before = datetime.now(timezone.utc) - timedelta(hours=1)
        state = AgentSessionState(session_id="ts-test", last_accessed=before)
        await mgr.save_session(state)
        # last_accessed should now be close to "now"
        assert state.last_accessed > before
        delta = datetime.now(timezone.utc) - state.last_accessed
        assert delta.total_seconds() < 2

    async def test_persisted_last_accessed_is_current(self, mgr):
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        state = AgentSessionState(session_id="la-persist", last_accessed=old_time)
        await mgr.save_session(state)
        reloaded = await mgr.load_session("la-persist")
        assert reloaded.last_accessed > old_time

    async def test_ttl_is_applied(self, redis_client, mgr):
        state = AgentSessionState(session_id="ttl-check")
        await mgr.save_session(state)
        ttl = await redis_client.ttl("agent:session:ttl-check")
        # TTL should be close to 86400 (allow a small epsilon)
        assert 86_390 <= ttl <= 86_400

    async def test_custom_ttl_is_respected(self, redis_client):
        short_mgr = RedisMemoryManager(redis_client, ttl_seconds=300)
        state = AgentSessionState(session_id="short-ttl")
        await short_mgr.save_session(state)
        ttl = await redis_client.ttl("agent:session:short-ttl")
        assert 290 <= ttl <= 300

    async def test_key_uses_correct_prefix(self, redis_client, mgr):
        state = AgentSessionState(session_id="prefix-test")
        await mgr.save_session(state)
        exists = await redis_client.exists("agent:session:prefix-test")
        assert exists == 1

    async def test_overwrite_updates_existing_key(self, mgr):
        state = AgentSessionState(session_id="overwrite-me")
        state.add_user_message("first")
        await mgr.save_session(state)

        state.add_user_message("second")
        await mgr.save_session(state)

        reloaded = await mgr.load_session("overwrite-me")
        assert len(reloaded.messages) == 2
