"""Tests for app.services.memory — AnalysisSessionStore TTL / eviction (Gap 15)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from app.domain.analysis_models import AnalysisSession
from app.services.memory import AnalysisSessionStore, InMemorySessionStore


# ── get_or_create ────────────────────────────────────────────────────── #


class TestAnalysisSessionStoreGetOrCreate:
    def test_get_or_create_creates_new_session_when_missing(self):
        # Arrange
        store = AnalysisSessionStore()

        # Act
        session = store.get_or_create("new-session")

        # Assert
        assert isinstance(session, AnalysisSession)
        assert session.session_id == "new-session"

    def test_get_or_create_returns_existing_session_on_second_call(self):
        # Arrange
        store = AnalysisSessionStore()
        first = store.get_or_create("s1")
        first.add_user_message("hello")

        # Act
        second = store.get_or_create("s1")

        # Assert — same object, message survives
        assert second is first
        assert len(second.messages) == 1

    def test_get_or_create_increments_active_sessions(self):
        store = AnalysisSessionStore()
        store.get_or_create("a")
        store.get_or_create("b")
        assert store.active_sessions == 2


# ── get ──────────────────────────────────────────────────────────────── #


class TestAnalysisSessionStoreGet:
    def test_get_returns_none_for_unknown_session(self):
        store = AnalysisSessionStore()
        assert store.get("unknown") is None

    def test_get_returns_session_after_save(self):
        store = AnalysisSessionStore()
        s = AnalysisSession(session_id="saved")
        store.save(s)
        assert store.get("saved") is s

    def test_get_touches_last_active_so_session_is_not_immediately_expired(self):
        store = AnalysisSessionStore(ttl_seconds=100)
        store.get_or_create("fresh")
        # Manually backdate the last_active to almost-expiry
        store._last_active["fresh"] = time.monotonic() - 99
        result = store.get("fresh")
        # Should still be alive (99 s < 100 s TTL) AND the touch should have reset it
        assert result is not None
        assert time.monotonic() - store._last_active["fresh"] < 1


# ── save ─────────────────────────────────────────────────────────────── #


class TestAnalysisSessionStoreSave:
    def test_save_updates_existing_entry(self):
        store = AnalysisSessionStore()
        s = AnalysisSession(session_id="upd")
        store.save(s)
        s.add_user_message("updated")
        store.save(s)
        retrieved = store.get("upd")
        assert len(retrieved.messages) == 1

    def test_save_refreshes_last_active_timestamp(self):
        store = AnalysisSessionStore(ttl_seconds=10)
        s = AnalysisSession(session_id="ts-test")
        store.save(s)
        before = store._last_active["ts-test"]
        time.sleep(0.05)
        store.save(s)
        after = store._last_active["ts-test"]
        assert after > before


# ── delete ───────────────────────────────────────────────────────────── #


class TestAnalysisSessionStoreDelete:
    def test_delete_removes_session_from_store(self):
        store = AnalysisSessionStore()
        store.get_or_create("to-delete")
        store.delete("to-delete")
        assert store.get("to-delete") is None

    def test_delete_removes_last_active_tracking(self):
        store = AnalysisSessionStore()
        store.get_or_create("track")
        store.delete("track")
        assert "track" not in store._last_active

    def test_delete_on_nonexistent_session_is_idempotent(self):
        store = AnalysisSessionStore()
        store.delete("ghost")  # should not raise


# ── TTL eviction ─────────────────────────────────────────────────────── #


class TestAnalysisSessionStoreTTL:
    def test_get_expired_ids_returns_nothing_for_fresh_sessions(self):
        store = AnalysisSessionStore(ttl_seconds=60)
        store.get_or_create("alive")
        assert store.get_expired_ids() == []

    def test_get_expired_ids_returns_stale_session_ids(self):
        store = AnalysisSessionStore(ttl_seconds=1)
        store.get_or_create("stale")
        time.sleep(1.1)
        expired = store.get_expired_ids()
        assert "stale" in expired

    def test_get_expired_ids_does_not_yet_evict(self):
        """get_expired_ids is a non-destructive query."""
        store = AnalysisSessionStore(ttl_seconds=1)
        store.get_or_create("pending")
        time.sleep(1.1)
        _ = store.get_expired_ids()
        # Session still in store — lazy eviction only on access
        assert "pending" in store._store

    def test_lazy_eviction_removes_expired_session_on_get_or_create(self):
        store = AnalysisSessionStore(ttl_seconds=1)
        store.get_or_create("expired")
        time.sleep(1.1)
        # Trigger lazy eviction via a new get_or_create
        store.get_or_create("trigger")
        assert store.get("expired") is None

    def test_lazy_eviction_does_not_remove_fresh_sessions(self):
        store = AnalysisSessionStore(ttl_seconds=60)
        store.get_or_create("long-lived")
        store.get_or_create("another")  # triggers _evict_expired
        assert store.get("long-lived") is not None


# ── on_evict callback ────────────────────────────────────────────────── #


class TestAnalysisSessionStoreOnEvict:
    def test_on_evict_callback_is_called_with_expired_session_id(self):
        evicted: list[str] = []
        store = AnalysisSessionStore(ttl_seconds=1, on_evict=evicted.append)
        store.get_or_create("victim")
        time.sleep(1.1)
        store.get_or_create("trigger")
        assert "victim" in evicted

    def test_on_evict_exception_does_not_block_eviction(self):
        def bad_callback(sid: str) -> None:
            raise RuntimeError("eviction hook crashed")

        store = AnalysisSessionStore(ttl_seconds=1, on_evict=bad_callback)
        store.get_or_create("fragile")
        time.sleep(1.1)
        # Should not raise even though the callback raises
        store.get_or_create("trigger")
        # The session is evicted despite the callback error
        assert store.get("fragile") is None


# ── active_sessions property ─────────────────────────────────────────── #


class TestAnalysisSessionStoreActiveCount:
    def test_active_sessions_is_zero_initially(self):
        store = AnalysisSessionStore()
        assert store.active_sessions == 0

    def test_active_sessions_reflects_current_store_size(self):
        store = AnalysisSessionStore()
        store.get_or_create("x")
        store.get_or_create("y")
        store.delete("x")
        assert store.active_sessions == 1


# ── InMemorySessionStore (legacy) ────────────────────────────────────── #


class TestInMemorySessionStore:
    def test_get_or_create_creates_agent_session(self):
        from app.domain.models import AgentSession

        store = InMemorySessionStore()
        s = store.get_or_create("legacy-001")
        assert isinstance(s, AgentSession)
        assert s.session_id == "legacy-001"

    def test_delete_removes_session(self):
        store = InMemorySessionStore()
        store.get_or_create("del-me")
        store.delete("del-me")
        # After delete, get_or_create creates a fresh session
        s2 = store.get_or_create("del-me")
        assert len(s2.messages) == 0
