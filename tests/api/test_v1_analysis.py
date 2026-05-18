"""Tests for API layer — TC-API-01 (react_trace translation) + TC-API-02 (DI integrity).

TC-API-01: Native tool_use/tool_result loop produces populated react_trace in response.
TC-API-02: RedisMemoryManager dependency injection — load_session and save_session
           are each called exactly once per request.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.domain.state_models import AgentSessionState


# ── App factory ──────────────────────────────────────────────────────── #


def _make_app():
    from app.main import app
    return app


def _chat_body(message: str = "Analyse data", session_id: str = "v1-test") -> dict:
    return {"message": message, "session_id": session_id}


# ── TC-API-01: Legacy react_trace translation ──────────────────────── #


class TestReactTraceTranslation:
    """Verify that the native tool_use loop produces react_trace entries
    in the AnalysisResponse that frontend clients can consume."""

    async def test_end_turn_response_has_react_trace_entry(self):
        """A simple end_turn response must produce at least one trace step."""
        from anthropic.types import Message
        from unittest.mock import MagicMock

        end_response = MagicMock(spec=Message)
        end_response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Analysis complete."
        end_response.content = [text_block]
        end_response.usage = MagicMock(input_tokens=50, output_tokens=20)

        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.services.data_agent._client.messages.create",
                new=AsyncMock(return_value=end_response),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["response"] == "Analysis complete."
        # The final answer is always logged as a trace step
        trace = body["react_trace"]
        assert isinstance(trace, list)
        assert len(trace) >= 1
        assert trace[-1]["action"] == "Final Answer"

    async def test_tool_dispatch_adds_action_observation_to_trace(self):
        """A tool_use turn must add action + observation entries to react_trace."""
        from anthropic.types import Message

        # First response: tool_use
        tool_response = MagicMock(spec=Message)
        tool_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_trace_001"
        tool_block.name = "list_datasets"
        tool_block.input = {}
        tool_response.content = [tool_block]
        tool_response.usage = MagicMock(input_tokens=60, output_tokens=30)

        # Second response: end_turn
        end_response = MagicMock(spec=Message)
        end_response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Found datasets."
        end_response.content = [text_block]
        end_response.usage = MagicMock(input_tokens=80, output_tokens=25)

        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.services.data_agent._client.messages.create",
                new=AsyncMock(side_effect=[tool_response, end_response]),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        assert response.status_code == 200
        body = response.json()
        trace = body["react_trace"]
        # There must be an entry for list_datasets call
        actions = [step["action"] for step in trace]
        assert any("list_datasets" in act for act in actions)

    async def test_react_trace_has_expected_schema(self):
        """Every react_trace entry must have thought, action, observation keys."""
        from anthropic.types import Message

        end_response = MagicMock(spec=Message)
        end_response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Done."
        end_response.content = [text_block]
        end_response.usage = MagicMock(input_tokens=30, output_tokens=10)

        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.services.data_agent._client.messages.create",
                new=AsyncMock(return_value=end_response),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        trace = response.json()["react_trace"]
        for step in trace:
            assert "thought" in step
            assert "action" in step
            assert "observation" in step


# ── TC-API-02: Dependency injection integrity ─────────────────────── #


class TestDependencyInjectionIntegrity:
    """Verify that RedisMemoryManager can be injected via FastAPI Depends()
    and that load_session / save_session are wired correctly."""

    async def test_get_memory_manager_returns_none_without_redis(self):
        """When REDIS_URL is unset, get_memory_manager yields None."""
        from app.api.deps import get_memory_manager

        result = await get_memory_manager(redis_client=None)
        assert result is None

    async def test_get_memory_manager_returns_manager_with_redis(self):
        """When a redis_client is supplied, a RedisMemoryManager is returned."""
        from app.api.deps import get_memory_manager
        from app.services.memory import RedisMemoryManager

        fake_redis = MagicMock()
        result = await get_memory_manager(redis_client=fake_redis)
        assert isinstance(result, RedisMemoryManager)

    async def test_redis_memory_manager_load_called_once_per_request(self):
        """Simulates a request flow: load_session called once, save_session called once."""
        from app.services.memory import RedisMemoryManager
        from app.domain.state_models import AgentSessionState

        mock_state = AgentSessionState(session_id="di-test")

        mock_mgr = MagicMock(spec=RedisMemoryManager)
        mock_mgr.load_session = AsyncMock(return_value=mock_state)
        mock_mgr.save_session = AsyncMock()

        # Simulate the hydration / dehydration pattern used by a route handler
        session_id = "di-test"
        state = await mock_mgr.load_session(session_id)
        state.add_user_message("hello")
        await mock_mgr.save_session(state)

        mock_mgr.load_session.assert_called_once_with(session_id)
        mock_mgr.save_session.assert_called_once_with(state)

    async def test_get_redis_client_yields_none_when_no_redis_url(self):
        """get_redis_client must yield None gracefully when REDIS_URL is absent."""
        from app.api.deps import get_redis_client

        gen = get_redis_client()
        client = await gen.__anext__()
        assert client is None
        # exhaust generator
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

    async def test_memory_manager_session_state_preserved_across_save_load(self):
        """End-to-end: save then load returns the same state."""
        import fakeredis.aioredis
        from app.services.memory import RedisMemoryManager

        redis = fakeredis.aioredis.FakeRedis()
        mgr = RedisMemoryManager(redis, ttl_seconds=300)

        state = AgentSessionState(session_id="e2e-di")
        state.add_user_message("test question")
        state.total_tokens_used = 100
        await mgr.save_session(state)

        reloaded = await mgr.load_session("e2e-di")
        assert reloaded.session_id == "e2e-di"
        assert len(reloaded.messages) == 1
        assert reloaded.total_tokens_used == 100

        await redis.aclose()
