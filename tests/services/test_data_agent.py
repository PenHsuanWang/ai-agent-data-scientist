"""Tests for app.services.data_agent.

Covers:
  - _call_claude_with_retry: LLM exception classification
  - run(): session rollback on LLM error
  - _run_loop: native tool-calling loop — end_turn, tool_use, unexpected stop_reason,
               unknown tool, max iterations
  - _apply_sliding_window: message history truncation
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.analysis_models import AnalysisSession
from app.domain.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMContextOverflowError,
    ReActLoopError,
)


# ── Mock builders ─────────────────────────────────────────────────────── #


def _make_end_turn_response(text: str) -> MagicMock:
    """Mock a Claude response with stop_reason='end_turn' and a text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _make_tool_use_response(
    tool_name: str,
    tool_id: str = "toolu_001",
    input_data: dict | None = None,
) -> MagicMock:
    """Mock a Claude response with stop_reason='tool_use'."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = input_data or {}
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def _make_unexpected_response() -> MagicMock:
    """Mock a Claude response with an unrecognised stop_reason."""
    resp = MagicMock()
    resp.stop_reason = "unexpected_stop"
    resp.content = []
    return resp


# ── fixtures ─────────────────────────────────────────────────────────── #


@pytest.fixture
def session():
    return AnalysisSession(session_id="agent-test-001")


@pytest.fixture
def agent():
    from app.services.data_agent import DataScienceAgentService
    return DataScienceAgentService()


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.execute.return_value = MagicMock(
        success=True, stdout="ok", stderr="", figures=[], execution_time_ms=1
    )
    runner.get_state.return_value = {}
    runner.get_figure_b64.return_value = None
    return runner


# ── _call_claude_with_retry: exception classification ────────────────── #


class TestCallClaudeWithRetry:
    async def test_maps_auth_error_to_llm_authentication_error(
        self, agent, session, anthropic_auth_error
    ):
        """AuthenticationError → LLMAuthenticationError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_auth_error()
            )
            with pytest.raises(LLMAuthenticationError):
                await agent._call_claude_with_retry(session)

    async def test_maps_context_overflow_to_llm_context_overflow_error(
        self, agent, session, anthropic_context_overflow_error
    ):
        """BadRequestError with 'prompt is too long' → LLMContextOverflowError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_context_overflow_error()
            )
            with pytest.raises(LLMContextOverflowError):
                await agent._call_claude_with_retry(session)

    async def test_maps_generic_bad_request_to_llm_api_error(
        self, agent, session, anthropic_bad_request_error
    ):
        """Non-context-overflow BadRequestError → LLMAPIError (not subclass)."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_bad_request_error("Unsupported parameter")
            )
            with pytest.raises(LLMAPIError) as exc_info:
                await agent._call_claude_with_retry(session)
            assert type(exc_info.value) is LLMAPIError

    async def test_maps_connection_error_to_llm_api_error(
        self, agent, session, anthropic_connection_error
    ):
        """APIConnectionError → LLMAPIError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_connection_error()
            )
            with pytest.raises(LLMAPIError):
                await agent._call_claude_with_retry(session)

    async def test_returns_response_on_success(self, agent, session):
        """Successful call returns the full Message response object."""
        mock_resp = _make_end_turn_response("Hello")
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_resp)
            response = await agent._call_claude_with_retry(session)
        assert response.stop_reason == "end_turn"


# ── run(): session rollback on LLM error ─────────────────────────────── #


class TestRunRollback:
    async def test_rolls_back_messages_on_llm_auth_error(
        self, agent, session, mock_runner, anthropic_auth_error
    ):
        """Session messages must revert to pre-run count on LLMAuthenticationError."""
        initial_count = len(session.messages)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_auth_error()
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(LLMAuthenticationError):
                    await agent.run(session, "analyse data")

        assert len(session.messages) == initial_count

    async def test_rolls_back_messages_on_context_overflow(
        self, agent, session, mock_runner, anthropic_context_overflow_error
    ):
        """Session messages must revert on LLMContextOverflowError."""
        initial_count = len(session.messages)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_context_overflow_error()
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(LLMContextOverflowError):
                    await agent.run(session, "too long request")

        assert len(session.messages) == initial_count

    async def test_rolls_back_to_checkpoint_not_absolute_zero(
        self, agent, session, mock_runner, anthropic_auth_error
    ):
        """Rollback must restore to pre-call state, not to empty, if session had prior messages."""
        session.add_user_message("prior message")
        session.add_assistant_message("prior reply")
        checkpoint = len(session.messages)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_auth_error()
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(LLMAuthenticationError):
                    await agent.run(session, "new request")

        assert len(session.messages) == checkpoint

    async def test_does_not_roll_back_on_react_loop_error(
        self, agent, session, mock_runner
    ):
        """ReActLoopError must NOT trigger rollback — trace is useful for debugging."""
        session.add_user_message("old msg")

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_unexpected_response()
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(ReActLoopError):
                    await agent.run(session, "analyse please")

        # User message was appended before the error (not rolled back)
        assert len(session.messages) > 1


# ── Native tool-calling loop ──────────────────────────────────────────── #


class TestToolCallingLoop:
    async def test_completes_on_end_turn_response(self, agent, session, mock_runner):
        """Loop terminates and returns text when stop_reason is 'end_turn'."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_end_turn_response("The mean efficiency is 42.5%")
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                result = await agent.run(session, "analyse data")

        assert "42.5%" in result

    async def test_dispatches_tool_then_completes(self, agent, session, mock_runner):
        """Loop dispatches tool_use block, appends result, then returns on end_turn."""
        responses = [
            _make_tool_use_response("list_datasets", "toolu_001", {}),
            _make_end_turn_response("Found 2 datasets."),
        ]
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(side_effect=responses)
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with patch("app.services.data_agent.list_datasets", return_value='["data.csv"]'):
                    result = await agent.run(session, "list datasets")

        assert "Found 2 datasets" in result

    async def test_tool_result_message_appended_to_session(
        self, agent, session, mock_runner
    ):
        """Tool results are stored as user messages with tool_result content blocks."""
        responses = [
            _make_tool_use_response("list_datasets", "toolu_001", {}),
            _make_end_turn_response("Done."),
        ]
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(side_effect=responses)
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with patch("app.services.data_agent.list_datasets", return_value='[]'):
                    await agent.run(session, "test")

        tool_result_msgs = [
            m for m in session.messages
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) >= 1

    async def test_unexpected_stop_reason_raises_react_loop_error(
        self, agent, session, mock_runner
    ):
        """An unrecognised stop_reason raises ReActLoopError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_unexpected_response()
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(ReActLoopError):
                    await agent.run(session, "test")

    async def test_unexpected_stop_reason_recorded_in_trace(
        self, agent, session, mock_runner
    ):
        """Unexpected stop_reason is recorded with __unexpected_stop__ sentinel."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_unexpected_response()
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(ReActLoopError):
                    await agent.run(session, "test")

        sentinel_steps = [
            s for s in session.react_trace
            if s.get("action") == "__unexpected_stop__"
        ]
        assert len(sentinel_steps) >= 1

    async def test_unknown_tool_observation_returned_not_raised(
        self, agent, session, mock_runner
    ):
        """Calling an unknown tool name produces an error observation, not an exception."""
        responses = [
            _make_tool_use_response("nonexistent_tool", "toolu_001", {}),
            _make_end_turn_response("Noted the error."),
        ]
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(side_effect=responses)
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                await agent.run(session, "test")

        error_steps = [
            s for s in session.react_trace
            if "Unknown tool" in s.get("observation", "")
        ]
        assert len(error_steps) >= 1

    async def test_max_iterations_raises_react_loop_error(
        self, agent, session, mock_runner
    ):
        """Exhausting max_react_iterations raises ReActLoopError."""
        with patch("app.services.data_agent._client") as mock_client:
            # Always return tool_use — loop never gets end_turn.
            mock_client.messages.create = AsyncMock(
                return_value=_make_tool_use_response("list_datasets", "toolu_001", {})
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with patch("app.services.data_agent.list_datasets", return_value="[]"):
                    with patch("app.services.data_agent.settings") as mock_cfg:
                        mock_cfg.max_react_iterations = 2
                        mock_cfg.max_context_messages = 100
                        mock_cfg.claude_model = "claude-test"
                        mock_cfg.max_tokens = 1024
                        with pytest.raises(ReActLoopError) as exc_info:
                            await agent.run(session, "test")

        assert exc_info.value.iterations == 2


# ── _apply_sliding_window ─────────────────────────────────────────────── #


class TestSlidingWindow:
    def test_no_truncation_when_under_limit(self):
        from app.services.data_agent import _apply_sliding_window

        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        assert _apply_sliding_window(messages, max_total=10) == messages

    def test_no_truncation_at_exact_limit(self):
        from app.services.data_agent import _apply_sliding_window

        messages = [{"role": "user", "content": str(i)} for i in range(5)]
        assert _apply_sliding_window(messages, max_total=5) == messages

    def test_truncates_to_max_total(self):
        from app.services.data_agent import _apply_sliding_window

        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(10)
        ]
        result = _apply_sliding_window(messages, max_total=4)
        assert len(result) <= 4
        assert result[-1] == messages[-1]

    def test_result_starts_with_user_message(self):
        from app.services.data_agent import _apply_sliding_window

        # After taking the last 4 of 5 messages the first kept entry is
        # assistant-role; the window should advance to the next user turn.
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        result = _apply_sliding_window(messages, max_total=4)
        assert result[0].get("role") == "user"

    def test_keeps_most_recent_messages(self):
        from app.services.data_agent import _apply_sliding_window

        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(8)
        ]
        result = _apply_sliding_window(messages, max_total=4)
        # The last message must be retained.
        assert result[-1] == messages[-1]
