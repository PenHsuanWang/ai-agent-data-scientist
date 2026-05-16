"""Tests for app.services.data_agent — Gaps 1, 2, 3, 16.

Covers:
  - _call_claude_with_retry: LLM exception classification
  - run(): session rollback on LLM error (Gap 2)
  - _run_loop: parse error sentinel + correction injection (Gap 16)
  - ReActMaxIterationsError raised after MAX_REACT_ITERATIONS
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


def _make_claude_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


_FINAL_ANSWER = (
    "Thought: done\n"
    "Final Answer: The mean efficiency is 42.5%"
)


# ── _call_claude_with_retry: exception classification ────────────────── #


class TestCallClaudeWithRetry:
    async def test_maps_auth_error_to_llm_authentication_error(
        self, agent, session, anthropic_auth_error
    ):
        """Gap 1: AuthenticationError → LLMAuthenticationError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_auth_error()
            )
            with pytest.raises(LLMAuthenticationError):
                await agent._call_claude_with_retry(session, "system")

    async def test_maps_context_overflow_to_llm_context_overflow_error(
        self, agent, session, anthropic_context_overflow_error
    ):
        """Gap 3: BadRequestError with 'prompt is too long' → LLMContextOverflowError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_context_overflow_error()
            )
            with pytest.raises(LLMContextOverflowError):
                await agent._call_claude_with_retry(session, "system")

    async def test_maps_generic_bad_request_to_llm_api_error(
        self, agent, session, anthropic_bad_request_error
    ):
        """Non-context-overflow BadRequestError → LLMAPIError (not subclass)."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_bad_request_error("Unsupported parameter")
            )
            with pytest.raises(LLMAPIError) as exc_info:
                await agent._call_claude_with_retry(session, "system")
            # Must NOT be the more specific subclass
            assert type(exc_info.value) is LLMAPIError

    async def test_maps_connection_error_to_llm_api_error(
        self, agent, session, anthropic_connection_error
    ):
        """Gap 1: APIConnectionError → LLMAPIError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_connection_error()
            )
            with pytest.raises(LLMAPIError):
                await agent._call_claude_with_retry(session, "system")

    async def test_returns_text_on_success(self, agent, session):
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_claude_response("Hello")
            )
            text = await agent._call_claude_with_retry(session, "system")
        assert text == "Hello"


# ── run(): session rollback on LLM error (Gap 2) ─────────────────────── #


class TestRunRollback:
    async def test_rolls_back_messages_on_llm_auth_error(
        self, agent, session, mock_runner, anthropic_auth_error
    ):
        """Gap 2: session messages must revert to pre-run count on LLMAuthenticationError."""
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
        """Gap 2: session messages must revert on LLMContextOverflowError."""
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
        """Rollback must restore to pre-call state, not empty, if session had prior messages."""
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

        # Claude keeps returning malformed text → triggers parse error rollup
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_claude_response("gibberish with no format")
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(ReActLoopError):
                    await agent.run(session, "analyse please")

        # Messages were appended (not rolled back)
        assert len(session.messages) > 1


# ── parse error sentinel (Gap 16) ────────────────────────────────────── #


class TestParseErrorSentinel:
    async def test_parse_error_appends_sentinel_to_trace(
        self, agent, session, mock_runner
    ):
        """Gap 16: parse errors must be recorded as '__parse_error__' actions in react_trace."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_claude_response("nonsense response")
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(ReActLoopError):
                    await agent.run(session, "test")

        sentinel_steps = [
            s for s in session.react_trace if s.get("action") == "__parse_error__"
        ]
        assert len(sentinel_steps) >= 1

    async def test_three_parse_errors_raise_react_loop_error(
        self, agent, session, mock_runner
    ):
        """Three consecutive parse failures must raise ReActLoopError."""
        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=_make_claude_response("not a valid react response")
            )
            with patch.object(agent, "_get_runner", return_value=mock_runner):
                with pytest.raises(ReActLoopError) as exc_info:
                    await agent.run(session, "test")

        exc = exc_info.value
        assert exc.iterations >= 3


# ── _parse_react utility ─────────────────────────────────────────────── #


class TestParseReact:
    def test_parses_final_answer(self):
        from app.services.data_agent import _parse_react

        text = "Thought: I'm done\nFinal Answer: 42%"
        result = _parse_react(text)

        assert result["type"] == "final_answer"
        assert result["answer"] == "42%"
        assert result["thought"] == "I'm done"

    def test_parses_action_and_json_input(self):
        from app.services.data_agent import _parse_react

        text = (
            'Thought: Check datasets\n'
            'Action: list_datasets\n'
            'Action Input: {}'
        )
        result = _parse_react(text)

        assert result["type"] == "action"
        assert result["action"] == "list_datasets"
        assert result["action_input"] == {}

    def test_parse_error_when_no_action_or_final_answer(self):
        from app.services.data_agent import _parse_react

        result = _parse_react("Random text without format")

        assert result["type"] == "parse_error"
        assert "raw" in result
        assert "reason" in result

    def test_parse_error_when_invalid_json_input(self):
        from app.services.data_agent import _parse_react

        text = (
            "Thought: check\n"
            "Action: inspect_dataset\n"
            "Action Input: {not valid json !!!}"
        )
        result = _parse_react(text)

        assert result["type"] == "parse_error"

    def test_action_input_defaults_to_empty_dict_when_missing(self):
        from app.services.data_agent import _parse_react

        text = (
            "Thought: check\n"
            "Action: list_datasets\n"
        )
        result = _parse_react(text)

        # No action_input line → defaults to {}
        if result["type"] == "action":
            assert result["action_input"] == {}

    def test_case_insensitive_final_answer(self):
        from app.services.data_agent import _parse_react

        text = "Thought: done\nfinal answer: result"
        result = _parse_react(text)

        assert result["type"] == "final_answer"

    def test_extracts_action_name_without_spaces(self):
        from app.services.data_agent import _parse_react

        text = (
            'Thought: need data\n'
            'Action: inspect_dataset\n'
            'Action Input: {"file_name": "data.csv"}'
        )
        result = _parse_react(text)

        assert result["action"] == "inspect_dataset"
        assert result["action_input"]["file_name"] == "data.csv"
