"""Tests for DataScienceAgentService — native tool-calling loop.

TC-AGT-01: Successful tool dispatch — handler called with correct kwargs,
           tool_result message appended to session.
TC-AGT-02: Error refinement loop — tool raises Exception → captured as
           tool_result with is_error=False (error embedded in content string).
TC-AGT-03: Parallel tool calling — two tool_use blocks in one response →
           both handlers executed, single user message with two tool_result
           blocks appended.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import Message, Usage

from app.domain.analysis_models import AnalysisSession
from app.services.data_agent import DataScienceAgentService


# ── Mock builders ─────────────────────────────────────────────────── #


def _tool_block(tool_id: str, name: str, input_: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_
    return block


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_response(tool_blocks: list, text_prefix: str = "") -> MagicMock:
    """Claude response asking to use one or more tools."""
    response = MagicMock(spec=Message)
    response.stop_reason = "tool_use"
    response.content = (
        [_text_block(text_prefix)] + tool_blocks if text_prefix else tool_blocks
    )
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


def _make_end_turn_response(text: str = "Final answer.") -> MagicMock:
    response = MagicMock(spec=Message)
    response.stop_reason = "end_turn"
    response.content = [_text_block(text)]
    response.usage = MagicMock()
    response.usage.input_tokens = 80
    response.usage.output_tokens = 20
    return response


# ── Fixtures ──────────────────────────────────────────────────────── #


@pytest.fixture
def session():
    return AnalysisSession(session_id="tc-agt-test")


@pytest.fixture
def agent():
    return DataScienceAgentService()


# ── TC-AGT-01: Successful tool dispatch ───────────────────────────── #


class TestSuccessfulToolDispatch:
    async def test_handler_called_with_correct_input(self, agent, session):
        tool_call_input = {"file_name": "power_plant.csv"}
        tool_response = _make_tool_response(
            [_tool_block("tu_001", "inspect_dataset", tool_call_input)]
        )
        end_response = _make_end_turn_response("Dataset has 200 rows.")

        handler = MagicMock(return_value='{"rows": 200}')

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={"inspect_dataset": handler},
        ):
            result = await agent.run(session, "Inspect the dataset")

        handler.assert_called_once_with(tool_call_input)

    async def test_tool_result_message_appended_to_session(self, agent, session):
        tool_response = _make_tool_response(
            [_tool_block("tu_002", "list_datasets", {})]
        )
        end_response = _make_end_turn_response("Here are the datasets.")

        handler = MagicMock(return_value='["ds1.csv", "ds2.csv"]')

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={"list_datasets": handler},
        ):
            await agent.run(session, "List datasets")

        # Find the tool_result user message
        tool_result_msgs = [
            m for m in session.messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"][0]["tool_use_id"] == "tu_002"

    async def test_final_answer_returned(self, agent, session):
        tool_response = _make_tool_response(
            [_tool_block("tu_003", "list_datasets", {})]
        )
        end_response = _make_end_turn_response("The answer is 42.")

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={"list_datasets": MagicMock(return_value="[]")},
        ):
            result = await agent.run(session, "What is the answer?")

        assert result == "The answer is 42."


# ── TC-AGT-02: Error refinement loop (self-correction) ────────────── #


class TestErrorRefinementLoop:
    async def test_tool_exception_captured_not_raised(self, agent, session):
        """A crashing tool must NOT propagate — error becomes tool_result content."""
        tool_response = _make_tool_response(
            [_tool_block("tu_err", "execute_python_code", {"code": "1/0"})]
        )
        end_response = _make_end_turn_response("I encountered an error.")

        def _crashing_handler(_inp):
            raise RuntimeError("division by zero")

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={"execute_python_code": _crashing_handler},
        ):
            # Must complete without raising
            result = await agent.run(session, "Run bad code")

        assert "I encountered an error" in result

    async def test_error_message_appended_as_tool_result(self, agent, session):
        tool_response = _make_tool_response(
            [_tool_block("tu_fail", "inspect_dataset", {"file_name": "missing.csv"})]
        )
        end_response = _make_end_turn_response("Dataset not found.")

        def _failing(_inp):
            raise FileNotFoundError("missing.csv not found")

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={"inspect_dataset": _failing},
        ):
            await agent.run(session, "Inspect missing dataset")

        # The tool_result content must contain the error text
        tool_result_msgs = [
            m for m in session.messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        obs = tool_result_msgs[0]["content"][0]["content"]
        assert "Error" in obs or "missing.csv" in obs

    async def test_loop_continues_after_tool_error(self, agent, session):
        """After the error tool_result, Claude gets another chance → end_turn."""
        tool_response = _make_tool_response(
            [_tool_block("tu_retry", "list_datasets", {})]
        )
        end_response = _make_end_turn_response("Recovered successfully.")

        call_count = {"n": 0}

        def _intermittent(_inp):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("transient failure")
            return "[]"

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={"list_datasets": _intermittent},
        ):
            result = await agent.run(session, "List datasets with retry")

        assert result == "Recovered successfully."


# ── TC-AGT-03: Parallel tool calling ─────────────────────────────── #


class TestParallelToolCalling:
    async def test_both_handlers_executed(self, agent, session):
        """LLM returns two tool_use blocks → both handlers must be called."""
        tool_response = _make_tool_response(
            [
                _tool_block("tu_A", "list_datasets", {}),
                _tool_block("tu_B", "list_domain_documents", {}),
            ]
        )
        end_response = _make_end_turn_response("Both tools done.")

        handler_a = MagicMock(return_value="[]")
        handler_b = MagicMock(return_value="[]")

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={
                "list_datasets": handler_a,
                "list_domain_documents": handler_b,
            },
        ):
            await agent.run(session, "Use both tools")

        handler_a.assert_called_once()
        handler_b.assert_called_once()

    async def test_single_user_message_with_two_tool_results(self, agent, session):
        """Two tool_use blocks → one user message containing two tool_result blocks."""
        tool_response = _make_tool_response(
            [
                _tool_block("tu_1", "list_datasets", {}),
                _tool_block("tu_2", "list_domain_documents", {}),
            ]
        )
        end_response = _make_end_turn_response("Done.")

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={
                "list_datasets": MagicMock(return_value="[]"),
                "list_domain_documents": MagicMock(return_value="[]"),
            },
        ):
            await agent.run(session, "Parallel calls")

        tool_result_msgs = [
            m for m in session.messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and all(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        assert len(tool_result_msgs[0]["content"]) == 2

    async def test_tool_result_ids_match_tool_use_ids(self, agent, session):
        """Each tool_result must reference the correct tool_use_id."""
        tool_response = _make_tool_response(
            [
                _tool_block("tu_X", "list_datasets", {}),
                _tool_block("tu_Y", "list_domain_documents", {}),
            ]
        )
        end_response = _make_end_turn_response("Matched.")

        with patch(
            "app.services.data_agent.DataScienceAgentService._call_claude_with_retry",
            new=AsyncMock(side_effect=[tool_response, end_response]),
        ), patch(
            "app.services.data_agent.DataScienceAgentService._build_tool_registry",
            return_value={
                "list_datasets": MagicMock(return_value="ds"),
                "list_domain_documents": MagicMock(return_value="docs"),
            },
        ):
            await agent.run(session, "Check IDs")

        tool_result_msgs = [
            m for m in session.messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        ids = {b["tool_use_id"] for b in tool_result_msgs[0]["content"]}
        assert "tu_X" in ids
        assert "tu_Y" in ids
