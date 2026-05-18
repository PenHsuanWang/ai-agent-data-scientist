"""Tests for app.domain.state_models — Pydantic data contracts.

TC-DOM-01: Session initialisation auto-populates UTC fields.
TC-DOM-02: Invalid role raises ValidationError.
TC-DOM-03: Full serialisation round-trip preserves complex state.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.domain.state_models import AgentMessage, AgentSessionState


# ── TC-DOM-01: Session initialisation ──────────────────────────────── #


class TestAgentSessionStateInit:
    def test_fresh_session_has_empty_messages(self):
        state = AgentSessionState(session_id="s1")
        assert state.messages == []

    def test_fresh_session_has_zero_tokens(self):
        state = AgentSessionState(session_id="s1")
        assert state.total_tokens_used == 0

    def test_created_at_is_auto_populated_utc(self):
        from datetime import timezone

        state = AgentSessionState(session_id="s1")
        assert state.created_at.tzinfo is not None
        assert state.created_at.tzinfo == timezone.utc

    def test_last_accessed_is_auto_populated_utc(self):
        from datetime import timezone

        state = AgentSessionState(session_id="s1")
        assert state.last_accessed.tzinfo is not None
        assert state.last_accessed.tzinfo == timezone.utc

    def test_session_id_is_preserved(self):
        state = AgentSessionState(session_id="my-session")
        assert state.session_id == "my-session"


# ── TC-DOM-02: Strict message validation ───────────────────────────── #


class TestAgentMessageValidation:
    def test_valid_user_message_string_content(self):
        msg = AgentMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_valid_assistant_message_list_content(self):
        blocks = [{"type": "text", "text": "hi"}]
        msg = AgentMessage(role="assistant", content=blocks)
        assert msg.content == blocks

    def test_valid_system_message(self):
        msg = AgentMessage(role="system", content="You are an agent.")
        assert msg.role == "system"

    def test_invalid_role_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentMessage(role="moderator", content="oops")  # type: ignore[arg-type]

    def test_empty_role_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentMessage(role="", content="text")  # type: ignore[arg-type]

    def test_none_role_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentMessage(role=None, content="text")  # type: ignore[arg-type]


# ── TC-DOM-03: Dehydration / Hydration round-trip ─────────────────── #


class TestAgentSessionStateRoundTrip:
    def test_empty_session_round_trip(self):
        original = AgentSessionState(session_id="rt-empty")
        json_str = original.model_dump_json()
        restored = AgentSessionState.model_validate_json(json_str)
        assert restored.session_id == original.session_id
        assert restored.messages == original.messages
        assert restored.total_tokens_used == 0

    def test_text_message_round_trip(self):
        state = AgentSessionState(session_id="rt-text")
        state.add_user_message("What is thermal efficiency?")
        state.add_assistant_message("It is the ratio of useful output to input energy.")
        json_str = state.model_dump_json()
        restored = AgentSessionState.model_validate_json(json_str)
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "What is thermal efficiency?"
        assert restored.messages[1].role == "assistant"

    def test_tool_use_block_round_trip(self):
        state = AgentSessionState(session_id="rt-tool")
        state.add_user_message("Run analysis")
        tool_use_block = [
            {
                "type": "tool_use",
                "id": "tu_123",
                "name": "execute_python_code",
                "input": {"code": "print('hello')"},
            }
        ]
        state.add_assistant_message(tool_use_block)
        state.add_tool_results(
            [{"type": "tool_result", "tool_use_id": "tu_123", "content": "hello"}]
        )
        json_str = state.model_dump_json()
        restored = AgentSessionState.model_validate_json(json_str)
        assert len(restored.messages) == 3
        assistant_msg = restored.messages[1]
        assert isinstance(assistant_msg.content, list)
        assert assistant_msg.content[0]["type"] == "tool_use"
        assert assistant_msg.content[0]["id"] == "tu_123"

    def test_total_tokens_round_trip(self):
        state = AgentSessionState(session_id="rt-tokens", total_tokens_used=4200)
        restored = AgentSessionState.model_validate_json(state.model_dump_json())
        assert restored.total_tokens_used == 4200

    def test_complex_mixed_content_round_trip(self):
        """State with text + tool_use + tool_result blocks survives round-trip."""
        state = AgentSessionState(session_id="rt-complex")
        state.add_user_message([{"type": "text", "text": "Analyse data"}])
        state.add_assistant_message(
            [
                {"type": "text", "text": "Let me check the dataset first."},
                {"type": "tool_use", "id": "tu_999", "name": "list_datasets", "input": {}},
            ]
        )
        state.add_tool_results(
            [{"type": "tool_result", "tool_use_id": "tu_999", "content": "[]"}]
        )
        state.add_assistant_message("No datasets available.")

        json_str = state.model_dump_json()
        restored = AgentSessionState.model_validate_json(json_str)

        assert len(restored.messages) == 4
        # First assistant turn has two blocks
        assert len(restored.messages[1].content) == 2  # type: ignore[arg-type]
        # Tool result user turn
        result_turn = restored.messages[2]
        assert result_turn.role == "user"
        assert isinstance(result_turn.content, list)
        assert result_turn.content[0]["type"] == "tool_result"
