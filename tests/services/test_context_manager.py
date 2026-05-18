"""Tests for app.services.context_manager — sliding window + cache injection.

TC-CTX-01: Sliding window truncates to max_history.
TC-CTX-02: Ephemeral cache_control injected on most recent user messages.
TC-CTX-03: Non-string content (list of blocks) handled correctly.
"""
from __future__ import annotations

import pytest

from app.domain.state_models import AgentMessage, AgentSessionState
from app.services.context_manager import optimize_context_window


def _make_messages(n: int) -> list[AgentMessage]:
    """Alternate user/assistant messages, n total."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(AgentMessage(role=role, content=f"message {i}"))
    return msgs


# ── TC-CTX-01: Sliding window ─────────────────────────────────────── #


class TestSlidingWindow:
    def test_truncates_to_max_history(self):
        msgs = _make_messages(50)
        result = optimize_context_window(msgs, max_history=20)
        assert len(result) == 20

    def test_no_truncation_when_under_limit(self):
        msgs = _make_messages(10)
        result = optimize_context_window(msgs, max_history=20)
        assert len(result) == 10

    def test_no_truncation_at_exact_limit(self):
        msgs = _make_messages(20)
        result = optimize_context_window(msgs, max_history=20)
        assert len(result) == 20

    def test_returns_most_recent_messages(self):
        msgs = _make_messages(10)
        result = optimize_context_window(msgs, max_history=4)
        # Last 4 messages have indices 6,7,8,9
        # User messages (even indices) get their content wrapped by cache injection;
        # extract the text from the wrapped block or plain string.
        def _text(m):
            c = m["content"]
            if isinstance(c, list):
                return c[0]["text"]
            return c

        assert _text(result[0]) == "message 6"
        assert _text(result[-1]) == "message 9"

    def test_result_starts_with_user_role(self):
        """After trimming, first message must be a user turn."""
        # Force a scenario where cutting would leave an assistant message first
        msgs = [
            AgentMessage(role="user", content="hello"),
            AgentMessage(role="assistant", content="hi"),
            AgentMessage(role="user", content="follow-up"),
        ]
        # max_history=2 keeps last 2: [assistant, user] → advance past assistant
        result = optimize_context_window(msgs, max_history=2)
        assert result[0]["role"] == "user"

    def test_empty_messages_returns_empty(self):
        result = optimize_context_window([], max_history=10)
        assert result == []

    def test_single_user_message_preserved(self):
        msgs = [AgentMessage(role="user", content="only")]
        result = optimize_context_window(msgs, max_history=5)
        assert len(result) == 1
        # Content is wrapped for cache injection — text should still be "only"
        content = result[0]["content"]
        if isinstance(content, list):
            assert content[0]["text"] == "only"
        else:
            assert content == "only"


# ── TC-CTX-02: Ephemeral cache injection ──────────────────────────── #


class TestEphemeralCacheInjection:
    def test_most_recent_user_message_gets_cache_control(self):
        msgs = [
            AgentMessage(role="user", content="first"),
            AgentMessage(role="assistant", content="reply"),
            AgentMessage(role="user", content="second"),
        ]
        result = optimize_context_window(msgs, max_history=10)
        last_user = next(m for m in reversed(result) if m["role"] == "user")
        content = last_user["content"]
        assert isinstance(content, list)
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_string_content_wrapped_to_list(self):
        msgs = [AgentMessage(role="user", content="plain text")]
        result = optimize_context_window(msgs, max_history=10)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "plain text"

    def test_up_to_two_user_messages_tagged(self):
        msgs = [
            AgentMessage(role="user", content="msg-a"),
            AgentMessage(role="assistant", content="rep-a"),
            AgentMessage(role="user", content="msg-b"),
            AgentMessage(role="assistant", content="rep-b"),
            AgentMessage(role="user", content="msg-c"),
        ]
        result = optimize_context_window(msgs, max_history=10)
        tagged = [
            m for m in result
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and m["content"][0].get("cache_control") == {"type": "ephemeral"}
        ]
        assert len(tagged) == 2

    def test_at_most_two_user_messages_tagged(self):
        """Even with 5 user messages, only 2 are tagged."""
        msgs = []
        for i in range(5):
            msgs.append(AgentMessage(role="user", content=f"user-{i}"))
            msgs.append(AgentMessage(role="assistant", content=f"asst-{i}"))
        result = optimize_context_window(msgs, max_history=20)
        tagged = [
            m for m in result
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(b.get("cache_control") for b in m["content"] if isinstance(b, dict))
        ]
        assert len(tagged) == 2

    def test_assistant_messages_not_tagged(self):
        msgs = [
            AgentMessage(role="user", content="q"),
            AgentMessage(role="assistant", content="a"),
        ]
        result = optimize_context_window(msgs, max_history=10)
        for m in result:
            if m["role"] == "assistant":
                content = m["content"]
                if isinstance(content, list):
                    for block in content:
                        assert "cache_control" not in block


# ── TC-CTX-03: Non-string (list) content handled correctly ────────── #


class TestNonStringContentHandling:
    def test_list_content_injects_into_last_text_block(self):
        blocks = [
            {"type": "text", "text": "some query"},
            {"type": "image", "source": {"type": "base64", "data": "..."}},
        ]
        msgs = [AgentMessage(role="user", content=blocks)]
        result = optimize_context_window(msgs, max_history=10)
        content = result[0]["content"]
        assert isinstance(content, list)
        # Last text block (index 0) should have cache_control
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert text_blocks[-1]["cache_control"] == {"type": "ephemeral"}

    def test_list_content_structure_is_not_broken(self):
        """Non-text blocks survive unchanged."""
        blocks = [
            {"type": "text", "text": "analyse this"},
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"},
        ]
        msgs = [AgentMessage(role="user", content=blocks)]
        result = optimize_context_window(msgs, max_history=10)
        content = result[0]["content"]
        tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["tool_use_id"] == "tu_1"

    def test_no_text_block_in_list_skips_injection(self):
        """If a user message list has no text block, it is left unchanged."""
        blocks = [{"type": "tool_result", "tool_use_id": "tu_x", "content": "data"}]
        msgs = [AgentMessage(role="user", content=blocks)]
        result = optimize_context_window(msgs, max_history=10)
        content = result[0]["content"]
        # cache_control should NOT appear since there's no text block to inject into
        for block in content:
            assert "cache_control" not in block

    def test_original_messages_not_mutated(self):
        """optimize_context_window must never mutate the input list."""
        original_content = [{"type": "text", "text": "hello"}]
        msgs = [AgentMessage(role="user", content=original_content)]
        optimize_context_window(msgs, max_history=10)
        # The original block should not have cache_control added
        assert "cache_control" not in original_content[0]

    def test_returns_dicts_not_agent_message_objects(self):
        msgs = [AgentMessage(role="user", content="hi")]
        result = optimize_context_window(msgs, max_history=10)
        assert isinstance(result[0], dict)
        assert "role" in result[0]
        assert "content" in result[0]
