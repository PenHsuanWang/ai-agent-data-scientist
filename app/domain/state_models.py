"""Pydantic-based state contracts for the stateless agent memory layer.

These models are the canonical serialisation format for session state
stored in Redis (or any other external store).  They sit in the domain
layer and therefore import **only** from the Python standard library and
Pydantic — no infrastructure or service imports allowed.

Classes
-------
AgentMessage
    A single turn in the conversation (user, assistant, or system).
    ``content`` supports both plain strings and the structured
    multi-modal / tool-calling block lists used by the Anthropic API.

AgentSessionState
    The full, serialisable state of one agent session.
    Designed for hydration/dehydration via
    ``model_validate_json`` / ``model_dump_json``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """One turn in the agent conversation.

    Args:
        role: Speaker role.  Must be ``"user"``, ``"assistant"``, or
              ``"system"``.
        content: Plain text **or** a list of Anthropic content blocks
                 (``{"type": "text", ...}``, ``{"type": "tool_use", ...}``,
                 ``{"type": "tool_result", ...}``, etc.).
    """

    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]]


class AgentSessionState(BaseModel):
    """Serialisable state for one agent session.

    All timestamps are UTC.  ``messages`` grows with every user/assistant
    turn and is trimmed by the context manager before API calls.

    Args:
        session_id: Unique session identifier.
        created_at: UTC timestamp when the session was first created.
            Auto-populated on construction.
        last_accessed: UTC timestamp of the most recent save.
            Updated by the memory manager on every ``save_session`` call.
        messages: Ordered list of conversation turns.
        total_tokens_used: Cumulative token count (updated externally by
            the agent loop after each API response).
    """

    session_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    messages: list[AgentMessage] = Field(default_factory=list)
    total_tokens_used: int = 0

    def add_user_message(self, content: str | list[dict[str, Any]]) -> None:
        """Append a user turn."""
        self.messages.append(AgentMessage(role="user", content=content))

    def add_assistant_message(self, content: str | list[dict[str, Any]]) -> None:
        """Append an assistant turn."""
        self.messages.append(AgentMessage(role="assistant", content=content))

    def add_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        """Append a user turn containing tool_result blocks."""
        self.messages.append(AgentMessage(role="user", content=tool_results))
