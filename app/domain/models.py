"""Core domain entities — original AgentSession from the MVP.

This module has ZERO external dependencies (Clean Architecture — Domain Layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSession:
    """A single conversation thread.

    ``messages`` follows the Anthropic API format:
      [{"role": "user" | "assistant", "content": str | list[ContentBlock]}, ...]
    """

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": tool_results})
