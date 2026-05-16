"""Shared pytest fixtures and helpers.

Sets the required ANTHROPIC_API_KEY env var before any imports so
pydantic-settings can build the Settings singleton.
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

# Must be set before app modules are imported so Settings validation passes.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-conftest")


# ── Domain helpers ───────────────────────────────────────────────────── #


@pytest.fixture
def session():
    """Fresh AnalysisSession with a fixed session_id."""
    from app.domain.analysis_models import AnalysisSession
    return AnalysisSession(session_id="test-session-001")


@pytest.fixture
def session_with_messages(session):
    """Session that already has two paired messages (simulates mid-conversation)."""
    session.add_user_message("Previous user question")
    session.add_assistant_message("Previous assistant answer")
    return session


# ── Anthropic exception builders ─────────────────────────────────────── #


def _make_httpx_response(status_code: int, body: dict | None = None) -> httpx.Response:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    content = json.dumps(body or {}).encode()
    return httpx.Response(status_code, request=req, content=content)


def _make_httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def anthropic_auth_error():
    """Factory for ``anthropic.AuthenticationError``."""
    import anthropic

    def _make(msg: str = "Invalid API key") -> anthropic.AuthenticationError:
        return anthropic.AuthenticationError(
            msg,
            response=_make_httpx_response(401),
            body=None,
        )

    return _make


@pytest.fixture
def anthropic_bad_request_error():
    """Factory for ``anthropic.BadRequestError`` with configurable message."""
    import anthropic

    def _make(msg: str = "Bad request") -> anthropic.BadRequestError:
        return anthropic.BadRequestError(
            msg,
            response=_make_httpx_response(400),
            body=None,
        )

    return _make


@pytest.fixture
def anthropic_context_overflow_error(anthropic_bad_request_error):
    """BadRequestError with a 'prompt is too long' message (context overflow)."""

    def _make() -> object:
        return anthropic_bad_request_error("prompt is too long: max 200000 tokens")

    return _make


@pytest.fixture
def anthropic_connection_error():
    """Factory for ``anthropic.APIConnectionError``."""
    import anthropic

    def _make(msg: str = "Connection error") -> anthropic.APIConnectionError:
        return anthropic.APIConnectionError(message=msg, request=_make_httpx_request())

    return _make


@pytest.fixture
def anthropic_rate_limit_error():
    """Factory for ``anthropic.RateLimitError``."""
    import anthropic

    def _make(msg: str = "Rate limit exceeded") -> anthropic.RateLimitError:
        return anthropic.RateLimitError(
            msg,
            response=_make_httpx_response(429),
            body=None,
        )

    return _make


# ── Claude response mock builder ─────────────────────────────────────── #


def make_claude_response(text: str) -> object:
    """Create a minimal mock of the Anthropic messages response object."""
    from unittest.mock import MagicMock

    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response
