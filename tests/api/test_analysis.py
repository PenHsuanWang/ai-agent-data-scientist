"""Tests for app.api.v1.analysis — Gaps 1, 3, 6, 12.

Covers:
  - LLMContextOverflowError → HTTP 400 (Gap 3)
  - LLMAuthenticationError  → HTTP 502 (Gap 1)
  - LLMAPIError             → HTTP 502 (Gap 1)
  - Corrupted base64 figure → HTTP 422 (Gap 6)
  - dataset_hint pre-load silently swallows errors (Gap 12)
"""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import AsyncClient, ASGITransport

from app.domain.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMContextOverflowError,
    ReActLoopError,
)


# ── App factory ──────────────────────────────────────────────────────── #


def _make_app():
    """Create a fresh FastAPI app without triggering lifespan events."""
    from app.main import app
    return app


# ── helpers ──────────────────────────────────────────────────────────── #


def _chat_body(message: str = "hello", session_id: str = "api-test-001") -> dict:
    return {"message": message, "session_id": session_id}


# ── LLM error → HTTP status code mapping ────────────────────────────── #


class TestLLMErrorHTTPMapping:
    async def test_context_overflow_returns_400(self):
        """Gap 3: LLMContextOverflowError must map to HTTP 400 Bad Request."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.api.v1.analysis.data_science_agent.run",
                new=AsyncMock(side_effect=LLMContextOverflowError("prompt too long")),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["error"] == "context_overflow"

    async def test_auth_error_returns_502(self):
        """Gap 1: LLMAuthenticationError must map to HTTP 502 Bad Gateway."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.api.v1.analysis.data_science_agent.run",
                new=AsyncMock(
                    side_effect=LLMAuthenticationError("Invalid API key")
                ),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        assert response.status_code == 502
        body = response.json()
        assert body["detail"]["error"] == "llm_auth_error"

    async def test_generic_llm_api_error_returns_502(self):
        """Gap 1: LLMAPIError must map to HTTP 502 Bad Gateway."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.api.v1.analysis.data_science_agent.run",
                new=AsyncMock(
                    side_effect=LLMAPIError("Anthropic service unavailable")
                ),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        assert response.status_code == 502
        body = response.json()
        assert body["detail"]["error"] == "llm_api_error"

    async def test_react_loop_error_returns_200_with_error_status(self):
        """ReActLoopError must not bubble up as HTTP 5xx — returns 200 with status='error'."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.api.v1.analysis.data_science_agent.run",
                new=AsyncMock(
                    side_effect=ReActLoopError("too many iterations", iterations=20)
                ),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "error"

    async def test_context_overflow_detail_suggests_new_session(self):
        """The 400 error body must instruct the user to start a new session."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.api.v1.analysis.data_science_agent.run",
                new=AsyncMock(side_effect=LLMContextOverflowError("too long")),
            ):
                response = await client.post(
                    "/api/v1/analysis/chat", json=_chat_body()
                )

        assert "new session" in response.json()["detail"]["message"]


# ── Figure retrieval: corrupted base64 → 422 (Gap 6) ────────────────── #


class TestFigureRetrieval:
    async def test_corrupted_figure_returns_422(self):
        """Gap 6: corrupted base64 in session.figures must return HTTP 422."""
        from app.services.memory import analysis_session_store

        session_id = "corrupt-fig-session"
        session = analysis_session_store.get_or_create(session_id)
        # Store deliberately corrupted base64
        session.figures["fig-bad"] = "!!NOT_VALID_BASE64!!"

        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/v1/analysis/{session_id}/figures/fig-bad"
            )

        assert response.status_code == 422

        # Cleanup
        analysis_session_store.delete(session_id)

    async def test_valid_figure_returns_200_with_png_content_type(self):
        """A valid PNG figure must be served with image/png content-type."""
        from app.services.memory import analysis_session_store

        session_id = "valid-fig-session"
        session = analysis_session_store.get_or_create(session_id)
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        session.figures["fig-ok"] = base64.b64encode(png_bytes).decode()

        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/v1/analysis/{session_id}/figures/fig-ok"
            )

        assert response.status_code == 200
        assert "image/png" in response.headers["content-type"]

        # Cleanup
        analysis_session_store.delete(session_id)

    async def test_unknown_session_returns_404(self):
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/analysis/ghost-session/figures/fig-x"
            )

        assert response.status_code == 404

    async def test_unknown_figure_in_valid_session_returns_404(self):
        from app.services.memory import analysis_session_store

        session_id = "known-session-no-fig"
        analysis_session_store.get_or_create(session_id)

        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/v1/analysis/{session_id}/figures/nonexistent"
            )

        assert response.status_code == 404

        # Cleanup
        analysis_session_store.delete(session_id)


# ── dataset_hint pre-load (Gap 12) ────────────────────────────────────── #


class TestDatasetHintPreload:
    async def test_dataset_hint_error_does_not_block_request(self):
        """Gap 12: pre-load failure must be swallowed, not returned as HTTP error."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.services.knowledge_tools.inspect_dataset",
                return_value='{"error": "file not found"}',
            ):
                with patch(
                    "app.api.v1.analysis.data_science_agent.run",
                    new=AsyncMock(return_value="The answer"),
                ):
                    response = await client.post(
                        "/api/v1/analysis/chat",
                        json={
                            "message": "analyse data",
                            "session_id": "hint-fail-test",
                            "dataset_hint": "missing.csv",
                        },
                    )

        # The request must complete normally despite the failed hint
        assert response.status_code == 200

    async def test_dataset_hint_exception_does_not_block_request(self):
        """Gap 12: exception inside _try_preload_dataset must be caught silently."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.services.knowledge_tools.inspect_dataset",
                side_effect=RuntimeError("unexpected crash"),
            ):
                with patch(
                    "app.api.v1.analysis.data_science_agent.run",
                    new=AsyncMock(return_value="answer"),
                ):
                    response = await client.post(
                        "/api/v1/analysis/chat",
                        json={
                            "message": "analyse",
                            "session_id": "hint-exc-test",
                            "dataset_hint": "bad.csv",
                        },
                    )

        assert response.status_code == 200

    async def test_successful_dataset_hint_registers_meta_in_session(self):
        """Gap 12: successful pre-load must register a DatasetMeta in the session."""
        from app.services.memory import analysis_session_store

        session_id = "hint-success-test"
        analysis_session_store.delete(session_id)  # fresh start

        mock_inspect_result = json.dumps({
            "file_name": "sample.csv",
            "format": "csv",
            "rows": 100,
            "columns": 3,
            "column_names": ["a", "b", "c"],
            "dtypes": {"a": "float64", "b": "float64", "c": "object"},
            "numeric_stats": {},
        })

        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch(
                "app.services.knowledge_tools.inspect_dataset",
                return_value=mock_inspect_result,
            ):
                with patch(
                    "app.api.v1.analysis.data_science_agent.run",
                    new=AsyncMock(return_value="done"),
                ):
                    response = await client.post(
                        "/api/v1/analysis/chat",
                        json={
                            "message": "go",
                            "session_id": session_id,
                            "dataset_hint": "sample.csv",
                        },
                    )

        assert response.status_code == 200

        # Verify DatasetMeta was registered
        session = analysis_session_store.get(session_id)
        assert session is not None
        assert "sample.csv" in session.datasets_loaded

        # Cleanup
        analysis_session_store.delete(session_id)


# ── Request schema validation ─────────────────────────────────────────── #


class TestRequestValidation:
    async def test_blank_message_returns_422(self):
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/analysis/chat",
                json={"message": "   "},  # blank after strip
            )

        assert response.status_code == 422

    async def test_invalid_session_id_returns_422(self):
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/analysis/chat",
                json={"message": "hello", "session_id": "../path/traversal"},
            )

        assert response.status_code == 422
