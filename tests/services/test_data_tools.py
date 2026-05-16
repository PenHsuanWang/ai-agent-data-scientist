"""Tests for app.services.data_tools — Gap 7: get_figure returns metadata, not base64."""
from __future__ import annotations

import base64
import json

import pytest

from app.domain.analysis_models import AnalysisSession


# ── helpers ──────────────────────────────────────────────────────────── #


def _make_session_with_figure(figure_id: str = "fig-001") -> AnalysisSession:
    """Return a session that has a small base64-encoded PNG registered."""
    session = AnalysisSession(session_id="data-tools-test")
    # register_figure stores a raw b64 string keyed by figure_id
    fake_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode()
    session.figures[figure_id] = fake_b64
    return session


# ── get_figure (Gap 7) ───────────────────────────────────────────────── #


class TestGetFigure:
    def test_returns_metadata_not_raw_base64(self):
        """Gap 7: figure data must NOT be returned inline to prevent context overflow."""
        from app.services.data_tools import get_figure

        session = _make_session_with_figure("fig-001")
        result = json.loads(get_figure("fig-001", session))

        assert "data" not in result, "Raw base64 data must not be returned in the tool response"
        assert "figure_id" in result
        assert result["figure_id"] == "fig-001"

    def test_returns_format_field(self):
        from app.services.data_tools import get_figure

        session = _make_session_with_figure("fig-002")
        result = json.loads(get_figure("fig-002", session))

        assert result["format"] == "png"

    def test_returns_size_bytes(self):
        from app.services.data_tools import get_figure

        session = _make_session_with_figure("fig-003")
        raw_b64 = session.figures["fig-003"]
        expected_approx = int(len(raw_b64) * 0.75)

        result = json.loads(get_figure("fig-003", session))

        assert result["size_bytes"] == expected_approx

    def test_returns_retrieval_url_pointing_to_api_route(self):
        from app.services.data_tools import get_figure

        session = _make_session_with_figure("fig-004")
        result = json.loads(get_figure("fig-004", session))

        assert "retrieval_url" in result
        assert "fig-004" in result["retrieval_url"]
        assert result["retrieval_url"].startswith("/api/")

    def test_retrieval_url_contains_session_id(self):
        from app.services.data_tools import get_figure

        session = AnalysisSession(session_id="my-session-xyz")
        session.figures["fig-5"] = base64.b64encode(b"fake").decode()
        result = json.loads(get_figure("fig-5", session))

        assert "my-session-xyz" in result["retrieval_url"]

    def test_returns_note_suggesting_use_of_url(self):
        from app.services.data_tools import get_figure

        session = _make_session_with_figure("fig-005")
        result = json.loads(get_figure("fig-005", session))

        assert "note" in result
        note_lower = result["note"].lower()
        assert "retrieval_url" in note_lower or "url" in note_lower

    def test_returns_error_json_for_unknown_figure_id(self):
        from app.services.data_tools import get_figure

        session = AnalysisSession(session_id="empty-fig")
        result = json.loads(get_figure("missing-fig", session))

        assert "error" in result
        assert "missing-fig" in result["error"]

    def test_error_response_includes_available_figures(self):
        from app.services.data_tools import get_figure

        session = AnalysisSession(session_id="listing-test")
        session.figures["existing"] = base64.b64encode(b"x").decode()
        result = json.loads(get_figure("not-there", session))

        assert "available_figures" in result
        assert "existing" in result["available_figures"]


# ── list_figures ─────────────────────────────────────────────────────── #


class TestListFigures:
    def test_returns_count_and_figure_ids(self):
        from app.services.data_tools import list_figures

        session = AnalysisSession(session_id="list-fig-test")
        session.figures["fig-a"] = base64.b64encode(b"a").decode()
        session.figures["fig-b"] = base64.b64encode(b"b").decode()

        result = json.loads(list_figures(session))

        assert result["count"] == 2
        assert "fig-a" in result["figure_ids"]
        assert "fig-b" in result["figure_ids"]

    def test_returns_zero_count_when_no_figures(self):
        from app.services.data_tools import list_figures

        session = AnalysisSession(session_id="empty-figs")
        result = json.loads(list_figures(session))

        assert result["count"] == 0
        assert result["figure_ids"] == []
