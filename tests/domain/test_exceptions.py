"""Tests for app.domain.exceptions — exception hierarchy, attributes, messages."""
from __future__ import annotations

import pytest

from app.domain.exceptions import (
    AgentError,
    CodeExecutionError,
    DatasetLoadError,
    KernelCrashError,
    LLMAPIError,
    LLMAuthenticationError,
    LLMContextOverflowError,
    PhysicalValidationError,
    ReActLoopError,
    ReActMaxIterationsError,
    ReActParseError,
    SessionNotFoundError,
)


# ── Hierarchy ────────────────────────────────────────────────────────── #


class TestExceptionHierarchy:
    def test_llm_api_error_is_agent_error(self):
        assert issubclass(LLMAPIError, AgentError)

    def test_llm_context_overflow_is_llm_api_error(self):
        assert issubclass(LLMContextOverflowError, LLMAPIError)

    def test_llm_authentication_error_is_llm_api_error(self):
        assert issubclass(LLMAuthenticationError, LLMAPIError)

    def test_kernel_crash_error_is_code_execution_error(self):
        assert issubclass(KernelCrashError, CodeExecutionError)

    def test_react_max_iterations_error_is_react_loop_error(self):
        assert issubclass(ReActMaxIterationsError, ReActLoopError)

    def test_all_custom_exceptions_are_agent_error(self):
        for cls in (
            SessionNotFoundError,
            ReActLoopError,
            ReActParseError,
            CodeExecutionError,
            PhysicalValidationError,
            DatasetLoadError,
            LLMAPIError,
            LLMContextOverflowError,
            LLMAuthenticationError,
            KernelCrashError,
            ReActMaxIterationsError,
        ):
            assert issubclass(cls, AgentError), f"{cls.__name__} must inherit AgentError"

    def test_llm_context_overflow_is_catchable_as_agent_error(self):
        exc = LLMContextOverflowError("context too long")
        assert isinstance(exc, AgentError)
        assert isinstance(exc, LLMAPIError)


# ── LLMAPIError attributes ───────────────────────────────────────────── #


class TestLLMAPIError:
    def test_llm_api_error_stores_message(self):
        exc = LLMAPIError("something went wrong")
        assert "something went wrong" in str(exc)

    def test_llm_api_error_status_code_default_is_none(self):
        exc = LLMAPIError("error")
        assert exc.status_code is None

    def test_llm_api_error_stores_status_code(self):
        exc = LLMAPIError("error", status_code=502)
        assert exc.status_code == 502

    def test_llm_context_overflow_has_no_status_code_by_default(self):
        exc = LLMContextOverflowError("prompt is too long")
        assert exc.status_code is None

    def test_llm_authentication_error_message(self):
        exc = LLMAuthenticationError("Invalid API key")
        assert "Invalid API key" in str(exc)


# ── KernelCrashError attributes ──────────────────────────────────────── #


class TestKernelCrashError:
    def test_kernel_crash_stores_session_id(self):
        exc = KernelCrashError(session_id="sess-123", reason="process died")
        assert exc.session_id == "sess-123"

    def test_kernel_crash_message_contains_session_and_reason(self):
        exc = KernelCrashError(session_id="sess-abc", reason="OOM killed")
        msg = str(exc)
        assert "sess-abc" in msg
        assert "OOM killed" in msg

    def test_kernel_crash_backend_is_jupyter(self):
        exc = KernelCrashError(session_id="s", reason="r")
        assert exc.backend == "jupyter"


# ── ReActLoopError & ReActMaxIterationsError ─────────────────────────── #


class TestReActLoopErrors:
    def test_react_loop_error_stores_iterations(self):
        exc = ReActLoopError("failed", iterations=7, last_thought="maybe try X")
        assert exc.iterations == 7

    def test_react_loop_error_stores_last_thought(self):
        exc = ReActLoopError("failed", iterations=3, last_thought="check dataset")
        assert exc.last_thought == "check dataset"

    def test_react_loop_error_message_contains_iteration_count(self):
        exc = ReActLoopError("bad parse", iterations=5)
        assert "5" in str(exc)

    def test_react_max_iterations_error_is_react_loop_error_instance(self):
        exc = ReActMaxIterationsError("max reached", iterations=20)
        assert isinstance(exc, ReActLoopError)
        assert exc.iterations == 20


# ── SessionNotFoundError ─────────────────────────────────────────────── #


class TestSessionNotFoundError:
    def test_session_not_found_stores_session_id(self):
        exc = SessionNotFoundError("sess-xyz")
        assert exc.session_id == "sess-xyz"

    def test_session_not_found_message_contains_session_id(self):
        exc = SessionNotFoundError("my-session")
        assert "my-session" in str(exc)


# ── DatasetLoadError ─────────────────────────────────────────────────── #


class TestDatasetLoadError:
    def test_dataset_load_error_stores_file_name_and_reason(self):
        exc = DatasetLoadError("data.csv", "file not found")
        assert exc.file_name == "data.csv"
        assert exc.reason == "file not found"

    def test_dataset_load_error_message(self):
        exc = DatasetLoadError("big.parquet", "too large")
        assert "big.parquet" in str(exc)
        assert "too large" in str(exc)
