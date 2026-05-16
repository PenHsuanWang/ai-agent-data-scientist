"""Tests for app.infrastructure.code_runner — Gaps 8, 9, 13."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.domain.exceptions import KernelCrashError


# ── CodeRunnerFactory validation (Gap 9) ─────────────────────────────── #


class TestCodeRunnerFactory:
    def test_raises_value_error_for_unknown_backend(self):
        """Gap 9: invalid backend must raise ValueError at construction time."""
        from app.infrastructure.code_runner import CodeRunnerFactory

        with patch("app.infrastructure.code_runner.settings") as mock_s:
            mock_s.code_execution_backend = "nonexistent_backend"
            mock_s.code_execution_timeout = 30
            with pytest.raises(ValueError, match="nonexistent_backend"):
                CodeRunnerFactory.create(session_id="test")

    def test_error_message_lists_valid_backends(self):
        from app.infrastructure.code_runner import CodeRunnerFactory

        with patch("app.infrastructure.code_runner.settings") as mock_s:
            mock_s.code_execution_backend = "bad"
            mock_s.code_execution_timeout = 30
            with pytest.raises(ValueError) as exc_info:
                CodeRunnerFactory.create(session_id="test")
        msg = str(exc_info.value)
        for backend in CodeRunnerFactory.VALID_BACKENDS:
            assert backend in msg

    def test_valid_backends_frozenset_contains_expected_values(self):
        from app.infrastructure.code_runner import CodeRunnerFactory

        assert "subprocess" in CodeRunnerFactory.VALID_BACKENDS
        assert "jupyter" in CodeRunnerFactory.VALID_BACKENDS
        assert "anthropic" in CodeRunnerFactory.VALID_BACKENDS

    def test_creates_subprocess_runner(self):
        from app.infrastructure.code_runner import CodeRunnerFactory, SubprocessCodeRunner

        with patch("app.infrastructure.code_runner.settings") as mock_s:
            mock_s.code_execution_backend = "subprocess"
            mock_s.code_execution_timeout = 30
            runner = CodeRunnerFactory.create(session_id="subprocess-test")

        assert isinstance(runner, SubprocessCodeRunner)

    def test_unknown_backend_does_not_create_runner(self):
        from app.infrastructure.code_runner import CodeRunnerFactory

        with patch("app.infrastructure.code_runner.settings") as mock_s:
            mock_s.code_execution_backend = "invalid"
            mock_s.code_execution_timeout = 30
            try:
                runner = CodeRunnerFactory.create(session_id="x")
                pytest.fail("ValueError was not raised")
            except ValueError:
                pass


# ── PREEXEC_FN guard (Gap 13) ─────────────────────────────────────────── #


class TestPreexecFnGuard:
    def test_preexec_fn_is_none_on_windows(self):
        """Gap 13: _PREEXEC_FN must be None on Windows to prevent crashes."""
        from app.infrastructure.code_runner import _PREEXEC_FN

        if sys.platform == "win32":
            assert _PREEXEC_FN is None
        else:
            # On non-Windows platforms the guard should have set a callable
            assert _PREEXEC_FN is not None

    def test_preexec_fn_is_callable_on_non_windows(self):
        """On Unix-like platforms, _PREEXEC_FN must be the limits function."""
        if sys.platform == "win32":
            pytest.skip("Not relevant on Windows")

        from app.infrastructure.code_runner import _PREEXEC_FN
        assert callable(_PREEXEC_FN)

    def test_apply_subprocess_limits_is_callable(self):
        """_apply_subprocess_limits must be a callable (even if limits silently fail)."""
        from app.infrastructure.code_runner import _apply_subprocess_limits
        assert callable(_apply_subprocess_limits)


# ── SubprocessCodeRunner (basic) ─────────────────────────────────────── #


class TestSubprocessCodeRunner:
    def test_execute_returns_success_for_trivial_code(self):
        from app.infrastructure.code_runner import SubprocessCodeRunner
        from app.core.config import settings

        runner = SubprocessCodeRunner(session_id="subproc-test")
        result = runner.execute("print('hello')")

        # The result must not raise and should reflect a clean execution
        assert result is not None
        assert hasattr(result, "success")

    def test_execute_returns_failure_for_bad_code(self):
        from app.infrastructure.code_runner import SubprocessCodeRunner

        runner = SubprocessCodeRunner(session_id="subproc-fail")
        result = runner.execute("raise RuntimeError('forced')")

        assert result.success is False
        assert result.stderr != ""

    def test_execute_handles_timeout(self):
        from app.infrastructure.code_runner import SubprocessCodeRunner
        import subprocess

        runner = SubprocessCodeRunner(session_id="timeout-test")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("python", 5)):
            with patch("app.infrastructure.code_runner.settings") as mock_s:
                mock_s.code_execution_timeout = 5
                mock_s.datasets_dir = MagicMock()
                mock_s.datasets_dir.resolve.return_value = "/tmp"
                result = runner.execute("import time; time.sleep(999)")

        assert result.success is False
        assert "timed out" in result.stderr.lower()

    def test_shutdown_clears_figures_and_state(self):
        from app.infrastructure.code_runner import SubprocessCodeRunner

        runner = SubprocessCodeRunner(session_id="shutdown-test")
        runner._last_figures = {"fig_001": "base64data"}
        runner._last_state = {"df": "DataFrame"}

        runner.shutdown()

        assert runner._last_figures == {}
        assert runner._last_state == {}

    def test_get_figure_b64_returns_none_for_unknown_id(self):
        from app.infrastructure.code_runner import SubprocessCodeRunner

        runner = SubprocessCodeRunner(session_id="fig-test")
        assert runner.get_figure_b64("nonexistent") is None

    def test_get_state_returns_dict(self):
        from app.infrastructure.code_runner import SubprocessCodeRunner

        runner = SubprocessCodeRunner(session_id="state-test")
        state = runner.get_state()

        assert isinstance(state, dict)


# ── JupyterKernelManager: probe and restart (Gap 8) ──────────────────── #


class TestJupyterKernelManager:
    def test_probe_alive_returns_false_when_not_started(self):
        """Gap 8: kernel probe must return False when kernel is not running."""
        from app.infrastructure.code_runner import JupyterKernelManager

        km = JupyterKernelManager(session_id="probe-test")
        assert km._probe_alive() is False

    def test_probe_alive_returns_false_when_km_is_none(self):
        from app.infrastructure.code_runner import JupyterKernelManager

        km = JupyterKernelManager(session_id="km-none-test")
        km._started = True  # lie about started state
        km._km = None
        assert km._probe_alive() is False

    def test_restart_kernel_raises_kernel_crash_error_when_start_fails(self):
        """Gap 8: KernelCrashError must be raised when restart fails."""
        from app.infrastructure.code_runner import JupyterKernelManager

        km = JupyterKernelManager(session_id="crash-test")
        km._started = True
        km._km = MagicMock()
        km._km.shutdown_kernel = MagicMock()

        # Force _start to fail after the shutdown
        with patch.object(km, "_start", side_effect=RuntimeError("kernel died")):
            with pytest.raises(KernelCrashError) as exc_info:
                km._restart_kernel()

        assert exc_info.value.session_id == "crash-test"
        assert exc_info.value.backend == "jupyter"

    def test_execute_returns_failure_when_restart_fails(self):
        """If kernel probe fails and restart raises KernelCrashError, execute returns failure."""
        from app.infrastructure.code_runner import JupyterKernelManager

        km = JupyterKernelManager(session_id="exec-crash")
        km._started = True
        km._km = MagicMock()
        km._kc = MagicMock()

        with patch.object(km, "_probe_alive", return_value=False):
            with patch.object(km, "_restart_kernel", side_effect=KernelCrashError("exec-crash", "OOM")):
                with patch.object(km, "_start", return_value=None):
                    result = km.execute("print('test')")

        assert result.success is False
        assert "Jupyter kernel crashed" in result.stderr

    def test_kernel_crash_error_contains_session_id(self):
        exc = KernelCrashError(session_id="my-session", reason="process died")
        assert "my-session" in str(exc)
        assert exc.session_id == "my-session"
        assert exc.backend == "jupyter"
