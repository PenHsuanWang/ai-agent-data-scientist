"""Code execution backends.

Three backends implement the CodeRunner abstract interface:
  SubprocessCodeRunner  — isolated subprocess, safe, no state
  JupyterKernelManager  — stateful kernel (optional, requires jupyter_client)
  AnthropicCodeExecRunner — Anthropic-hosted sandbox (stub, needs beta)

The factory picks the correct backend based on settings.code_execution_backend.
"""
from __future__ import annotations

import abc
import logging
import subprocess
import sys
import textwrap
import time
import uuid
from typing import Any

from app.core.config import settings
from app.domain.analysis_models import AnalysisResult
from app.infrastructure.style_config import STYLE_PREAMBLE

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────── #
# Abstract interface                                                    #
# ──────────────────────────────────────────────────────────────────── #


class CodeRunner(abc.ABC):
    """Abstract interface for code execution backends.

    Contract:
    - execute() NEVER raises — returns AnalysisResult(success=False) on error.
    - get_state() returns {name: type_name} for current variables.
    - get_figure_b64() returns None if figure_id unknown.
    - shutdown() is idempotent.
    """

    @abc.abstractmethod
    def execute(self, code: str) -> AnalysisResult:
        ...

    @abc.abstractmethod
    def get_state(self) -> dict[str, str]:
        ...

    @abc.abstractmethod
    def get_figure_b64(self, figure_id: str) -> str | None:
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        ...


# ──────────────────────────────────────────────────────────────────── #
# Subprocess backend (default)                                          #
# ──────────────────────────────────────────────────────────────────── #

_FIGURE_POSTAMBLE = textwrap.dedent("""
import json as _json
print("__FIGURES__:" + _json.dumps(_FIGURES))
print("__STATE__:" + _json.dumps({k: type(v).__name__ for k, v in list(globals().items()) if not k.startswith('_') and k not in ('plt','pd','np','sns','json','base64','io','sys','os','matplotlib','seaborn')}))
""")


class SubprocessCodeRunner(CodeRunner):
    """Executes code in an isolated subprocess.

    Variables do NOT persist between calls — each execute() is independent.
    Figures are captured by patching plt.show().
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._last_figures: dict[str, str] = {}   # figure_id → b64 PNG
        self._last_state: dict[str, str] = {}

    def execute(self, code: str) -> AnalysisResult:
        start = time.monotonic()
        datasets_dir = str(settings.datasets_dir.resolve())
        preamble = STYLE_PREAMBLE.replace("{datasets_dir!r}", repr(datasets_dir))
        full_code = preamble + "\n" + code + "\n" + _FIGURE_POSTAMBLE

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=settings.code_execution_timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.monotonic() - start) * 1000)
            return AnalysisResult(
                success=False,
                stderr=f"Execution timed out after {settings.code_execution_timeout}s",
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            logger.error("Subprocess error: %s", exc, exc_info=True)
            return AnalysisResult(
                success=False,
                stderr=str(exc),
                execution_time_ms=elapsed,
            )

        elapsed = int((time.monotonic() - start) * 1000)
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Parse embedded figure/state JSON lines
        figures_captured: dict[str, str] = {}
        state_captured: dict[str, str] = {}
        clean_lines: list[str] = []

        for line in stdout.splitlines():
            if line.startswith("__FIGURES__:"):
                try:
                    import json
                    figures_captured = json.loads(line[len("__FIGURES__:"):])
                except Exception:
                    pass
            elif line.startswith("__STATE__:"):
                try:
                    import json
                    state_captured = json.loads(line[len("__STATE__:"):])
                except Exception:
                    pass
            else:
                clean_lines.append(line)

        self._last_figures = figures_captured
        self._last_state = state_captured

        clean_stdout = "\n".join(clean_lines)
        success = result.returncode == 0

        if not success:
            logger.warning(
                "Code execution failed (session=%s): %s",
                self._session_id,
                stderr[:500],
            )

        return AnalysisResult(
            success=success,
            stdout=clean_stdout,
            stderr=stderr if not success else "",
            figures=list(figures_captured.keys()),
            execution_time_ms=elapsed,
        )

    def get_state(self) -> dict[str, str]:
        return dict(self._last_state)

    def get_figure_b64(self, figure_id: str) -> str | None:
        return self._last_figures.get(figure_id)

    def shutdown(self) -> None:
        self._last_figures.clear()
        self._last_state.clear()


# ──────────────────────────────────────────────────────────────────── #
# Jupyter backend (optional)                                            #
# ──────────────────────────────────────────────────────────────────── #


class JupyterKernelManager(CodeRunner):
    """Stateful Jupyter kernel backend.

    Requires: pip install jupyter_client ipykernel
    Variables persist between execute() calls (true REPL state).
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._km = None
        self._kc = None
        self._figures: dict[str, str] = {}
        self._state: dict[str, str] = {}
        self._figure_counter = 0
        self._started = False

    def _start(self) -> None:
        if self._started:
            return
        try:
            import jupyter_client  # noqa: F401
            from jupyter_client.manager import KernelManager
            self._km = KernelManager(kernel_name="python3")
            self._km.start_kernel()
            self._kc = self._km.client()
            self._kc.start_channels()
            self._kc.wait_for_ready(timeout=30)
            self._started = True
            logger.info("Jupyter kernel started for session %s", self._session_id)
        except ImportError:
            raise RuntimeError(
                "jupyter_client is not installed. "
                "Install with: pip install jupyter_client ipykernel"
            )

    def execute(self, code: str) -> AnalysisResult:
        try:
            self._start()
        except Exception as exc:
            return AnalysisResult(success=False, stderr=str(exc))

        start = time.monotonic()
        msg_id = self._kc.execute(code)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        new_figures: list[str] = []

        try:
            while True:
                msg = self._kc.get_iopub_msg(timeout=settings.code_execution_timeout)
                content = msg["content"]
                msg_type = msg["msg_type"]

                if msg_type == "stream":
                    if content.get("name") == "stdout":
                        stdout_parts.append(content.get("text", ""))
                    else:
                        stderr_parts.append(content.get("text", ""))

                elif msg_type == "display_data" or msg_type == "execute_result":
                    if "image/png" in content.get("data", {}):
                        fid = f"fig_{self._figure_counter:03d}"
                        self._figures[fid] = content["data"]["image/png"]
                        self._figure_counter += 1
                        new_figures.append(fid)
                        stdout_parts.append(f"[Figure captured: {fid}]")

                elif msg_type == "error":
                    stderr_parts.append("\n".join(content.get("traceback", [])))

                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break

        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return AnalysisResult(
                success=False,
                stderr=str(exc),
                execution_time_ms=elapsed,
            )

        elapsed = int((time.monotonic() - start) * 1000)
        success = not any(stderr_parts)

        return AnalysisResult(
            success=success,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            figures=new_figures,
            execution_time_ms=elapsed,
        )

    def get_state(self) -> dict[str, str]:
        if not self._started:
            return {}
        try:
            result = self.execute(
                "import json as _j; "
                "print('__VARS__:' + _j.dumps({k: type(v).__name__ for k,v in globals().items() if not k.startswith('_')}))"
            )
            for line in result.stdout.splitlines():
                if line.startswith("__VARS__:"):
                    import json
                    return json.loads(line[len("__VARS__:"):])
        except Exception:
            pass
        return {}

    def get_figure_b64(self, figure_id: str) -> str | None:
        return self._figures.get(figure_id)

    def shutdown(self) -> None:
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
        self._started = False
        logger.info("Jupyter kernel shut down for session %s", self._session_id)


# ──────────────────────────────────────────────────────────────────── #
# Factory                                                               #
# ──────────────────────────────────────────────────────────────────── #


class CodeRunnerFactory:
    """Creates the appropriate CodeRunner based on configuration."""

    @staticmethod
    def create(session_id: str | None = None) -> CodeRunner:
        sid = session_id or str(uuid.uuid4())
        backend = settings.code_execution_backend.lower()

        if backend == "subprocess":
            return SubprocessCodeRunner(session_id=sid)
        elif backend == "jupyter":
            return JupyterKernelManager(session_id=sid)
        else:
            raise ValueError(
                f"Unknown code execution backend: '{backend}'. "
                "Valid options: subprocess, jupyter, anthropic"
            )
