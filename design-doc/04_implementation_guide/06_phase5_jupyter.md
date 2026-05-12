# Phase 5 — Jupyter Kernel Bridge

## Overview

The Jupyter bridge provides **stateful** Python execution: variables defined in one call persist into the next. This is essential for multi-step analyses (load data → compute → plot) without reloading the dataset each time.

Architecture:
```
DataScienceAgentService._dispatch("execute_python_code")
    -> data_tools.execute_python_code(code, session)
        -> JupyterKernelManager.execute_cell(session_id, code)
            -> jupyter_client.BlockingKernelClient.execute(code)
            <- iopub messages: stream, display_data, execute_result
        <- AnalysisResult(stdout, figures, variables)
```

---

## 1. Installation

```bash
uv add jupyter_client>=8.0 nbformat>=5.10

# Install Python3 kernel (only needed once per machine)
uv add --dev ipykernel
python -m ipykernel install --user --name python3 --display-name "Python 3"

# Verify
jupyter kernelspec list
# python3    /Users/user/Library/Jupyter/kernels/python3
```

Set in `.env`:

```bash
CODE_EXECUTION_BACKEND=jupyter
ENABLE_JUPYTER_BRIDGE=true
CODE_EXECUTION_TIMEOUT=60      # jupyter kernels are slower to start
```

---

## 2. JupyterKernelManager — Complete Implementation

```python
# app/infrastructure/jupyter_bridge.py
"""
JupyterKernelManager: stateful Python execution via Jupyter kernel protocol.

One kernel per session_id. Kernels are started on first use and
kept alive for the session duration (idle timeout: 30 minutes).

Variables, DataFrames, and matplotlib figures persist between execute_cell() calls.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import nbformat
import nbformat.v4 as nbv4

from app.domain.analysis_models import AnalysisResult
from app.domain.exceptions import CodeExecutionError

logger = logging.getLogger(__name__)

# Import lazily to avoid hard dependency when backend != jupyter
try:
    import jupyter_client
    from jupyter_client import KernelManager
    from jupyter_client.blocking import BlockingKernelClient
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


# ── KERNEL_PREAMBLE ────────────────────────────────────────────────────────
# Executed once when a kernel is started to configure the environment.

KERNEL_PREAMBLE = """\
import io, base64, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Auto-capture any plt.show() call as a base64 PNG display
_original_show = plt.show
def _show_and_capture(*args, **kwargs):
    fig = plt.gcf()
    if fig.get_axes():
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        from IPython.display import display, Image
        display(Image(data=buf.read(), format='png'))
        plt.close(fig)
plt.show = _show_and_capture
print("Kernel ready: pandas", pd.__version__, "numpy", np.__version__)
"""


class _KernelEntry:
    """Tracks a running kernel and its last-activity timestamp."""
    def __init__(self, km: Any, kc: Any):
        self.km = km   # KernelManager
        self.kc = kc   # BlockingKernelClient
        self.last_used = time.monotonic()

    def touch(self):
        self.last_used = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_used


class JupyterKernelManager:
    """
    Manages a pool of Jupyter kernels, one per session_id.

    Thread safety: all access to _kernels is protected by _lock.
    Kernels are started in background threads (jupyter_client is synchronous).
    """

    IDLE_TIMEOUT_SECONDS = 1800  # 30 minutes

    def __init__(self):
        if not JUPYTER_AVAILABLE:
            raise ImportError(
                "jupyter_client is not installed. "
                "Run: uv add jupyter_client ipykernel"
            )
        self._kernels: dict[str, _KernelEntry] = {}
        self._lock = threading.Lock()

    def _start_kernel(self, session_id: str) -> _KernelEntry:
        """Synchronously start a new kernel and run the preamble."""
        logger.info("Starting Jupyter kernel for session %s", session_id)
        km = KernelManager(kernel_name="python3")
        km.start_kernel()
        kc = km.blocking_client()
        kc.start_channels()

        try:
            kc.wait_for_ready(timeout=30)
        except RuntimeError as e:
            km.shutdown_kernel(now=True)
            raise CodeExecutionError(
                f"Kernel failed to start: {e}",
                backend="jupyter",
            )

        # Run preamble
        msg_id = kc.execute(KERNEL_PREAMBLE, silent=False)
        self._drain_iopub(kc, msg_id, timeout=30)

        entry = _KernelEntry(km, kc)
        with self._lock:
            self._kernels[session_id] = entry
        return entry

    def _get_or_start(self, session_id: str) -> _KernelEntry:
        with self._lock:
            entry = self._kernels.get(session_id)
        if entry is None:
            entry = self._start_kernel(session_id)
        entry.touch()
        return entry

    def _drain_iopub(
        self,
        kc: Any,
        msg_id: str,
        timeout: int = 30,
    ) -> tuple[str, dict[str, str]]:
        """
        Drain iopub messages until kernel reaches 'idle' status.

        Returns:
            (stdout_combined, figures_dict)
        """
        stdout_parts: list[str] = []
        figures: dict[str, str] = {}
        fig_counter = 0
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodeExecutionError(
                    f"Kernel iopub timed out after {timeout}s",
                    backend="jupyter",
                    timeout=True,
                )
            try:
                msg = kc.get_iopub_msg(timeout=min(remaining, 5.0))
            except Exception:
                # Timeout on get_iopub_msg -> keep waiting
                continue

            msg_type = msg.get("msg_type", "")
            content = msg.get("content", {})
            parent_id = msg.get("parent_header", {}).get("msg_id", "")

            # Only process messages from our execution
            if parent_id != msg_id:
                continue

            if msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

            elif msg_type == "stream":
                if content.get("name") in ("stdout", "stderr"):
                    stdout_parts.append(content.get("text", ""))

            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data", {})
                if "image/png" in data:
                    fid = f"fig_{fig_counter:03d}"
                    fig_counter += 1
                    figures[fid] = data["image/png"]
                elif "text/plain" in data:
                    stdout_parts.append(data["text/plain"] + "\n")

            elif msg_type == "error":
                traceback = "\n".join(content.get("traceback", []))
                ename = content.get("ename", "Error")
                evalue = content.get("evalue", "")
                stdout_parts.append(f"{ename}: {evalue}\n{traceback}\n")

        return "".join(stdout_parts), figures

    async def execute_cell(
        self,
        session_id: str,
        code: str,
        timeout: int = 30,
    ) -> AnalysisResult:
        """
        Execute code in the kernel for the given session.

        Runs in asyncio executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_cell_sync,
            session_id,
            code,
            timeout,
        )

    def _execute_cell_sync(
        self,
        session_id: str,
        code: str,
        timeout: int,
    ) -> AnalysisResult:
        """Synchronous kernel execution (called from thread pool)."""
        start_ms = int(time.monotonic() * 1000)
        try:
            entry = self._get_or_start(session_id)
            kc = entry.kc

            msg_id = kc.execute(code, silent=False, store_history=True)
            stdout, figures = self._drain_iopub(kc, msg_id, timeout=timeout)

            elapsed = int(time.monotonic() * 1000) - start_ms

            # Detect errors in stdout (Jupyter prints tracebacks to stream)
            error_str = ""
            if any(line.startswith(("Error", "Exception", "Traceback"))
                   for line in stdout.splitlines()):
                error_str = stdout.splitlines()[0] if stdout.strip() else "Unknown error"

            return AnalysisResult(
                stdout=stdout,
                figures=figures,
                error=error_str,
                execution_time_ms=elapsed,
            )

        except CodeExecutionError as e:
            elapsed = int(time.monotonic() * 1000) - start_ms
            return AnalysisResult(
                error=str(e),
                timed_out=e.timeout,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            logger.exception("Unexpected kernel error for session %s", session_id)
            elapsed = int(time.monotonic() * 1000) - start_ms
            return AnalysisResult(error=f"{type(e).__name__}: {e}", execution_time_ms=elapsed)

    async def get_kernel_variables(self, session_id: str) -> dict[str, Any]:
        """
        Inspect kernel namespace for JSON-serialisable variables.
        Uses the %whos magic to list variables, then retrieves them.
        """
        result = await self.execute_cell(
            session_id,
            code=(
                "import json\n"
                "_ns = {k: v for k, v in globals().items() "
                "if not k.startswith('_') and not callable(v)}\n"
                "_safe = {}\n"
                "for k, v in _ns.items():\n"
                "    try:\n"
                "        json.dumps(v)\n"
                "        _safe[k] = v\n"
                "    except: pass\n"
                "print(json.dumps(_safe))"
            ),
            timeout=10,
        )
        try:
            last_line = result.stdout.strip().splitlines()[-1]
            return json.loads(last_line)
        except Exception:
            return {}

    async def export_notebook(
        self,
        session_id: str,
        cells: list[dict[str, str]],
        title: str,
        output_dir: Path = Path("outputs/notebooks"),
    ) -> Path:
        """
        Build and save a .ipynb from a list of cell dicts.

        Args:
            cells: [{"type": "code"|"markdown", "source": "..."}]
            title: Notebook title (used in first markdown cell)
            output_dir: Directory to save the notebook

        Returns:
            Path to the saved .ipynb file
        """
        nb = nbv4.new_notebook()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

        nb.cells.append(nbv4.new_markdown_cell(f"# {title}"))
        nb.cells.append(nbv4.new_code_cell(
            "import pandas as pd\nimport numpy as np\n"
            "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "%matplotlib inline"
        ))

        for cell in cells:
            if cell.get("type") == "code":
                nb.cells.append(nbv4.new_code_cell(cell["source"]))
            elif cell.get("type") == "markdown":
                nb.cells.append(nbv4.new_markdown_cell(cell["source"]))

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:50]
        nb_path = output_dir / f"{safe_title.replace(' ', '_')}_{session_id[:8]}.ipynb"

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return nb_path

    async def shutdown_kernel(self, session_id: str) -> None:
        """Gracefully shut down a kernel for a session."""
        with self._lock:
            entry = self._kernels.pop(session_id, None)
        if entry is None:
            return
        try:
            entry.kc.stop_channels()
            entry.km.shutdown_kernel(now=False, timeout=10)
            logger.info("Kernel shut down for session %s", session_id)
        except Exception as e:
            logger.warning("Error shutting down kernel %s: %s", session_id, e)

    async def shutdown_idle_kernels(self) -> int:
        """Shut down kernels idle for more than IDLE_TIMEOUT_SECONDS. Returns count."""
        with self._lock:
            idle_ids = [
                sid for sid, entry in self._kernels.items()
                if entry.idle_seconds > self.IDLE_TIMEOUT_SECONDS
            ]
        count = 0
        for sid in idle_ids:
            await self.shutdown_kernel(sid)
            count += 1
        return count


# Module-level singleton — used by data_tools.py when backend=="jupyter"
jupyter_manager = JupyterKernelManager() if JUPYTER_AVAILABLE else None
```

---

## 3. execute_cell() iopub Message Loop — Detailed

The iopub channel broadcasts all kernel output. Message types:

| msg_type | When | What to extract |
|---|---|---|
| `status` | Before/after execution | `execution_state`: "busy" / "idle" |
| `execute_input` | When kernel starts | (skip) |
| `stream` | print(), stderr | `content.name` + `content.text` |
| `execute_result` | Jupyter In/Out cells | `content.data["text/plain"]` |
| `display_data` | plt.show(), IPython.display | `content.data["image/png"]` (base64) |
| `error` | Python exception | `content.ename`, `content.evalue`, `content.traceback` |

The loop exits when `status.execution_state == "idle"` **AND** the `parent_header.msg_id` matches our execution's `msg_id`. This is critical — the kernel may broadcast messages from other sessions.

```python
# Pattern: always check parent_id
parent_id = msg.get("parent_header", {}).get("msg_id", "")
if parent_id != msg_id:
    continue  # skip messages from other executions
```

---

## 4. Figure Extraction from display_data

Matplotlib figures are sent as base64 PNG in `display_data` messages:

```python
# Inside _drain_iopub():
elif msg_type in ("display_data", "execute_result"):
    data = content.get("data", {})
    if "image/png" in data:
        fid = f"fig_{fig_counter:03d}"
        fig_counter += 1
        # data["image/png"] is ALREADY base64-encoded by IPython
        figures[fid] = data["image/png"]
```

The KERNEL_PREAMBLE patches `plt.show()` to call `IPython.display.display(Image(...))`, which sends a `display_data` message. This way, user code that calls `plt.show()` automatically sends the figure over the iopub channel.

---

## 5. nbformat Notebook Export

```python
import nbformat
import nbformat.v4 as nbv4

# Create notebook
nb = nbv4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

# Add cells
nb.cells.append(nbv4.new_markdown_cell("# My Analysis"))
nb.cells.append(nbv4.new_code_cell("import pandas as pd\ndf = pd.read_csv('data.csv')"))
nb.cells.append(nbv4.new_code_cell("print(df.describe())"))

# Add output to a cell (optional — shows what was printed)
code_cell = nbv4.new_code_cell("print('hello')")
code_cell.outputs = [
    nbv4.new_output(output_type="stream", name="stdout", text="hello\n")
]
nb.cells.append(code_cell)

# Save
with open("analysis.ipynb", "w") as f:
    nbformat.write(nb, f)
```

---

## 6. Kernel Lifecycle Management

```
Session created
    |
    v
First execute_cell() call
    |
    v
_get_or_start() -> _start_kernel() -> kernel starts (3-10 sec)
    |                                  KERNEL_PREAMBLE executes
    |                                  kernel enters "idle"
    v
execute_cell() calls (seconds apart)
    |
    v (after 30 minutes of inactivity)
shutdown_idle_kernels() -> km.shutdown_kernel()
    |
    v
Session ends / FastAPI lifespan cleanup
    -> shutdown all remaining kernels
```

**Background task for idle cleanup** — add to `main.py`:

```python
import asyncio
from app.infrastructure.jupyter_bridge import jupyter_manager

async def _cleanup_idle_kernels():
    """Background task: shut down idle kernels every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        if jupyter_manager:
            count = await jupyter_manager.shutdown_idle_kernels()
            if count > 0:
                logger.info("Shut down %d idle kernels", count)

# In lifespan:
asyncio.create_task(_cleanup_idle_kernels())
```

---

## 7. Enabling in config

`.env` settings:

```bash
CODE_EXECUTION_BACKEND=jupyter
ENABLE_JUPYTER_BRIDGE=true
CODE_EXECUTION_TIMEOUT=60
```

`config.py` usage in `data_tools.py`:

```python
def execute_python_code(code: str, session: AnalysisSession) -> str:
    backend = settings.code_execution_backend

    if backend == "subprocess":
        runner = _get_runner(session)
        result = runner.execute(code)
    elif backend == "jupyter":
        if not settings.enable_jupyter_bridge:
            return "ERROR: Jupyter bridge not enabled. Set ENABLE_JUPYTER_BRIDGE=true."
        from app.infrastructure.jupyter_bridge import jupyter_manager
        if jupyter_manager is None:
            return "ERROR: jupyter_client not installed. Run: uv add jupyter_client ipykernel"
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            jupyter_manager.execute_cell(
                session.session_id,
                code,
                timeout=settings.code_execution_timeout,
            )
        )
    else:
        result = AnalysisResult(error=f"Unknown backend: {backend}")

    if result.figures:
        session.merge_figures(result.figures)

    return result.to_observation_string()
```

---

## 8. Testing Jupyter Bridge

```python
# tests/test_phase5_jupyter.py
"""
Integration tests for JupyterKernelManager.
Requires: pip install ipykernel (installs python3 kernel)
These tests start a REAL kernel — they are slower than unit tests.
Run with: pytest tests/test_phase5_jupyter.py -v -m jupyter
"""
import pytest
import asyncio

pytest_plugins = ("anyio",)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def km():
    """Shared JupyterKernelManager for this test module."""
    try:
        from app.infrastructure.jupyter_bridge import JupyterKernelManager
        manager = JupyterKernelManager()
        yield manager
        # Cleanup: shut down all kernels
        for sid in list(manager._kernels.keys()):
            await manager.shutdown_kernel(sid)
    except ImportError:
        pytest.skip("jupyter_client not installed")


@pytest.mark.jupyter
@pytest.mark.asyncio
async def test_simple_print(km):
    result = await km.execute_cell("test_session_1", "print('hello from kernel')")
    assert "hello from kernel" in result.stdout
    assert result.success


@pytest.mark.jupyter
@pytest.mark.asyncio
async def test_state_persistence(km):
    """Variables defined in call 1 must be available in call 2."""
    await km.execute_cell("test_session_2", "x = 42")
    result = await km.execute_cell("test_session_2", "print(x * 2)")
    assert "84" in result.stdout


@pytest.mark.jupyter
@pytest.mark.asyncio
async def test_pandas_operation(km):
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})\n"
        "print(df['a'].mean())"
    )
    result = await km.execute_cell("test_session_3", code)
    assert "2.0" in result.stdout
    assert result.success


@pytest.mark.jupyter
@pytest.mark.asyncio
async def test_figure_capture(km):
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3], [4, 5, 6])\n"
        "ax.set_title('Test Figure')\n"
        "plt.show()"
    )
    result = await km.execute_cell("test_session_4", code)
    assert len(result.figures) == 1
    fid = list(result.figures.keys())[0]
    assert result.figures[fid]  # non-empty base64 string


@pytest.mark.jupyter
@pytest.mark.asyncio
async def test_error_in_code(km):
    result = await km.execute_cell("test_session_5", "x = undefined_variable")
    assert result.error or "NameError" in result.stdout


@pytest.mark.jupyter
@pytest.mark.asyncio
async def test_shutdown_kernel(km):
    await km.execute_cell("test_session_shutdown", "y = 100")
    await km.shutdown_kernel("test_session_shutdown")
    # After shutdown, a new kernel should start for the same session
    result = await km.execute_cell("test_session_shutdown", "print('new kernel')")
    assert "new kernel" in result.stdout
```

Run only jupyter tests:

```bash
pytest tests/test_phase5_jupyter.py -v -m jupyter
```

---

## Subprocess vs Jupyter Comparison

| Feature | SubprocessCodeRunner | JupyterKernelManager |
|---|---|---|
| State persistence | No (isolated per call) | Yes (kernel namespace) |
| Startup overhead | ~50ms per call | ~5s first call, ~10ms subsequent |
| Figure capture | Via JSON envelope | Via iopub display_data |
| Crash recovery | Automatic (new process) | Manual kernel restart needed |
| Security | Fully isolated | Shared kernel (same process) |
| Memory usage | Low | Higher (kernel always running) |
| Best for | Simple one-shot analysis | Multi-step iterative analysis |

---

## Checkpoint

After Phase 5:

```
app/infrastructure/jupyter_bridge.py  -- JupyterKernelManager with stateful execution
tests/test_phase5_jupyter.py          -- Integration tests (require ipykernel)
```

-> Next: 07_phase6_api.md -- FastAPI integration
