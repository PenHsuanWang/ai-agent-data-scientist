# Jupyter Bridge Design — Interactive Kernel Management

## Overview

The Jupyter Bridge provides a **true Python REPL** for each analysis session by managing a real IPython kernel process via `jupyter_client`. Unlike the `SubprocessCodeRunner` (which re-spawns a fresh process each call), the Jupyter Bridge keeps one kernel alive per `AnalysisSession` — so `import pandas as pd; df = pd.read_csv("data.csv")` in cell 1 is still accessible in cell 10.

This is the **recommended backend** when:
- Users expect notebook export (`.ipynb`)
- Analysis spans many dependent cells (fit model in cell 5, predict in cell 9)
- Interactive widgets (`ipywidgets`) are needed
- Plotly/Bokeh interactive figures are desired

---

## Architecture

```
AnalysisSession (session_id)
       │
       ▼
JupyterKernelManager  (singleton, manages all sessions)
       │
       ├── session_id_A  →  KernelManager_A + BlockingKernelClient_A
       ├── session_id_B  →  KernelManager_B + BlockingKernelClient_B
       └── ...
                              │
                    ┌─────────▼─────────────────────────────┐
                    │   IPython Kernel Subprocess             │
                    │   (python3 kernel, one per session)     │
                    │                                         │
                    │   shell channel:   execute_request      │
                    │   iopub channel:   stream, display_data,│
                    │                   execute_result, error │
                    │   stdin channel:   (disabled)           │
                    └─────────────────────────────────────────┘
```

---

## Class Design

```python
# app/infrastructure/jupyter_bridge.py

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jupyter_client
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output

from app.domain.analysis_models import AnalysisResult

logger = logging.getLogger(__name__)

_KERNEL_TIMEOUT = 60   # seconds to wait for kernel ready
_EXECUTE_TIMEOUT = 120  # seconds to wait for cell execution


@dataclass
class KernelSession:
    """Holds the kernel manager + client pair for one AnalysisSession."""
    kernel_manager: jupyter_client.KernelManager
    kernel_client: jupyter_client.BlockingKernelClient
    session_id: str
    cells: list[dict] = field(default_factory=list)  # for notebook export


class JupyterKernelManager:
    """
    Manages one IPython kernel per AnalysisSession.

    Public API:
        get_or_start_kernel(session_id)    → KernelSession
        execute_cell(session_id, code)     → AnalysisResult
        export_notebook(session_id, title) → Path
        shutdown_kernel(session_id)        → None
        shutdown_all()                     → None
    """

    def __init__(self) -> None:
        self._kernels: dict[str, KernelSession] = {}

    # ── Lifecycle ──────────────────────────────────────────────────── #

    def get_or_start_kernel(self, session_id: str) -> KernelSession:
        """Return existing kernel session, or start a new one."""
        if session_id in self._kernels:
            ks = self._kernels[session_id]
            if ks.kernel_manager.is_alive():
                return ks
            logger.warning("Kernel for session %s died — restarting", session_id)
            self._cleanup(session_id)

        logger.info("Starting IPython kernel for session %s", session_id)
        km = jupyter_client.KernelManager(kernel_name="python3")
        km.start_kernel()

        kc = km.client()
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=_KERNEL_TIMEOUT)
        except RuntimeError as exc:
            km.shutdown_kernel(now=True)
            raise RuntimeError(f"Kernel failed to start: {exc}") from exc

        ks = KernelSession(kernel_manager=km, kernel_client=kc, session_id=session_id)
        self._kernels[session_id] = ks
        logger.info("Kernel ready for session %s (pid=%s)", session_id, km.kernel.pid)
        return ks

    def shutdown_kernel(self, session_id: str) -> None:
        """Gracefully shut down the kernel for a session."""
        if session_id in self._kernels:
            self._cleanup(session_id)
            logger.info("Kernel shut down for session %s", session_id)

    def shutdown_all(self) -> None:
        """Shut down all kernels — call on application shutdown."""
        for session_id in list(self._kernels.keys()):
            self._cleanup(session_id)

    def _cleanup(self, session_id: str) -> None:
        ks = self._kernels.pop(session_id, None)
        if ks:
            try:
                ks.kernel_client.stop_channels()
                ks.kernel_manager.shutdown_kernel(now=True)
            except Exception as exc:
                logger.warning("Error shutting down kernel: %s", exc)

    # ── Execution ──────────────────────────────────────────────────── #

    def execute_cell(self, session_id: str, code: str) -> AnalysisResult:
        """
        Execute Python code in the session kernel.

        Returns AnalysisResult with stdout, stderr, figures (base64 PNGs),
        and list of newly defined variable names.
        """
        import time
        start_ms = time.time() * 1000

        ks = self.get_or_start_kernel(session_id)
        kc = ks.kernel_client

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        figures: dict[str, str] = {}   # figure_id → base64
        execute_result: str = ""
        error_text: str = ""
        fig_counter = len([k for k in figures])

        # Send execute_request on shell channel
        msg_id = kc.execute(
            code,
            silent=False,
            store_history=True,
            allow_stdin=False,
        )

        # Drain iopub until execution_state → idle
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=_EXECUTE_TIMEOUT)
            except Exception:
                error_text = "Execution timed out"
                break

            msg_type = msg["msg_type"]
            content = msg.get("content", {})

            if msg_type == "stream":
                if content.get("name") == "stdout":
                    stdout_parts.append(content.get("text", ""))
                elif content.get("name") == "stderr":
                    stderr_parts.append(content.get("text", ""))

            elif msg_type == "display_data":
                data = content.get("data", {})
                if "image/png" in data:
                    fig_id = f"fig_{fig_counter:03d}"
                    fig_counter += 1
                    figures[fig_id] = data["image/png"]   # already base64

            elif msg_type == "execute_result":
                text = content.get("data", {}).get("text/plain", "")
                execute_result = text
                # Also capture images from execute_result
                data = content.get("data", {})
                if "image/png" in data:
                    fig_id = f"fig_{fig_counter:03d}"
                    fig_counter += 1
                    figures[fig_id] = data["image/png"]

            elif msg_type == "error":
                tb = "\n".join(content.get("traceback", []))
                error_text = f"{content.get('ename')}: {content.get('evalue')}\n{tb}"

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break   # execution complete

        # Append to cell history for notebook export
        outputs = []
        if stdout_parts:
            outputs.append({"output_type": "stream", "name": "stdout", "text": "".join(stdout_parts)})
        if stderr_parts:
            outputs.append({"output_type": "stream", "name": "stderr", "text": "".join(stderr_parts)})
        if execute_result:
            outputs.append({"output_type": "execute_result", "execution_count": None,
                            "data": {"text/plain": execute_result}, "metadata": {}})
        for fid, b64 in figures.items():
            outputs.append({"output_type": "display_data",
                            "data": {"image/png": b64}, "metadata": {}})

        ks.cells.append({"source": code, "outputs": outputs})

        elapsed = time.time() * 1000 - start_ms
        success = not error_text

        return AnalysisResult(
            code=code,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts) + error_text,
            figures=list(figures.keys()),
            variables_defined=self._get_new_variables(kc),
            execution_time_ms=elapsed,
            success=success,
            figures_data=figures,   # pass base64 data upstream
        )

    def _get_new_variables(self, kc: jupyter_client.BlockingKernelClient) -> list[str]:
        """Run %who_ls magic to get current namespace variables."""
        try:
            msg_id = kc.execute("%who_ls", silent=True, store_history=False)
            while True:
                msg = kc.get_iopub_msg(timeout=10)
                if msg["msg_type"] == "execute_result":
                    text = msg["content"]["data"].get("text/plain", "[]")
                    return json.loads(text.replace("'", '"'))
                if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                    break
        except Exception:
            pass
        return []

    # ── Notebook Export ────────────────────────────────────────────── #

    def export_notebook(self, session_id: str, title: str, output_dir: Path) -> Path:
        """
        Export the session's cell history as a .ipynb file.

        Returns the path to the saved notebook.
        """
        ks = self._kernels.get(session_id)
        cells_data = ks.cells if ks else []

        nb = new_notebook()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
        nb.metadata["title"] = title

        # Title cell
        title_cell = nbformat.v4.new_markdown_cell(f"# {title}\n\nGenerated by Data Scientist AI-Agent")
        nb.cells.append(title_cell)

        # Code cells with outputs
        for cell_data in cells_data:
            cell = new_code_cell(source=cell_data["source"])
            for out in cell_data.get("outputs", []):
                out_type = out["output_type"]
                if out_type == "stream":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="stream",
                        name=out["name"],
                        text=out["text"],
                    ))
                elif out_type == "execute_result":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="execute_result",
                        execution_count=1,
                        data=out["data"],
                        metadata=out.get("metadata", {}),
                    ))
                elif out_type == "display_data":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="display_data",
                        data=out["data"],
                        metadata=out.get("metadata", {}),
                    ))
            nb.cells.append(cell)

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
        path = output_dir / f"{safe_title}.ipynb"

        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        logger.info("Notebook exported: %s (%d cells)", path, len(nb.cells))
        return path


# Module-level singleton
jupyter_kernel_manager = JupyterKernelManager()
```

---

## Kernel Lifecycle State Machine

```
              ┌─────────────────┐
              │    NOT STARTED  │
              └────────┬────────┘
                       │ get_or_start_kernel()
              ┌────────▼────────┐
              │    STARTING     │  km.start_kernel()
              │                 │  kc.wait_for_ready()
              └────────┬────────┘
                       │ ready
              ┌────────▼────────┐
              │     IDLE        │◄──────────────────────┐
              └────────┬────────┘                       │
                       │ execute_cell()                 │ execution_state: idle
              ┌────────▼────────┐                       │
              │    EXECUTING    │───────────────────────┘
              └────────┬────────┘
                       │ idle timeout OR shutdown_kernel()
              ┌────────▼────────┐
              │    DEAD/STOPPED │
              └─────────────────┘
                       │ get_or_start_kernel() again
                       └────────► STARTING (restart)
```

---

## iopub Message Types Reference

| Message type | When it appears | What to extract |
|---|---|---|
| `status` | Before and after every execution | `execution_state`: "busy" → "idle" |
| `stream` | Any `print()` call | `name` ("stdout"/"stderr"), `text` |
| `display_data` | `plt.show()`, `IPython.display.display()` | `data["image/png"]` (base64) |
| `execute_result` | Expression result (last line value) | `data["text/plain"]` or `data["image/png"]` |
| `error` | Exception raised | `ename`, `evalue`, `traceback` |
| `execute_input` | Echo of sent code | Ignore |

---

## Figure Extraction Detail

Matplotlib figures appear in `display_data` messages when either:
1. `plt.show()` is called explicitly
2. Cell ends with a figure object (IPython rich output)
3. `IPython.display.display(fig)` is called

The `image/png` value is **already base64-encoded** by the kernel. No re-encoding needed:

```python
# Inside execute_cell() — image comes pre-encoded
if "image/png" in data:
    fig_id = f"fig_{fig_counter:03d}"
    figures[fig_id] = data["image/png"]   # store directly
```

To retrieve: `base64.b64decode(figures[fig_id])` → raw PNG bytes.

---

## Session Scoping and Concurrency

- **One kernel per `session_id`** — each AnalysisSession has its own Python namespace
- **No shared state between sessions** — kernel A cannot read kernel B's variables
- **Serial execution per kernel** — `execute_cell()` is blocking; don't call concurrently on the same session

For async FastAPI handlers, wrap in `asyncio.to_thread()`:

```python
# In DataScienceAgentService:
result = await asyncio.to_thread(
    jupyter_kernel_manager.execute_cell,
    session.session_id,
    code,
)
```

---

## Configuration

Add to `app/core/config.py`:

```python
enable_jupyter_bridge: bool = Field(False, alias="ENABLE_JUPYTER_BRIDGE")
jupyter_kernel_name: str = Field("python3", alias="JUPYTER_KERNEL_NAME")
jupyter_startup_timeout: int = Field(60, alias="JUPYTER_STARTUP_TIMEOUT")
jupyter_execute_timeout: int = Field(120, alias="JUPYTER_EXECUTE_TIMEOUT")
```

Add to `.env`:
```
ENABLE_JUPYTER_BRIDGE=true
CODE_EXECUTION_BACKEND=jupyter
```

---

## FastAPI Lifespan Integration

Register kernel manager shutdown in `main.py`:

```python
from contextlib import asynccontextmanager
from app.infrastructure.jupyter_bridge import jupyter_kernel_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    jupyter_kernel_manager.shutdown_all()   # cleanup on shutdown

app = FastAPI(lifespan=lifespan)
```

---

## Error Recovery

| Scenario | Detection | Recovery |
|---|---|---|
| Kernel process dies | `km.is_alive()` returns False | `_cleanup()` + restart on next call |
| Cell hangs forever | `get_iopub_msg(timeout=120)` raises | Return AnalysisResult with error, kernel still alive |
| Memory OOM (kernel killed by OS) | iopub never returns idle | Timeout → `_cleanup()` → restart |
| Syntax error in cell | `error` message in iopub | Capture ename/evalue, kernel stays alive |
| Kernel startup failure | `wait_for_ready()` raises RuntimeError | Propagate as `CodeExecutionError` |

---

## Testing the Jupyter Bridge

```python
# tests/test_jupyter_bridge.py
import pytest
from app.infrastructure.jupyter_bridge import JupyterKernelManager

@pytest.fixture
def manager():
    m = JupyterKernelManager()
    yield m
    m.shutdown_all()

def test_basic_execution(manager):
    result = manager.execute_cell("test-session", "x = 42; print(x)")
    assert result.success
    assert "42" in result.stdout

def test_state_persistence(manager):
    manager.execute_cell("test-session", "import numpy as np; arr = np.arange(10)")
    result = manager.execute_cell("test-session", "print(arr.sum())")
    assert "45" in result.stdout   # state persisted across cells

def test_figure_capture(manager):
    code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()
"""
    result = manager.execute_cell("test-session-fig", code)
    assert result.success
    assert len(result.figures) > 0

def test_error_handling(manager):
    result = manager.execute_cell("test-session-err", "1/0")
    assert not result.success
    assert "ZeroDivisionError" in result.stderr

def test_notebook_export(tmp_path, manager):
    manager.execute_cell("nb-session", "x = 1 + 1")
    manager.execute_cell("nb-session", "print(x)")
    path = manager.export_notebook("nb-session", "Test Analysis", tmp_path)
    assert path.exists()
    assert path.suffix == ".ipynb"
```
