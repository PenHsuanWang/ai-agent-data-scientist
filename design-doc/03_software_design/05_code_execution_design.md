# Code Execution Infrastructure — Three Backends

**Document Version:** 1.0  
**Status:** Approved  
**Bounded Context:** Infrastructure — Code Execution and State Management  

---

## 1. Interface Design

All three code execution backends implement a common abstract interface. The service layer and tools only depend on this interface — they never import a concrete backend directly.

```python
# infrastructure/code_runner.py
from __future__ import annotations

import abc
from app.domain.analysis_models import AnalysisResult


class CodeRunner(abc.ABC):
    """
    Abstract interface for code execution backends.

    Contract:
    - execute() NEVER raises exceptions — it returns AnalysisResult with success=False on error
    - get_state() returns variable names and type strings (no actual values — safe to serialize)
    - get_figure_b64() returns None if figure_id is unknown
    - Session state (variables, figures) persists across execute() calls on the same instance
    """

    @abc.abstractmethod
    def execute(self, code: str) -> AnalysisResult:
        """
        Executes Python code and returns the result.
        Never raises. On infrastructure failure, returns AnalysisResult(success=False).
        """
        ...

    @abc.abstractmethod
    def get_state(self) -> dict[str, str]:
        """
        Returns current variable namespace as {name: type_name}.
        Used by get_execution_variables() tool.
        """
        ...

    @abc.abstractmethod
    def get_figure_b64(self, figure_id: str) -> str | None:
        """
        Returns base64-encoded PNG for a figure generated during execution.
        Returns None if figure_id is not found.
        """
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        """
        Releases all resources held by this runner.
        Called when the AnalysisSession expires or is deleted.
        Idempotent — safe to call multiple times.
        """
        ...
```

### Factory Pattern

```python
class CodeRunnerFactory:
    """Creates the appropriate CodeRunner based on configuration."""

    @staticmethod
    def create(backend: str, session_id: str) -> CodeRunner:
        """
        Factory method. backend must be one of:
        - "subprocess"  → SubprocessCodeRunner (default, safest)
        - "jupyter"     → JupyterKernelManager (richest, native notebooks)
        - "anthropic"   → AnthropicCodeExecRunner (hosted, no local execution)

        Raises ValueError for unknown backends.
        """
        if backend == "subprocess":
            return SubprocessCodeRunner(session_id=session_id)
        elif backend == "jupyter":
            from app.infrastructure.jupyter_bridge import JupyterKernelManager
            return JupyterKernelManager(session_id=session_id)
        elif backend == "anthropic":
            from app.infrastructure.anthropic_code_exec import AnthropicCodeExecRunner
            return AnthropicCodeExecRunner(session_id=session_id)
        else:
            raise ValueError(
                f"Unknown code execution backend: {backend!r}. "
                f"Choose 'subprocess', 'jupyter', or 'anthropic'."
            )
```

---

## 2. Backend Comparison Table

| Dimension | SubprocessCodeRunner | JupyterKernelManager | AnthropicCodeExecRunner |
|---|---|---|---|
| **State persistence** | Pickle-based between calls | True Python kernel state | Anthropic container session |
| **Isolation** | OS process boundary | OS process boundary | Anthropic cloud sandbox |
| **Figure capture** | Agg backend + file injection | display_data iopub messages | Embedded in response |
| **Notebook export** | Rebuilt from jupyter_cells | Native nbformat | Not directly supported |
| **Startup latency** | ~50ms | ~1–3s (kernel boot) | ~200ms (first request) |
| **Resource control** | setrlimit (Unix) | Kernel restart on OOM | Anthropic-managed |
| **Available on Bedrock/Vertex** | ✅ Yes | ✅ Yes | ❌ No |
| **Requires local Python** | ✅ Yes (same env) | ✅ Yes (kernel) | ❌ No |
| **Security model** | AST check + subprocess | Kernel isolation | Anthropic sandbox |
| **Best for** | Development, testing | Production data science | Cloud-only deployments |

---

## 3. Backend Selection Logic

```python
# core/config.py (extended)
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Existing fields (unchanged)
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192

    # New fields with defaults
    code_execution_backend: str = "subprocess"       # "subprocess" | "jupyter" | "anthropic"
    datasets_dir: Path = Path("./datasets")
    domain_docs_dir: Path = Path("./docs")
    figures_dir: Path = Path("./figures")
    notebooks_dir: Path = Path("./notebooks")
    max_react_iterations: int = 20
    max_code_output_bytes: int = 102_400              # 100KB
    code_execution_timeout_seconds: int = 30
    jupyter_kernel_timeout_seconds: int = 3600        # 1 hour idle before cleanup

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

```
# .env (local dev)
ANTHROPIC_API_KEY=sk-ant-...
CODE_EXECUTION_BACKEND=subprocess

# .env (production with Jupyter)
ANTHROPIC_API_KEY=sk-ant-...
CODE_EXECUTION_BACKEND=jupyter
JUPYTER_KERNEL_TIMEOUT_SECONDS=3600
```

---

## 4. SubprocessCodeRunner Design

### 4.1 Security Model

Before any code is executed, a static AST analysis pass rejects dangerous patterns:

```python
import ast
from app.domain.exceptions import CodeSecurityError

_BANNED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "ctypes", "multiprocessing", "threading",
    "__builtins__",
})

_BANNED_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "open",
    "input", "breakpoint",
})

class CodeSecurityAnalyzer(ast.NodeVisitor):
    """AST-based code security checker. Rejects banned imports and calls."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BANNED_MODULES:
                self.violations.append(f"Banned import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in _BANNED_MODULES:
                self.violations.append(f"Banned import from: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_BUILTINS:
            self.violations.append(f"Banned builtin call: '{node.func.id}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "__class__" and isinstance(node.ctx, ast.Load):
            self.violations.append("Accessing __class__ attribute is restricted")
        self.generic_visit(node)


def _check_code_security(code: str) -> None:
    """Raises CodeSecurityError if code contains banned patterns."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeSecurityError(f"Syntax error in code: {e}")

    analyzer = CodeSecurityAnalyzer()
    analyzer.visit(tree)

    if analyzer.violations:
        raise CodeSecurityError(
            "Security violations detected:\n" + "\n".join(f"  - {v}" for v in analyzer.violations)
        )
```

### 4.2 Process Isolation

```python
# The subprocess receives code via stdin and writes results to stdout as JSON
# A wrapper script is injected around user code to capture figures and variables

_WRAPPER_TEMPLATE = '''
import sys
import json
import time
import traceback
import base64
import pickle
from pathlib import Path

# Load persisted state from previous call
_STATE_FILE = Path({state_file!r})
_namespace = {{}}
if _STATE_FILE.exists():
    try:
        with open(_STATE_FILE, "rb") as f:
            _namespace = pickle.load(f)
    except Exception:
        _namespace = {{}}

# Set up matplotlib before any user code
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.close("all")

# Execute user code
_stdout_lines = []
_stderr_lines = []
_figures = []
_success = True
_start = time.monotonic()

import io, contextlib
_stdout_buf = io.StringIO()
_stderr_buf = io.StringIO()

try:
    with contextlib.redirect_stdout(_stdout_buf), contextlib.redirect_stderr(_stderr_buf):
        exec(compile({code!r}, "<agent_code>", "exec"), _namespace)
    
    # Capture figures
    for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
        fig_id = f"fig_{{len(_figures):03d}}"
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        _figures.append({{
            "id": fig_id,
            "b64": base64.b64encode(buf.getvalue()).decode(),
        }})
        plt.close(fig)

except Exception as e:
    _success = False
    _stderr_lines.append(traceback.format_exc())

_elapsed = (time.monotonic() - _start) * 1000

# Extract variable names (skip private and module-level builtins)
_var_names = [
    k for k, v in _namespace.items()
    if not k.startswith("_") and not callable(v) or isinstance(v, (int, float, str, list, dict))
]

# Persist state for next call
try:
    _pickle_ns = {{k: v for k, v in _namespace.items() if not k.startswith("_")}}
    with open(_STATE_FILE, "wb") as f:
        pickle.dump(_pickle_ns, f, protocol=pickle.HIGHEST_PROTOCOL)
except Exception:
    pass

# Output result as JSON
print(json.dumps({{
    "stdout": _stdout_buf.getvalue()[:102400],
    "stderr": _stderr_buf.getvalue()[:102400] + "\\n".join(_stderr_lines),
    "figures": _figures,
    "variables_defined": _var_names,
    "success": _success,
}}))
'''
```

### 4.3 State Persistence via Pickle

```python
class SubprocessCodeRunner(CodeRunner):
    """
    Executes code in an isolated subprocess. State is persisted between
    calls via pickle serialization to a session-scoped temp file.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        # Use a session-specific directory in the project's data dir (NOT /tmp)
        self._state_dir = settings.figures_dir.parent / "runner_state" / session_id
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "state.pkl"
        self._figures: dict[str, str] = {}   # figure_id → base64
        self._variables: dict[str, str] = {} # name → type

    def execute(self, code: str) -> AnalysisResult:
        import json
        import subprocess
        import time
        from app.domain.analysis_models import AnalysisResult
        from app.domain.exceptions import CodeSecurityError, CodeTimeoutError

        # Security check
        try:
            _check_code_security(code)
        except CodeSecurityError as e:
            return AnalysisResult(
                code=code, stdout="", stderr=str(e),
                figures=[], variables_defined=[],
                execution_time_ms=0.0, success=False,
            )

        # Build wrapper
        wrapper = _WRAPPER_TEMPLATE.format(
            state_file=str(self._state_file),
            code=code,
        )

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["python", "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=settings.code_execution_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return AnalysisResult(
                code=code,
                stdout="",
                stderr=f"Execution timed out after {settings.code_execution_timeout_seconds}s",
                figures=[],
                variables_defined=[],
                execution_time_ms=elapsed,
                success=False,
            )
        except Exception as e:
            return AnalysisResult(
                code=code, stdout="", stderr=f"Subprocess error: {e}",
                figures=[], variables_defined=[],
                execution_time_ms=(time.monotonic() - start) * 1000,
                success=False,
            )

        elapsed = (time.monotonic() - start) * 1000

        # Parse JSON output from subprocess stdout
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return AnalysisResult(
                code=code,
                stdout=result.stdout[:1000],
                stderr=result.stderr or "Failed to parse runner output",
                figures=[], variables_defined=[],
                execution_time_ms=elapsed,
                success=False,
            )

        # Register figures
        fig_ids = []
        for fig_data in data.get("figures", []):
            fig_id = fig_data["id"]
            self._figures[fig_id] = fig_data["b64"]
            fig_ids.append(fig_id)

        # Update variable tracking
        self._variables = {
            name: "unknown" for name in data.get("variables_defined", [])
        }

        return AnalysisResult(
            code=code,
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            figures=fig_ids,
            variables_defined=data.get("variables_defined", []),
            execution_time_ms=elapsed,
            success=data.get("success", False),
        )

    def get_state(self) -> dict[str, str]:
        return dict(self._variables)

    def get_figure_b64(self, figure_id: str) -> str | None:
        return self._figures.get(figure_id)

    def shutdown(self) -> None:
        import shutil
        if self._state_dir.exists():
            shutil.rmtree(self._state_dir, ignore_errors=True)
        self._figures.clear()
        self._variables.clear()
```

---

## 5. JupyterKernelManager Design

(Full detail in File 7 — `07_jupyter_bridge_design.md`)

### 5.1 Summary

`JupyterKernelManager` implements `CodeRunner` using `jupyter_client`. Key differences from SubprocessCodeRunner:

- True Python REPL state (no pickle serialization needed)
- Native figure capture via `display_data` iopub messages with `image/png`
- `jupyter_cells` appended on each execution → enables `export_notebook()`
- One kernel process per `AnalysisSession`
- Kernel lifecycle: start on first `execute()`, shutdown on session expiry

```python
class JupyterKernelManager(CodeRunner):
    """CodeRunner backed by a live Jupyter kernel."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._km: "jupyter_client.KernelManager | None" = None
        self._kc: "jupyter_client.BlockingKernelClient | None" = None
        self._figures: dict[str, str] = {}
        self._variables: dict[str, str] = {}

    def execute(self, code: str) -> AnalysisResult:
        self._ensure_kernel_started()
        # See full implementation in 07_jupyter_bridge_design.md
        ...

    def get_state(self) -> dict[str, str]:
        return dict(self._variables)

    def get_figure_b64(self, figure_id: str) -> str | None:
        return self._figures.get(figure_id)

    def shutdown(self) -> None:
        if self._km and self._km.is_alive():
            self._km.shutdown_kernel(now=True)
        self._km = None
        self._kc = None
```

---

## 6. AnthropicCodeExecRunner Design

### 6.1 How Anthropic Code Execution Works

Anthropic provides a hosted code execution sandbox via the `computer_use` toolset. As of the `code_execution_20260120` server tool, Claude can execute Python in an Anthropic-managed container.

**Key facts:**
- Available only via `api.anthropic.com` — NOT on AWS Bedrock or Google Vertex AI
- The container session persists within a single `messages.create()` call
- Files uploaded via the Files API are available in the container's filesystem
- Results are returned in `server_tool_use` response content blocks

### 6.2 Files API Integration

```python
class AnthropicCodeExecRunner(CodeRunner):
    """
    CodeRunner that delegates execution to Anthropic's hosted code execution.
    
    NOTE: Not available on Bedrock or Vertex. Requires direct Anthropic API.
    """

    def __init__(self, session_id: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._session_id = session_id
        self._figures: dict[str, str] = {}
        self._variables: dict[str, str] = {}
        self._uploaded_files: dict[str, str] = {}  # local_path → file_id
        self._cell_outputs: list[dict] = []

    def _upload_dataset(self, path: "Path") -> str:
        """Uploads a dataset file to Anthropic Files API. Returns file_id."""
        path_str = str(path)
        if path_str in self._uploaded_files:
            return self._uploaded_files[path_str]
        
        with open(path, "rb") as f:
            response = self._client.beta.files.upload(
                file=(path.name, f, "text/csv"),
            )
        file_id = response.id
        self._uploaded_files[path_str] = file_id
        return file_id

    def execute(self, code: str) -> AnalysisResult:
        """
        Sends code to Anthropic's hosted execution environment.
        Extracts results from server_tool_use response blocks.
        """
        import time
        from app.domain.analysis_models import AnalysisResult

        start = time.monotonic()
        
        try:
            response = self._client.beta.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_tokens,
                tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Execute this Python code and return the output:\n```python\n{code}\n```"
                        ),
                    }
                ],
                betas=["code-execution-2025-05-22"],
            )
        except Exception as e:
            return AnalysisResult(
                code=code, stdout="", stderr=f"Anthropic API error: {e}",
                figures=[], variables_defined=[],
                execution_time_ms=(time.monotonic() - start) * 1000,
                success=False,
            )

        elapsed = (time.monotonic() - start) * 1000

        # Extract output from response content blocks
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        fig_ids: list[str] = []
        success = True

        for block in response.content:
            block_type = getattr(block, "type", "")
            
            if block_type == "server_tool_use":
                # Code was sent for execution
                pass
            elif block_type == "server_tool_result":
                # Execution result
                content = getattr(block, "content", [])
                for item in content:
                    item_type = getattr(item, "type", "")
                    if item_type == "text":
                        stdout_parts.append(item.text)
                    elif item_type == "image":
                        # base64-encoded figure
                        fig_id = f"fig_{len(self._figures):03d}"
                        self._figures[fig_id] = item.data
                        fig_ids.append(fig_id)
            elif block_type == "text":
                # Claude's commentary (not code output)
                pass

        stdout = "\n".join(stdout_parts)

        return AnalysisResult(
            code=code,
            stdout=stdout,
            stderr="\n".join(stderr_parts),
            figures=fig_ids,
            variables_defined=[],  # Anthropic runner cannot introspect variables
            execution_time_ms=elapsed,
            success=success,
        )

    def get_state(self) -> dict[str, str]:
        # Anthropic runner cannot enumerate variables between separate API calls
        return {"note": "Variable introspection not available with Anthropic backend"}

    def get_figure_b64(self, figure_id: str) -> str | None:
        return self._figures.get(figure_id)

    def shutdown(self) -> None:
        self._figures.clear()
        self._uploaded_files.clear()
```

---

## 7. Security Considerations

### SubprocessCodeRunner Security Layers

1. **AST Pre-check** (before any execution): Blocks `os`, `sys`, `subprocess`, `socket`, `open()`, `exec()`, `eval()`, `__import__`
2. **Process isolation**: Code runs in a separate OS process. It cannot access the parent process memory.
3. **Timeout**: `subprocess.run(..., timeout=30)` kills the child process after 30 seconds
4. **No network access** (enforced by blocking `socket` import)
5. **Limited filesystem access** (enforced by blocking `open`, `os`, `pathlib`)

### JupyterKernelManager Security Layers

1. **Kernel isolation**: Each session has its own kernel process
2. **No AST pre-check** (trusted user assumption for production; add if needed)
3. **Kernel timeout**: Idle kernels shut down after `jupyter_kernel_timeout_seconds`
4. **Resource limits**: System-level cgroups (production) or ulimit (development)

### AnthropicCodeExecRunner Security

Anthropic manages container isolation. No local security policy needed.

---

## 8. Error Handling Per Backend

```
┌─────────────────────────────────────────────────────────────────────┐
│  Error Type         │ Subprocess         │ Jupyter          │ Anthropic │
├─────────────────────┼────────────────────┼──────────────────┼───────────┤
│ Syntax error        │ Captured in stderr │ Captured in      │ In stdout │
│                     │ via traceback      │ "error" message  │           │
├─────────────────────┼────────────────────┼──────────────────┼───────────┤
│ Runtime exception   │ Captured in stderr │ Captured in      │ In stdout │
│                     │ via traceback      │ "error" message  │           │
├─────────────────────┼────────────────────┼──────────────────┼───────────┤
│ Timeout             │ success=False,     │ Kernel restart,  │ API error │
│                     │ stderr="timed out" │ success=False    │ exception │
├─────────────────────┼────────────────────┼──────────────────┼───────────┤
│ OOM / memory        │ Process killed     │ Kernel dies,     │ Anthropic │
│                     │ by OS, timeout     │ restart policy   │ managed   │
├─────────────────────┼────────────────────┼──────────────────┼───────────┤
│ Banned import       │ CodeSecurityError  │ N/A (no AST      │ N/A       │
│                     │ → success=False    │ check)           │           │
├─────────────────────┼────────────────────┼──────────────────┼───────────┤
│ Pickle failure      │ State lost,        │ N/A              │ N/A       │
│                     │ logged to stderr   │                  │           │
└─────────────────────┴────────────────────┴──────────────────┴───────────┘
```

All errors ultimately result in `AnalysisResult(success=False, stderr="...")` — the tool function wraps this as `json.dumps(result.to_json_dict())` and the ReAct loop sees it as an Observation. Claude can then decide to retry with different code or report the error.

---

## 9. Complete code_runner.py

```python
# infrastructure/code_runner.py
"""
Abstract CodeRunner interface and SubprocessCodeRunner implementation.
"""
from __future__ import annotations

import abc
import ast
import base64
import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import settings
from app.domain.analysis_models import AnalysisResult
from app.domain.exceptions import CodeSecurityError

if TYPE_CHECKING:
    pass

# ── Security ─────────────────────────────────────────────────────────────────

_BANNED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil",
    "importlib", "ctypes", "multiprocessing", "threading",
})

_BANNED_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "open", "input", "breakpoint",
})


class _SecurityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in _BANNED_MODULES:
                self.violations.append(f"Banned import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in _BANNED_MODULES:
            self.violations.append(f"Banned import from: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_BUILTINS:
            self.violations.append(f"Banned call: '{node.func.id}()'")
        self.generic_visit(node)


def _check_security(code: str) -> list[str]:
    """Returns list of violation strings. Empty list means code is clean."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]
    visitor = _SecurityVisitor()
    visitor.visit(tree)
    return visitor.violations


# ── Abstract Base ────────────────────────────────────────────────────────────

class CodeRunner(abc.ABC):
    @abc.abstractmethod
    def execute(self, code: str) -> AnalysisResult: ...

    @abc.abstractmethod
    def get_state(self) -> dict[str, str]: ...

    @abc.abstractmethod
    def get_figure_b64(self, figure_id: str) -> str | None: ...

    @abc.abstractmethod
    def shutdown(self) -> None: ...


# ── SubprocessCodeRunner ──────────────────────────────────────────────────────

_WRAPPER = '''\
import sys, json, time, traceback, base64, pickle, io, contextlib
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.close("all")

_sf = Path({state_file!r})
_ns = {{}}
if _sf.exists():
    try:
        with open(_sf, "rb") as _f: _ns = pickle.load(_f)
    except Exception: _ns = {{}}

_ob, _eb = io.StringIO(), io.StringIO()
_figs, _success = [], True
_t0 = time.monotonic()
try:
    with contextlib.redirect_stdout(_ob), contextlib.redirect_stderr(_eb):
        exec(compile({code!r}, "<agent>", "exec"), _ns)
    for _fn in plt.get_fignums():
        _fig = plt.figure(_fn)
        _buf = io.BytesIO()
        _fig.savefig(_buf, format="png", bbox_inches="tight", dpi=100)
        _id = f"fig_{{len(_figs):03d}}"
        _figs.append({{"id": _id, "b64": base64.b64encode(_buf.getvalue()).decode()}})
        plt.close(_fig)
except Exception:
    _success = False
    _eb.write(traceback.format_exc())
_elapsed = (time.monotonic() - _t0) * 1000
_vars = [k for k in _ns if not k.startswith("_")]
try:
    with open(_sf, "wb") as _f: pickle.dump({{k: v for k, v in _ns.items() if not k.startswith("_")}}, _f)
except Exception: pass
print(json.dumps({{"stdout": _ob.getvalue()[:102400], "stderr": _eb.getvalue()[:102400], "figures": _figs, "variables_defined": _vars, "success": _success, "elapsed_ms": _elapsed}}))
'''


class SubprocessCodeRunner(CodeRunner):
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._state_dir = settings.figures_dir.parent / "runner_state" / session_id
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "state.pkl"
        self._figures: dict[str, str] = {}
        self._variables: dict[str, str] = {}

    def execute(self, code: str) -> AnalysisResult:
        violations = _check_security(code)
        if violations:
            return AnalysisResult(
                code=code, stdout="",
                stderr="Security check failed:\n" + "\n".join(f"  - {v}" for v in violations),
                figures=[], variables_defined=[], execution_time_ms=0.0, success=False,
            )

        wrapper = _WRAPPER.format(state_file=str(self._state_file), code=code)
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                ["python", "-c", wrapper],
                capture_output=True, text=True,
                timeout=settings.code_execution_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return AnalysisResult(
                code=code, stdout="",
                stderr=f"Timed out after {settings.code_execution_timeout_seconds}s",
                figures=[], variables_defined=[],
                execution_time_ms=(time.monotonic() - t0) * 1000, success=False,
            )
        except Exception as e:
            return AnalysisResult(
                code=code, stdout="", stderr=f"Subprocess launch error: {e}",
                figures=[], variables_defined=[],
                execution_time_ms=(time.monotonic() - t0) * 1000, success=False,
            )

        elapsed = (time.monotonic() - t0) * 1000

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return AnalysisResult(
                code=code, stdout=proc.stdout[:500],
                stderr=proc.stderr or "Could not parse runner output",
                figures=[], variables_defined=[], execution_time_ms=elapsed, success=False,
            )

        fig_ids = []
        for fig in data.get("figures", []):
            self._figures[fig["id"]] = fig["b64"]
            fig_ids.append(fig["id"])

        self._variables = {k: "object" for k in data.get("variables_defined", [])}

        return AnalysisResult(
            code=code,
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            figures=fig_ids,
            variables_defined=data.get("variables_defined", []),
            execution_time_ms=elapsed,
            success=data.get("success", False),
        )

    def get_state(self) -> dict[str, str]:
        return dict(self._variables)

    def get_figure_b64(self, figure_id: str) -> str | None:
        return self._figures.get(figure_id)

    def shutdown(self) -> None:
        import shutil
        shutil.rmtree(self._state_dir, ignore_errors=True)
        self._figures.clear()
        self._variables.clear()


# ── Factory ──────────────────────────────────────────────────────────────────

class CodeRunnerFactory:
    @staticmethod
    def create(backend: str, session_id: str) -> CodeRunner:
        if backend == "subprocess":
            return SubprocessCodeRunner(session_id=session_id)
        elif backend == "jupyter":
            from app.infrastructure.jupyter_bridge import JupyterKernelManager
            return JupyterKernelManager(session_id=session_id)
        elif backend == "anthropic":
            from app.infrastructure.anthropic_code_exec import AnthropicCodeExecRunner
            return AnthropicCodeExecRunner(session_id=session_id)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")
```
