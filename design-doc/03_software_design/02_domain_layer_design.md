# Domain Layer Design — Entities and Value Objects

**Document Version:** 1.0  
**Status:** Approved  
**Bounded Context:** Core Domain — Data Analysis Session Management  

---

## 1. Design Principles

The domain layer is the innermost ring of the architecture. It must be protected from infrastructure concerns.

**Principle 1: Zero External Dependencies**  
`domain/models.py` and `domain/analysis_models.py` import only Python stdlib. No `pandas`, no `anthropic`, no `pint`. If a domain object requires external computation, it accepts pre-computed values as constructor arguments.

**Principle 2: Pure Python, No Framework Leakage**  
Domain entities are plain `@dataclass` objects. No Pydantic `BaseModel` (that is the API layer's concern), no SQLAlchemy models, no FastAPI dependencies.

**Principle 3: Testable Without Mocks**  
Every domain object can be instantiated and its methods exercised in a single `pytest` function with no external services running. This guarantees that business logic is isolated and regression-safe.

**Principle 4: Immutable Value Objects**  
`DatasetMeta`, `AnalysisResult`, and `PhysicalUnit` are frozen dataclasses. Once created, they cannot be mutated. Only `AgentSession` and `AnalysisSession` are mutable aggregates.

**Principle 5: Explicit Invariants**  
Business rules are enforced in `__post_init__` methods and documented as docstring invariants. No silent corruption of state.

---

## 2. Existing Entities (from MVP — Unchanged)

### 2.1 AgentSession

`AgentSession` is the aggregate root for a single conversation. It owns the mutable message list that is fed to the Claude API.

```
AgentSession
├── session_id: str          (UUID4, immutable identifier)
└── messages: list[dict]     (mutable, ordered list of Claude API messages)
```

**Invariants:**
- `messages` alternates roles: `user` → `assistant` → `user` → ... (with tool_result blocks)
- `session_id` is never empty
- Message dicts conform to the Anthropic API `messages` parameter schema

**Existing Methods (reference, do not modify):**

```python
@dataclass
class AgentSession:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Appends a user-role message with a text content block."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: Any) -> None:
        """Appends an assistant-role message. Content may be str or list of blocks."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, tool_results: list[dict]) -> None:
        """Appends a user-role message containing tool_result blocks."""
        self.messages.append({"role": "user", "content": tool_results})
```

---

## 3. New Entity: AnalysisSession

### 3.1 Design Rationale

`AnalysisSession` extends `AgentSession` (inheritance, not composition) because:

1. The Claude API message history (`self.messages`) must be present on every analysis session — it is not optional.
2. All methods defined on `AgentSession` (`add_user_message`, `add_assistant_message`, `add_tool_results`) are directly needed by the ReAct loop in `DataScienceAgentService`.
3. `InMemorySessionStore` already manages `AgentSession` objects; `AnalysisSession` is a proper subtype (Liskov substitution holds — every `AnalysisSession` IS-A `AgentSession`).
4. Composition would require duplicating the `AgentSession` field and forwarding all message methods, providing no benefit.

**Why NOT Pydantic:** `AnalysisSession` is mutated in-place throughout the ReAct loop (figures added, react_trace appended). Pydantic's immutability semantics do not fit this lifecycle.

### 3.2 Full Class Definition

```python
@dataclass
class AnalysisSession(AgentSession):
    """
    Aggregate root for a Data Scientist Agent conversation.

    Extends AgentSession with:
    - Loaded dataset metadata (name → DatasetMeta)
    - Generated figure registry (figure_id → base64 PNG)
    - Jupyter notebook cell history
    - Physical unit validation log
    - ReAct reasoning trace

    Invariants:
    - session_id inherited from AgentSession (non-empty string)
    - datasets_loaded keys are file basenames without path traversal characters
    - figures values are valid base64-encoded PNG strings
    - react_trace is append-only (entries are never removed)
    - code_runner_state is managed exclusively by the active CodeRunner
    """

    datasets_loaded: dict[str, DatasetMeta] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)          # figure_id → base64 PNG
    jupyter_cells: list[dict] = field(default_factory=list)        # [{source, outputs, ...}]
    unit_context: list[PhysicalUnit] = field(default_factory=list) # ordered validation log
    react_trace: list[dict] = field(default_factory=list)          # {thought, action, observation}
    code_runner_state: dict = field(default_factory=dict)          # opaque to domain layer
```

### 3.3 Mutation Methods to Add

These methods enforce safe mutation semantics and keep business rules centralized:

```python
    def register_dataset(self, meta: DatasetMeta) -> None:
        """
        Registers a loaded dataset in this session.
        
        Raises ValueError if file_name contains path traversal characters.
        """
        if ".." in meta.file_name or "/" in meta.file_name:
            raise ValueError(f"Unsafe file_name in DatasetMeta: {meta.file_name!r}")
        self.datasets_loaded[meta.file_name] = meta

    def add_figure(self, figure_id: str, base64_png: str) -> None:
        """
        Stores a generated figure. figure_id must be a non-empty string.
        base64_png must be a non-empty string (validity not checked here — 
        infrastructure layer is responsible for correct encoding).
        """
        if not figure_id:
            raise ValueError("figure_id cannot be empty")
        if not base64_png:
            raise ValueError("base64_png cannot be empty")
        self.figures[figure_id] = base64_png

    def append_react_step(self, thought: str, action: str, observation: str) -> None:
        """
        Appends one complete ReAct iteration to the trace.
        All three parts are required; empty strings are acceptable for 
        actions that produce no observable output.
        """
        self.react_trace.append({
            "thought": thought,
            "action": action,
            "observation": observation,
        })

    def log_unit_validation(self, unit: PhysicalUnit) -> None:
        """Appends a PhysicalUnit validation result to the session log."""
        self.unit_context.append(unit)

    def add_jupyter_cell(self, source: str, outputs: list[dict]) -> None:
        """
        Appends one executed Jupyter cell to the notebook history.
        Used by JupyterKernelManager when building the exported notebook.
        """
        self.jupyter_cells.append({
            "cell_type": "code",
            "source": source,
            "outputs": outputs,
            "execution_count": len(self.jupyter_cells) + 1,
        })

    def get_dataset(self, file_name: str) -> "DatasetMeta":
        """
        Returns DatasetMeta for file_name.
        Raises DatasetNotFoundError if not loaded.
        """
        from app.domain.exceptions import DatasetNotFoundError
        if file_name not in self.datasets_loaded:
            raise DatasetNotFoundError(file_name)
        return self.datasets_loaded[file_name]
```

---

## 4. New Value Object: DatasetMeta

### 4.1 Design Rationale

`DatasetMeta` is a **frozen value object** (immutable snapshot) because:
- It is a description of a file at the time of inspection, not a live object
- Multiple parts of the system read it concurrently; immutability prevents races
- Equality by value is correct: two `DatasetMeta` for the same file are equivalent

### 4.2 Full Class Definition

```python
@dataclass(frozen=True)
class DatasetMeta:
    """
    Immutable snapshot of a dataset's structure and statistics.

    Created once when inspect_dataset() is called. Not updated 
    if the underlying file changes — call inspect_dataset() again 
    to refresh.

    Fields:
        file_name:       Basename of the file (e.g., "power_plant.csv")
        path:            Absolute Path to the file on disk
        shape:           (n_rows, n_cols) tuple
        dtypes:          Column name → pandas dtype string (e.g., "float64", "object")
        sample_rows:     First 5 rows as list of {col: value} dicts (JSON-serializable)
        null_counts:     Column name → count of null/NaN values
        file_size_bytes: File size in bytes at inspection time
    """
    file_name: str
    path: Path
    shape: tuple[int, int]
    dtypes: dict[str, str]
    sample_rows: list[dict]
    null_counts: dict[str, int]
    file_size_bytes: int

    def __post_init__(self) -> None:
        if not self.file_name:
            raise ValueError("DatasetMeta.file_name cannot be empty")
        if self.file_size_bytes < 0:
            raise ValueError("DatasetMeta.file_size_bytes cannot be negative")
        if len(self.shape) != 2:
            raise ValueError("DatasetMeta.shape must be a 2-tuple (rows, cols)")
```

### 4.3 Factory Method (lives in knowledge_tools.py, not domain)

```python
# In services/knowledge_tools.py — NOT in domain layer
def _dataset_meta_from_df(file_name: str, path: Path, df_head: pd.DataFrame, total_shape: tuple[int, int]) -> DatasetMeta:
    """
    Builds DatasetMeta from a pandas DataFrame head sample.
    total_shape must be passed separately (computed cheaply with len(df) 
    after full load, or estimated for large files).
    """
    return DatasetMeta(
        file_name=file_name,
        path=path,
        shape=total_shape,
        dtypes={col: str(dtype) for col, dtype in df_head.dtypes.items()},
        sample_rows=df_head.head(5).to_dict(orient="records"),
        null_counts={col: int(df_head[col].isna().sum()) for col in df_head.columns},
        file_size_bytes=path.stat().st_size,
    )
```

### 4.4 JSON Serialization Helper

```python
    def to_json_dict(self) -> dict:
        """Returns a JSON-serializable dict (safe for tool return strings)."""
        return {
            "file_name": self.file_name,
            "path": str(self.path),
            "shape": list(self.shape),
            "dtypes": self.dtypes,
            "sample_rows": self.sample_rows,
            "null_counts": self.null_counts,
            "file_size_bytes": self.file_size_bytes,
        }
```

---

## 5. New Value Object: AnalysisResult

### 5.1 Design Rationale

`AnalysisResult` captures the complete outcome of a single code execution. It is frozen because it is a historical record — once code runs, the result does not change. Multiple systems consume it (API layer, ReAct loop, notebook exporter).

### 5.2 Full Class Definition

```python
@dataclass(frozen=True)
class AnalysisResult:
    """
    Immutable record of one code execution.

    Fields:
        code:                The exact Python code that was executed
        stdout:              Text written to stdout during execution (max 100KB)
        stderr:              Text written to stderr (warnings, tracebacks)
        figures:             List of figure_ids registered during execution
        variables_defined:   Names of new top-level variables after execution
        execution_time_ms:   Wall-clock time in milliseconds
        success:             True if execution completed without unhandled exception
    
    Semantics:
        - success=True AND stderr non-empty: warnings occurred (non-fatal)
        - success=False: stderr contains the full traceback
        - figures non-empty: caller must retrieve each via get_figure()
    """
    code: str
    stdout: str
    stderr: str
    figures: list[str]
    variables_defined: list[str]
    execution_time_ms: float
    success: bool

    def __post_init__(self) -> None:
        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms cannot be negative")

    @property
    def has_output(self) -> bool:
        """True if execution produced any text output."""
        return bool(self.stdout.strip())

    @property
    def has_figures(self) -> bool:
        """True if any figures were generated."""
        return len(self.figures) > 0

    def to_json_dict(self) -> dict:
        return {
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "figures": self.figures,
            "variables_defined": self.variables_defined,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
        }
```

---

## 6. New Value Object: PhysicalUnit

### 6.1 Design Rationale

`PhysicalUnit` is a validation result, not a live unit object. It records: what was validated, whether it passed, and why. The `pint` library never appears in the domain layer — all parsing happens in `infrastructure/unit_registry.py`. The domain only sees the validated result.

### 6.2 Full Class Definition

```python
@dataclass(frozen=True)
class PhysicalUnit:
    """
    Record of a single physical quantity validation.

    Fields:
        quantity:         Human-readable quantity name (e.g., "thermal_efficiency")
        value:            The numeric value that was validated
        unit_str:         The unit string as provided (e.g., "%", "MW", "°C")
        reasonable_range: (min, max) inclusive range in the canonical unit
        is_valid:         True if value parsed, dimensions correct, magnitude in range
        message:          Human-readable explanation (reason for failure, or "OK")

    Semantics:
        - is_valid=True:  All three validation stages passed
        - is_valid=False, message starts with "Unit parse error":   Stage 1 failed
        - is_valid=False, message starts with "Dimensional error":  Stage 2 failed
        - is_valid=False, message starts with "Magnitude warning":  Stage 3 failed
    """
    quantity: str
    value: float
    unit_str: str
    reasonable_range: tuple[float, float]
    is_valid: bool
    message: str

    def __post_init__(self) -> None:
        if not self.quantity:
            raise ValueError("PhysicalUnit.quantity cannot be empty")
        if len(self.reasonable_range) != 2:
            raise ValueError("reasonable_range must be a 2-tuple (min, max)")
        if self.reasonable_range[0] > self.reasonable_range[1]:
            raise ValueError("reasonable_range[0] must be ≤ reasonable_range[1]")

    def to_json_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "value": self.value,
            "unit_str": self.unit_str,
            "reasonable_range": list(self.reasonable_range),
            "is_valid": self.is_valid,
            "message": self.message,
        }
```

---

## 7. Exception Hierarchy Extension

### 7.1 Inheritance Diagram

```
Exception
└── AgentError                          (EXISTING — base for all agent errors)
    ├── AgentLoopError                  (EXISTING — agentic loop failures)
    ├── ReActLoopError                  (NEW — ReAct-specific loop failures)
    │   ├── ReActParseError             (NEW — malformed Thought/Action text)
    │   └── ReActMaxIterationsError     (NEW — exceeded MAX_REACT_ITERATIONS)
    ├── CodeExecutionError              (NEW — code runner failures)
    │   ├── CodeTimeoutError            (NEW — subprocess/kernel timeout)
    │   └── CodeSecurityError           (NEW — banned import or AST violation)
    ├── UnitValidationError             (NEW — pint parsing failure)
    └── DatasetNotFoundError            (NEW — requested dataset not in session)
```

### 7.2 Complete Extended exceptions.py

```python
# domain/exceptions.py

class AgentError(Exception):
    """Base class for all agent errors. Existing — do not modify."""
    pass


class AgentLoopError(AgentError):
    """Raised when the MVP agentic loop fails. Existing — do not modify."""
    pass


# ── New exceptions below ──────────────────────────────────────────────────────

class ReActLoopError(AgentError):
    """
    Base class for errors occurring within the ReAct reasoning loop.
    Distinct from AgentLoopError to allow callers to handle them separately.
    """
    pass


class ReActParseError(ReActLoopError):
    """
    Raised when the LLM response cannot be parsed as a valid ReAct turn.

    This happens when Claude's output does not match the expected
    "Thought: ... Action: ... Action Input: ..." format.

    Attributes:
        raw_response: The unparsed string from Claude.
        iteration:    Which ReAct iteration produced the malformed response.
    """
    def __init__(self, raw_response: str, iteration: int) -> None:
        self.raw_response = raw_response
        self.iteration = iteration
        super().__init__(
            f"ReAct parse error at iteration {iteration}. "
            f"Response did not match expected format. "
            f"First 200 chars: {raw_response[:200]!r}"
        )


class ReActMaxIterationsError(ReActLoopError):
    """
    Raised when the ReAct loop reaches MAX_REACT_ITERATIONS without
    producing a "Final Answer:" block.

    Attributes:
        iterations: The number of iterations that were executed.
        last_thought: The last Thought extracted before giving up.
    """
    def __init__(self, iterations: int, last_thought: str = "") -> None:
        self.iterations = iterations
        self.last_thought = last_thought
        super().__init__(
            f"ReAct loop exceeded maximum of {iterations} iterations "
            f"without reaching a Final Answer."
        )


class CodeExecutionError(AgentError):
    """
    Raised when a code execution attempt fails at the infrastructure level
    (not a user-code error — those are captured in AnalysisResult.stderr).

    This is for infrastructure failures: backend crash, IPC timeout, 
    serialization failure.

    Attributes:
        backend: Which backend failed ("subprocess", "jupyter", "anthropic").
        cause:   The underlying exception, if any.
    """
    def __init__(self, message: str, backend: str, cause: Exception | None = None) -> None:
        self.backend = backend
        self.cause = cause
        super().__init__(f"[{backend}] {message}")


class CodeTimeoutError(CodeExecutionError):
    """
    Raised when code execution exceeds the configured timeout.

    Attributes:
        timeout_seconds: How long we waited before killing the process.
    """
    def __init__(self, timeout_seconds: int, backend: str) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Code execution timed out after {timeout_seconds}s",
            backend=backend,
        )


class CodeSecurityError(CodeExecutionError):
    """
    Raised when AST analysis detects a banned operation before execution.
    The code is never executed — this is a pre-execution check.

    Attributes:
        violation: Description of what was detected (e.g., "import os detected").
    """
    def __init__(self, violation: str) -> None:
        self.violation = violation
        super().__init__(
            f"Code security check failed: {violation}",
            backend="ast_checker",
        )


class UnitValidationError(AgentError):
    """
    Raised when pint cannot parse the provided unit string at all.
    Distinct from a failed validation (which returns PhysicalUnit(is_valid=False)).

    This exception means the validation infrastructure itself failed,
    not that the value was out of range.

    Attributes:
        unit_str: The unit string that could not be parsed.
    """
    def __init__(self, unit_str: str, cause: Exception | None = None) -> None:
        self.unit_str = unit_str
        self.cause = cause
        super().__init__(f"Cannot parse unit string: {unit_str!r}")


class DatasetNotFoundError(AgentError):
    """
    Raised when a tool or service requests a dataset that has not been
    loaded into the current AnalysisSession.

    Attributes:
        file_name: The requested dataset name.
    """
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        super().__init__(
            f"Dataset {file_name!r} is not loaded in this session. "
            f"Call inspect_dataset('{file_name}') first."
        )
```

---

## 8. Complete analysis_models.py

```python
# domain/analysis_models.py
"""
New domain entities and value objects for the Data Scientist Agent.

Design constraints:
- Zero imports from app.services, app.infrastructure, or any third-party library.
- Only stdlib: dataclasses, pathlib, typing.
- All value objects are frozen (immutable).
- AnalysisSession is the mutable aggregate root, extending AgentSession.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.domain.models import AgentSession

if TYPE_CHECKING:
    pass  # No runtime circular imports


@dataclass(frozen=True)
class DatasetMeta:
    """Immutable snapshot of a dataset at inspection time."""
    file_name: str
    path: Path
    shape: tuple[int, int]
    dtypes: dict[str, str]
    sample_rows: list[dict]
    null_counts: dict[str, int]
    file_size_bytes: int

    def __post_init__(self) -> None:
        if not self.file_name:
            raise ValueError("DatasetMeta.file_name cannot be empty")
        if self.file_size_bytes < 0:
            raise ValueError("DatasetMeta.file_size_bytes cannot be negative")
        if len(self.shape) != 2:
            raise ValueError("DatasetMeta.shape must be a 2-tuple (rows, cols)")

    def to_json_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "path": str(self.path),
            "shape": list(self.shape),
            "dtypes": self.dtypes,
            "sample_rows": self.sample_rows,
            "null_counts": self.null_counts,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Immutable record of one code execution."""
    code: str
    stdout: str
    stderr: str
    figures: list[str]
    variables_defined: list[str]
    execution_time_ms: float
    success: bool

    def __post_init__(self) -> None:
        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms cannot be negative")

    @property
    def has_output(self) -> bool:
        return bool(self.stdout.strip())

    @property
    def has_figures(self) -> bool:
        return len(self.figures) > 0

    def to_json_dict(self) -> dict:
        return {
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "figures": self.figures,
            "variables_defined": self.variables_defined,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
        }


@dataclass(frozen=True)
class PhysicalUnit:
    """Record of one physical quantity validation result."""
    quantity: str
    value: float
    unit_str: str
    reasonable_range: tuple[float, float]
    is_valid: bool
    message: str

    def __post_init__(self) -> None:
        if not self.quantity:
            raise ValueError("PhysicalUnit.quantity cannot be empty")
        if len(self.reasonable_range) != 2:
            raise ValueError("reasonable_range must be a 2-tuple (min, max)")
        if self.reasonable_range[0] > self.reasonable_range[1]:
            raise ValueError("reasonable_range min must be ≤ max")

    def to_json_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "value": self.value,
            "unit_str": self.unit_str,
            "reasonable_range": list(self.reasonable_range),
            "is_valid": self.is_valid,
            "message": self.message,
        }


@dataclass
class AnalysisSession(AgentSession):
    """
    Mutable aggregate root for a Data Scientist Agent conversation.
    Extends AgentSession (inherits session_id + messages).
    """
    datasets_loaded: dict[str, DatasetMeta] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    jupyter_cells: list[dict] = field(default_factory=list)
    unit_context: list[PhysicalUnit] = field(default_factory=list)
    react_trace: list[dict] = field(default_factory=list)
    code_runner_state: dict = field(default_factory=dict)

    def register_dataset(self, meta: DatasetMeta) -> None:
        if ".." in meta.file_name or "/" in meta.file_name:
            raise ValueError(f"Unsafe file_name: {meta.file_name!r}")
        self.datasets_loaded[meta.file_name] = meta

    def add_figure(self, figure_id: str, base64_png: str) -> None:
        if not figure_id:
            raise ValueError("figure_id cannot be empty")
        if not base64_png:
            raise ValueError("base64_png cannot be empty")
        self.figures[figure_id] = base64_png

    def append_react_step(self, thought: str, action: str, observation: str) -> None:
        self.react_trace.append({
            "thought": thought,
            "action": action,
            "observation": observation,
        })

    def log_unit_validation(self, unit: PhysicalUnit) -> None:
        self.unit_context.append(unit)

    def add_jupyter_cell(self, source: str, outputs: list[dict]) -> None:
        self.jupyter_cells.append({
            "cell_type": "code",
            "source": source,
            "outputs": outputs,
            "execution_count": len(self.jupyter_cells) + 1,
        })

    def get_dataset(self, file_name: str) -> DatasetMeta:
        from app.domain.exceptions import DatasetNotFoundError
        if file_name not in self.datasets_loaded:
            raise DatasetNotFoundError(file_name)
        return self.datasets_loaded[file_name]
```

---

## 9. Unit Test Examples

Every entity is tested without mocks or external services.

```python
# tests/domain/test_analysis_models.py
import pytest
from pathlib import Path
from app.domain.analysis_models import (
    DatasetMeta, AnalysisResult, PhysicalUnit, AnalysisSession
)
from app.domain.exceptions import DatasetNotFoundError


# ── DatasetMeta ──────────────────────────────────────────────────────────────

def test_dataset_meta_creation():
    meta = DatasetMeta(
        file_name="test.csv",
        path=Path("/data/test.csv"),
        shape=(100, 5),
        dtypes={"a": "float64", "b": "object"},
        sample_rows=[{"a": 1.0, "b": "x"}],
        null_counts={"a": 0, "b": 2},
        file_size_bytes=4096,
    )
    assert meta.file_name == "test.csv"
    assert meta.shape == (100, 5)


def test_dataset_meta_rejects_empty_name():
    with pytest.raises(ValueError, match="cannot be empty"):
        DatasetMeta("", Path("/"), (0, 0), {}, [], {}, 0)


def test_dataset_meta_rejects_negative_size():
    with pytest.raises(ValueError, match="negative"):
        DatasetMeta("x.csv", Path("/x.csv"), (1, 1), {}, [], {}, -1)


def test_dataset_meta_frozen():
    meta = DatasetMeta("f.csv", Path("/f.csv"), (1, 2), {}, [], {}, 100)
    with pytest.raises(Exception):  # FrozenInstanceError
        meta.file_name = "other.csv"  # type: ignore


def test_dataset_meta_to_json_dict():
    meta = DatasetMeta("t.csv", Path("/t.csv"), (3, 2), {"x": "int64"}, [], {"x": 0}, 512)
    d = meta.to_json_dict()
    assert d["shape"] == [3, 2]
    assert d["path"] == "/t.csv"


# ── AnalysisResult ───────────────────────────────────────────────────────────

def test_analysis_result_success():
    result = AnalysisResult(
        code="print('hello')",
        stdout="hello\n",
        stderr="",
        figures=[],
        variables_defined=[],
        execution_time_ms=12.5,
        success=True,
    )
    assert result.has_output is True
    assert result.has_figures is False


def test_analysis_result_with_figures():
    result = AnalysisResult("", "", "", ["fig_001"], [], 5.0, True)
    assert result.has_figures is True


def test_analysis_result_negative_time_rejected():
    with pytest.raises(ValueError, match="negative"):
        AnalysisResult("", "", "", [], [], -1.0, True)


# ── PhysicalUnit ─────────────────────────────────────────────────────────────

def test_physical_unit_valid():
    pu = PhysicalUnit(
        quantity="thermal_efficiency",
        value=0.42,
        unit_str="%",
        reasonable_range=(0.1, 0.6),
        is_valid=True,
        message="OK",
    )
    assert pu.is_valid is True


def test_physical_unit_invalid_range():
    with pytest.raises(ValueError, match="min must be"):
        PhysicalUnit("x", 1.0, "m", (10.0, 5.0), True, "")


def test_physical_unit_empty_quantity():
    with pytest.raises(ValueError, match="cannot be empty"):
        PhysicalUnit("", 1.0, "m", (0.0, 10.0), True, "OK")


# ── AnalysisSession ──────────────────────────────────────────────────────────

def _make_session() -> AnalysisSession:
    return AnalysisSession(session_id="test-session-001")


def test_analysis_session_inherits_agent_session():
    s = _make_session()
    s.add_user_message("hello")
    assert len(s.messages) == 1
    assert s.messages[0]["role"] == "user"


def test_register_dataset():
    s = _make_session()
    meta = DatasetMeta("data.csv", Path("/data.csv"), (10, 3), {}, [], {}, 1024)
    s.register_dataset(meta)
    assert "data.csv" in s.datasets_loaded


def test_register_dataset_rejects_path_traversal():
    s = _make_session()
    meta = DatasetMeta("../etc/passwd", Path("/"), (1, 1), {}, [], {}, 0)
    with pytest.raises(ValueError, match="Unsafe"):
        s.register_dataset(meta)


def test_get_dataset_not_found():
    s = _make_session()
    with pytest.raises(DatasetNotFoundError):
        s.get_dataset("missing.csv")


def test_react_trace_append():
    s = _make_session()
    s.append_react_step("thinking", "inspect_dataset", "result data")
    assert len(s.react_trace) == 1
    assert s.react_trace[0]["thought"] == "thinking"


def test_add_figure():
    s = _make_session()
    s.add_figure("fig_001", "abc123base64")
    assert s.figures["fig_001"] == "abc123base64"


def test_add_figure_rejects_empty_id():
    s = _make_session()
    with pytest.raises(ValueError, match="cannot be empty"):
        s.add_figure("", "base64data")
```

---

## 10. Design Decisions

### Decision 1: Inheritance vs. Composition for AnalysisSession

**Decision:** `AnalysisSession(AgentSession)` — inheritance.

**Rejected alternative:** `AnalysisSession` contains a field `agent_session: AgentSession` and forwards methods.

**Reason:** The ReAct loop in `DataScienceAgentService` calls `session.add_user_message()`, `session.add_assistant_message()`, and reads `session.messages` directly. These are defined on `AgentSession`. Inheritance makes `AnalysisSession` a proper drop-in, satisfying Liskov substitution. The `InMemorySessionStore` already typed as `dict[str, AgentSession]` can store `AnalysisSession` objects without modification.

### Decision 2: Frozen Dataclasses for Value Objects

**Decision:** `DatasetMeta`, `AnalysisResult`, `PhysicalUnit` are `@dataclass(frozen=True)`.

**Reason:** These are historical records, not live entities. Freezing makes them:
1. Safe to cache (hash-based lookup)
2. Safe to pass across async boundaries without defensive copying
3. Self-documenting: callers know they cannot mutate them

### Decision 3: No Pydantic in Domain Layer

**Decision:** Plain `@dataclass` throughout the domain.

**Reason:** Pydantic is an API/serialization concern. Domain objects do not care about JSON schema generation, OpenAPI docs, or `model_validate()`. Adding Pydantic to the domain layer would create a dependency from the innermost ring outward, violating Clean Architecture. The API layer's Pydantic schemas reference domain objects but remain separate.

### Decision 4: Exception Strategy

**Decision:** New exceptions extend `AgentError`, not `Exception` directly.

**Reason:** Callers (API handlers, the ReAct loop) can catch `AgentError` for any agent-related failure and handle uniformly. Specific subclasses allow precise handling when needed. The existing `except AgentError` clauses in MVP code remain valid.
