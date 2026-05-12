# Phase 1 — Domain Models and Configuration

## Overview

Phase 1 establishes the pure-Python domain layer. All classes here have **zero framework dependencies** — no FastAPI, no Anthropic SDK, no database. They can be unit-tested with plain `pytest`, no mocks required.

**Deliverables:**
- Extended `exceptions.py` with agent-specific error types
- New `analysis_models.py` with full data model for analysis sessions
- Extended `config.py` with all new settings
- Fixture data generation script
- pytest test suite for all domain models

---

## Step 1: Extend `app/domain/exceptions.py`

Add these four new exception types after the existing exceptions. Keep all existing exceptions untouched.

```python
# app/domain/exceptions.py

# ── Existing exceptions (keep as-is) ─────────────────────────────────────
class AgentError(Exception):
    """Base exception for all agent errors."""
    pass

class SessionNotFoundError(AgentError):
    """Raised when a session_id cannot be found in the store."""
    def __init__(self, session_id: str):
        super().__init__(f"Session '{session_id}' not found")
        self.session_id = session_id

class ToolExecutionError(AgentError):
    """Raised when a tool fails during execution."""
    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Tool '{tool_name}' failed: {reason}")
        self.tool_name = tool_name
        self.reason = reason

# ── NEW: Data science agent exceptions ───────────────────────────────────

class ReActLoopError(AgentError):
    """
    Raised when the ReAct loop cannot converge to a Final Answer.

    Common causes:
    - Exceeded MAX_REACT_ITERATIONS without reaching Final Answer
    - Claude repeatedly produces malformed Thought/Action blocks
    - A required tool is unavailable and the loop gets stuck

    Attributes:
        iterations: How many iterations were attempted before giving up.
        last_thought: The last parsed Thought string, for diagnostics.
    """
    def __init__(
        self,
        reason: str,
        iterations: int = 0,
        last_thought: str = "",
    ):
        super().__init__(f"ReAct loop error after {iterations} iterations: {reason}")
        self.iterations = iterations
        self.last_thought = last_thought


class CodeExecutionError(AgentError):
    """
    Raised when Python code execution fails in any backend.

    Attributes:
        backend: 'subprocess', 'jupyter', or 'anthropic'
        stderr: Raw stderr output from the execution attempt.
        timeout: True if the error was caused by a timeout.
    """
    def __init__(
        self,
        message: str,
        backend: str = "subprocess",
        stderr: str = "",
        timeout: bool = False,
    ):
        super().__init__(message)
        self.backend = backend
        self.stderr = stderr
        self.timeout = timeout

    @classmethod
    def from_timeout(cls, backend: str, timeout_seconds: int) -> "CodeExecutionError":
        return cls(
            message=f"Code execution timed out after {timeout_seconds}s",
            backend=backend,
            stderr="",
            timeout=True,
        )

    @classmethod
    def from_stderr(cls, backend: str, stderr: str) -> "CodeExecutionError":
        first_line = stderr.strip().splitlines()[0] if stderr.strip() else "unknown error"
        return cls(
            message=f"Code execution failed: {first_line}",
            backend=backend,
            stderr=stderr,
            timeout=False,
        )


class UnitValidationError(AgentError):
    """
    Raised when a physical quantity fails unit validation.

    This is raised by the physical validation gate before returning
    a Final Answer, or by individual tool calls to validate_physical_units.

    Attributes:
        quantity_name: Human-readable name of the quantity (e.g. 'thermal efficiency').
        value: The numeric value that was checked.
        unit: The unit string (e.g. 'percent', 'degC').
        expected_range: (min, max) tuple for the expected range, if applicable.
        dimensionality_error: True if the unit has wrong dimensions entirely.
    """
    def __init__(
        self,
        quantity_name: str,
        value: float,
        unit: str,
        reason: str,
        expected_range: tuple[float, float] | None = None,
        dimensionality_error: bool = False,
    ):
        super().__init__(
            f"Physical validation failed for '{quantity_name}' = {value} {unit}: {reason}"
        )
        self.quantity_name = quantity_name
        self.value = value
        self.unit = unit
        self.reason = reason
        self.expected_range = expected_range
        self.dimensionality_error = dimensionality_error


class DatasetNotFoundError(AgentError):
    """
    Raised when a requested dataset file cannot be located.

    Attributes:
        file_name: The file name or path that was requested.
        search_dir: The directory that was searched.
        available: List of files that ARE available (for suggestions).
    """
    def __init__(
        self,
        file_name: str,
        search_dir: str = "data",
        available: list[str] | None = None,
    ):
        available_str = ", ".join(available or []) or "none"
        super().__init__(
            f"Dataset '{file_name}' not found in '{search_dir}'. "
            f"Available files: {available_str}"
        )
        self.file_name = file_name
        self.search_dir = search_dir
        self.available = available or []
```

---

## Step 2: Create `app/domain/analysis_models.py`

This is the core domain model for the data science agent. All fields are fully typed.

```python
# app/domain/analysis_models.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── PhysicalUnit ──────────────────────────────────────────────────────────

@dataclass
class PhysicalUnit:
    """
    A validated physical quantity with unit and optional range context.

    Created by the unit validation layer and stored on AnalysisResult
    so the final response can include physical interpretation.
    """
    name: str            # human-readable, e.g. "thermal efficiency"
    value: float         # numeric magnitude
    unit: str            # pint-compatible unit string, e.g. "percent"
    is_valid: bool       # passed range + dimensionality check
    warning: str = ""    # non-empty if suspicious but not invalid
    expected_range: tuple[float, float] | None = None

    @property
    def formatted(self) -> str:
        return f"{self.value:.4g} {self.unit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "is_valid": self.is_valid,
            "warning": self.warning,
            "expected_range": list(self.expected_range) if self.expected_range else None,
        }


# ── DatasetMeta ───────────────────────────────────────────────────────────

@dataclass
class DatasetMeta:
    """
    Lightweight descriptor for a dataset file on disk.

    Populated by knowledge_tools.inspect_dataset() without loading the
    full file into memory. Used to build Claude's context about what
    data is available.
    """
    file_name: str          # basename only, e.g. "power_plant_data.csv"
    file_format: str        # "csv" | "parquet" | "excel" | "hdf5"
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]  # column -> pandas dtype string
    sample_rows: list[dict[str, Any]]  # first 3 rows
    file_size_bytes: int
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dataframe(
        cls,
        file_path: Path,
        df: Any,  # pd.DataFrame — avoid hard import at domain layer
    ) -> "DatasetMeta":
        return cls(
            file_name=file_path.name,
            file_format=file_path.suffix.lstrip("."),
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            sample_rows=df.head(3).to_dict(orient="records"),
            file_size_bytes=file_path.stat().st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_format": self.file_format,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "sample_rows": self.sample_rows,
            "file_size_bytes": self.file_size_bytes,
        }

    def summary_for_prompt(self) -> str:
        """Short one-liner for injecting into the system prompt."""
        return (
            f"  • {self.file_name} ({self.file_format.upper()}, "
            f"{self.row_count} rows × {self.column_count} cols): "
            f"{', '.join(self.columns[:6])}"
            + ("..." if len(self.columns) > 6 else "")
        )


# ── AnalysisResult ────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Output from a single code execution call.

    Returned by SubprocessCodeRunner, JupyterKernelManager, and
    AnthropicCodeExecRunner through the same interface.
    """
    stdout: str = ""
    stderr: str = ""
    figures: dict[str, str] = field(default_factory=dict)
    # figures: { "fig_000": "<base64-png>", "fig_001": ... }
    variables: dict[str, Any] = field(default_factory=dict)
    # variables: snapshot of Python namespace after execution
    error: str = ""         # non-empty if execution failed
    timed_out: bool = False
    execution_time_ms: int = 0

    @property
    def success(self) -> bool:
        return not self.error and not self.timed_out

    @property
    def figure_ids(self) -> list[str]:
        return sorted(self.figures.keys())

    def merge(self, other: "AnalysisResult") -> "AnalysisResult":
        """Combine results from multiple execution calls in a session."""
        merged_figures = {**self.figures, **other.figures}
        merged_vars = {**self.variables, **other.variables}
        return AnalysisResult(
            stdout=self.stdout + "\n" + other.stdout,
            stderr=self.stderr + "\n" + other.stderr,
            figures=merged_figures,
            variables=merged_vars,
            error=other.error or self.error,
            timed_out=self.timed_out or other.timed_out,
            execution_time_ms=self.execution_time_ms + other.execution_time_ms,
        )

    def to_observation_string(self) -> str:
        """Convert to a string suitable for injecting as a ReAct Observation."""
        parts = []
        if self.error:
            parts.append(f"ERROR: {self.error}")
            if self.stderr:
                parts.append(f"Stderr: {self.stderr[:500]}")
            return "\n".join(parts)
        if self.stdout.strip():
            parts.append(self.stdout.strip()[:2000])
        if self.figures:
            parts.append(f"Figures generated: {', '.join(self.figure_ids)}")
        if not parts:
            parts.append("(code executed successfully, no output)")
        return "\n".join(parts)


# ── ReActStep ─────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    """
    One parsed step from Claude's text response.

    A response can be either:
    - An action step: thought + action + action_input
    - A final answer: thought (optional) + final_answer

    The ReActParser populates this from raw text.
    """
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    final_answer: str = ""
    raw_text: str = ""

    @property
    def is_final(self) -> bool:
        return bool(self.final_answer)

    @property
    def has_action(self) -> bool:
        return bool(self.action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "final_answer": self.final_answer,
        }


# ── AnalysisSession ───────────────────────────────────────────────────────

@dataclass
class AnalysisSession:
    """
    Full state for one data science analysis conversation.

    Holds:
    - The message history for the Anthropic API
    - The react_trace for debugging and transparency
    - Accumulated figures from code execution
    - Dataset metadata discovered during the session
    - Physical units extracted and validated
    - Path to any exported notebook

    This is the primary object passed through the entire agent pipeline.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Conversation history (Anthropic messages format)
    messages: list[dict[str, str]] = field(default_factory=list)

    # ReAct execution trace for transparency/debugging
    react_trace: list[dict[str, Any]] = field(default_factory=list)

    # Accumulated figures across all code executions in this session
    # key: "fig_000", value: base64-encoded PNG string
    figures: dict[str, str] = field(default_factory=dict)

    # Datasets discovered and inspected during this session
    datasets: dict[str, DatasetMeta] = field(default_factory=dict)

    # Physical units validated during this session
    validated_units: list[PhysicalUnit] = field(default_factory=list)

    # Path to exported notebook (set after export_notebook tool call)
    notebook_path: Path | None = None

    # Which data directory was specified for this session
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Status: "running" | "completed" | "error" | "timeout"
    status: str = "running"

    # Final answer text (set when ReAct loop completes)
    final_answer: str = ""

    # Current code execution state (Python variables) — runner-managed
    execution_state: dict[str, Any] = field(default_factory=dict)

    # ── Mutation helpers ──────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.updated_at = datetime.now(timezone.utc)

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.updated_at = datetime.now(timezone.utc)

    def add_react_step(
        self,
        step: ReActStep,
        observation: str = "",
    ) -> None:
        record = step.to_dict()
        record["observation"] = observation
        self.react_trace.append(record)
        self.updated_at = datetime.now(timezone.utc)

    def merge_figures(self, new_figures: dict[str, str]) -> None:
        """Add figures from an AnalysisResult, renaming to avoid collisions."""
        offset = len(self.figures)
        for i, (old_key, b64) in enumerate(new_figures.items()):
            new_key = f"fig_{offset + i:03d}"
            self.figures[new_key] = b64
        self.updated_at = datetime.now(timezone.utc)

    def register_dataset(self, meta: DatasetMeta) -> None:
        self.datasets[meta.file_name] = meta
        self.updated_at = datetime.now(timezone.utc)

    def add_validated_unit(self, unit: PhysicalUnit) -> None:
        self.validated_units.append(unit)
        self.updated_at = datetime.now(timezone.utc)

    def mark_complete(self, final_answer: str) -> None:
        self.final_answer = final_answer
        self.status = "completed"
        self.updated_at = datetime.now(timezone.utc)

    def mark_error(self, reason: str) -> None:
        self.status = "error"
        self.final_answer = f"[Error: {reason}]"
        self.updated_at = datetime.now(timezone.utc)

    # ── Factory methods ───────────────────────────────────────────────────

    @classmethod
    def new(cls, user_message: str, data_dir: str | Path = "data") -> "AnalysisSession":
        """Create a fresh session with the first user message already added."""
        session = cls(data_dir=Path(data_dir))
        session.add_user_message(user_message)
        return session

    # ── Query helpers ─────────────────────────────────────────────────────

    @property
    def iteration_count(self) -> int:
        return len(self.react_trace)

    @property
    def figure_ids(self) -> list[str]:
        return sorted(self.figures.keys())

    @property
    def notebook_available(self) -> bool:
        return self.notebook_path is not None and self.notebook_path.exists()

    @property
    def dataset_names(self) -> list[str]:
        return list(self.datasets.keys())

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "iterations": self.iteration_count,
            "figures": self.figure_ids,
            "datasets": self.dataset_names,
            "notebook_available": self.notebook_available,
            "validated_units": len(self.validated_units),
            "created_at": self.created_at.isoformat(),
        }
```

---

## Step 3: Extend `app/core/config.py`

Show the full file with new fields clearly marked:

```python
# app/core/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Existing MVP settings ─────────────────────────────────────────────
    anthropic_api_key: str
    app_name: str = "Data Scientist Agent"
    debug: bool = False

    # ── NEW: Model configuration ──────────────────────────────────────────
    # Upgraded from claude-3-7-sonnet-20250219 for better reasoning
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192

    # ── NEW: Data directories ─────────────────────────────────────────────
    # Root directory containing domain_docs/ and datasets/ subdirectories
    data_dir: Path = Path("data")

    # ── NEW: Code execution ───────────────────────────────────────────────
    # Backend for running Python code generated by the agent
    # Options: "subprocess" | "jupyter" | "anthropic"
    code_execution_backend: str = "subprocess"

    # Seconds before a code execution is forcibly terminated
    code_execution_timeout: int = 30

    # Enable the Jupyter kernel bridge (requires jupyter_client + ipykernel)
    enable_jupyter_bridge: bool = False

    # ── NEW: ReAct loop ───────────────────────────────────────────────────
    # Hard limit on Thought/Action/Observation iterations
    max_react_iterations: int = 20

    # ── NEW: Output paths ─────────────────────────────────────────────────
    figures_dir: Path = Path("outputs/figures")
    notebooks_dir: Path = Path("outputs/notebooks")

    # ── Derived properties ────────────────────────────────────────────────
    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def domain_docs_dir(self) -> Path:
        return self.data_dir / "domain_docs"

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist. Call at startup."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
```

---

## Step 4: Generate Fixture Data

```bash
# Run from project root
python scripts/create_sample_data.py

# Verify
ls -la data/domain_docs/
ls -la data/datasets/
```

---

## Step 5: Verify Phase 1 — pytest Tests

Create `tests/test_phase1_domain.py`:

```python
# tests/test_phase1_domain.py
"""
Pure-Python unit tests for Phase 1 domain models.
No mocks, no FastAPI, no Anthropic SDK.
"""
import pytest
from datetime import timezone
from pathlib import Path

from app.domain.exceptions import (
    ReActLoopError,
    CodeExecutionError,
    UnitValidationError,
    DatasetNotFoundError,
)
from app.domain.analysis_models import (
    AnalysisSession,
    AnalysisResult,
    DatasetMeta,
    PhysicalUnit,
    ReActStep,
)


# ── Exception tests ────────────────────────────────────────────────────────

class TestReActLoopError:
    def test_basic_creation(self):
        err = ReActLoopError("exceeded limit", iterations=20, last_thought="I should stop")
        assert "20" in str(err)
        assert err.iterations == 20
        assert err.last_thought == "I should stop"

    def test_default_values(self):
        err = ReActLoopError("something went wrong")
        assert err.iterations == 0
        assert err.last_thought == ""


class TestCodeExecutionError:
    def test_from_timeout(self):
        err = CodeExecutionError.from_timeout("subprocess", 30)
        assert err.timeout is True
        assert err.backend == "subprocess"
        assert "30" in str(err)

    def test_from_stderr(self):
        stderr = "NameError: name 'df' is not defined\nTraceback..."
        err = CodeExecutionError.from_stderr("jupyter", stderr)
        assert err.backend == "jupyter"
        assert "NameError" in str(err)
        assert err.stderr == stderr
        assert err.timeout is False


class TestUnitValidationError:
    def test_full_construction(self):
        err = UnitValidationError(
            quantity_name="thermal efficiency",
            value=150.0,
            unit="percent",
            reason="exceeds 100% maximum",
            expected_range=(30.0, 50.0),
            dimensionality_error=False,
        )
        assert "thermal efficiency" in str(err)
        assert err.value == 150.0
        assert err.expected_range == (30.0, 50.0)


class TestDatasetNotFoundError:
    def test_with_suggestions(self):
        err = DatasetNotFoundError(
            "missing.csv",
            search_dir="data/datasets",
            available=["power_plant_data.csv", "turbine.parquet"],
        )
        assert "missing.csv" in str(err)
        assert "power_plant_data.csv" in str(err)
        assert err.available == ["power_plant_data.csv", "turbine.parquet"]


# ── PhysicalUnit tests ─────────────────────────────────────────────────────

class TestPhysicalUnit:
    def test_formatted(self):
        pu = PhysicalUnit("efficiency", 0.37, "dimensionless", is_valid=True)
        assert "0.37" in pu.formatted

    def test_to_dict(self):
        pu = PhysicalUnit(
            "steam temp", 540.0, "degC", is_valid=True,
            expected_range=(520.0, 600.0),
        )
        d = pu.to_dict()
        assert d["name"] == "steam temp"
        assert d["expected_range"] == [520.0, 600.0]


# ── DatasetMeta tests ──────────────────────────────────────────────────────

class TestDatasetMeta:
    def test_summary_for_prompt(self):
        meta = DatasetMeta(
            file_name="power_plant_data.csv",
            file_format="csv",
            row_count=500,
            column_count=7,
            columns=["timestamp", "steam_temp_C", "pressure_MPa",
                     "power_MW", "heat_rate", "efficiency", "co2"],
            dtypes={"timestamp": "datetime64[ns]", "power_MW": "float64"},
            sample_rows=[{"timestamp": "2024-01-01", "power_MW": 620.0}],
            file_size_bytes=40_000,
        )
        summary = meta.summary_for_prompt()
        assert "power_plant_data.csv" in summary
        assert "CSV" in summary
        assert "500 rows" in summary

    def test_to_dict_roundtrip(self):
        meta = DatasetMeta(
            file_name="test.csv", file_format="csv", row_count=10,
            column_count=3, columns=["a", "b", "c"],
            dtypes={"a": "float64"}, sample_rows=[],
            file_size_bytes=1024,
        )
        d = meta.to_dict()
        assert d["file_name"] == "test.csv"
        assert d["row_count"] == 10


# ── AnalysisResult tests ───────────────────────────────────────────────────

class TestAnalysisResult:
    def test_success_property(self):
        ok = AnalysisResult(stdout="42\n")
        assert ok.success is True

        fail = AnalysisResult(error="NameError: x not defined")
        assert fail.success is False

        timeout = AnalysisResult(timed_out=True)
        assert timeout.success is False

    def test_to_observation_string_success(self):
        result = AnalysisResult(stdout="Mean efficiency: 36.7%\n")
        obs = result.to_observation_string()
        assert "36.7%" in obs

    def test_to_observation_string_error(self):
        result = AnalysisResult(error="SyntaxError", stderr="line 3: invalid syntax")
        obs = result.to_observation_string()
        assert "ERROR" in obs

    def test_to_observation_string_figures(self):
        result = AnalysisResult(figures={"fig_000": "abc123"})
        obs = result.to_observation_string()
        assert "fig_000" in obs

    def test_merge(self):
        r1 = AnalysisResult(stdout="first\n", figures={"fig_000": "aaa"})
        r2 = AnalysisResult(stdout="second\n", figures={"fig_001": "bbb"})
        merged = r1.merge(r2)
        assert "first" in merged.stdout
        assert "second" in merged.stdout
        assert len(merged.figures) == 2


# ── ReActStep tests ────────────────────────────────────────────────────────

class TestReActStep:
    def test_action_step(self):
        step = ReActStep(
            thought="I need to load the data.",
            action="inspect_dataset",
            action_input={"file_name": "power_plant_data.csv"},
        )
        assert step.has_action is True
        assert step.is_final is False

    def test_final_step(self):
        step = ReActStep(final_answer="The mean efficiency is 36.7%.")
        assert step.is_final is True
        assert step.has_action is False

    def test_to_dict(self):
        step = ReActStep(thought="Analyzing...", action="list_datasets", action_input={})
        d = step.to_dict()
        assert d["action"] == "list_datasets"
        assert d["final_answer"] == ""


# ── AnalysisSession tests ──────────────────────────────────────────────────

class TestAnalysisSession:
    def test_factory_new(self):
        session = AnalysisSession.new("Analyze the power plant data.", data_dir="data")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.status == "running"

    def test_add_messages(self):
        session = AnalysisSession.new("Hello")
        session.add_assistant_message("I'll help.")
        session.add_user_message("Thanks.")
        assert len(session.messages) == 3
        assert session.messages[-1]["role"] == "user"

    def test_merge_figures_no_collision(self):
        session = AnalysisSession.new("test")
        session.merge_figures({"fig_000": "aaa", "fig_001": "bbb"})
        assert len(session.figures) == 2
        session.merge_figures({"fig_000": "ccc"})  # colliding key
        assert len(session.figures) == 3            # renamed to fig_002

    def test_mark_complete(self):
        session = AnalysisSession.new("test")
        session.mark_complete("The answer is 42.")
        assert session.status == "completed"
        assert session.final_answer == "The answer is 42."

    def test_mark_error(self):
        session = AnalysisSession.new("test")
        session.mark_error("loop exceeded max iterations")
        assert session.status == "error"

    def test_notebook_available_false_when_no_path(self):
        session = AnalysisSession.new("test")
        assert session.notebook_available is False

    def test_notebook_available_false_when_path_missing(self):
        session = AnalysisSession.new("test")
        session.notebook_path = Path("/nonexistent/notebook.ipynb")
        assert session.notebook_available is False

    def test_iteration_count(self):
        session = AnalysisSession.new("test")
        step = ReActStep(thought="x", action="list_datasets", action_input={})
        session.add_react_step(step, observation="[file.csv]")
        assert session.iteration_count == 1

    def test_to_summary_dict(self):
        session = AnalysisSession.new("test")
        summary = session.to_summary_dict()
        assert summary["status"] == "running"
        assert summary["iterations"] == 0
        assert isinstance(summary["figures"], list)
```

Run the tests:

```bash
pytest tests/test_phase1_domain.py -v
```

Expected output:
```
tests/test_phase1_domain.py::TestReActLoopError::test_basic_creation PASSED
tests/test_phase1_domain.py::TestReActLoopError::test_default_values PASSED
...
tests/test_phase1_domain.py::TestAnalysisSession::test_to_summary_dict PASSED

24 passed in 0.12s
```

---

## Step 6: Common Mistakes When Extending AgentSession

### Mistake 1: Mutating messages list directly

**Wrong:**
```python
session.messages.append({"role": "user", "content": "hello"})
# updated_at is NOT updated — tracking broken
```

**Right:**
```python
session.add_user_message("hello")
# Always use mutation helpers — they update timestamps
```

### Mistake 2: Figure key collision when merging results

**Wrong:**
```python
session.figures.update(result.figures)
# If result has "fig_000" and session already has "fig_000" → silent overwrite
```

**Right:**
```python
session.merge_figures(result.figures)
# merge_figures() renames with offset: fig_002, fig_003, etc.
```

### Mistake 3: Checking notebook_available before the path is set

**Wrong:**
```python
# Calling export_notebook writes the file but you check before it returns
if session.notebook_available:  # Always False here
    ...
```

**Right:**
```python
path = await export_notebook(session=session, title="Analysis")
# export_notebook sets session.notebook_path internally
assert session.notebook_available  # True now
```

---

## Checkpoint

After Phase 1, your project should have:

```
✅ app/domain/exceptions.py      — 4 new exception classes
✅ app/domain/analysis_models.py — 6 dataclasses, all fields typed
✅ app/core/config.py            — 8 new settings fields
✅ data/domain_docs/             — 2 markdown knowledge files
✅ data/datasets/                — 3 sample datasets (CSV, Parquet, Excel)
✅ tests/test_phase1_domain.py   — 24 passing tests
```

→ **Next**: [03_phase2_tools.md](03_phase2_tools.md) — Building the tool suite
