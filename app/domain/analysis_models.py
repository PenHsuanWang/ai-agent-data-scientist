"""Data Scientist Agent domain entities.

Zero external dependencies — stdlib only.
All value objects are frozen (immutable once created).
AnalysisSession is the mutable aggregate root for analysis conversations.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.domain.models import AgentSession


# ──────────────────────────────────────────────────────────────────── #
# Value Objects (frozen)                                                #
# ──────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class DatasetMeta:
    """Immutable metadata snapshot of a loaded dataset.

    Invariants:
    - rows >= 0
    - columns >= 0
    - file_name is a basename (no path components)
    """

    file_name: str
    format: str          # "csv" | "parquet" | "xlsx" | "hdf5"
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]           # column → dtype name
    numeric_stats: dict[str, Any]    # column → {min, max, mean, std}
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.rows < 0:
            raise ValueError("rows must be >= 0")
        if self.columns < 0:
            raise ValueError("columns must be >= 0")
        if "/" in self.file_name or "\\" in self.file_name:
            raise ValueError(f"file_name must be a basename, got '{self.file_name}'")


@dataclass(frozen=True)
class AnalysisResult:
    """Immutable result from a single code execution.

    Invariants:
    - If success is False, stdout may be empty but stderr should explain the failure.
    """

    success: bool
    stdout: str = ""
    stderr: str = ""
    figures: list[str] = field(default_factory=list)  # list of figure_ids
    execution_time_ms: int = 0

    @property
    def output(self) -> str:
        """Combined readable output for the ReAct Observation block."""
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.figures:
            parts.append(f"[Figures generated: {', '.join(self.figures)}]")
        if not self.success and self.stderr.strip():
            parts.append(f"[Error: {self.stderr.strip()}]")
        return "\n".join(parts) if parts else "(no output)"


@dataclass(frozen=True)
class PhysicalUnit:
    """Result of a single physical unit validation check.

    Invariants:
    - is_valid=True implies message is "OK" or a positive confirmation.
    - is_valid=False implies message describes why the check failed.
    """

    quantity: str         # e.g. "thermal_efficiency"
    value: float
    unit: str             # e.g. "%" or "MW"
    is_valid: bool
    message: str          # human-readable verdict
    canonical_value: float | None = None   # value converted to SI base unit
    canonical_unit: str | None = None


@dataclass(frozen=True)
class NotebookCell:
    """A single cell in the exported Jupyter notebook."""

    cell_type: str    # "code" | "markdown"
    source: str
    outputs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────── #
# Aggregate Root: AnalysisSession                                       #
# ──────────────────────────────────────────────────────────────────── #


@dataclass
class AnalysisSession(AgentSession):
    """Aggregate root for a Data Scientist Agent conversation.

    Extends AgentSession with:
    - Loaded dataset metadata registry
    - Generated figure registry (figure_id → base64 PNG)
    - Jupyter notebook cell history
    - Physical unit validation log
    - ReAct reasoning trace (append-only)
    - Opaque code_runner_state managed by the active CodeRunner

    Invariants:
    - session_id is non-empty (inherited)
    - datasets_loaded keys are file basenames (no traversal chars)
    - react_trace is append-only
    """

    datasets_loaded: dict[str, DatasetMeta] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)          # figure_id → base64 PNG
    notebook_cells: list[NotebookCell] = field(default_factory=list)
    unit_context: list[PhysicalUnit] = field(default_factory=list)
    react_trace: list[dict[str, str]] = field(default_factory=list)
    code_runner_state: dict[str, Any] = field(default_factory=dict)
    notebook_path: str | None = None

    # ── Factory ─────────────────────────────────────────────────────── #

    @classmethod
    def new(cls) -> "AnalysisSession":
        """Create a fresh session with a UUID4 session_id."""
        return cls(session_id=str(uuid.uuid4()))

    # ── Mutation helpers ─────────────────────────────────────────────── #

    def register_dataset(self, meta: DatasetMeta) -> None:
        """Record that a dataset has been loaded into this session."""
        self.datasets_loaded[meta.file_name] = meta

    def register_figure(self, figure_id: str, b64_png: str) -> None:
        """Store a base64-encoded PNG under figure_id."""
        if not figure_id or not b64_png:
            raise ValueError("figure_id and b64_png must be non-empty")
        self.figures[figure_id] = b64_png

    def append_react_step(
        self,
        thought: str,
        action: str,
        observation: str,
    ) -> None:
        """Append one Thought/Action/Observation triple to the trace."""
        self.react_trace.append(
            {"thought": thought, "action": action, "observation": observation}
        )

    def append_notebook_cell(self, cell: NotebookCell) -> None:
        self.notebook_cells.append(cell)

    def log_unit_validation(self, unit: PhysicalUnit) -> None:
        self.unit_context.append(unit)

    @property
    def figure_count(self) -> int:
        return len(self.figures)

    @property
    def next_figure_id(self) -> str:
        return f"fig_{self.figure_count:03d}"
