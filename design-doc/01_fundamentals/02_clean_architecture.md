# Clean Architecture for AI Agents

> *"The architecture of a software system is the set of structures needed to reason about the system."*
> — Bass, Clements & Kazman, *Software Architecture in Practice*

---

## Table of Contents

1. [The Four Layers](#1-the-four-layers)
2. [The Dependency Rule](#2-the-dependency-rule)
3. [How the MVP Implements It](#3-how-the-mvp-implements-it)
4. [Extension Points](#4-extension-points)
5. [New File Map](#5-new-file-map)
6. [SOLID Principles in Practice](#6-solid-principles-in-practice)
7. [Dependency Injection](#7-dependency-injection)
8. [Testing Strategy](#8-testing-strategy)

---

## 1. The Four Layers

Clean Architecture (Robert C. Martin, 2017) organizes software into concentric layers. Each layer has a single responsibility, and each layer only depends on layers closer to the center. For an AI agent, this produces a system where you can swap the LLM provider, the code execution backend, or the API framework without touching any business logic.

### Layer 1: Domain (Innermost)

**What it contains**: Pure data structures (dataclasses), value objects, and domain-specific enumerations. Zero dependencies on any framework, database, or external library (with the single exception of Python's `dataclasses` stdlib module and possibly `pydantic` for validation).

**What it does NOT contain**: I/O, network calls, framework decorators, database queries, business logic that depends on external state.

**Why it's the center**: Domain models are what everything else is ultimately about. They change only when the fundamental concepts of the domain change — which is rare. By keeping them pure, you guarantee they can always be instantiated, tested, and reasoned about without any infrastructure.

**Domain models in the data scientist agent**:
```python
# domain/models.py
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

class AnalysisStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

@dataclass
class DatasetMeta:
    """Metadata about a dataset file — discovered without loading it."""
    name: str
    path: str
    format: str           # "csv", "parquet", "hdf5", "excel", "json"
    size_bytes: int
    row_count: Optional[int] = None   # None until dataset is loaded/inspected
    column_count: Optional[int] = None
    columns: list[str] = field(default_factory=list)
    description: Optional[str] = None  # From accompanying .meta.yaml if present

@dataclass
class PhysicalUnit:
    """A quantity with its unit and dimensional metadata."""
    quantity_name: str       # e.g., "thermal_efficiency"
    value: float
    unit_str: str            # e.g., "percent", "MW", "kg/s"
    dimension: Optional[str] = None   # e.g., "dimensionless", "[power]"
    domain: Optional[str] = None      # e.g., "thermodynamics", "fluid_dynamics"

@dataclass
class ValidationResult:
    """Result of a physical validation check."""
    valid: bool
    quantity: PhysicalUnit
    violation: Optional[str] = None     # Null if valid
    recommendation: Optional[str] = None
    carnot_limit: Optional[float] = None
    design_range: Optional[tuple[float, float]] = None

@dataclass
class ExecutionResult:
    """Result from a code execution backend."""
    stdout: str
    stderr: str
    success: bool
    figures: list[str] = field(default_factory=list)  # Base64-encoded PNGs
    execution_time_ms: Optional[float] = None

@dataclass
class NotebookCell:
    """A single cell in the exported Jupyter notebook."""
    cell_type: str           # "code" or "markdown"
    source: str
    outputs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    execution_count: Optional[int] = None

@dataclass
class AnalysisSession:
    """Extends AgentSession with data science specific state."""
    session_id: str
    messages: list[dict] = field(default_factory=list)
    status: AnalysisStatus = AnalysisStatus.PENDING
    cells: list[NotebookCell] = field(default_factory=list)  # For notebook export
    thought_trace: list[str] = field(default_factory=list)
    datasets_loaded: list[str] = field(default_factory=list)
    documents_read: list[str] = field(default_factory=list)
    iteration_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    final_answer: Optional[str] = None
```

### Layer 2: Infrastructure (Outermost Technical)

**What it contains**: All code that talks to the external world — files, databases, APIs, subprocesses, network sockets. Also includes adapters that translate external formats into domain models.

**What it does NOT contain**: Business logic, orchestration, routing, or any decision-making about what to do with data.

**Infrastructure components in the data scientist agent**:

- `LocalDocumentStore` — reads markdown/PDF files from the filesystem into strings
- `LocalDatasetStore` — lists dataset files, reads CSV/Parquet/HDF5/Excel/JSON using pandas
- `SubprocessCodeRunner` — spawns a Python subprocess, captures stdout/stderr
- `JupyterKernelManager` — creates and manages a Jupyter kernel in-process
- `AnthropicCodeExecRunner` — uses the Anthropic `code_execution_20260120` server tool
- `NotebookSerializer` — converts `list[NotebookCell]` to `.ipynb` JSON via `nbformat`

```python
# infrastructure/dataset_store.py
import pandas as pd
from pathlib import Path
from app.domain.models import DatasetMeta

class LocalDatasetStore:
    """Discovers and loads datasets from a configured directory."""
    
    def __init__(self, datasets_dir: str) -> None:
        self._dir = Path(datasets_dir)
    
    def list_datasets(self) -> list[DatasetMeta]:
        """Return metadata for all dataset files in the datasets directory."""
        SUPPORTED = {".csv", ".parquet", ".h5", ".hdf5", ".xlsx", ".xls", ".json"}
        results = []
        for path in sorted(self._dir.rglob("*")):
            if path.suffix.lower() in SUPPORTED:
                results.append(DatasetMeta(
                    name=path.stem,
                    path=str(path),
                    format=path.suffix.lower().lstrip("."),
                    size_bytes=path.stat().st_size,
                ))
        return results
    
    def load_dataframe(self, name: str) -> pd.DataFrame:
        """Load a dataset by name and return a pandas DataFrame."""
        candidates = list(self._dir.rglob(f"{name}.*"))
        if not candidates:
            raise FileNotFoundError(f"No dataset named '{name}' found in {self._dir}")
        path = candidates[0]
        loaders = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
            ".h5": lambda p: pd.read_hdf(p),
            ".hdf5": lambda p: pd.read_hdf(p),
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".json": pd.read_json,
        }
        loader = loaders.get(path.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return loader(path)
```

### Layer 3: Application (Business Logic)

**What it contains**: The orchestration of domain models and infrastructure. This is where the ReAct loop lives, where tools are dispatched, where validation is applied, where sessions are managed.

**What it does NOT contain**: HTTP concerns (routing, request parsing, response serialization), database connection strings, or filesystem paths.

**Application components in the data scientist agent**:

- `DataScienceAgentService` — the ReAct loop, tool dispatch, session management
- `ToolExecutor` — registry + dispatcher for all 14 tools
- `PhysicalValidationEngine` — coordinates pint + domain ranges + law checks
- `NotebookExporter` — translates session cells to nbformat structure

```python
# application/agent_service.py
import anthropic
from app.domain.models import AnalysisSession, AnalysisStatus
from app.application.tool_executor import ToolExecutor
from app.application.validation_engine import PhysicalValidationEngine

class DataScienceAgentService:
    """
    Orchestrates the ReAct loop for data science analysis sessions.
    
    Depends on abstractions (ToolExecutor, ValidationEngine) not on
    concrete infrastructure classes — enabling testability and swap-ability.
    """
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        tool_executor: ToolExecutor,
        validation_engine: PhysicalValidationEngine,
        model: str = "claude-sonnet-4-6",
        max_iterations: int = 20,
    ) -> None:
        self._client = client
        self._executor = tool_executor
        self._validator = validation_engine
        self._model = model
        self._max_iterations = max_iterations
    
    def run(self, session: AnalysisSession, user_message: str) -> str:
        """Add user message and run the ReAct loop until completion."""
        session.messages.append({"role": "user", "content": user_message})
        session.status = AnalysisStatus.RUNNING
        # ... (full loop implementation in 03_software_design/07_agent_service.md)
```

### Layer 4: Presentation (Outermost)

**What it contains**: HTTP routing (FastAPI routers), request/response Pydantic models, middleware, authentication, and serialization. This layer translates between HTTP and the application layer.

**What it does NOT contain**: Business logic, direct database access, or domain model construction.

```python
# api/v1/analysis.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.application.agent_service import DataScienceAgentService
from app.core.dependencies import get_analysis_service, get_session_store

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    iteration_count: int
    status: str

@router.post("/chat", response_model=ChatResponse)
async def analysis_chat(
    request: ChatRequest,
    service: DataScienceAgentService = Depends(get_analysis_service),
    sessions = Depends(get_session_store),
):
    # Get or create session
    session = sessions.get(request.session_id) if request.session_id else None
    if session is None:
        session = sessions.create_new()
    
    response_text = service.run(session, request.message)
    sessions.save(session)
    
    return ChatResponse(
        session_id=session.session_id,
        response=response_text,
        iteration_count=session.iteration_count,
        status=session.status.value,
    )
```

---

## 2. The Dependency Rule

The fundamental rule of Clean Architecture is: **dependencies point inward only**.

```
Presentation  →  Application  →  Infrastructure
     ↘               ↘               ↘
         Domain  ←  ←  ←  ←  ←  ←  ←  ←
```

More precisely: **outer layers may depend on inner layers; inner layers must never depend on outer layers**.

What this means in practice:
- `domain/models.py` imports nothing from `infrastructure/`, `application/`, or `api/`
- `application/agent_service.py` imports from `domain/` but not from `api/`
- `infrastructure/code_runner.py` imports from `domain/` but not from `application/` or `api/`
- `api/v1/analysis.py` imports from `application/` and `domain/` but orchestrates everything

When you feel the urge to import an application-layer class in a domain model, stop. You are about to violate the dependency rule. The solution is always to invert the dependency: define an abstract interface in the inner layer, and implement it in the outer layer.

```
                    ┌──────────────────────────────────────────┐
                    │              DOMAIN                       │
                    │  AnalysisSession, DatasetMeta,           │
                    │  PhysicalUnit, ExecutionResult           │
                    └──────────────────────────────────────────┘
                           ↑ imported by
      ┌────────────────────┼─────────────────────────┐
      │              APPLICATION                      │
      │  DataScienceAgentService                     │
      │  ToolExecutor                                │
      │  PhysicalValidationEngine                    │
      └────────────────────┬─────────────────────────┘
                           ↑ imported by
      ┌────────────────────┼─────────────────────────┐
      │           INFRASTRUCTURE & PRESENTATION        │
      │  LocalDatasetStore  SubprocessCodeRunner       │
      │  FastAPI routers   Pydantic schemas            │
      └───────────────────────────────────────────────┘
```

---

## 3. How the MVP Implements It

The MVP (`ai-agent-mvp`) is a clean implementation of this pattern. Understanding it makes the extension points obvious.

### MVP File Structure

```
ai-agent-mvp/
├── domain/
│   └── models.py          # AgentSession dataclass — pure Python
├── application/
│   └── services/
│       └── agent.py        # AgentService — calls Anthropic, dispatches tools
├── infrastructure/
│   └── document_store.py   # Reads files from disk
└── api/
    └── v1/
        └── chat.py         # POST /api/v1/chat FastAPI router
```

### `domain/models.py`

```python
# MVP domain model — no imports beyond stdlib
from dataclasses import dataclass, field

@dataclass
class AgentSession:
    session_id: str
    messages: list[dict] = field(default_factory=list)
```

Zero dependencies. The session is just an ID and a message list. Any layer can import this freely.

### `application/services/agent.py`

```python
# MVP agent service — imports domain, uses Anthropic SDK
import anthropic
from app.domain.models import AgentSession

_MAX_LOOP_ITERATIONS = 10

class AgentService:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client
    
    def chat(self, session: AgentSession, user_message: str) -> str:
        session.messages.append({"role": "user", "content": user_message})
        
        for _ in range(_MAX_LOOP_ITERATIONS):
            response = self._client.messages.create(
                model="claude-opus-4-7",
                max_tokens=8096,
                tools=self._build_tools(),
                messages=session.messages,
            )
            
            session.messages.append({
                "role": "assistant",
                "content": response.content,
            })
            
            if response.stop_reason == "end_turn":
                return self._extract_text(response)
            
            if response.stop_reason == "tool_use":
                tool_results = self._execute_tools(response)
                session.messages.append({
                    "role": "user",
                    "content": tool_results,
                })
        
        return "Error: Max iterations reached."
```

This service depends on:
1. `anthropic.Anthropic` — the SDK client (comes from constructor, not hardcoded)
2. `AgentSession` — the domain model (inner layer)
3. Nothing from `api/` or `infrastructure/` (no violations)

### `api/v1/chat.py`

```python
# MVP presentation layer — imports application layer
from fastapi import APIRouter, Depends
from app.application.services.agent import AgentService
from app.core.dependencies import get_agent_service, get_session_store

router = APIRouter()

@router.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest, service = Depends(get_agent_service)):
    session = get_or_create_session(request.session_id)
    response = service.chat(session, request.message)
    return {"response": response, "session_id": session.session_id}
```

This only knows about `AgentService` (application layer) and HTTP concerns. No database queries, no filesystem operations, no Anthropic SDK calls directly.

---

## 4. Extension Points

### Adding `DataScienceAgentService` Without Touching `AgentService`

The key principle: **extend, don't modify**. We create a new service class that follows the same interface pattern as `AgentService`, without touching `AgentService` itself.

```python
# application/services/data_science_agent.py — NEW FILE
# Never modifies application/services/agent.py

import anthropic
from app.domain.models import AnalysisSession, AnalysisStatus
from app.application.tool_executor import ToolExecutor

class DataScienceAgentService:
    """
    A new service for data science analysis.
    Shares the Anthropic client with AgentService but has its own:
    - Tool set (14 tools vs 2)
    - System prompt (domain-aware)
    - Iteration limit (20 vs 10)
    - Session type (AnalysisSession vs AgentSession)
    - Loop logic (ReAct with trace vs simple tool-use)
    """
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        tool_executor: ToolExecutor,
    ) -> None:
        self._client = client
        self._executor = tool_executor
    
    def chat(self, session: AnalysisSession, user_message: str) -> str:
        # New loop logic, entirely independent of AgentService
        ...
```

This approach means:
- `AgentService` is never broken by data science changes
- `DataScienceAgentService` can be deleted without affecting any chat functionality
- Both services share the same `Anthropic` client instance (memory/connection efficient)
- Both services can be tested independently

### Adding a New Tool

Adding a new tool requires:
1. Implement the tool function in `application/tools/` (new file or add to existing)
2. Register it in `ToolExecutor`
3. Add the tool definition to the tool catalog
4. Update the system prompt to mention it

Zero changes to: `AgentService`, `DataScienceAgentService`, any API routes, any domain models, any tests of other tools.

---

## 5. New File Map

Complete file tree for the data scientist agent, with layer annotations:

```
data-scientist-agent/
├── app/
│   ├── domain/                          # [DOMAIN LAYER]
│   │   ├── __init__.py
│   │   └── models.py                   # AnalysisSession, DatasetMeta, PhysicalUnit, etc.
│   │
│   ├── application/                     # [APPLICATION LAYER]
│   │   ├── __init__.py
│   │   ├── services/
│   │   │   ├── agent.py                 # MVP AgentService (unchanged)
│   │   │   └── data_science_agent.py   # NEW: DataScienceAgentService + ReAct loop
│   │   ├── tools/                       # Tool implementations
│   │   │   ├── __init__.py
│   │   │   ├── knowledge_tools.py       # list_docs, read_doc, search_docs, list_datasets, inspect_dataset
│   │   │   ├── data_tools.py            # load_dataset, describe_dataset, execute_python, get_figure
│   │   │   ├── validation_tools.py      # validate_units, convert_units, check_magnitude
│   │   │   └── output_tools.py          # export_notebook, save_figure
│   │   ├── tool_executor.py             # ToolExecutor: registry + dispatcher
│   │   └── validation_engine.py         # PhysicalValidationEngine
│   │
│   ├── infrastructure/                  # [INFRASTRUCTURE LAYER]
│   │   ├── __init__.py
│   │   ├── document_store.py            # LocalDocumentStore (extended from MVP)
│   │   ├── dataset_store.py             # LocalDatasetStore: discover + load datasets
│   │   ├── code_runners/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                  # CodeRunner abstract base class
│   │   │   ├── subprocess_runner.py     # SubprocessCodeRunner
│   │   │   ├── jupyter_runner.py        # JupyterKernelManager
│   │   │   └── anthropic_runner.py      # AnthropicCodeExecRunner
│   │   └── notebook_serializer.py       # Converts NotebookCells to .ipynb via nbformat
│   │
│   ├── api/                             # [PRESENTATION LAYER]
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── chat.py                  # MVP chat endpoint (unchanged)
│   │       ├── analysis.py              # NEW: POST /analysis/chat, GET /analysis/{id}/notebook
│   │       └── datasets.py              # NEW: GET /datasets
│   │
│   ├── core/
│   │   ├── config.py                    # Settings singleton (pydantic-settings)
│   │   ├── dependencies.py              # FastAPI dependency injection factories
│   │   └── prompts.py                   # System prompts (REACT_SYSTEM_PROMPT, etc.)
│   │
│   └── main.py                          # FastAPI app creation, router registration
│
├── datasets/                            # Sample datasets for testing
│   └── turbine_data.csv
├── domain_docs/                         # Domain knowledge documents
│   └── power_plant_thermodynamics.md
├── tests/
│   ├── domain/
│   │   └── test_models.py
│   ├── application/
│   │   ├── test_tool_executor.py
│   │   └── test_validation_engine.py
│   ├── infrastructure/
│   │   └── test_dataset_store.py
│   └── api/
│       └── test_analysis_endpoints.py
└── pyproject.toml
```

---

## 6. SOLID Principles in Practice

### S — Single Responsibility Principle

**"A module should have one and only one reason to change."**

Each class in this system has a single responsibility:

| Class | Responsibility | Reason to change |
|-------|---------------|-----------------|
| `AnalysisSession` | Carry session state | Domain model changes |
| `LocalDatasetStore` | Load datasets from filesystem | Filesystem/format changes |
| `ToolExecutor` | Dispatch tool calls by name | Tool registry changes |
| `PhysicalValidationEngine` | Check physical plausibility | Validation logic changes |
| `DataScienceAgentService` | Orchestrate the ReAct loop | Agent behavior changes |

**Example violation to avoid**:
```python
# WRONG: DataScienceAgentService doing file I/O directly
class DataScienceAgentService:
    def _read_dataset(self, name: str):
        import pandas as pd
        return pd.read_csv(f"/data/{name}.csv")  # File path hardcoded in application layer!
```

**Correct**:
```python
# RIGHT: Inject the store dependency
class DataScienceAgentService:
    def __init__(self, ..., dataset_store: LocalDatasetStore) -> None:
        self._dataset_store = dataset_store
    
    # Tool executor calls dataset_store.load_dataframe(name) — not the agent service
```

### O — Open/Closed Principle

**"Open for extension, closed for modification."**

The `CodeRunner` abstract base class is the canonical example in this system:

```python
# infrastructure/code_runners/base.py
from abc import ABC, abstractmethod
from app.domain.models import ExecutionResult

class CodeRunner(ABC):
    """Abstract base for all code execution backends."""
    
    @abstractmethod
    def execute(self, code: str, timeout_seconds: float = 30.0) -> ExecutionResult:
        """Execute Python code and return the result."""
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset execution state (clear variables, restart kernel, etc.)."""
        ...

# infrastructure/code_runners/subprocess_runner.py
class SubprocessCodeRunner(CodeRunner):
    def execute(self, code: str, timeout_seconds: float = 30.0) -> ExecutionResult:
        # subprocess implementation
        ...

# infrastructure/code_runners/jupyter_runner.py
class JupyterKernelManager(CodeRunner):
    def execute(self, code: str, timeout_seconds: float = 30.0) -> ExecutionResult:
        # jupyter_client implementation
        ...

# Adding a new backend: ZERO changes to existing files
# infrastructure/code_runners/docker_runner.py
class DockerCodeRunner(CodeRunner):
    def execute(self, code: str, timeout_seconds: float = 30.0) -> ExecutionResult:
        # Docker-based isolated execution
        ...
```

The tool executor is open for extension via registration, closed for modification:

```python
# application/tool_executor.py
class ToolExecutor:
    def __init__(self) -> None:
        self._registry: dict[str, Callable] = {}
    
    def register(self, name: str, fn: Callable) -> None:
        """Register a new tool. No modification of existing code."""
        self._registry[name] = fn
    
    def execute(self, name: str, args: dict) -> str:
        if name not in self._registry:
            return f"Error: Unknown tool '{name}'"
        return self._registry[name](**args)

# In application startup:
executor = ToolExecutor()
executor.register("list_local_documents", list_local_documents_fn)
executor.register("read_local_document", read_local_document_fn)
# Adding a new tool: one line, no changes to ToolExecutor
executor.register("my_new_tool", my_new_tool_fn)
```

### L — Liskov Substitution Principle

**"Objects of a subtype must be substitutable for their base type."**

Every `CodeRunner` subclass must be usable anywhere a `CodeRunner` is expected:

```python
# This works regardless of whether runner is Subprocess, Jupyter, or Anthropic
def execute_with_timeout(runner: CodeRunner, code: str) -> ExecutionResult:
    try:
        return runner.execute(code, timeout_seconds=30.0)
    except TimeoutError:
        return ExecutionResult(stdout="", stderr="Execution timed out", success=False)

# All three backends satisfy this contract:
subprocess_result = execute_with_timeout(SubprocessCodeRunner(), "print(1+1)")
jupyter_result = execute_with_timeout(JupyterKernelManager(), "print(1+1)")
anthropic_result = execute_with_timeout(AnthropicCodeExecRunner(client), "print(1+1)")
```

### I — Interface Segregation Principle

**"No client should be forced to depend on methods it does not use."**

The validation engine exposes separate methods for different validation types:

```python
class PhysicalValidationEngine:
    def validate_unit_consistency(self, quantity: PhysicalUnit) -> ValidationResult:
        """Check dimensional consistency only."""
        ...
    
    def validate_range(self, quantity: PhysicalUnit, domain: str) -> ValidationResult:
        """Check against domain-specific empirical ranges only."""
        ...
    
    def validate_physical_law(self, law: str, quantities: dict[str, PhysicalUnit]) -> ValidationResult:
        """Check against a named physical law only."""
        ...
    
    # The tool function calls whichever validation is needed,
    # not a monolithic "validate everything" method
```

### D — Dependency Inversion Principle

**"High-level modules should not depend on low-level modules. Both should depend on abstractions."**

`DataScienceAgentService` (high-level) depends on `CodeRunner` (abstraction), not on `SubprocessCodeRunner` (low-level):

```python
# application/services/data_science_agent.py
from app.infrastructure.code_runners.base import CodeRunner  # abstraction

class DataScienceAgentService:
    def __init__(self, ..., code_runner: CodeRunner) -> None:
        self._runner = code_runner  # Injected — could be any implementation
```

The concrete implementation is chosen at startup:

```python
# core/dependencies.py
from app.core.config import Settings
from app.infrastructure.code_runners.subprocess_runner import SubprocessCodeRunner
from app.infrastructure.code_runners.jupyter_runner import JupyterKernelManager

def get_code_runner(settings: Settings = Depends(get_settings)) -> CodeRunner:
    if settings.code_execution_backend == "jupyter":
        return JupyterKernelManager()
    elif settings.code_execution_backend == "anthropic":
        return AnthropicCodeExecRunner(get_anthropic_client())
    else:
        return SubprocessCodeRunner()
```

---

## 7. Dependency Injection

FastAPI's `Depends()` system provides constructor injection at the request level. For singleton resources (clients, stores), we use module-level factories with lazy initialization.

### The Settings Singleton

```python
# core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    anthropic_api_key: str
    model: str = "claude-sonnet-4-6"
    max_iterations: int = 20
    datasets_dir: str = "./datasets"
    documents_dir: str = "./domain_docs"
    code_execution_backend: str = "subprocess"  # "subprocess" | "jupyter" | "anthropic"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

### The Anthropic Client Singleton

```python
# core/dependencies.py
from functools import lru_cache
import anthropic
from app.core.config import get_settings

@lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)
```

`@lru_cache(maxsize=1)` ensures the client is created once and reused. This matters because `anthropic.Anthropic()` creates an HTTP connection pool.

### The Service Graph

```python
# core/dependencies.py (continued)
from fastapi import Depends
from app.application.tool_executor import ToolExecutor
from app.application.services.data_science_agent import DataScienceAgentService

def get_tool_executor(
    settings: Settings = Depends(get_settings),
    client: anthropic.Anthropic = Depends(get_anthropic_client),
) -> ToolExecutor:
    from app.infrastructure.dataset_store import LocalDatasetStore
    from app.infrastructure.document_store import LocalDocumentStore
    from app.application.tools import build_all_tools
    
    dataset_store = LocalDatasetStore(settings.datasets_dir)
    document_store = LocalDocumentStore(settings.documents_dir)
    code_runner = get_code_runner(settings)
    
    executor = ToolExecutor()
    build_all_tools(executor, dataset_store, document_store, code_runner)
    return executor

def get_analysis_service(
    client: anthropic.Anthropic = Depends(get_anthropic_client),
    executor: ToolExecutor = Depends(get_tool_executor),
    settings: Settings = Depends(get_settings),
) -> DataScienceAgentService:
    return DataScienceAgentService(
        client=client,
        tool_executor=executor,
        model=settings.model,
        max_iterations=settings.max_iterations,
    )
```

---

## 8. Testing Strategy

Clean Architecture's greatest practical benefit is testability. Each layer can be tested independently, with progressively heavier dependencies.

### Domain Layer Tests — Pure Python, Zero Mocks

```python
# tests/domain/test_models.py
from app.domain.models import AnalysisSession, DatasetMeta, PhysicalUnit, AnalysisStatus

def test_analysis_session_defaults():
    session = AnalysisSession(session_id="test-123")
    assert session.messages == []
    assert session.status == AnalysisStatus.PENDING
    assert session.iteration_count == 0
    assert session.cells == []

def test_physical_unit_construction():
    unit = PhysicalUnit(
        quantity_name="efficiency",
        value=36.2,
        unit_str="percent",
        dimension="dimensionless",
        domain="thermodynamics",
    )
    assert unit.value == 36.2
    assert unit.domain == "thermodynamics"

def test_dataset_meta_format_inference():
    meta = DatasetMeta(
        name="turbine_data",
        path="/data/turbine_data.csv",
        format="csv",
        size_bytes=1024,
    )
    assert meta.row_count is None  # Not loaded yet
```

### Infrastructure Layer Tests — Mocked Filesystem and Network

```python
# tests/infrastructure/test_dataset_store.py
import pytest
import pandas as pd
from pathlib import Path
from app.infrastructure.dataset_store import LocalDatasetStore

@pytest.fixture
def temp_datasets_dir(tmp_path):
    # Create a CSV file in the temporary directory
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    df.to_csv(tmp_path / "test_data.csv", index=False)
    return tmp_path

def test_list_datasets_finds_csv(temp_datasets_dir):
    store = LocalDatasetStore(str(temp_datasets_dir))
    datasets = store.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].name == "test_data"
    assert datasets[0].format == "csv"

def test_load_dataframe_returns_correct_shape(temp_datasets_dir):
    store = LocalDatasetStore(str(temp_datasets_dir))
    df = store.load_dataframe("test_data")
    assert df.shape == (3, 2)
    assert list(df.columns) == ["x", "y"]

def test_load_nonexistent_raises():
    store = LocalDatasetStore("/nonexistent")
    with pytest.raises(FileNotFoundError):
        store.load_dataframe("ghost_dataset")
```

### Application Layer Tests — Mocked Infrastructure

```python
# tests/application/test_tool_executor.py
import pytest
from unittest.mock import MagicMock
from app.application.tool_executor import ToolExecutor

def test_executor_dispatches_registered_tool():
    executor = ToolExecutor()
    mock_fn = MagicMock(return_value="success")
    executor.register("my_tool", mock_fn)
    
    result = executor.execute("my_tool", {"arg1": "val1"})
    
    assert result == "success"
    mock_fn.assert_called_once_with(arg1="val1")

def test_executor_returns_error_for_unknown_tool():
    executor = ToolExecutor()
    result = executor.execute("nonexistent_tool", {})
    assert result.startswith("Error:")

def test_executor_catches_tool_exception():
    executor = ToolExecutor()
    def failing_tool(**kwargs):
        raise ValueError("something went wrong")
    executor.register("bad_tool", failing_tool)
    
    result = executor.execute("bad_tool", {})
    assert "Error:" in result
    assert "ValueError" in result
```

### Presentation Layer Tests — FastAPI TestClient

```python
# tests/api/test_analysis_endpoints.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_analysis_chat_creates_session(client):
    mock_response = "Analysis complete: efficiency is 36.2%"
    
    with patch("app.core.dependencies.get_analysis_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.run.return_value = mock_response
        mock_factory.return_value = mock_service
        
        response = client.post("/api/v1/analysis/chat", json={
            "message": "Analyze the turbine dataset",
            "session_id": None,
        })
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["response"] == mock_response

def test_analysis_chat_rejects_empty_message(client):
    response = client.post("/api/v1/analysis/chat", json={"message": ""})
    assert response.status_code == 422  # Validation error
```

---

*Next: [03_physical_domain_modeling.md](03_physical_domain_modeling.md) — How the system validates that results are physically meaningful.*
