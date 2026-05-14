# Data Scientist AI Agent — Project Overview

> **Comprehensive technical report** covering architecture, agent loop, tool system, data flow,
> and implementation details.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Technology Stack](#2-technology-stack)
3. [Repository Structure](#3-repository-structure)
4. [Architecture — Four-Layer Clean Architecture](#4-architecture--four-layer-clean-architecture)
5. [Domain Layer](#5-domain-layer)
6. [Core: The ReAct Agent Loop](#6-core-the-react-agent-loop)
7. [Core: The Python Function Toolbox](#7-core-the-python-function-toolbox)
8. [Infrastructure Layer](#8-infrastructure-layer)
9. [Presentation Layer — API](#9-presentation-layer--api)
10. [Configuration](#10-configuration)
11. [Data & Domain Knowledge](#11-data--domain-knowledge)
12. [Design Principles & Conventions](#12-design-principles--conventions)
13. [Known Issues & Gaps](#13-known-issues--gaps)

---

## 1. Project Summary

**Data Scientist AI Agent** is a domain-aware AI system that accepts natural language data science
requests via a REST API and returns reasoned answers, generated visualisations, and exportable
Jupyter notebooks.

Key differentiators:

| Feature | Description |
|---|---|
| **Text-based ReAct Loop** | Full `Thought → Action → Observation` reasoning loop via plain-text regex parsing — NOT Anthropic's native tool-use API |
| **15-Function Toolbox** | 7 knowledge + 6 execution + 3 physical validation tools |
| **Physical Unit Validation** | `pint`-backed registry enforces thermodynamic constraints (efficiency ≤ 100%, T > 0 K, etc.) |
| **Multi-backend Code Execution** | Subprocess (stateless, default) or Jupyter kernel (stateful, optional) |
| **Notebook Export** | Every session's reasoning trace exportable as `.ipynb` with code cells |
| **Domain Knowledge** | Agent reads/searches Markdown domain docs before analysing data |
| **Full Reasoning Transparency** | Every `Thought / Action / Observation` triple returned in the API response |

---

## 2. Technology Stack

| Component | Library / Version |
|---|---|
| Web framework | `FastAPI >= 0.115` + `Uvicorn` |
| AI SDK | `anthropic >= 0.50.0` (`AsyncAnthropic`) |
| Configuration | `pydantic-settings >= 2.0` + `.env` file |
| Data validation | `pydantic >= 2.0` |
| Physical units | `pint >= 0.24` |
| Data manipulation | `pandas >= 2.0`, `numpy >= 1.26` |
| Visualisation | `matplotlib >= 3.8`, `seaborn >= 0.13` |
| Notebook export | `nbformat >= 5.10` |
| Excel support | `openpyxl >= 3.1` |
| Parquet support | `pyarrow >= 14.0` |
| Package manager | `uv` |
| Python | `>= 3.12` (uses new-style generics: `list[str]`, `int \| None`) |
| Optional (Jupyter backend) | `jupyter_client >= 8.0`, `ipykernel >= 6.29` |
| Dev tooling | `pytest`, `pytest-asyncio`, `httpx`, `ruff` |

---

## 3. Repository Structure

```
ai-agent-data-scientist/
├── app/
│   ├── api/v1/
│   │   ├── analysis.py          # POST /chat, GET figures/notebook
│   │   └── datasets.py          # GET /datasets, GET /datasets/{name}/schema
│   ├── core/
│   │   └── config.py            # pydantic-settings — all env vars
│   ├── domain/
│   │   ├── analysis_models.py   # AnalysisSession, DatasetMeta, AnalysisResult, PhysicalUnit, NotebookCell
│   │   ├── exceptions.py        # AgentError hierarchy
│   │   └── models.py            # AgentSession (base)
│   ├── infrastructure/
│   │   ├── code_runner.py       # CodeRunner ABC, SubprocessCodeRunner, JupyterKernelManager, Factory
│   │   ├── notebook_exporter.py # nbformat export + figure save
│   │   ├── style_config.py      # matplotlib rcParams, color palette, STYLE_PREAMBLE
│   │   └── unit_registry.py     # pint singleton, DOMAIN_RANGES, 3 validation tools
│   ├── schemas/
│   │   ├── analysis.py          # AnalysisRequest, AnalysisResponse, ReActStep, FigureRef
│   │   └── datasets.py          # DatasetInfo, DatasetListResponse, DatasetSchemaResponse
│   ├── services/
│   │   ├── data_agent.py        # DataScienceAgentService — the ReAct loop
│   │   ├── data_tools.py        # Group B: 6 execution tools
│   │   ├── knowledge_tools.py   # Group A: 7 knowledge tools
│   │   ├── memory.py            # InMemorySessionStore, AnalysisSessionStore
│   │   └── tool_definitions.py  # 16 JSON schemas for all tools
│   └── main.py                  # FastAPI app factory + lifespan
├── data/
│   ├── datasets/                # CSV, Parquet, Excel, HDF5, JSON files
│   └── domain_docs/             # Markdown knowledge files (read by agent)
├── outputs/
│   ├── figures/                 # Saved PNG plots
│   └── notebooks/               # Exported .ipynb files
├── scripts/
│   ├── create_sample_data.py    # Generates sample datasets + domain docs
│   └── verify_install.py        # Dependency check
├── tests/                       # (empty — no tests yet)
├── design-doc/                  # Architecture diagrams and reference guides
├── .env.example
├── CLAUDE.md                    # Short developer reference
├── README.md
└── pyproject.toml
```

---

## 4. Architecture — Four-Layer Clean Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Presentation Layer                         │
│           app/api/v1/analysis.py · datasets.py              │
│       FastAPI routers — HTTP in, Pydantic models out         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Application Layer                           │
│   app/services/data_agent.py     ← ReAct Loop (core)        │
│   app/services/knowledge_tools.py   Group A  (7 tools)      │
│   app/services/data_tools.py        Group B  (6 tools)      │
│   app/services/tool_definitions.py  tool JSON schemas       │
│   app/services/memory.py            session stores          │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│               Infrastructure Layer                           │
│   app/infrastructure/code_runner.py     subprocess/jupyter  │
│   app/infrastructure/unit_registry.py  pint  Group C tools  │
│   app/infrastructure/notebook_exporter.py   nbformat        │
│   app/infrastructure/style_config.py        plot preamble   │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                    Domain Layer                              │
│   app/domain/analysis_models.py  AnalysisSession, entities  │
│   app/domain/models.py           AgentSession (base)        │
│   app/domain/exceptions.py       AgentError hierarchy       │
│   ZERO external imports — stdlib only                        │
└─────────────────────────────────────────────────────────────┘
```

**Dependency rule**: each layer only imports from layers below it. The domain layer has no imports
from any framework or third-party library.

---

## 5. Domain Layer

### `AgentSession` (`models.py`)

Base conversation container. Holds the flat message list in Anthropic wire format:

```python
@dataclass
class AgentSession:
    session_id: str
    messages: list[dict[str, Any]]   # [{"role": "user"|"assistant", "content": str|list}, ...]

    def add_user_message(content: str) -> None
    def add_assistant_message(content: Any) -> None
    def add_tool_results(tool_results: list[dict]) -> None
```

### `AnalysisSession` (`analysis_models.py`)

The **aggregate root** for one data science conversation. Extends `AgentSession` with:

```python
@dataclass
class AnalysisSession(AgentSession):
    datasets_loaded: dict[str, DatasetMeta]   # file_name → metadata snapshot
    figures:         dict[str, str]            # figure_id → base64 PNG
    notebook_cells:  list[NotebookCell]        # for explicit notebook building
    unit_context:    list[PhysicalUnit]        # all validation results this session
    react_trace:     list[dict[str, str]]      # append-only Thought/Action/Observation
    code_runner_state: dict[str, Any]          # opaque backend state
    notebook_path:   str | None                # set after export_notebook tool call
```

Factory method: `AnalysisSession.new()` creates a fresh session with a `uuid4` session ID.

### Value Objects (all frozen/immutable)

| Class | Fields | Purpose |
|---|---|---|
| `DatasetMeta` | `file_name, format, rows, columns, column_names, dtypes, numeric_stats, size_bytes` | Snapshot of a loaded dataset |
| `AnalysisResult` | `success, stdout, stderr, figures, execution_time_ms` | Result of one code execution |
| `PhysicalUnit` | `quantity, value, unit, is_valid, message, canonical_value, canonical_unit` | Validation verdict |
| `NotebookCell` | `cell_type, source, outputs, metadata` | One Jupyter notebook cell |

### Exception Hierarchy

```
AgentError (base)
├── SessionNotFoundError        — session_id not in store
├── AgentLoopError              — tool-use loop safety cap exceeded
├── ReActLoopError              — ReAct loop cannot converge (iterations, last_thought)
├── ReActParseError             — parser cannot extract Action from response text
├── CodeExecutionError          — code execution failed (backend, stderr, timeout)
├── PhysicalValidationError     — hard physical constraint violation
└── DatasetLoadError            — dataset cannot be loaded or parsed
```

---

## 6. Core: The ReAct Agent Loop

### Overview

`DataScienceAgentService` in `app/services/data_agent.py` orchestrates the entire reasoning loop.
It uses **text-based ReAct** (Reason + Act) — the agent is not using Anthropic's native
`tools=` parameter. Instead, Claude is prompted via a system prompt to output structured text,
which is regex-parsed by the service.

### The Conversation Protocol

Every Claude response must conform to exactly one of two shapes:

```
# Shape 1 — Tool invocation
Thought: I need to inspect the dataset first to understand its structure.
Action: inspect_dataset
Action Input: {"file_name": "power_plant_data.csv"}

# Shape 2 — Terminal (done)
Thought: I now have all data needed to answer the question.
Final Answer: The mean thermal efficiency is 36.73% (range: 33.1%–40.2%).
              Physical validation: ✅ within expected range [35–48%].
```

### The ReAct Loop — Step by Step

```
User message
    │
    ▼
session.add_user_message(user_message)
    │
    │  ┌──────────────────── for iteration in range(20) ─────────────────────┐
    │  │                                                                     │
    │  │  1. AsyncAnthropic.messages.create(                                 │
    │  │         model=settings.claude_model,                                │
    │  │         max_tokens=settings.max_tokens,                             │
    │  │         system=system_prompt,        ◄── tool list + rules          │
    │  │         messages=session.messages    ◄── full history               │
    │  │     )                                                               │
    │  │                  │ raw_text (plain text)                            │
    │  │                  ▼                                                  │
    │  │  2. parsed = _parse_react(raw_text)                                 │
    │  │                  │                                                  │
    │  │          ┌───────┼───────────────────┐                              │
    │  │    "final_answer" │            "parse_error"   "action"             │
    │  │          │        │                  │              │               │
    │  │        return   (terminal)     < 3 failures:  dispatch tool         │
    │  │        answer ◄──┘           inject correction    │                 │
    │  │                               ≥ 3 failures:        ▼                │
    │  │                               raise ReActLoopError  observation     │
    │  │                                                     │               │
    │  │  3. session.append_react_step(thought, action, observation)         │
    │  │  4. session.add_assistant_message("Thought:..Action:..Input:..")    │
    │  │  5. session.add_user_message("Observation: " + observation) ────────┘
    │  └─────────────────────────────────────────────────────────────────────┘
    │
    ▼
raise ReActLoopError if max_react_iterations reached without Final Answer
```

### The Parser (`_parse_react`)

Four compiled regex patterns applied in order:

```python
_THOUGHT_RE      = re.compile(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", re.DOTALL | re.IGNORECASE)
_ACTION_RE       = re.compile(r"Action:\s*([a-z_][a-z0-9_]*)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})\s*$", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)
```

Return values:

```python
{"type": "final_answer", "thought": str, "answer": str}
{"type": "action",       "thought": str, "action": str, "action_input": dict}
{"type": "parse_error",  "raw": str,     "reason": str}
```

JSON parsing fallback: if `json.loads()` fails on the `Action Input` value, the parser retries
with `ast.literal_eval()` to handle Python-dict-style strings (single quotes, trailing commas).

### The Tool Registry

Built fresh on every `.run()` call as `dict[str, Callable[..., str]]`. All callables are
lambdas that close over the current `session` and `runner`:

```python
def _build_tool_registry(session, runner) -> dict[str, Callable]:
    return {
        # Group A — pure functions, no session needed
        "list_domain_documents":   lambda _:   list_domain_documents(),
        "read_domain_document":    lambda inp: read_domain_document(inp["file_name"]),
        "search_domain_knowledge": lambda inp: search_domain_knowledge(inp["query"], inp.get("top_k", 3)),
        "get_coding_standards":    lambda _:   get_coding_standards(),
        "list_datasets":           lambda _:   list_datasets(),
        "inspect_dataset":         lambda inp: inspect_dataset(inp["file_name"]),
        "describe_columns":        lambda inp: describe_columns(inp["file_name"], inp["columns"]),
        # Group B — stateful, require session + runner
        "execute_python_code":     lambda inp: execute_python_code(inp["code"], session, runner),
        "get_execution_variables": lambda _:   get_execution_variables(runner),
        "get_figure":              lambda inp: get_figure(inp["figure_id"], session),
        "list_figures":            lambda _:   list_figures(session),
        "export_notebook":         lambda inp: export_notebook_tool(session, inp["title"]),
        "save_figure":             lambda inp: save_figure_tool(session, inp["figure_id"], inp["filename"]),
        # Group C — pint-backed, stateless
        "validate_physical_units": lambda inp: validate_physical_units(inp["quantity"], inp["value"], inp["unit"]),
        "convert_units":           lambda inp: convert_units(inp["value"], inp["from_unit"], inp["to_unit"]),
        "check_magnitude":         lambda inp: check_magnitude(inp["quantity"], inp["value"], inp["unit"]),
    }
```

Unknown tool name → `observation = "Error: Unknown tool '...'. Available: [...]"` — Claude
reads this and self-corrects on the next iteration.

### Message History Structure

After each tool call, **two messages** are appended to `session.messages`:

```python
# Assistant "said" the tool invocation
{"role": "assistant", "content": "Thought: ...\nAction: inspect_dataset\nAction Input: {\"file_name\": \"...\"}"}

# User "replied" with the observation
{"role": "user", "content": "Observation: {\"rows\": 500, \"columns\": 10, ...}"}
```

This creates a multi-turn conversation where Claude always sees the full reasoning history,
allowing it to build on previous observations.

### Observation Safeguard

Tool outputs longer than **8000 characters** are truncated:

```python
MAX_OBS = 8000
if len(observation) > MAX_OBS:
    observation = observation[:MAX_OBS] + f"\n[...truncated at {MAX_OBS} chars]"
```

### Error Recovery

| Scenario | Behaviour |
|---|---|
| Unknown tool name | Observation = error string listing valid tools |
| Tool raises exception | Caught by try/except in loop, observation = `"Error: Tool '...' failed — <exc>"` |
| Parse error (< 3) | Inject correction prompt, do NOT count as iteration failure |
| Parse error (≥ 3) | Raise `ReActLoopError` |
| Exceeded `max_react_iterations` | Raise `ReActLoopError` with last thought |

### System Prompt

The system prompt embeds:
- **Tool summary** (auto-generated from `TOOL_DEFINITIONS`)
- **12 workflow rules** (e.g., always call `get_coding_standards` first, always use `print()`)
- **Physical law reminders** (efficiency > 100% violates First Law)
- **Reporting standard** (structured summary at end of every Final Answer)

---

## 7. Core: The Python Function Toolbox

**Contract**: every tool function returns `str`, never raises. Errors are returned as
`"Error: <description>"` strings that Claude reads and self-corrects from.

---

### Group A — Knowledge Tools (`knowledge_tools.py`) — 7 functions

All functions are **pure** (read-only, no session state mutation), **path-safe**, and **stateless**.

#### Path Safety Guard

Every file access passes through `_safe_resolve()`:

```python
def _safe_resolve(base_dir: Path, file_name: str) -> Path | None:
    if ".." in file_name or file_name.startswith("/") or file_name.startswith("\\"):
        return None
    target = (base_dir / file_name).resolve()
    if base_dir.resolve() not in target.parents:
        return None
    return target
```

This prevents directory traversal attacks (e.g., `../../etc/passwd`).

#### Tool Details

**`list_domain_documents()`**
- Scans `data/domain_docs/` for `.md`, `.txt`, `.rst` files
- Returns `JSON: [{"file_name": str, "size_bytes": int}, ...]`

**`read_domain_document(file_name)`**
- Reads full file content, UTF-8 with replacement on bad bytes
- Hard cap: **50 KB** — truncated with notice
- Returns raw text or `"Error: ..."` string

**`search_domain_knowledge(query, top_k=3)`**
- Term-frequency keyword search across all `.md` and `.txt` files
- Scoring: count of query terms found in document
- Snippet extraction: finds best-density window (±150 chars around best match position)
- Returns `JSON: [{"file": str, "score": int, "snippet": str}, ...]` sorted by score

**`list_datasets()`**
- Scans `data/datasets/` for `.csv`, `.parquet`, `.xlsx`, `.xls`, `.h5`, `.hdf5`, `.json`
- Returns `JSON: [{"file_name": str, "format": str, "size_bytes": int}, ...]`

**`inspect_dataset(file_name)`**
- Loads dataset with `pandas` (auto-detects format from extension)
- Computes numeric stats: `{min, max, mean, std, null_count}` per numeric column
- Returns 5 sample rows
- Returns `JSON: {file_name, format, rows, columns, column_names, dtypes, numeric_stats, sample_rows}`

**`describe_columns(file_name, columns)`**
- Loads dataset, computes deep per-column stats:
  - **Numeric**: `count, null_count, min, max, mean, median, std, q25, q75, skewness`
  - **Datetime**: `count, null_count, min, max`
  - **Categorical**: `count, null_count, unique_count, top_values (top 10)`
- Returns `JSON: {column_name: {type: "numeric"|"datetime"|"categorical", stats...}}`

**`get_coding_standards()`**
- Thin wrapper: calls `read_domain_document("coding_standards.md")`
- Returns the full coding standards Markdown document

---

### Group B — Execution Tools (`data_tools.py`) — 6 functions

These functions are **stateful** — they read/write `AnalysisSession` and delegate to a
`CodeRunner` instance.

#### `execute_python_code(code, session, runner)`

The most important tool. Full execution pipeline:

```
code (str)
    │
    ▼
runner.execute(code)
    │  SubprocessCodeRunner:
    │    1. Prepend STYLE_PREAMBLE (imports, rcParams, color palette, helpers, plt.show() patch)
    │    2. Append _FIGURE_POSTAMBLE (emits __FIGURES__ and __STATE__ JSON lines on stdout)
    │    3. subprocess.run([sys.executable, "-c", full_code], timeout=30s)
    │    4. Parse stdout for __FIGURES__: and __STATE__: sentinel lines
    │    5. Return AnalysisResult(success, stdout, stderr, figures[], execution_time_ms)
    │
    ▼
For each new figure in result.figures:
    b64 = runner.get_figure_b64(fig_id)
    session.register_figure(session.next_figure_id, b64)
    │
    ▼
Return JSON: {
    "success": bool,
    "stdout": str,            # clean stdout (sentinel lines stripped)
    "stderr": str,            # only on failure
    "figures": [str, ...],    # session figure IDs registered this call
    "execution_time_ms": int
}
```

**Pre-imported in every execution** (via STYLE_PREAMBLE):
`pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `seaborn as sns`,
`json`, `base64`, `io`, `sys`, `os`, `pathlib.Path`

**Pre-configured helpers** (also injected):
`label_bars()`, `add_reference_line()`, `format_axis_units()`, `engineering_plot()`

#### `get_execution_variables(runner)`
- Calls `runner.get_state()` → `{variable_name: type_name}`
- Returns JSON dict (e.g., `{"df": "DataFrame", "efficiency_mean": "float64"}`)
- Only meaningful after at least one `execute_python_code` call in the same session

#### `get_figure(figure_id, session)`
- Looks up `session.figures[figure_id]` (base64 PNG string)
- Returns `JSON: {figure_id, format: "png", encoding: "base64", data: "<b64>"}`
- On miss: returns `JSON: {error: "...", available_figures: [...]}`

#### `list_figures(session)`
- Returns `JSON: {count: int, figure_ids: [str, ...]}`

#### `export_notebook_tool(session, title)`
- Delegates to `infrastructure/notebook_exporter.export_notebook()`
- Converts `session.react_trace` → notebook cells:
  - Each `Thought` → Markdown cell
  - Each `execute_python_code` action → Code cell with its output
  - Other actions → Markdown cell with action + observation
- Saves to `outputs/notebooks/<session_id[:8]>_<safe_title>.ipynb`
- Sets `session.notebook_path`
- Returns `JSON: {notebook_path, filename, cell_count, download_url}`

#### `save_figure_tool(session, figure_id, filename)`
- Decodes base64 PNG from `session.figures`
- Sanitizes filename (alphanumeric, `_`, `-` only)
- Writes to `outputs/figures/<filename>.png`
- Returns `JSON: {saved_to, figure_id, filename, size_bytes}`

---

### Group C — Physical Validation Tools (`unit_registry.py`) — 3 functions

These tools use `pint` for unit parsing and a domain range registry for plausibility checks.

#### `pint` Registry Setup

A module-level singleton is lazily initialised on first use:

```python
_ureg = pint.UnitRegistry()
# Custom units registered:
ureg.define("percent = 0.01 * [] = pct")
ureg.define("ppm = 1e-6 * [] = ppm")
ureg.define("ppb = 1e-9 * [] = ppb")
```

#### Domain Range Registry

```python
DOMAIN_RANGES: dict[str, tuple[float, float, str]] = {
    # quantity_key: (min, max, canonical_unit)
    "thermal_efficiency":    (0.0,     100.0,   "percent"),
    "isentropic_efficiency": (0.0,     100.0,   "percent"),
    "mechanical_efficiency": (0.0,     100.0,   "percent"),
    "efficiency":            (0.0,     100.0,   "percent"),
    "steam_temperature":     (-50.0,   700.0,   "degC"),
    "flue_gas_temperature":  (50.0,    500.0,   "degC"),
    "temperature":           (-273.15, 5000.0,  "degC"),
    "steam_pressure":        (0.001,   35.0,    "MPa"),
    "pressure":              (0.0,     1000.0,  "MPa"),
    "gross_power":           (0.0,     5000.0,  "MW"),
    "net_power":             (0.0,     5000.0,  "MW"),
    "power":                 (-10000.0,10000.0, "MW"),
    "heat_rate":             (2000.0,  20000.0, "kJ/kWh"),
    "co2_emission":          (0.0,     2000.0,  "g/kWh"),
    "mass_flow":             (0.0,     10000.0, "kg/s"),
}
```

**Fuzzy quantity matching** (`_fuzzy_match_quantity`):
1. Exact key match (case-insensitive)
2. Substring match — `"plant_thermal_efficiency"` → `"thermal_efficiency"`

#### `validate_physical_units(quantity, value, unit)`

Two-stage validation:

```
Stage 1: pint.Quantity(value, unit)
    → unit parse error? return PhysicalUnit(is_valid=False, message="Unit parse error: ...")

Stage 2: _fuzzy_match_quantity(quantity) → domain range
    → value outside [min, max]? return PhysicalUnit(is_valid=False, message="Magnitude ... outside [lo, hi]")
    → OK? return PhysicalUnit(is_valid=True, message="OK — ...")
```

Returns `JSON: {quantity, value, unit, is_valid, message, canonical_value, canonical_unit}`

#### `convert_units(value, from_unit, to_unit)`

```python
q = ureg.Quantity(value, from_unit)
converted = q.to(to_unit)
```

Returns `JSON: {original_value, original_unit, converted_value, converted_unit, success}`

#### `check_magnitude(quantity, value, unit)`

Plausibility-only check (no unit parsing):

```python
lo, hi, canonical = DOMAIN_RANGES[matched_key]
is_plausible = lo <= value <= hi
```

Returns `JSON: {quantity, matched_domain_key, value, unit, expected_range, is_plausible, message}`

---

## 8. Infrastructure Layer

### Code Execution (`code_runner.py`)

#### Abstract Interface

```python
class CodeRunner(abc.ABC):
    def execute(self, code: str) -> AnalysisResult: ...     # NEVER raises
    def get_state(self) -> dict[str, str]: ...              # var_name → type_name
    def get_figure_b64(self, figure_id: str) -> str | None: ...
    def shutdown(self) -> None: ...                          # idempotent
```

#### `SubprocessCodeRunner` (default)

- **Stateless**: each `execute()` call runs in a fresh Python subprocess — variables do NOT
  persist between calls
- Figures captured by patching `plt.show()` in `STYLE_PREAMBLE`
- Embedded JSON sentinels on stdout: `__FIGURES__:{...}` and `__STATE__:{...}`
- Hard timeout: `CODE_EXECUTION_TIMEOUT` (default 30s) via `subprocess.TimeoutExpired`

#### `JupyterKernelManager` (optional)

- **Stateful**: uses `jupyter_client` to start a persistent `python3` kernel
- Variables persist between `execute()` calls (true REPL)
- Figures captured from `display_data` IOPub messages (`image/png` MIME type)
- Requires: `pip install jupyter_client ipykernel`
- Kernel started lazily on first `execute()` call

#### `CodeRunnerFactory`

```python
class CodeRunnerFactory:
    @staticmethod
    def create(session_id: str | None = None) -> CodeRunner:
        if backend == "subprocess": return SubprocessCodeRunner(session_id)
        if backend == "jupyter":    return JupyterKernelManager(session_id)
        raise ValueError(f"Unknown backend: '{backend}'")
```

One `CodeRunner` instance is maintained per session in
`DataScienceAgentService._runners: dict[str, CodeRunner]`.

### Style Configuration (`style_config.py`)

`STYLE_PREAMBLE` is a raw string prepended to every code execution. It provides:

| Component | Detail |
|---|---|
| Global rcParams | 150 DPI, white background, grid, no top/right spines, bold titles |
| `COLORS` dict | `blue, orange, green, purple, teal, amber, brown, bluegrey` (colorblind-safe) |
| `PALETTE` list | `list(COLORS.values())` — passed to `sns.set_theme` |
| Semantic colors | `C_GOOD (#4CAF50)`, `C_WARN (#FFC107)`, `C_BAD (#F44336)`, `C_NEUTRAL (#9E9E9E)` |
| `label_bars(ax)` | Annotates bar chart bars with their values |
| `add_reference_line(ax, value)` | Draws horizontal reference/limit line |
| `format_axis_units(ax, xlabel, ylabel, title)` | Sets labels + `tight_layout()` |
| `engineering_plot(nrows, ncols)` | Creates pre-styled figure with optional title |
| Figure capture | `plt.show` patched → base64 PNG → `_FIGURES` dict → `__FIGURES__:` stdout line |

### Notebook Exporter (`notebook_exporter.py`)

`export_notebook(session, title)` converts `session.react_trace` to a `.ipynb` file:

```
react_trace step → notebook cells
─────────────────────────────────
thought              → Markdown cell: "**Thought N:** ..."
execute_python_code  → Code cell (code extracted via regex from action string)
                       + output stream cell with observation text
other actions        → Markdown cell: "**Action:** ... **Observation:** ..."
session.notebook_cells → appended verbatim
```

Notebook metadata includes `kernelspec: python3`.
Filename: `<session_id[:8]>_<safe_title[:50]>.ipynb`

---

## 9. Presentation Layer — API

### FastAPI App (`main.py`)

```python
app = FastAPI(title="Data Scientist AI Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.include_router(analysis_router, prefix="/api/v1/analysis")
app.include_router(datasets_router, prefix="/api/v1/datasets")
```

Lifespan: configures logging + calls `settings.ensure_directories()` on startup.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/analysis/chat` | Main endpoint — runs ReAct loop, returns full trace |
| `GET` | `/api/v1/analysis/{session_id}/figures/{figure_id}` | Returns figure as `image/png` |
| `GET` | `/api/v1/analysis/{session_id}/notebook` | Downloads `.ipynb` (requires prior export) |
| `GET` | `/api/v1/datasets` | Lists all available datasets |
| `GET` | `/api/v1/datasets/{name}/schema` | Returns schema + stats + sample rows |
| `GET` | `/health` | Liveness probe — returns env, model, backend |

### Request / Response Models

**`AnalysisRequest`**:

```python
class AnalysisRequest(BaseModel):
    message:      str        # required, 1–10000 chars, stripped
    session_id:   str | None # optional — alphanumeric+hyphens, max 64 chars
    dataset_hint: str | None # optional — not yet consumed by the agent
```

**`AnalysisResponse`**:

```python
class AnalysisResponse(BaseModel):
    response:          str            # Final Answer text
    session_id:        str
    react_trace:       list[ReActStep]      # full Thought/Action/Observation list
    figures:           list[FigureRef]      # figure_id + retrieval URL
    notebook_available: bool
    unit_validations:  list[dict]           # all PhysicalUnit verdicts
    iterations_used:   int
    model:             str                  # e.g. "claude-sonnet-4-6"
    status:            str                  # "completed" | "error"
```

### Session Management

`AnalysisSessionStore` is a plain in-memory `dict[str, AnalysisSession]` (no persistence):

```python
class AnalysisSessionStore:
    def get_or_create(session_id: str) -> AnalysisSession
    def get(session_id: str) -> AnalysisSession | None
    def save(session: AnalysisSession) -> None
    def delete(session_id: str) -> None
    def active_sessions -> int   # property
```

Sessions are lost on server restart. The store is noted as "swappable to Redis".

---

## 10. Configuration

All settings are in `app/core/config.py` via `pydantic-settings` (reads `.env` file):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Wrapped in `SecretStr` — never logged |
| `ANTHROPIC_BASE_URL` | `None` | Optional base URL override |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model identifier |
| `MAX_TOKENS` | `8192` | Max tokens per API call |
| `MAX_RETRIES` | `2` | Anthropic SDK retry count |
| `APP_ENV` | `development` | Environment label |
| `DEBUG` | `False` | Sets log level to DEBUG if true |
| `DATA_DIR` | `data/` | Root data directory |
| `DOMAIN_DOCS_DIR` | `data/domain_docs/` | Domain knowledge documents |
| `DATASETS_DIR` | `data/datasets/` | Dataset files |
| `CODE_EXECUTION_BACKEND` | `subprocess` | `subprocess` or `jupyter` |
| `CODE_EXECUTION_TIMEOUT` | `30` | Seconds before code timeout |
| `ENABLE_JUPYTER_BRIDGE` | `False` | Enable Jupyter bridge (currently unused) |
| `MAX_REACT_ITERATIONS` | `20` | Max ReAct loop iterations per request |
| `FIGURES_DIR` | `outputs/figures/` | Saved PNG output |
| `NOTEBOOKS_DIR` | `outputs/notebooks/` | Exported `.ipynb` output |

---

## 11. Data & Domain Knowledge

### Sample Datasets (generated by `scripts/create_sample_data.py`)

| File | Format | Rows | Key Columns |
|---|---|---|---|
| `power_plant_data.csv` | CSV | 500 | `timestamp, steam_temp_C, steam_pressure_MPa, gross_power_MW, heat_rate_kJ_kWh, efficiency_pct, co2_emission_g_kWh` |
| `turbine_efficiency.parquet` | Parquet | 100 | `load_pct, isentropic_efficiency, mechanical_efficiency, stage` |
| `sensor_readings.xlsx` | Excel | 50 | `sensor_id, location, temp_K, pressure_kPa, flow_kg_s` |

### Domain Documents

| File | Content |
|---|---|
| `power_plant_thermodynamics.md` | Rankine cycle overview, KPI table (typical ranges + units), efficiency formulas, hard physical constraints |
| `unit_definitions.md` | Unit systems for temperature, pressure, power, mass flow, efficiency, emission factors |
| `coding_standards.md` | Python conventions, 7-step analysis workflow, visualization recipes (loaded via `get_coding_standards` tool) |

---

## 12. Design Principles & Conventions

| Principle | Implementation |
|---|---|
| **Tools never raise** | All 16 tool functions return `"Error: ..."` strings on failure; Claude self-corrects |
| **Domain layer purity** | `app/domain/` imports only stdlib — no fastapi, pydantic, pandas, etc. |
| **Path traversal protection** | `_safe_resolve()` guards all knowledge tool file access |
| **SecretStr** | `ANTHROPIC_API_KEY` wrapped in Pydantic `SecretStr` — never appears in logs or `repr()` |
| **Observation cap** | Tool outputs truncated at 8000 chars to avoid token overflow |
| **Parse error recovery** | Up to 3 parse failures tolerated — correction prompt injected before raising |
| **Figure ID scoping** | Runner uses its own counter; session assigns its own sequential IDs (`fig_000`, `fig_001`, ...) |
| **Stateless tools vs stateful tools** | Group A (knowledge) = pure functions; Group B (execution) = mutates session |
| **Python 3.12+ generics** | `list[str]`, `dict[str, Any]`, `int \| None` — no `typing.List`, `typing.Optional` |
| **Async throughout** | Presentation + service layers use `async/await`; infrastructure is sync (subprocess is blocking) |

---

## 13. Known Issues & Gaps

### Bug: Missing `import textwrap` in `code_runner.py`

**File**: `app/infrastructure/code_runner.py`, line 62

```python
_FIGURE_POSTAMBLE = textwrap.dedent("""...""")  # ← NameError: textwrap not imported
```

`textwrap` is used but never imported. This raises `NameError` at module load time, preventing
the entire application from starting. **Fix**: add `import textwrap` to the imports section.

### No Tests

The `tests/` directory is empty. There are no unit, integration, or end-to-end tests. Key areas
that need coverage:
- `_parse_react()` — the most fragile part of the system
- `_safe_resolve()` — security-critical path guard
- Tool functions (all return-str contract)
- `validate_physical_units()` / `check_magnitude()` domain range logic

### `dataset_hint` Field Unimplemented

`AnalysisRequest.dataset_hint` is accepted in the schema but silently ignored by
`data_agent.py`. It was intended to pre-load a dataset before the ReAct loop starts.

### Subprocess Backend is Stateless

The default `SubprocessCodeRunner` starts a fresh Python process per `execute_python_code` call.
Variables do NOT persist. Claude must reload datasets in every code block. The Jupyter backend
solves this, but requires optional dependencies (`jupyter_client`, `ipykernel`).

### In-Memory Sessions — No Persistence

All `AnalysisSession` objects live in a Python dict. Server restart loses all sessions.
The code comment acknowledges this is "swappable to Redis" but no implementation exists.

### Tool Count Discrepancy

`CLAUDE.md`, `README.md`, and docstrings say "15 tools" but `TOOL_DEFINITIONS` actually contains
**16 entries**: Group A has 7 (not 6 — `get_coding_standards` is a separate tool), Group B has 6,
Group C has 3.

### No Rate Limiting or Auth

The API has no authentication, rate limiting, or session ownership checks. Any client can
read any session's figures or notebook by guessing the session ID (UUID4, so hard to guess but
not impossible in a shared deployment).

---

*Generated from codebase analysis — `ai-agent-data-scientist` v0.1.0*
