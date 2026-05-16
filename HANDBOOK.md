# Data Scientist AI Agent — Engineering Handbook

> **Audience:** Engineers onboarding to the project, code reviewers, and anyone extending or maintaining the system.
> This document covers architecture, data flow, the ReAct loop lifecycle, every exception class and its propagation path, all 16 robustness improvements, and the test strategy.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Architecture — Four Layers](#3-architecture--four-layers)
4. [Domain Model](#4-domain-model)
5. [The ReAct Reasoning Loop](#5-the-react-reasoning-loop)
6. [The 16 Tools](#6-the-16-tools)
7. [Code Execution Backends](#7-code-execution-backends)
8. [Exception Hierarchy](#8-exception-hierarchy)
9. [Exception Propagation Map](#9-exception-propagation-map)
10. [16 Robustness Improvements (Gap Audit)](#10-16-robustness-improvements-gap-audit)
11. [Session Lifecycle & TTL Eviction](#11-session-lifecycle--ttl-eviction)
12. [Physical Unit Validation](#12-physical-unit-validation)
13. [API Reference](#13-api-reference)
14. [Configuration Reference](#14-configuration-reference)
15. [Running the Service](#15-running-the-service)
16. [Test Strategy & Coverage](#16-test-strategy--coverage)
17. [Adding a New Tool](#17-adding-a-new-tool)
18. [Adding a New Exception](#18-adding-a-new-exception)
19. [Operational Runbook](#19-operational-runbook)

---

## 1. Project Overview

This project is a **domain-aware Data Scientist AI Agent** built on:

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI 0.115+ |
| LLM | Anthropic Claude (SDK `AsyncAnthropic`) |
| Reasoning protocol | ReAct (Reason + Act) text loop |
| Code execution | subprocess (default) or Jupyter kernel |
| Physical validation | `pint` unit registry |
| Notebook export | `nbformat` |
| Configuration | `pydantic-settings` v2 with `.env` |
| Python version | 3.12+ |

**Core value proposition:** The agent can answer data science questions that require reading domain documents, loading datasets, executing Python code, generating plots, and validating physical quantities (e.g., thermal efficiency must be 0–100 %).

---

## 2. Repository Layout

```
ai-agent-data-scientist/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── analysis.py        # POST /chat, GET /figures, GET /notebook
│   │       └── datasets.py        # GET /datasets listing
│   ├── core/
│   │   └── config.py              # pydantic-settings Settings singleton
│   ├── domain/
│   │   ├── analysis_models.py     # AnalysisSession, DatasetMeta, AnalysisResult, PhysicalUnit
│   │   ├── exceptions.py          # Full exception hierarchy (zero external deps)
│   │   └── models.py              # AgentSession base (legacy MVP)
│   ├── infrastructure/
│   │   ├── code_runner.py         # SubprocessCodeRunner, JupyterKernelManager, factory
│   │   ├── notebook_exporter.py   # nbformat-based .ipynb writer
│   │   ├── style_config.py        # STYLE_PREAMBLE injected before user code
│   │   └── unit_registry.py       # pint registry + 13 domain quantity ranges
│   ├── services/
│   │   ├── data_agent.py          # DataScienceAgentService — ReAct loop orchestrator
│   │   ├── data_tools.py          # Group B: 6 execution tools (session-stateful)
│   │   ├── knowledge_tools.py     # Group A: 7 knowledge tools (read-only, pure)
│   │   ├── memory.py              # InMemorySessionStore, AnalysisSessionStore (TTL)
│   │   └── tool_definitions.py    # 16 JSON schemas for TOOL_DEFINITIONS
│   └── main.py                    # FastAPI app factory, lifespan, GC background task
├── data/
│   ├── datasets/                  # CSV / Parquet / Excel datasets
│   └── domain_docs/               # Markdown domain knowledge files
├── outputs/
│   ├── figures/                   # Saved PNG figures
│   └── notebooks/                 # Exported .ipynb files
├── tests/
│   ├── conftest.py                # Shared fixtures, Anthropic exception builders
│   ├── domain/test_exceptions.py
│   ├── services/
│   │   ├── test_memory.py
│   │   ├── test_knowledge_tools.py
│   │   ├── test_data_tools.py
│   │   └── test_data_agent.py
│   ├── infrastructure/test_code_runner.py
│   └── api/test_analysis.py
├── scripts/
│   ├── create_sample_data.py
│   └── verify_install.py
├── design-doc/                    # SRS, architecture diagrams, ADRs
├── pyproject.toml
└── HANDBOOK.md                    ← this file
```

---

## 3. Architecture — Four Layers

```
┌─────────────────────────────────────────────────────────┐
│  Presentation  (app/api/v1/)                            │
│  FastAPI routers — HTTP ↔ domain object translation     │
│  Exception → HTTP status code mapping                   │
├─────────────────────────────────────────────────────────┤
│  Application   (app/services/)                          │
│  DataScienceAgentService — ReAct loop, tool dispatch    │
│  AnalysisSessionStore — in-memory session with TTL      │
│  16 tools in knowledge_tools.py + data_tools.py         │
├─────────────────────────────────────────────────────────┤
│  Infrastructure  (app/infrastructure/)                  │
│  CodeRunner (subprocess / Jupyter / Anthropic)          │
│  UnitRegistry (pint)                                    │
│  NotebookExporter (nbformat)                            │
├─────────────────────────────────────────────────────────┤
│  Domain  (app/domain/)                                  │
│  AnalysisSession, DatasetMeta, AnalysisResult           │
│  PhysicalUnit, NotebookCell                             │
│  Full exception hierarchy — ZERO external imports       │
└─────────────────────────────────────────────────────────┘
```

**Key rule:** The domain layer imports nothing outside the Python stdlib. All other layers may import from layers below them; they must never import from layers above.

---

## 4. Domain Model

### `AnalysisSession` (aggregate root)

```python
@dataclass
class AnalysisSession(AgentSession):
    datasets_loaded: dict[str, DatasetMeta]   # file_name → meta
    figures: dict[str, str]                   # figure_id → base64 PNG
    notebook_cells: list[NotebookCell]
    unit_context: list[PhysicalUnit]          # all validation verdicts
    react_trace: list[dict[str, str]]         # append-only T/A/O triples
    code_runner_state: dict[str, Any]
    notebook_path: str | None
```

**Mutation helpers** (all inside the session class):
- `register_dataset(meta)` — keyed by `meta.file_name`
- `register_figure(figure_id, b64_png)` — enforces non-empty
- `append_react_step(thought, action, observation)` — append-only trace
- `log_unit_validation(unit)` — appends `PhysicalUnit`
- `append_notebook_cell(cell)`

### `DatasetMeta` (frozen value object)

Invariants enforced in `__post_init__`: `rows >= 0`, `columns >= 0`, `file_name` is a basename (no `/` or `\`).

### `AnalysisResult` (frozen value object)

`success: bool` — `True` means exit code 0. If `False`, `stderr` must explain the failure. Never raises.

### Message History (`session.messages`)

The `messages` list is the Anthropic-compatible chat history passed to `_client.messages.create(…)`. It grows with every `add_user_message` / `add_assistant_message` call. On LLM API error it is **rolled back** to a checkpoint taken before the call (see §10 Gap 2).

---

## 5. The ReAct Reasoning Loop

### Text Protocol

Claude is instructed to emit **one of two formats** per response:

**Action format:**
```
Thought: <reasoning about what to do next>
Action: <tool_name>
Action Input: {"param": "value"}
```

**Final answer format:**
```
Thought: <final reasoning>
Final Answer: <complete answer to the user>
```

### Loop Lifecycle (per `run()` call)

```
run(session, user_message)
│
├─ checkpoint = len(session.messages)          ← snapshot for rollback
│
└─ _run_loop(session, user_message, ...)
   │
   ├─ session.add_user_message(user_message)
   │
   └─ for iteration in range(MAX_REACT_ITERATIONS):
      │
      ├─ _call_claude_with_retry(session, system_prompt)
      │   ├─ OK  → raw_text
      │   └─ Exc → LLMAuthenticationError | LLMContextOverflowError | LLMAPIError
      │              (raised → propagates to run() → ROLLBACK → re-raise)
      │
      ├─ _parse_react(raw_text)
      │   ├─ {"type": "final_answer"} → return answer ✓
      │   ├─ {"type": "action"}       → dispatch tool
      │   └─ {"type": "parse_error"}
      │       ├─ count < 3 → inject correction message, continue
      │       └─ count >= 3 → raise ReActLoopError (NO rollback)
      │
      ├─ tool_registry[action_name](action_input)
      │   ├─ OK  → observation string
      │   └─ Exception → caught here → observation = "Error: …"
      │
      └─ session.add_assistant_message + session.add_user_message(observation)
         session.append_react_step(thought, action, observation)
   │
   └─ raise ReActMaxIterationsError (MAX_REACT_ITERATIONS exhausted)
```

### Parse Error Sentinel

Every parse failure is recorded in `react_trace` with `action="__parse_error__"`. This distinguishes genuine tool calls from protocol failures when inspecting the trace later.

### Rollback Semantics

| Exception type | Rollback? | Reason |
|----------------|-----------|--------|
| `LLMAuthenticationError` | ✅ Yes | Key is invalid; session state stays clean for retry |
| `LLMContextOverflowError` | ✅ Yes | History is too long; user must start new session |
| `LLMAPIError` (generic) | ✅ Yes | Transient error; partial state would be inconsistent |
| `ReActLoopError` | ❌ No | Partial trace is valuable for debugging |
| `ReActMaxIterationsError` | ❌ No | Same as above |

---

## 6. The 16 Tools

Tools are registered in `DataScienceAgentService._build_tool_registry()`. All tool functions return `str`. **They never raise** — errors are returned as `"Error: ..."` strings so Claude can self-correct.

### Group A — Knowledge Tools (7)

| Tool | Module | Description |
|------|--------|-------------|
| `list_domain_documents` | `knowledge_tools` | Lists `.md/.txt/.rst` files in `domain_docs_dir`. Per-file `stat()` errors yield `size_bytes=0`. |
| `read_domain_document` | `knowledge_tools` | Reads a single doc, truncates at 50 KB. Path-safety guarded. |
| `search_domain_knowledge` | `knowledge_tools` | Keyword search across all `.md/.txt` files. Returns ranked snippets. |
| `list_datasets` | `knowledge_tools` | Lists datasets with format and size. Per-file `stat()` errors yield `size_bytes=0`. |
| `inspect_dataset` | `knowledge_tools` | Loads dataset, returns schema + stats. **Rejects files > `max_dataset_bytes` (200 MB default).** |
| `describe_columns` | `knowledge_tools` | Per-column statistics. **Same size guard.** |
| `get_coding_standards` | `knowledge_tools` | Reads `coding_standards.md`. Called at the start of every analysis. |

### Group B — Execution Tools (6)

| Tool | Module | Description |
|------|--------|-------------|
| `execute_python_code` | `data_tools` | Runs code in a `CodeRunner`. Captures stdout + figures. |
| `get_execution_variables` | `data_tools` | Returns `{name: type_name}` dict from last execution state. |
| `get_figure` | `data_tools` | Returns **metadata + `retrieval_url`** only — **no raw base64** (Gap 7). |
| `list_figures` | `data_tools` | Returns `{count, figure_ids}`. |
| `export_notebook` | `data_tools` | Delegates to `notebook_exporter.export_notebook()`. |
| `save_figure` | `data_tools` | Delegates to `notebook_exporter.save_figure()`. |

> **Why no base64 from `get_figure`?** Returning the raw PNG inline to the LLM would consume the entire context window for even modest images. The figure is served separately via `GET /api/v1/analysis/{session_id}/figures/{figure_id}`.

### Group C — Validation Tools (3)

| Tool | Module | Description |
|------|--------|-------------|
| `validate_physical_units` | `unit_registry` | Validates unit parseability + domain range. Logs to `session.unit_context`. |
| `convert_units` | `unit_registry` | Converts using `pint`. Returns JSON with original and converted values. |
| `check_magnitude` | `unit_registry` | Checks plausibility without requiring unit conversion. |

---

## 7. Code Execution Backends

### Selection

`CODE_EXECUTION_BACKEND` setting → `CodeRunnerFactory.create(session_id)`.

`VALID_BACKENDS = frozenset({"subprocess", "jupyter", "anthropic"})`. Any other value raises `ValueError` at **startup** (fail-fast, Gap 9).

### `SubprocessCodeRunner` (default)

- Each `execute()` spawns a fresh Python subprocess.
- The STYLE_PREAMBLE is prepended (imports `pd`, `np`, `plt`, `sns`; patches `plt.show()`).
- On **Unix**: `preexec_fn=_apply_subprocess_limits` is called after fork to set `RLIMIT_AS=512 MB` and `RLIMIT_CPU=timeout+10s` (Gap 13).
- On **Windows**: `_PREEXEC_FN = None` (resource module unavailable).
- Timeout → `AnalysisResult(success=False, stderr="Execution timed out…")`.
- Any `subprocess.run` failure → `AnalysisResult(success=False, stderr=…)`.
- **Variables do not persist** between calls.

### `JupyterKernelManager` (optional)

- Requires `pip install jupyter_client ipykernel` (`uv sync --extra jupyter`).
- Variables **persist** between `execute()` calls (true REPL state).
- Before every `execute()`:
  1. `_probe_alive()` — sends `kernel_info` request, waits 3 s for reply.
  2. If dead → `_restart_kernel()` — shuts down old kernel, starts new one.
  3. If restart fails → raises `KernelCrashError` → `execute()` returns `AnalysisResult(success=False)`.

### `AnthropicCodeExecRunner` (stub)

- Recognised backend name, but raises `ValueError("Backend 'anthropic' is recognised but not yet implemented.")`.

---

## 8. Exception Hierarchy

```
Exception
└── AgentError                          # base for all custom exceptions
    ├── SessionNotFoundError(session_id)
    ├── AgentLoopError
    ├── ReActLoopError(reason, iterations, last_thought)
    │   └── ReActMaxIterationsError     # loop hit MAX_REACT_ITERATIONS
    ├── ReActParseError(raw_text, reason)
    ├── CodeExecutionError(message, backend, stderr, timeout)
    │   └── KernelCrashError(session_id, reason)   # backend="jupyter"
    ├── PhysicalValidationError(quantity, reason)
    ├── DatasetLoadError(file_name, reason)
    └── LLMAPIError(message, status_code=None)
        ├── LLMContextOverflowError     # "prompt is too long" → HTTP 400
        └── LLMAuthenticationError     # invalid API key → HTTP 502
```

### Class Details

| Class | Attributes | Typical Cause |
|-------|-----------|---------------|
| `SessionNotFoundError` | `session_id: str` | Client uses an expired/unknown session_id |
| `ReActLoopError` | `iterations: int`, `last_thought: str` | Loop could not converge; parse failures; tool kept returning errors |
| `ReActMaxIterationsError` | inherits above | MAX_REACT_ITERATIONS (default 20) exhausted |
| `ReActParseError` | `raw_text: str`, `reason: str` | Claude response didn't match expected format |
| `CodeExecutionError` | `backend: str`, `stderr: str`, `timeout: bool` | Subprocess exit ≠ 0, Python exception inside code |
| `KernelCrashError` | `session_id: str` (+ `backend="jupyter"`) | Jupyter kernel died and could not be restarted |
| `PhysicalValidationError` | `quantity: str`, `reason: str` | A computed value fails hard domain range check |
| `DatasetLoadError` | `file_name: str`, `reason: str` | File missing, unsupported format, or corrupted |
| `LLMAPIError` | `status_code: int \| None` | Any unclassified Anthropic API error |
| `LLMContextOverflowError` | inherits `status_code` | Message history too long for model context window |
| `LLMAuthenticationError` | inherits `status_code` | API key is invalid, expired, or revoked |

---

## 9. Exception Propagation Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  anthropic SDK raises                                               │
│  AuthenticationError → LLMAuthenticationError                      │
│  BadRequestError("prompt is too long")→ LLMContextOverflowError    │
│  BadRequestError (other) → LLMAPIError(status_code=400)           │
│  APIError (any other)   → LLMAPIError(status_code=exc.status_code) │
│  Exception (unexpected) → LLMAPIError(status_code=None)           │
└────────────────────┬────────────────────────────────────────────────┘
                     │ raised by _call_claude_with_retry()
                     │
         ┌───────────▼──────────────┐
         │  run()  — catches ALL    │
         │  LLMAPIError subclasses  │
         │  → rolls back messages   │
         │  → re-raises             │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────────────────────────────────────┐
         │  analysis_chat()  in  app/api/v1/analysis.py             │
         │                                                           │
         │  LLMContextOverflowError  → HTTP 400  {"error":          │
         │                              "context_overflow"}         │
         │  LLMAuthenticationError   → HTTP 502  {"error":          │
         │                              "llm_auth_error"}           │
         │  LLMAPIError              → HTTP 502  {"error":          │
         │                              "llm_api_error"}            │
         │  ReActLoopError           → HTTP 200  {"status":"error"} │
         │  AgentError (any other)   → HTTP 200  {"status":"error"} │
         │  Exception (unhandled)    → HTTP 200  {"status":"error"} │
         └──────────────────────────────────────────────────────────┘

Tool errors (inside _run_loop):
  tool_registry[name](input)
  ├─ Returns "Error: …" string → Claude reads it, self-corrects
  └─ Exception raised by tool  → caught in _run_loop
       → observation = f"Error: Tool '{name}' failed — {exc}"
       → logged as ERROR; loop continues

Jupyter probe/restart errors:
  _probe_alive() → False
  └─ _restart_kernel()
     ├─ Success → execution continues
     └─ Exception → KernelCrashError raised
        └─ execute() catches KernelCrashError
           → AnalysisResult(success=False, stderr=str(exc))
           → returned as "Error: …" observation to Claude

Figure base64 decoding (GET /figures/{figure_id}):
  base64.b64decode(b64)
  └─ Exception → HTTP 422 {"error": "corrupted_figure"}
```

---

## 10. Sixteen Robustness Improvements (Gap Audit)

These were identified by auditing the original codebase against expected production behaviour. Each gap has a corresponding test.

| # | Gap | File(s) changed | What was done |
|---|-----|----------------|---------------|
| 1 | LLM exceptions not classified | `data_agent.py` | `_call_claude_with_retry()` maps every `anthropic.*Error` to `LLM*Error` subclass |
| 2 | No session rollback on LLM error | `data_agent.py` | `checkpoint = len(messages)` before loop; `messages[:checkpoint]` on `LLMAPIError` |
| 3 | Context overflow → generic 500 | `analysis.py` | `LLMContextOverflowError` → HTTP 400 with `"context_overflow"` detail |
| 4 | No dataset file-size guard | `knowledge_tools.py` | `inspect_dataset` + `describe_columns` check `settings.max_dataset_bytes` before loading |
| 5 | Leaked sessions (no eviction) | `memory.py`, `main.py` | TTL eviction in `AnalysisSessionStore`; GC background task in lifespan |
| 6 | Corrupted figure base64 → 500 | `analysis.py` | `base64.b64decode` guarded → HTTP 422 `"corrupted_figure"` |
| 7 | `get_figure` returns raw base64 to LLM | `data_tools.py` | Returns `{figure_id, format, size_bytes, retrieval_url, note}` — no data field |
| 8 | Dead Jupyter kernel not recovered | `code_runner.py` | `_probe_alive()` + `_restart_kernel()` called before every `execute()` |
| 9 | Invalid backend config fails silently | `code_runner.py`, `main.py` | `VALID_BACKENDS` frozenset + startup probe (`sys.exit(1)` on `ValueError`) |
| 10 | `stat()` errors crash listing tools | `knowledge_tools.py` | Per-file `try/except OSError` in `list_domain_documents` + `list_datasets` |
| 11 | Notebook export errors not logged | `notebook_exporter.py` | `logger.debug()` before markdown fallback in inner `except` |
| 12 | No `dataset_hint` pre-load | `analysis.py` | `_try_preload_dataset()` swallows all errors; pre-registers `DatasetMeta` |
| 13 | No subprocess resource limits | `code_runner.py` | `_apply_subprocess_limits()` preexec_fn: `RLIMIT_AS=512 MB`, `RLIMIT_CPU=timeout+10s` |
| 14 | Hardcoded CORS origins | `config.py`, `main.py` | `cors_origins: list[str]` configurable via `CORS_ORIGINS` env var |
| 15 | No session TTL config | `config.py`, `memory.py` | `session_ttl_seconds: int = 3600`; `AnalysisSessionStore(ttl_seconds=…)` |
| 16 | Parse errors invisible in trace | `data_agent.py` | `action="__parse_error__"` sentinel in `react_trace`; correction message injected |

---

## 11. Session Lifecycle & TTL Eviction

### Session creation and access

```
POST /api/v1/analysis/chat
  │
  ├─ session_id provided?
  │   YES → analysis_session_store.get_or_create(session_id)
  │   NO  → uuid4() → get_or_create(new_id)
  │
  ├─ _try_preload_dataset(session, request.dataset_hint)   ← silent, Gap 12
  │
  ├─ data_science_agent.run(session, message)
  │
  └─ analysis_session_store.save(session)                  ← resets last_active
```

### TTL eviction (lazy + eager)

**Lazy eviction:** Every `get_or_create()` and `get()` call triggers `_evict_expired()` which removes sessions whose `monotonic() - last_active > ttl_seconds`.

**Eager eviction (GC task):** `_gc_sessions_task()` runs every 5 minutes. It calls `get_expired_ids()` (non-destructive query), then for each expired ID:
1. `data_science_agent.shutdown_session(sid)` — calls `runner.shutdown()` (closes Jupyter kernel socket, frees memory)
2. `analysis_session_store.delete(sid)` — removes from store + clears last_active

### `on_evict` callback

`AnalysisSessionStore(on_evict=callback)` — called with the session_id before removal. If the callback raises, the exception is **logged as warning** and eviction proceeds regardless.

### Key settings

| Setting | Env var | Default | Description |
|---------|---------|---------|-------------|
| `session_ttl_seconds` | `SESSION_TTL_SECONDS` | `3600` | Seconds of inactivity before expiry |

---

## 12. Physical Unit Validation

### Domain ranges

Defined in `DOMAIN_RANGES` dict in `unit_registry.py`:

| Quantity key | Range | Canonical unit |
|-------------|-------|----------------|
| `thermal_efficiency` | 0–100 | `percent` |
| `temperature` | −273.15–5000 | `degC` |
| `steam_pressure` | 0.001–35 | `MPa` |
| `gross_power` / `net_power` | 0–5000 | `MW` |
| `heat_rate` | 2000–20000 | `kJ/kWh` |
| `co2_emission` | 0–2000 | `g/kWh` |
| `mass_flow` | 0–10000 | `kg/s` |

### Validation flow

```
validate_physical_units(quantity, value, unit)
  1. Parse unit string via pint → fail → PhysicalUnit(is_valid=False)
  2. Fuzzy-match quantity name to DOMAIN_RANGES key
     (exact → substring → None)
  3. If matched: check lo <= value <= hi
     → Out of range → PhysicalUnit(is_valid=False, message explains)
  4. Return PhysicalUnit(is_valid=True) + log to session.unit_context
```

Claude is instructed to always call `validate_physical_units` after computing efficiency, temperature, pressure, or power values, and to investigate before reporting any anomaly.

---

## 13. API Reference

### `POST /api/v1/analysis/chat`

**Request:**
```json
{
  "message": "Analyse power_plant_data.csv and compute thermal efficiency",
  "session_id": "abc-123",        // optional; UUID created if omitted
  "dataset_hint": "power_plant_data.csv"  // optional pre-load
}
```

**Success response (200):**
```json
{
  "response": "The mean thermal efficiency is 38.4% ...",
  "session_id": "abc-123",
  "react_trace": [{"thought": "...", "action": "...", "observation": "..."}],
  "figures": [{"figure_id": "fig_000", "retrieval_url": "/api/v1/analysis/abc-123/figures/fig_000"}],
  "notebook_available": false,
  "unit_validations": [...],
  "iterations_used": 7,
  "model": "claude-sonnet-4-6",
  "status": "completed"
}
```

**Error responses:**

| Condition | HTTP | `detail.error` |
|-----------|------|----------------|
| Context too long | 400 | `"context_overflow"` — start a new session |
| LLM auth failure | 502 | `"llm_auth_error"` — contact support |
| LLM API failure | 502 | `"llm_api_error"` — retry later |
| ReAct loop failure | 200 | `status="error"` in body |

### `GET /api/v1/analysis/{session_id}/figures/{figure_id}`

Returns raw `image/png` bytes. **422** if the stored base64 data is corrupted.

### `GET /api/v1/analysis/{session_id}/notebook`

Returns the `.ipynb` file as `application/x-ipynb+json`. **404** if notebook not yet exported.

### `GET /health`

```json
{"status": "ok", "env": "development", "model": "claude-sonnet-4-6", "code_backend": "subprocess"}
```

---

## 14. Configuration Reference

All settings are in `app/core/config.py` and read from `.env` (or environment variables).

| Setting | Env var | Default | Description |
|---------|---------|---------|-------------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | **required** | Stored as `SecretStr` — never logged |
| `anthropic_base_url` | `ANTHROPIC_BASE_URL` | `None` | Override API endpoint |
| `claude_model` | `CLAUDE_MODEL` | `"claude-sonnet-4-6"` | Model ID |
| `max_tokens` | `MAX_TOKENS` | `8192` | Max tokens per Claude response |
| `max_retries` | `MAX_RETRIES` | `2` | SDK-level retry count |
| `app_env` | `APP_ENV` | `"development"` | Environment label |
| `debug` | `DEBUG` | `False` | Enables DEBUG logging |
| `data_dir` | `DATA_DIR` | `"data"` | Root data directory |
| `domain_docs_dir` | `DOMAIN_DOCS_DIR` | `"data/domain_docs"` | Domain knowledge docs |
| `datasets_dir` | `DATASETS_DIR` | `"data/datasets"` | Dataset files |
| `code_execution_backend` | `CODE_EXECUTION_BACKEND` | `"subprocess"` | `subprocess` \| `jupyter` \| `anthropic` |
| `code_execution_timeout` | `CODE_EXECUTION_TIMEOUT` | `30` | Seconds before subprocess kill |
| `max_react_iterations` | `MAX_REACT_ITERATIONS` | `20` | ReAct loop iteration cap |
| `cors_origins` | `CORS_ORIGINS` | `["http://localhost:3000","http://localhost:8001"]` | JSON array of allowed origins |
| `max_dataset_bytes` | `MAX_DATASET_BYTES` | `209715200` (200 MB) | File size limit for dataset tools |
| `session_ttl_seconds` | `SESSION_TTL_SECONDS` | `3600` | Idle session expiry |
| `figures_dir` | `FIGURES_DIR` | `"outputs/figures"` | Saved PNG destination |
| `notebooks_dir` | `NOTEBOOKS_DIR` | `"outputs/notebooks"` | Exported .ipynb destination |

**Minimal `.env`:**
```ini
ANTHROPIC_API_KEY=sk-ant-...
```

**Production `.env` additions:**
```ini
APP_ENV=production
DEBUG=false
CLAUDE_MODEL=claude-sonnet-4-6
MAX_REACT_ITERATIONS=20
CODE_EXECUTION_BACKEND=subprocess
CODE_EXECUTION_TIMEOUT=60
MAX_DATASET_BYTES=209715200
SESSION_TTL_SECONDS=1800
CORS_ORIGINS=["https://app.example.com"]
```

---

## 15. Running the Service

### Install

```bash
# Install all dependencies with uv
uv sync

# Install with Jupyter backend support
uv sync --extra jupyter

# Install dev dependencies (tests, linter)
uv sync --extra dev
```

### Development server

```bash
uv run uvicorn app.main:app --reload --port 8001
```

### Create sample data

```bash
python scripts/create_sample_data.py
```

### Verify installation

```bash
python scripts/verify_install.py
```

### Run tests

```bash
ANTHROPIC_API_KEY=sk-ant-test123 uv run --extra dev pytest tests/ --tb=short -q
```

**Current baseline:** 120 tests, 0 failures.

### Run with coverage

```bash
ANTHROPIC_API_KEY=sk-ant-test123 uv run --extra dev pytest tests/ --cov=app --cov-report=term-missing -q
```

### Lint

```bash
uv run --extra dev ruff check app/ tests/
```

---

## 16. Test Strategy & Coverage

Tests follow the **AAA (Arrange-Act-Assert)** pattern. Class names describe the unit under test; method names describe the behaviour being verified.

### Test File Map

| File | Gaps / units covered | Key techniques |
|------|--------------------|----------------|
| `conftest.py` | Shared fixtures | `anthropic_auth_error`, `anthropic_context_overflow_error`, etc. factory fixtures; `make_claude_response()` helper |
| `domain/test_exceptions.py` | Hierarchy, attributes, messages | `issubclass`, `isinstance`, `str(exc)` checks |
| `services/test_memory.py` | Gaps 5, 15 — TTL, eviction, callback | `AnalysisSessionStore(ttl_seconds=1)` + `time.sleep(1.1)` for fast expiry |
| `services/test_knowledge_tools.py` | Gaps 4, 10 — size guard, stat fallback | `tmp_path`, `patch("…settings")`, `MagicMock(spec=Path)` for stat error |
| `services/test_data_tools.py` | Gap 7 — `get_figure` metadata-only | Direct session fixture; assert `"data" not in result` |
| `services/test_data_agent.py` | Gaps 1, 2, 3, 16 — classify, rollback, sentinel | `patch("app.services.data_agent._client")` with `AsyncMock` |
| `infrastructure/test_code_runner.py` | Gaps 8, 9, 13 — factory, limits, probe | `CodeRunnerFactory.create()` with bad backend; `JupyterKernelManager` with mocked `_start` |
| `api/test_analysis.py` | Gaps 1, 3, 6, 12 — HTTP codes, preload | `httpx.AsyncClient(transport=ASGITransport(app))` without lifespan |

### Anthropic exception builders

Constructing Anthropic exceptions requires `httpx.Request` and `httpx.Response` objects. The `conftest.py` provides ready-made fixtures:

```python
# In a test:
async def test_auth_error(agent, session, anthropic_auth_error):
    with patch("app.services.data_agent._client") as mock_client:
        mock_client.messages.create = AsyncMock(side_effect=anthropic_auth_error())
        with pytest.raises(LLMAuthenticationError):
            await agent._call_claude_with_retry(session, "system")
```

### Mocking the module-level `_client`

The `AsyncAnthropic` client is instantiated at module import time as `_client`. Patch it as:

```python
with patch("app.services.data_agent._client") as mock_client:
    mock_client.messages.create = AsyncMock(return_value=make_claude_response("Final Answer: 42"))
```

### Async tests

`pyproject.toml` sets `asyncio_mode = "auto"` — all `async def test_*` functions run automatically without `@pytest.mark.asyncio`.

### API tests: lifespan not triggered

`httpx.AsyncClient(transport=ASGITransport(app=app))` does NOT trigger the FastAPI lifespan. This means the startup probe and GC task are not started — safe for unit tests. If you need the lifespan, use `starlette.testclient.TestClient` instead.

---

## 17. Adding a New Tool

1. **Implement** the function in `knowledge_tools.py` (read-only, pure) or `data_tools.py` (stateful, takes `session` and/or `runner`).  
   - Return type: always `str`  
   - On error: return `"Error: ..."` — do **not** raise  
   - If reading files: use `_safe_resolve(base_dir, file_name)` for path safety

2. **Add the JSON schema** in `tool_definitions.py`. Add to `KNOWLEDGE_TOOLS`, `EXECUTION_TOOLS`, or `VALIDATION_TOOLS` as appropriate.

3. **Register in `data_agent.py`** inside `_build_tool_registry()`:
   ```python
   "my_new_tool": lambda inp: my_new_tool(inp["param1"], session),
   ```

4. **Write a test** in `tests/services/test_knowledge_tools.py` or `test_data_tools.py`.

5. Nothing else changes — the system prompt is auto-generated from `TOOL_DEFINITIONS`.

---

## 18. Adding a New Exception

1. Add the class to `app/domain/exceptions.py` only. Zero external imports allowed.
2. Inherit from the most specific existing parent (`AgentError` → specific subclass).
3. Add `__init__` with meaningful attributes if needed.
4. Update `app/services/data_agent.py` or `app/api/v1/analysis.py` if the new exception needs special handling in the loop or needs mapping to an HTTP status.
5. Add tests to `tests/domain/test_exceptions.py`.

**Template:**
```python
class MyNewError(AgentError):
    """Raised when X happens.

    Attributes:
        context_field: description.
    """
    def __init__(self, context_field: str, reason: str) -> None:
        super().__init__(f"My error for '{context_field}': {reason}")
        self.context_field = context_field
        self.reason = reason
```

---

## 19. Operational Runbook

### Invalid API key at startup

**Symptom:** Service starts but every `/chat` request returns HTTP 502 with `"llm_auth_error"`.

**Diagnosis:** Check logs for `LLMAuthenticationError`. Verify `ANTHROPIC_API_KEY` in `.env`.

**Resolution:** Update the key and restart the service. The client is instantiated at import time; a restart is required.

---

### Session history grows too large → HTTP 400

**Symptom:** Long-running sessions start returning HTTP 400 with `"context_overflow"`.

**Cause:** `session.messages` exceeds Claude's context window.

**Resolution:** Client must start a new session (`session_id=None` in next request). The rolled-back session is left in the store but is no longer usable.

---

### Jupyter kernel dies mid-session

**Symptom:** `execute_python_code` observations contain `"Jupyter kernel crashed for session …"`.

**Cause:** OOM, signal, or kernel process crash.

**What happened:** `_probe_alive()` detected the dead kernel, `_restart_kernel()` was attempted, but failed → `KernelCrashError` → `AnalysisResult(success=False)` → Claude received the error as an observation.

**Recovery:** Claude will attempt to re-execute the last code block. If it keeps failing, the session's CodeRunner is in a degraded state. Deleting the session (or letting it TTL-expire) and starting fresh resolves it.

---

### Subprocess code execution times out

**Symptom:** `execute_python_code` returns `"Execution timed out after 30s"`.

**Resolution options:**
- Increase `CODE_EXECUTION_TIMEOUT` in `.env`.
- Ask Claude to optimise the code (sample a smaller dataframe, reduce iterations).

---

### GC task not running

**Symptom:** Memory grows unbounded; `analysis_session_store.active_sessions` keeps increasing.

**Diagnosis:** Check if the lifespan started normally. Look for `"Starting Data Scientist Agent"` in logs. The GC task is created as `asyncio.create_task(_gc_sessions_task())` inside lifespan.

**Note:** The GC task does not fire during tests (lifespan not triggered by `ASGITransport`).

---

### High memory usage from datasets

**Symptom:** Process memory grows after many `inspect_dataset` calls.

**Diagnosis:** `inspect_dataset` loads the full DataFrame into memory for stats. With `max_dataset_bytes=200 MB`, the in-process peak can be ~3× the file size after pandas parsing.

**Mitigations:**
- Reduce `MAX_DATASET_BYTES` in `.env`.
- Use Parquet format (columnar; pandas loads only requested columns).
- Session TTL eviction + GC task will reclaim memory when CodeRunner is shut down.

---

### Figures not rendering in downstream client

**Symptom:** Client calls `GET /figures/{figure_id}` and gets HTTP 422.

**Cause:** The base64 data stored in `session.figures[figure_id]` was corrupted (truncated write, encoding mismatch).

**Diagnosis:** The error log will contain `"Figure '%s' in session '%s' has corrupted base64 data"`.

**Resolution:** Re-run the analysis that generated the figure. Use a new session if the session state is suspect.

---

*Last updated: 2026-05-16 by the engineering team.*
