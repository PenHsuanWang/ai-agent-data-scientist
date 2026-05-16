# Exception Handling Design — Per-Layer Error Architecture

**Document Version:** 1.0  
**Status:** Draft  
**Scope:** All four architectural layers — Domain, Infrastructure, Application (ReAct), Presentation (API)  
**Depends on:** `SRS_robustness.md` (requirements), `03_react_service_design.md`, `05_code_execution_design.md`, `08_api_design.md`

---

## 1. Design Principles

This document defines **how exceptions are classified, propagated, caught, and surfaced** at
every layer of the Data Scientist AI Agent.

### Principle 1 — Tools Never Raise (Layer Boundary)

Every tool function (Group A, B, C) returns `str`. On any failure it returns
`"Error: <human-readable reason>"`. This is the primary isolation mechanism: the ReAct loop
never sees a Python exception from a tool — only an Observation string that Claude can
reason about and self-correct.

### Principle 2 — Session State is Committed on Success Only

`AnalysisSession.messages` is the single mutable buffer shared between turns. Before any
`run()` call begins writing, a **checkpoint** of `len(session.messages)` is recorded.
If the run raises before completion, `session.messages` is truncated to the checkpoint.
This prevents half-written message history from poisoning future turns.

### Principle 3 — Propagation Stops at the API Layer

Exceptions that escape the service layer are caught in the FastAPI handler and converted
to structured `AnalysisResponse(status="error")` objects. No raw Python traceback
text ever appears in an HTTP response body.

### Principle 4 — Distinguish Retryable from Fatal

| Exception Class | Category | Action |
|---|---|---|
| `anthropic.APIConnectionError` | Transient | Retry with back-off |
| `anthropic.RateLimitError` | Transient (quota) | Retry with extended back-off |
| `anthropic.AuthenticationError` | Fatal (config) | Fail fast, no retry |
| `anthropic.BadRequestError` (prompt too long) | Recoverable (trim) | Trim context, retry once |
| `anthropic.BadRequestError` (other) | Fatal | Fail with error code |
| `subprocess.TimeoutExpired` | Expected (user code) | Return AnalysisResult(success=False) |
| `ReActLoopError` | Application | Partial response with react_trace |
| `SessionNotFoundError` | Domain | HTTP 404 |

### Principle 5 — Log Correlation by session_id

Every log entry related to a request includes `session_id` as a structured field.
This enables filtering all log entries for a single user session during post-incident review.

---

## 2. Exception Taxonomy

```
AgentError (base)
├── SessionNotFoundError           — session_id lookup failed
├── AgentLoopError                 — generic loop safety cap (legacy MVP)
├── ReActLoopError                 — ReAct convergence failure
│   ├── .iterations: int           — how many iterations ran
│   └── .last_thought: str         — last known thought for diagnostics
├── ReActParseError                — parser could not extract Action from Claude text
├── CodeExecutionError             — code runner infrastructure failure
│   ├── .backend: str              — 'subprocess' | 'jupyter' | 'anthropic'
│   ├── .stderr: str               — raw error output
│   └── .timeout: bool             — True if caused by timeout
├── PhysicalValidationError        — hard physical law violation (not just a warning)
│   ├── .quantity: str
│   └── .reason: str
└── DatasetLoadError               — dataset file cannot be loaded/parsed
    ├── .file_name: str
    └── .reason: str

NEW (required by SRS_robustness.md):
├── LLMAPIError                    — Anthropic API call failed after all retries
│   ├── .status_code: int | None   — HTTP status from Anthropic (429, 500, etc.)
│   ├── .retries_attempted: int
│   └── .is_auth_error: bool       — True for 401/403 (config error, not transient)
├── LLMContextOverflowError        — prompt exceeds model context window
│   └── .estimated_tokens: int
└── SessionResourceLeakWarning     — (not raised; emitted as log.warning)
    — signals that CodeRunner for a session was GC'd without explicit shutdown()
```

---

## 3. Layer-by-Layer Exception Handling

### 3.1 Domain Layer (`app/domain/`)

The domain layer contains pure dataclasses with no external dependencies. It enforces
business invariants in `__post_init__`:

```python
# analysis_models.py — DatasetMeta invariants (existing, correct)
def __post_init__(self) -> None:
    if self.rows < 0:
        raise ValueError("rows must be >= 0")
    if "/" in self.file_name or "\\" in self.file_name:
        raise ValueError(f"file_name must be a basename, got '{self.file_name}'")
```

**Rule:** Domain `ValueError` raised from `__post_init__` propagates to the caller
(tool or service layer), which is responsible for converting it to a tool error string
or an `AgentError` subclass.

---

### 3.2 Infrastructure Layer (`app/infrastructure/`)

#### 3.2.1 CodeRunner — Contract

`CodeRunner.execute()` **never raises**. All execution failures return
`AnalysisResult(success=False, stderr="<reason>")`.

Error capture points in `SubprocessCodeRunner.execute()`:

```
subprocess.run()
├── TimeoutExpired          → AnalysisResult(success=False, stderr="Timed out after Xs")
├── OSError / ValueError    → AnalysisResult(success=False, stderr=str(exc))
├── returncode != 0         → AnalysisResult(success=False, stderr=result.stderr)
└── JSON parse of __FIGURES__/__STATE__ fails → silently skip (figures={}, state={})
```

Error capture points in `JupyterKernelManager.execute()`:

```
_start()
├── ImportError (jupyter_client missing)  → AnalysisResult(success=False, stderr="...")
├── RuntimeError (kernel timeout)         → AnalysisResult(success=False, stderr="...")
└── Any other exception                   → AnalysisResult(success=False, stderr=str(exc))

_kc.get_iopub_msg(timeout=N)
├── queue.Empty / TimeoutError            → AnalysisResult(success=False, stderr="Kernel timeout")
└── Any other exception                   → AnalysisResult(success=False, stderr=str(exc))

[NEW] Kernel health check before execute():
├── self._km.is_alive() == False          → attempt one restart
│   ├── restart succeeds                  → log WARNING, re-execute code
│   └── restart fails                     → AnalysisResult(success=False, stderr="Kernel restart failed")
```

#### 3.2.2 UnitRegistry — pint Initialization

```
_get_ureg()
├── ImportError (pint not installed)      → propagates as ImportError to tool function
│   └── validate_physical_units() catches → returns JSON {"is_valid": false, "message": "Error: ..."}
└── pint.UnitRegistry() unexpected error  → same catch path

_register_custom_units()  — each ureg.define() is individually wrapped in try/except
```

#### 3.2.3 NotebookExporter

```
export_notebook()
├── ImportError (nbformat missing)        → returns JSON {"error": "nbformat not installed"}
├── Code extraction regex failure         → [NEW] log WARNING, degrade gracefully to markdown cell
│                                           (NOT silent except Exception: pass)
├── nbformat.write() / OSError            → returns JSON {"error": str(exc)}
└── Any other exception                   → log ERROR, returns JSON {"error": str(exc)}

save_figure()
├── Figure not found in session           → returns JSON {"error": "Figure '...' not found"}
├── base64.b64decode() failure            → returns JSON {"error": str(exc)}
└── file write OSError                    → returns JSON {"error": str(exc)}
```

---

### 3.3 Application Layer — ReAct Loop (`app/services/data_agent.py`)

This is the most complex error domain. The full state machine with error transitions is:

```
run(session, user_message):

  [PRE-CONDITION] Snapshot message cursor for rollback:
    checkpoint = len(session.messages)

  [ENTRY] session.add_user_message(user_message)

  [LOOP 0..MAX_REACT_ITERATIONS]:

    ── STEP A: Claude API Call ──────────────────────────────────────────
    try:
        response = await _call_claude_with_retry(session, system_prompt)
    except LLMAuthenticationError:
        _rollback_session(session, checkpoint)
        raise                            # propagates to API layer → HTTP 502
    except LLMContextOverflowError:
        _rollback_session(session, checkpoint)
        raise                            # propagates to API layer → error response
    except LLMAPIError:
        _rollback_session(session, checkpoint)
        raise                            # propagates to API layer → HTTP 502

    ── STEP B: Parse ReAct Format ────────────────────────────────────────
    parsed = _parse_react(raw_text)

    if parsed["type"] == "parse_error":
        [NEW] session.append_react_step(
            thought="", action="__parse_error__", observation=parsed["reason"]
        )
        parse_error_count += 1
        if parse_error_count >= 3:
            _rollback_session(session, checkpoint)
            raise ReActLoopError(reason=parsed["reason"], iterations=iteration+1)
        # inject correction, continue

    if parsed["type"] == "final_answer":
        return final_answer              # SUCCESS EXIT

    ── STEP C: Tool Dispatch ─────────────────────────────────────────────
    handler = tool_registry.get(action_name)
    if handler is None:
        observation = "Error: Unknown tool '...'. Available: ..."
    else:
        try:
            observation = handler(action_input)
        except Exception as exc:         # last-resort catch (tools should never raise)
            logger.error("Tool '%s' raised unexpectedly: %s", action_name, exc, exc_info=True)
            observation = f"Error: Tool '{action_name}' failed — {exc}"

    ── STEP D: Append Observation ────────────────────────────────────────
    [truncate observation to MAX_OBS=8000 chars]
    session.append_react_step(thought, action, observation)
    session.add_assistant_message(assistant_content)
    session.add_user_message(f"Observation: {observation}")

  [LOOP END — MAX ITERATIONS REACHED]
  raise ReActLoopError(
      reason=f"Reached {MAX_REACT_ITERATIONS} iterations without Final Answer",
      iterations=MAX_REACT_ITERATIONS,
      last_thought=last_thought,
  )
  # NOTE: No rollback here — partial react_trace is preserved as audit trail
  # Session messages end on a complete assistant turn (Step D was last)
```

#### 3.3.1 `_call_claude_with_retry()` — New Helper

```python
async def _call_claude_with_retry(
    session: AnalysisSession,
    system_prompt: str,
    max_retries: int = settings.max_llm_retries,      # default: 3
    backoff_base: float = settings.llm_retry_backoff, # default: 1.0 s
) -> anthropic.types.Message:
    """
    Wraps _client.messages.create() with retry logic.

    Retry policy:
    - anthropic.APIConnectionError  → retry with exponential back-off
    - anthropic.RateLimitError      → retry after retry_after header or 60s
    - anthropic.AuthenticationError → raise LLMAuthenticationError immediately (no retry)
    - anthropic.BadRequestError     → if "prompt is too long": raise LLMContextOverflowError
                                     else: raise LLMAPIError (no retry)
    - anthropic.APIStatusError (5xx)→ retry up to max_retries
    """
    import asyncio
    from anthropic import (
        APIConnectionError, RateLimitError, AuthenticationError,
        BadRequestError, APIStatusError,
    )

    for attempt in range(max_retries + 1):
        try:
            return await _client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_tokens,
                system=system_prompt,
                messages=session.messages,
            )
        except AuthenticationError as exc:
            logger.error("Anthropic authentication failed (session=%s)", session.session_id)
            raise LLMAuthenticationError(str(exc)) from exc

        except BadRequestError as exc:
            if "prompt is too long" in str(exc).lower():
                raise LLMContextOverflowError(str(exc)) from exc
            raise LLMAPIError(str(exc), status_code=exc.status_code, retries_attempted=0) from exc

        except RateLimitError as exc:
            if attempt >= max_retries:
                raise LLMAPIError(
                    "Rate limit exceeded after retries", status_code=429, retries_attempted=attempt
                ) from exc
            wait = getattr(exc, "retry_after", None) or (60.0)
            logger.warning(
                "Rate limited by Anthropic API (session=%s), waiting %.1fs (attempt %d/%d)",
                session.session_id, wait, attempt + 1, max_retries,
            )
            await asyncio.sleep(wait)

        except (APIConnectionError, APIStatusError) as exc:
            if attempt >= max_retries:
                raise LLMAPIError(
                    str(exc),
                    status_code=getattr(exc, "status_code", None),
                    retries_attempted=attempt,
                ) from exc
            wait = backoff_base * (2 ** attempt)
            logger.warning(
                "Anthropic API error (session=%s): %s — retry %d/%d in %.1fs",
                session.session_id, exc, attempt + 1, max_retries, wait,
            )
            await asyncio.sleep(wait)
    # Unreachable, but satisfies type checker:
    raise LLMAPIError("Exhausted retries", retries_attempted=max_retries)
```

#### 3.3.2 `_rollback_session()` — New Helper

```python
def _rollback_session(session: AnalysisSession, checkpoint: int) -> None:
    """
    Truncates session.messages back to the checkpoint index.
    Called when run() must abort before a successful Final Answer.
    Ensures the session history is consistent for the next run() call.
    """
    if len(session.messages) > checkpoint:
        logger.debug(
            "Rolling back session '%s' messages from %d to %d",
            session.session_id, len(session.messages), checkpoint,
        )
        session.messages = session.messages[:checkpoint]
```

---

### 3.4 Knowledge Tools (`app/services/knowledge_tools.py`)

All tools follow the **never-raise, return-error-string** contract.

Updated error capture map:

| Tool | Input Validation | Filesystem Guard | New Guards Required |
|---|---|---|---|
| `list_domain_documents` | — | `docs_dir.exists()` | Per-file `f.stat()` try/except |
| `read_domain_document` | `_safe_resolve` | `target.exists()` | ✅ Already has outer try/except |
| `search_domain_knowledge` | top_k clamp | per-file `except Exception: continue` | ✅ Already safe |
| `list_datasets` | — | `datasets_dir.exists()` | Per-file `f.stat()` try/except |
| `inspect_dataset` | `_safe_resolve` | `target.exists()` | **NEW: size guard before pd.read_*()** |
| `describe_columns` | `_safe_resolve` | `target.exists()` | **NEW: size guard before pd.read_*()** |
| `get_coding_standards` | — | delegates to `read_domain_document` | ✅ Inherited safety |

#### Size Guard Pattern (applies to `inspect_dataset` and `describe_columns`)

```python
MAX_DATASET_BYTES = settings.max_dataset_bytes  # default: 268_435_456 (256 MB)

size = target.stat().st_size
if size > MAX_DATASET_BYTES:
    return (
        f"Error: Dataset '{file_name}' is {size // 1_048_576} MB, "
        f"which exceeds the maximum load size of {MAX_DATASET_BYTES // 1_048_576} MB. "
        "Use chunked processing or reduce the dataset before analysis."
    )
```

#### Robust `list_domain_documents` / `list_datasets`

```python
result = []
for f in files:
    try:
        result.append({"file_name": f.name, "size_bytes": f.stat().st_size})
    except OSError as exc:
        logger.warning("Could not stat '%s': %s", f, exc)
        result.append({"file_name": f.name, "size_bytes": -1})
return json.dumps(result)
```

---

### 3.5 Data Tools (`app/services/data_tools.py`)

#### `get_figure` — Return Metadata Only

```python
def get_figure(figure_id: str, session: "AnalysisSession") -> str:
    b64 = session.figures.get(figure_id)
    if b64 is None:
        return json.dumps({
            "error": f"Figure '{figure_id}' not found.",
            "available_figures": list(session.figures.keys()),
        })
    # Return metadata ONLY — never raw base64 in a Claude Observation
    import base64
    try:
        size_bytes = len(base64.b64decode(b64))
    except Exception:
        size_bytes = -1
    return json.dumps({
        "figure_id": figure_id,
        "format": "png",
        "size_bytes": size_bytes,
        "retrieval_url": f"/api/v1/analysis/{session.session_id}/figures/{figure_id}",
        "note": "Use the retrieval_url endpoint to download the image.",
    })
```

---

### 3.6 Presentation Layer (`app/api/v1/analysis.py`)

#### Full Exception Catch Map for `analysis_chat()`

```
analysis_chat(request):

  try:
      session = analysis_session_store.get_or_create(session_id)
      answer  = await data_science_agent.run(session, request.message)
      analysis_session_store.save(session)
      return AnalysisResponse(status="completed", ...)

  except LLMAuthenticationError as exc:
      logger.critical("LLM auth error (session=%s): %s", session_id, exc)
      # HTTP 502 — config problem, not user error
      raise HTTPException(status_code=502, detail={"error": "llm_auth_error", ...})

  except LLMContextOverflowError as exc:
      logger.warning("Context overflow (session=%s): %s", session_id, exc)
      return AnalysisResponse(
          status="error", error_code="context_overflow",
          response="The conversation history is too long. Please start a new session.",
          ...
      )

  except LLMAPIError as exc:
      logger.error("LLM API error (session=%s): %s", session_id, exc, exc_info=True)
      raise HTTPException(status_code=502, detail={"error": "llm_api_error", ...})

  except ReActLoopError as exc:
      logger.warning("ReAct loop failed (session=%s): %s", session_id, exc)
      return AnalysisResponse(
          status="error", error_code="react_loop_exhausted",
          response=f"Analysis incomplete after {exc.iterations} steps. Last thought: {exc.last_thought}",
          iterations_used=exc.iterations,
          ...
      )

  except AgentError as exc:
      logger.warning("Agent error (session=%s): %s", session_id, exc)
      return AnalysisResponse(
          status="error", response=f"Agent error: {exc}", ...
      )

  except Exception as exc:
      logger.error("Unhandled error (session=%s): %s", session_id, exc, exc_info=True)
      raise HTTPException(status_code=500, detail={"error": "internal_error"})
```

#### Guarded Figure Endpoint

```python
@router.get("/{session_id}/figures/{figure_id}")
async def get_figure(session_id: str, figure_id: str) -> Response:
    session = analysis_session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    b64 = session.figures.get(figure_id)
    if b64 is None:
        raise HTTPException(
            status_code=404,
            detail=f"Figure '{figure_id}' not found. Available: {list(session.figures.keys())}",
        )
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=422, detail="Figure data is corrupted.")
    return Response(content=img_bytes, media_type="image/png")
```

#### Guarded Notebook Endpoint

```python
@router.get("/{session_id}/notebook")
async def get_notebook(session_id: str) -> FileResponse:
    session = analysis_session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if not session.notebook_path:
        raise HTTPException(status_code=404, detail="No notebook exported yet.")
    from pathlib import Path
    if not Path(session.notebook_path).exists():
        raise HTTPException(status_code=404, detail="Notebook file no longer available on disk.")
    return FileResponse(...)
```

---

## 4. Session Lifecycle and Resource Management

### 4.1 Session Garbage Collection

A background `asyncio` task runs every `SESSION_GC_INTERVAL_SECONDS` (default: 300 s).
Sessions inactive for more than `SESSION_TTL_SECONDS` (default: 3600 s) are evicted.

```
AnalysisSessionStore._store: dict[session_id → AnalysisSession]
    └── last_active_at: datetime  ← updated on every .save()

Background GC task (asyncio.create_task in lifespan):
  while True:
      await asyncio.sleep(SESSION_GC_INTERVAL_SECONDS)
      now = datetime.utcnow()
      for sid, session in list(store._store.items()):
          if (now - session.last_active_at).total_seconds() > SESSION_TTL_SECONDS:
              data_science_agent.shutdown_session(sid)
              store.delete(sid)
              logger.info("GC evicted session '%s'", sid)
```

### 4.2 Startup Backend Probe

```
lifespan startup:
  try:
      probe_runner = CodeRunnerFactory.create(session_id="__probe__")
      probe_runner.shutdown()
      logger.info("Code backend '%s' probed successfully", settings.code_execution_backend)
  except ValueError as exc:
      logger.critical("Invalid CODE_EXECUTION_BACKEND: %s — halting.", exc)
      raise SystemExit(1)
```

### 4.3 Graceful Shutdown

```
lifespan shutdown:
  for sid in list(data_science_agent._runners.keys()):
      data_science_agent.shutdown_session(sid)
  logger.info("All %d session runners shut down.", count)
```

---

## 5. HTTP Error Code Reference (Updated)

| Error Condition | HTTP Status | `error_code` | Retry? |
|---|---|---|---|
| Invalid request body (Pydantic) | 422 | `validation_error` | Fix request |
| Session not found | 404 | `session_not_found` | Use valid session_id |
| Figure not found | 404 | `figure_not_found` | Check figure_ids |
| Figure data corrupted | 422 | `corrupted_figure` | Report bug |
| Notebook not generated | 404 | `notebook_not_available` | Call export_notebook first |
| Notebook file deleted from disk | 404 | `notebook_file_missing` | Re-export |
| Dataset not found | 404 | `dataset_not_found` | Check filename |
| ReAct max iterations exceeded | 200 | `react_loop_exhausted` | Simplify request |
| ReAct parse failure (×3) | 200 | `react_parse_failed` | Report bug |
| Context window overflow | 200 | `context_overflow` | Start new session |
| Claude API auth error | 502 | `llm_auth_error` | Fix API key |
| Claude API transient error | 502 | `llm_api_error` | Retry after delay |
| Code execution timeout | 200¹ | `code_timeout` | Simplify code |
| Generic server error | 500 | `internal_error` | Contact support |

> ¹ Code execution timeouts appear as Claude tool errors inside a successful `run()` response
> (Claude self-corrects). They surface as HTTP 200 with the timeout detail in `react_trace`.

---

## 6. Observability Checklist

Every module must emit structured log entries at the appropriate level:

| Event | Level | Required Fields |
|---|---|---|
| ReAct loop start | `DEBUG` | `session_id`, `iteration`, `message_count` |
| Tool dispatched | `INFO` | `session_id`, `tool_name`, `input_preview` |
| Tool raised unexpectedly | `ERROR` | `session_id`, `tool_name`, `exc_info=True` |
| Parse error correction | `WARNING` | `session_id`, `attempt_number`, `reason` |
| Claude API retry | `WARNING` | `session_id`, `attempt`, `wait_seconds` |
| Claude API auth failure | `CRITICAL` | `session_id`, error message (no key!) |
| Context overflow | `WARNING` | `session_id`, `estimated_tokens` |
| Jupyter kernel restart | `WARNING` | `session_id` |
| Session GC eviction | `INFO` | `session_id`, `age_seconds` |
| Backend probe failure | `CRITICAL` | `backend_name`, exit |
| Dataset size guard triggered | `INFO` | `file_name`, `size_bytes`, `limit_bytes` |
| Notebook code extraction failed | `WARNING` | `session_id`, `step_index` |
