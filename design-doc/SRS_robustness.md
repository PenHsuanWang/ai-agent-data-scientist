# Software Requirements Specification — Robustness & Exception Handling

**Product Name:** Data Scientist AI Agent  
**Document Version:** 1.1  
**Status:** Draft — Pending Review  
**Authors:** System Analysis (derived from gap audit, 2026-05-16)  
**Supersedes:** N/A — New document, addendum to the original architecture design  

---

## 1. Product Overview

- **Product Name:** Data Scientist AI Agent
- **Business Goal:** Provide a domain-aware, physically-validated AI agent for data science
  analysis that is **robust under adversarial conditions**, recovers gracefully from
  infrastructure failures, and never silently corrupts session state.
- **Target Audience:**
  - **Primary:** Engineers and scientists performing quantitative analysis via REST API
  - **Secondary:** Platform operators running the agent in long-lived server deployments

### 1.1 Motivation for This SRS

A gap audit (2026-05-16) identified **16 exception-handling deficiencies** across the agent's
four architectural layers. These deficiencies fall into three severity tiers:

| Tier | Count | Representative Risk |
|---|---|---|
| 🔴 Critical | 3 | Unguarded Anthropic API call corrupts session state |
| 🟠 High | 4 | OOM on large dataset; base64 decode 500; figure data truncation |
| 🟡 Medium–Low | 9 | Jupyter kernel leak; invalid backend config; CORS wildcard |

This SRS translates those findings into actionable, testable requirements.

---

## 2. Epics & User Stories

---

### EPIC-1: ReAct Loop Resilience

**Description:** The ReAct reasoning loop must remain safe and predictable when the Anthropic
API fails, the context window overflows, or Claude produces malformed output. Session state
must never be left in a half-written, inconsistent state.

---

#### USER STORY-1.1: Anthropic API Error Handling with Retry

- **User Story:** As a data scientist, I want the agent to automatically retry on transient
  Anthropic API failures so that a brief network glitch or rate-limit event does not terminate
  my analysis session.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Transient connection error:**
    - **Given** a valid `AnalysisRequest` is submitted
    - **And** the Anthropic API returns `APIConnectionError` on the first attempt
    - **When** the ReAct loop makes the Claude API call
    - **Then** the system retries up to `MAX_LLM_RETRIES` times (default: 3)
    - **And** applies exponential back-off starting at 1 s
    - **And** returns `AnalysisResponse(status="error", error_code="llm_api_error")` if all retries exhaust

  - **Scenario 2 — Rate limit (429):**
    - **Given** `anthropic.RateLimitError` is raised
    - **When** the ReAct loop calls the Claude API
    - **Then** the system backs off for `retry_after` seconds (from response header) or 60 s default
    - **And** retries once before returning a structured error response

  - **Scenario 3 — Authentication failure (401):**
    - **Given** `anthropic.AuthenticationError` is raised
    - **When** the ReAct loop calls the Claude API
    - **Then** the system does **not** retry
    - **And** raises `LLMAuthenticationError` immediately
    - **And** returns HTTP 502 with `error_code="llm_auth_error"`
    - **And** logs the error at `ERROR` level without revealing the API key

  - **Scenario 4 — Non-retryable API error:**
    - **Given** `anthropic.APIStatusError` with a 4xx status (not 429) is raised
    - **When** the ReAct loop calls the Claude API
    - **Then** the system does **not** retry
    - **And** returns `AnalysisResponse(status="error", error_code="llm_api_error")`

- **Business Rules:**
  - `ANTHROPIC_API_KEY` must never appear in logs, error messages, or HTTP responses.
  - Retry policy must be configurable via `MAX_LLM_RETRIES` and `LLM_RETRY_BACKOFF_BASE` env vars.

---

#### USER STORY-1.2: Session State Integrity on Mid-Loop Failure

- **User Story:** As a developer integrating the agent, I want the session's message history
  to remain consistent even when a loop iteration fails, so that subsequent requests to the
  same session do not send malformed context to Claude.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — API failure after user message appended:**
    - **Given** a session exists with valid history
    - **And** `session.add_user_message(user_message)` has been called
    - **When** the subsequent Claude API call raises any exception
    - **Then** the trailing unacknowledged user message is **rolled back** from `session.messages`
    - **And** the session history is identical to its state before `run()` was called
    - **And** the session is saved back to the store in the clean state

  - **Scenario 2 — Partial loop state on max-iterations:**
    - **Given** the ReAct loop reaches `MAX_REACT_ITERATIONS`
    - **When** `ReActLoopError` is raised
    - **Then** all react_trace steps accumulated are preserved (audit trail)
    - **And** the session `messages` list ends on a complete assistant turn
    - **And** the session remains usable for the next user request

- **Business Rules:**
  - Session state mutations during a `run()` call must follow a **commit-on-success** pattern:
    snap the message list length at entry; roll back to that length on any unhandled exception.

---

#### USER STORY-1.3: Context Window Overflow Protection

- **User Story:** As a data scientist running a long multi-turn analysis, I want the system
  to gracefully handle context-window limits instead of crashing with an unhandled 400 error.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Context overflow detected before API call:**
    - **Given** the session messages have grown to exceed `CONTEXT_WINDOW_SAFE_THRESHOLD` tokens
    - **When** the user submits a new analysis request
    - **Then** the system summarizes or truncates old observations (keeping the last N turns)
    - **And** injects a `[Context summarized for length]` marker into the message history
    - **And** proceeds with the API call using the trimmed context

  - **Scenario 2 — `BadRequestError` from Claude (prompt too long):**
    - **Given** `anthropic.BadRequestError` with reason containing "prompt is too long" is raised
    - **When** the ReAct loop calls the Claude API
    - **Then** the system catches the error
    - **And** returns `AnalysisResponse(status="error", error_code="context_overflow")`
    - **And** includes a human-readable message recommending starting a new session
    - **And** does **not** corrupt the session message list

- **Business Rules:**
  - The context window budget is modelled as `max_tokens × 0.8` to leave headroom for the response.
  - A `CONTEXT_WINDOW_SAFE_THRESHOLD` config var controls when proactive trimming begins.

---

#### USER STORY-1.4: Parse-Error Steps Visible in ReAct Trace

- **User Story:** As a developer debugging an analysis, I want to see when Claude produced
  malformed output and how the system corrected it, so I can diagnose prompt engineering issues.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Parse correction injected:**
    - **Given** Claude produces a response that fails `_parse_react()`
    - **When** the correction prompt is injected
    - **Then** the react_trace includes an entry: `{thought: "...", action: "__parse_error__", observation: "<reason>"}`
    - **And** the returned `AnalysisResponse.react_trace` includes this entry

- **Business Rules:** Parse errors do not count against `MAX_REACT_ITERATIONS`.

---

### EPIC-2: Safe Data Ingestion

**Description:** Dataset loading operations must enforce size and memory limits to prevent
a single oversized file from exhausting server memory.

---

#### USER STORY-2.1: Dataset Size Guard Before Loading

- **User Story:** As a platform operator, I want the agent to refuse to load datasets that
  exceed a configurable size limit, so that a large uploaded file cannot OOM the server process.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — File exceeds size limit:**
    - **Given** a dataset file with `file_size_bytes > MAX_DATASET_BYTES` (default: 256 MB)
    - **When** `inspect_dataset(file_name)` is called
    - **Then** the tool returns `"Error: Dataset '...' exceeds the maximum load size of 256 MB. Use chunked processing."`
    - **And** pandas is never called
    - **And** the server memory is unaffected

  - **Scenario 2 — File within limit:**
    - **Given** a dataset file within the size limit
    - **When** `inspect_dataset(file_name)` is called
    - **Then** the tool loads and returns schema + sample rows as normal

  - **Scenario 3 — `describe_columns` same guard:**
    - **Given** a dataset file exceeding the limit
    - **When** `describe_columns(file_name, columns)` is called
    - **Then** the same size-guard error is returned

- **Business Rules:**
  - `MAX_DATASET_BYTES` defaults to `268_435_456` (256 MB) and is configurable via env var.
  - The check is performed by `file.stat().st_size` **before** `pd.read_*()`.

---

#### USER STORY-2.2: Robust `list_domain_documents` Under Filesystem Errors

- **User Story:** As a knowledge-tools consumer, I want `list_domain_documents` to return a
  partial result (skipping broken files) rather than failing entirely when a symlink or
  permission error is encountered.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Broken symlink in docs directory:**
    - **Given** the domain docs directory contains a broken symlink
    - **When** `list_domain_documents()` is called
    - **Then** the broken entry is silently skipped
    - **And** all valid documents are returned in the result
    - **And** the error is logged at `WARNING` level

- **Business Rules:** `f.stat()` calls must be wrapped per-file; a single failure must not abort the listing.

---

### EPIC-3: Resource Management and Lifecycle

**Description:** Long-running deployments must not accumulate unreleased CodeRunners, Jupyter
kernel processes, or unbounded session objects. Cleanup must be automatic.

---

#### USER STORY-3.1: CodeRunner Automatic Shutdown on Session Expiry

- **User Story:** As a platform operator, I want CodeRunner instances (especially Jupyter
  kernels) to be automatically released when a session expires, so that kernel processes
  do not accumulate and exhaust server memory.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Session TTL expiry:**
    - **Given** a session has been inactive for more than `SESSION_TTL_SECONDS` (default: 3600 s)
    - **When** the background cleanup task runs (every `SESSION_GC_INTERVAL_SECONDS`, default: 300 s)
    - **Then** `DataScienceAgentService.shutdown_session(session_id)` is called
    - **And** `AnalysisSessionStore.delete(session_id)` is called
    - **And** any associated Jupyter kernel subprocess is terminated

  - **Scenario 2 — Explicit session cleanup on server shutdown:**
    - **Given** the FastAPI `lifespan` shutdown event fires
    - **When** the server begins graceful shutdown
    - **Then** `shutdown_session()` is called for every active session
    - **And** all Jupyter kernel processes are confirmed terminated within 10 s

  - **Scenario 3 — Invalid backend config detected at startup:**
    - **Given** `CODE_EXECUTION_BACKEND` is set to an unsupported value (e.g., `"anthropic"`)
    - **When** the FastAPI app starts (`lifespan` startup)
    - **Then** the startup sequence calls `CodeRunnerFactory.create()` with a dummy session_id as a probe
    - **And** if `ValueError` is raised, the app logs `CRITICAL` and **exits immediately**
    - **And** the error is clear: `"Unknown backend 'anthropic'. Halting."`

- **Business Rules:**
  - `SESSION_TTL_SECONDS` and `SESSION_GC_INTERVAL_SECONDS` are configurable env vars.
  - The GC task runs as an `asyncio` background task started in `lifespan`.

---

#### USER STORY-3.2: Jupyter Kernel Crash Recovery

- **User Story:** As a data scientist using the Jupyter backend, I want the agent to
  automatically restart a crashed kernel rather than silently failing every execution,
  so my analysis can continue without starting a new session.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Kernel crash detected:**
    - **Given** a Jupyter kernel is running for a session
    - **And** the kernel process has exited unexpectedly (returncode is not None)
    - **When** `execute_python_code` is called
    - **Then** `JupyterKernelManager` detects the dead kernel via `is_alive()` check
    - **And** automatically calls `shutdown()` then `_start()` to restart the kernel
    - **And** re-executes the code once in the fresh kernel
    - **And** logs `WARNING: Jupyter kernel for session <id> died; restarted.`

  - **Scenario 2 — Kernel restart fails:**
    - **Given** the kernel restart also fails
    - **When** `_start()` raises an exception
    - **Then** `execute()` returns `AnalysisResult(success=False, stderr="Kernel restart failed: ...")`
    - **And** the runner is marked as permanently failed for that session

- **Business Rules:** At most one automatic restart attempt per `execute()` call.

---

### EPIC-4: API Endpoint Robustness

**Description:** All HTTP endpoints must return structured error responses, never raw Python
tracebacks. Binary operations (base64 decode, file serving) must be guarded.

---

#### USER STORY-4.1: Guarded Figure Retrieval Endpoint

- **User Story:** As an API consumer, I want the figure retrieval endpoint to return a
  structured 422 or 500 error (not a raw traceback) when a stored figure is corrupt,
  so my client can handle it gracefully.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Valid figure:**
    - **Given** a valid session and figure_id exist
    - **When** `GET /api/v1/analysis/{session_id}/figures/{figure_id}` is called
    - **Then** HTTP 200 with `Content-Type: image/png` is returned

  - **Scenario 2 — Corrupted base64:**
    - **Given** the stored base64 for a figure is malformed
    - **When** the endpoint calls `base64.b64decode(b64)`
    - **Then** the `binascii.Error` is caught
    - **And** HTTP 422 is returned with body `{"detail": "Figure data is corrupted."}`

  - **Scenario 3 — Notebook file deleted after export:**
    - **Given** a session has `notebook_path` set
    - **And** the file has been deleted from disk since export
    - **When** `GET /api/v1/analysis/{session_id}/notebook` is called
    - **Then** HTTP 404 is returned with `{"detail": "Notebook file no longer available on disk."}`

- **Business Rules:** No Python traceback text may appear in any HTTP response body.

---

#### USER STORY-4.2: Figure Tool Returns Metadata, Not Raw Bytes

- **User Story:** As a Claude reasoning step, I want the `get_figure` tool to return figure
  metadata (not raw base64 data), so that the Observation fits within the 8000-character budget
  and is not silently truncated.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Figure exists:**
    - **Given** `figure_id` is valid in the current session
    - **When** Claude calls `get_figure({"figure_id": "fig_000"})`
    - **Then** the Observation is: `{"figure_id": "fig_000", "format": "png", "size_bytes": 18432, "retrieval_url": "/api/v1/analysis/.../figures/fig_000", "note": "Use the retrieval_url to download the image."}`
    - **And** the base64 data is **NOT** included in the Observation
    - **And** the total Observation length is under 400 characters

  - **Scenario 2 — Figure does not exist:**
    - **Given** `figure_id` is not in the session
    - **When** Claude calls `get_figure({"figure_id": "fig_999"})`
    - **Then** the Observation is: `{"error": "Figure 'fig_999' not found.", "available_figures": [...]}`

- **Business Rules:** Base64 PNG data must never be included in any Claude Observation.
  It is only served via the HTTP `GET /figures/{figure_id}` endpoint.

---

### EPIC-5: Code Execution Safety

**Description:** Python code execution must be bounded in time, memory, and filesystem access.
The subprocess backend must prevent executed code from accessing paths outside the data
directories or making outbound network connections in production deployments.

---

#### USER STORY-5.1: Subprocess Resource Limits

- **User Story:** As a platform operator, I want subprocess code execution to be bounded
  by CPU time and memory so that a runaway loop or memory-intensive operation cannot
  exhaust server resources.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Code exceeds memory limit:**
    - **Given** `CODE_EXEC_MEMORY_LIMIT_MB` is set (default: 512 MB on Linux)
    - **When** executed Python code allocates memory exceeding the limit
    - **Then** the subprocess receives `SIGKILL` from the OS
    - **And** `execute()` returns `AnalysisResult(success=False, stderr="MemoryError: process killed by OS")`
    - **And** the main server process is unaffected

  - **Scenario 2 — Execution timeout (existing, verify):**
    - **Given** `CODE_EXECUTION_TIMEOUT` is set (default: 30 s)
    - **When** executed code runs longer than the timeout
    - **Then** `subprocess.TimeoutExpired` is caught
    - **And** `AnalysisResult(success=False, stderr="Execution timed out after 30s")` is returned

- **Business Rules:**
  - Resource limits are applied via `resource.setrlimit()` on Linux (no-op on Windows/macOS in development).
  - `CODE_EXEC_MEMORY_LIMIT_MB` defaults to 0 (disabled) in development, 512 in production.

---

### EPIC-6: Security Hardening

**Description:** The API must enforce CORS restrictions in production, and cross-site requests
from untrusted origins must be rejected.

---

#### USER STORY-6.1: Environment-Aware CORS Policy

- **User Story:** As a security engineer, I want CORS to be restricted to allowed origins
  in production, so that the API cannot be called from arbitrary browser origins.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Development environment:**
    - **Given** `APP_ENV=development`
    - **When** the FastAPI app starts
    - **Then** `allow_origins=["*"]` is used (permissive for local dev)

  - **Scenario 2 — Production environment:**
    - **Given** `APP_ENV=production`
    - **And** `CORS_ALLOW_ORIGINS=https://myapp.example.com`
    - **When** the FastAPI app starts
    - **Then** `allow_origins=["https://myapp.example.com"]` is used
    - **And** requests from any other origin receive HTTP 403

- **Business Rules:**
  - `CORS_ALLOW_ORIGINS` is a comma-separated env var.
  - If `APP_ENV=production` and `CORS_ALLOW_ORIGINS` is not set, the app logs `WARNING` and defaults to `[]` (block all cross-origin).

---

### EPIC-7: Notebook Export Reliability

**Description:** Notebook export must produce a valid `.ipynb` with all code cells correctly
populated, and must not silently degrade to empty markdown cells when code extraction fails.

---

#### USER STORY-7.1: Robust Code Extraction for Notebook Cells

- **User Story:** As a data scientist, I want exported Jupyter notebooks to contain the
  actual Python code from my analysis, not blank placeholders, so the notebook is reproducible.
- **Acceptance Criteria (BDD Format):**

  - **Scenario 1 — Code extracted successfully:**
    - **Given** a session has executed Python code via `execute_python_code`
    - **When** `export_notebook` is called
    - **Then** each code cell in the notebook contains the full Python source

  - **Scenario 2 — Code extraction regex fails:**
    - **Given** the action string for `execute_python_code` contains multi-line code with nested JSON
    - **And** the current regex fails to extract it
    - **When** `export_notebook` is called
    - **Then** the notebook cell contains a `# [Code extraction failed — raw action recorded below]` comment followed by the raw action string
    - **And** a `WARNING` log is emitted: `"export_notebook: code extraction failed for step {i}"`
    - **And** the notebook is still exported (no silent error swallowing)

- **Business Rules:** `export_notebook` must log every degraded cell at `WARNING` level.

---

## 3. Non-Functional Requirements (NFRs)

### 3.1 Resilience

- **NF-R1:** The ReAct loop must handle Anthropic API transient errors with exponential
  back-off (max 3 retries). A single `APIConnectionError` must not terminate a session.
- **NF-R2:** Session state (`messages`, `react_trace`) must be fully consistent before and
  after any failed `run()` call. No partial writes permitted.
- **NF-R3:** The system must remain responsive under `MAX_REACT_ITERATIONS=20` regardless
  of tool output size (via the 8000-char truncation cap, which must be enforced).

### 3.2 Safety

- **NF-S1:** No dataset larger than `MAX_DATASET_BYTES` (256 MB default) may be fully loaded
  into memory by any tool function.
- **NF-S2:** Executed Python code must run in an isolated subprocess with a configurable
  wall-clock timeout (`CODE_EXECUTION_TIMEOUT`, default 30 s).
- **NF-S3:** Resource limits (CPU time, memory) must be applied to subprocess execution on
  Linux via `resource.setrlimit`. On other platforms, the timeout-only guard applies.
- **NF-S4:** `ANTHROPIC_API_KEY` must never appear in logs, tracebacks, HTTP responses,
  or `react_trace` entries. It must be handled exclusively as `SecretStr`.

### 3.3 Observability

- **NF-O1:** Every exception caught at the API layer must be logged with `session_id`,
  `error_code`, and `exc_info=True` (for 5xx errors).
- **NF-O2:** Parse-error correction injections must be recorded in `react_trace` under
  the action name `"__parse_error__"`.
- **NF-O3:** Jupyter kernel restarts must be logged at `WARNING` with session_id.
- **NF-O4:** Session GC events (session deleted, runner shut down) must be logged at `INFO`.

### 3.4 Operability

- **NF-OP1:** Invalid `CODE_EXECUTION_BACKEND` values must cause a startup failure with a
  `CRITICAL` log, not a silent runtime error on first use.
- **NF-OP2:** Session TTL must be configurable via `SESSION_TTL_SECONDS` env var.
- **NF-OP3:** All retry and back-off parameters must be configurable via env vars.

### 3.5 Security

- **NF-SEC1:** CORS `allow_origins` must be restricted in `APP_ENV=production`.
- **NF-SEC2:** All tool file-name inputs must be validated against directory traversal
  (`..`, absolute paths, null bytes) before any filesystem operation.
- **NF-SEC3:** `base64.b64decode()` in HTTP endpoint handlers must be wrapped in
  try/except to prevent raw exception propagation.

---

## 4. Glossary & Definitions

| Term | Definition |
|---|---|
| **ReAct Loop** | The Reason + Act iteration cycle: Claude produces Thought/Action/Observation triples until `Final Answer:` is emitted |
| **AnalysisSession** | Aggregate root for a single conversation: holds message history, figures, react_trace, unit_context, notebook_cells |
| **CodeRunner** | Abstract interface for code execution; implementations: `SubprocessCodeRunner`, `JupyterKernelManager` |
| **Observation** | The tool result string injected back to Claude as a user message in format `"Observation: ..."` |
| **parse_error** | A Claude response that does not match any known ReAct format pattern |
| **Context window** | Maximum total token count accepted by the Claude API (200K for claude-sonnet-4-6) |
| **Session TTL** | Time-to-live: max inactivity duration before a session and its CodeRunner are garbage-collected |
| **MAX_DATASET_BYTES** | Configurable upper bound on dataset file size before refusing to load into memory |
| **DOMAIN_RANGES** | Dict mapping physical quantity names to (min, max, canonical_unit) tuples for range validation |
| **llm_api_error** | Error code returned when the Anthropic API call fails after all retries |
| **context_overflow** | Error code returned when `BadRequestError` from Claude indicates prompt length exceeded |
| **session_gc** | Background garbage-collection task that evicts expired sessions and their CodeRunners |
