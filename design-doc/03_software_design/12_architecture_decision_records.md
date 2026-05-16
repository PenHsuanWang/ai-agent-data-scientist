# Architecture Decision Records (ADRs)

**Document Version:** 1.0  
**Status:** Approved  
**Scope:** Data Scientist AI-Agent — Key Architectural Decisions  

ADRs capture the context, reasoning, and consequences of significant design decisions.
They are permanent records — superseded ADRs are marked as such rather than deleted.

---

## ADR-001: Text-Mode ReAct vs Anthropic Native Tool-Use API

**Date:** 2025  
**Status:** Accepted

### Context

Anthropic's Python SDK supports two modes for agentic workflows:

1. **Native Tool-Use Mode:** The client sends a `tools=[]` list; Claude returns structured
   `tool_use` content blocks with parsed JSON inputs. The SDK handles the Thought/Action
   parsing automatically.

2. **Text-Mode ReAct:** Claude generates free-text responses following the
   `Thought: / Action: / Action Input:` template. The application parses these via
   regex/AST.

The project must choose one mode as the primary architecture for the ReAct loop.

### Decision

**Use text-mode ReAct (Option 2).**

### Rationale

| Criterion | Native Tool-Use | Text-Mode ReAct | Decision |
|---|---|---|---|
| Full reasoning transparency | ❌ Thoughts are implicit | ✅ `Thought:` block is explicit | Text-Mode |
| `react_trace` with thought strings | ❌ No structured thought output | ✅ Every step has `thought`, `action`, `observation` | Text-Mode |
| JSON schema validation | ✅ SDK validates inputs | 🟡 Regex + `ast.literal_eval` fallback | Native |
| Multi-provider compatibility | ❌ Anthropic-specific | ✅ Any model following the ReAct template | Text-Mode |
| Prompt engineering control | 🟡 System prompt supplements schema | ✅ Full control over format and instructions | Text-Mode |
| Domain-specific prompting | 🟡 Limited | ✅ Physical validation rules injected inline | Text-Mode |

The explicit `Thought:` block is a first-class requirement: it populates `react_trace`
in API responses, enabling operators and users to audit the agent's reasoning. Native
Tool-Use does not expose per-step thoughts in a structured way.

### Consequences

**Positive:**
- Full reasoning trace available in every API response
- System prompt can be tuned with domain-specific rules (physical validation, tool guidance)
- Compatible with any LLM that follows the ReAct format

**Negative:**
- `ReActParser` must handle malformed output (see Gap 16 and `09_exception_handling_design.md`)
- JSON parsing requires `ast.literal_eval` fallback for single-quoted strings from Claude
- Parse errors add latency (format-correction injection adds a loop iteration)

**Mitigation:** `ReActParser.handle_malformed()` injects a format-correction observation
and records `__parse_error__` in `react_trace` to detect prompt quality issues.

---

## ADR-002: In-Memory Session Store vs Redis

**Date:** 2025  
**Status:** Accepted (for development); Superseded in production by Redis migration

### Context

`AnalysisSession` objects must be persisted across multiple HTTP requests (multi-turn
conversations). A session store is required. Two options:

1. **In-Memory (`dict`):** Zero dependencies, zero latency, zero configuration.
2. **Redis:** Persistent, survives restarts, enables horizontal scaling.

### Decision

**Use `InMemorySessionStore` (Option 1) for the current milestone.**  
A `AbstractSessionStore` interface is defined now to allow Redis substitution without
changing the API layer.

### Rationale

- Current deployment is single-instance; no horizontal scaling requirement exists today
- Redis adds an operational dependency (Redis server, `redis-py`, connection pool config)
- The interface (`get`, `set`, `delete`) is intentionally minimal — a `RedisSessionStore`
  can implement it without API changes (see `10_hld_architecture.md §9.1`)

### Consequences

**Positive:**
- Zero infrastructure dependencies for local development and testing
- Session lookup is O(1) with no network hop

**Negative / Risks:**
- **Data loss on restart:** All sessions are lost when the process restarts
- **Unbounded memory growth:** No TTL eviction (Gap 5 in exception audit); mitigated by
  the Session GC design in `09_exception_handling_design.md §4`
- **No horizontal scaling:** A second uvicorn instance would not share session state

**Migration path to Redis:**
1. Implement `RedisSessionStore(AbstractSessionStore)` in `app/infrastructure/`
2. Wire via FastAPI `Depends()` using the corrected DIP pattern (see `10_hld_architecture.md §9.1`)
3. Set `SESSION_BACKEND=redis` in `.env`

---

## ADR-003: Three-Backend Code Execution Strategy

**Date:** 2025  
**Status:** Accepted

### Context

The agent must execute Python code submitted by Claude. Three execution backends are
needed:

1. **Subprocess:** Spawn a fresh Python child process per execution. State persisted
   via pickle between calls.
2. **Jupyter Kernel:** A persistent `jupyter_client` kernel per session. True REPL state.
3. **Anthropic Code Execution:** Anthropic's hosted execution sandbox via the
   `code_execution_20260120` server tool.

The project must decide the default backend and the interface contract.

### Decision

**Default backend: `subprocess`. All three backends implement the `CodeRunner` interface
(Strategy pattern). Selected via `CODE_EXECUTION_BACKEND` environment variable.**

### Rationale

| Criterion | Subprocess | Jupyter | Anthropic |
|---|---|---|---|
| Zero infrastructure dependency | ✅ | ❌ (requires Jupyter) | ❌ (requires direct API) |
| True REPL state (no pickle) | ❌ | ✅ | ✅ |
| Notebook export support | ❌ | ✅ (`jupyter_cells`) | ❌ |
| AST security pre-check | ✅ | ❌ | ❌ (Anthropic manages) |
| Resource limits (setrlimit) | ✅ | 🟡 (cgroups) | ✅ (Anthropic manages) |
| Available on AWS Bedrock/Vertex | ✅ | ✅ | ❌ |

Subprocess is the safe default for development. Jupyter is preferred for production when
notebook export is a primary use case. Anthropic backend is reserved for managed cloud
deployments on direct `api.anthropic.com`.

### Consequences

**Positive:**
- `CodeRunner` interface (Strategy pattern) allows backend switching without changing
  the ReAct loop or tool implementations
- OCP: new backends can be added by implementing `CodeRunner` and registering in
  `CodeRunnerFactory` — `DataScienceAgentService` is never modified
- LSP: all three backends are substitutable; `execute()` always returns `AnalysisResult`

**Negative:**
- Subprocess backend uses pickle for state — risks `PickleError` on complex objects
  (silent state loss, not a crash)
- Jupyter backend introduces Gap 8 (kernel crash, no restart logic) — mitigated by
  `_restart_kernel()` design in `05_code_execution_design.md §9.2`
- Three backends increase test surface area

---

## ADR-004: Modular Monolith vs Microservices

**Date:** 2025  
**Status:** Accepted

### Context

The system can be built as either a single FastAPI application or decomposed into
multiple independent services (LLM Gateway, Code Execution, Analysis API).

Evaluated against `system_architecture_hld.md §1`.

### Decision

**Build as a Modular Monolith. Define internal module boundaries that map to future
microservice extraction boundaries (see `10_hld_architecture.md §10`).**

### Rationale

Per HLD reference §1: *"Before splitting a service, confirm that the operational
overhead (deployment, monitoring, network latency) is justified by the domain boundary."*

Current scale assessment:
- Single-tenant deployment (one user or small team)
- No independent scaling requirement for LLM calls vs code execution
- No multi-team ownership boundary requiring service isolation
- Operational overhead of microservices (service discovery, inter-service auth,
  distributed tracing) exceeds the benefit at this scale

The module structure (`domain/`, `services/`, `infrastructure/`, `api/`) is Clean
Architecture-compliant and maps directly to the proposed future service boundaries.

### Consequences

**Positive:**
- Simple deployment (single Docker image, single port)
- No network serialization overhead between components
- Easier to debug and trace (single process, single log stream)

**Negative:**
- A runaway code execution subprocess can affect the API server process
  (mitigated by `setrlimit` resource limits — see `05_code_execution_design.md §9.1`)
- Cannot independently scale code execution vs API serving

**Extraction triggers (re-evaluate when any is true):**
- Code execution needs GPU access or independent horizontal scaling
- A second LLM provider is integrated
- Multi-tenant isolation requirements demand process-level separation

---

## ADR-005: `pint` for Physical Unit Validation

**Date:** 2025  
**Status:** Accepted

### Context

The agent analyzes power plant, thermodynamic, and materials-science data. Numeric
values with physical units (efficiency %, temperature K/°C, pressure bar, power MW)
must be validated against known physical bounds to prevent the model from reporting
physically impossible results (e.g., 120% efficiency, −500 K temperature).

Two options were considered:
1. **`pint` library:** Standards-compliant unit registry, conversion, and dimensional
   analysis. Active maintenance, 5k+ GitHub stars.
2. **Custom lookup table:** A dict of `{unit: (min, max)}` bounds, no external dep.

### Decision

**Use `pint` for unit conversion and dimensional analysis. Complement with a custom
domain range table (`DOMAIN_RANGES`) for physical plausibility checks.**

### Rationale

- `pint` handles unit conversion (°C → K, bar → kPa) correctly via dimensional analysis,
  which would require significant implementation effort in a custom table
- Domain range validation (is this value physically plausible for this domain?) is
  NOT `pint`'s responsibility — it is a business rule. The custom `DOMAIN_RANGES` dict
  captures this separately and can be extended without touching `pint` integration
- `pint` is a zero-runtime-side-effect library (no network calls, no singletons)

### Consequences

**Positive:**
- Correct unit conversion across all SI and customary systems
- `PhysicalValidationError` is raised only on domain range violations, not on unit
  ambiguity (unit ambiguity produces a warning appended to the Final Answer)
- The `UnitRegistry` infrastructure adapter wraps `pint` — the domain layer never
  imports `pint` directly (DIP compliance)

**Negative:**
- `pint` adds ~2 MB to the dependency footprint
- Unit string parsing (e.g., `"kWh/m²"`) can fail on non-standard notations;
  all such failures return `"Error: ..."` strings (tools-never-raise contract)

---

## ADR-006: No Authentication in v1

**Date:** 2025  
**Status:** Accepted (Deferred — v2 planned)

### Context

The current API has no authentication or authorization. All endpoints are publicly
accessible. This is intentional for the development milestone. The system is designed
for local or trusted-network deployment only.

### Decision

**No authentication for v1. Plan OAuth 2.0 / API key authentication for v2.**

### Rationale

- v1 is single-tenant (personal or small-team deployment)
- Adding auth at this stage would block progress on core AI agent functionality
- The `CORS allow_origins=["*"]` issue (Gap 14 / EPIC-6) is mitigated by the env-aware
  CORS config planned in `08_api_design.md §13`

### Consequences

**Risk:** If deployed publicly without a network-level firewall, the agent's Anthropic
API key could be abused by third parties making unlimited API calls.

**Mitigations required before any public deployment:**
1. Deploy behind a network firewall or VPN
2. Set `CORS_ORIGINS` to specific allowed origins
3. Add API key header (`X-API-Key`) validation in a FastAPI middleware
4. Rate-limit with `slowapi` or an API Gateway

**v2 authentication design (out of scope for current milestone):**
- API key issued per user, stored hashed in PostgreSQL
- JWT short-lived access tokens for session continuity
- Route-level authorization via FastAPI `Depends(require_auth)`
