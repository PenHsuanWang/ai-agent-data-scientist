# System Overview — C4 Architecture Diagrams

**Document Version:** 1.0  
**Status:** Approved  
**Scope:** Data Scientist AI-Agent — Full System Architecture  

---

## 1. Introduction & Scope

This document describes the complete system architecture of the **Data Scientist AI-Agent**, a conversational AI assistant that combines Claude's language reasoning with real code execution, physical-unit validation, and interactive Jupyter notebook generation.

**In-Scope:**
- FastAPI backend with ReAct agentic loop
- Three code-execution backends (Subprocess, Jupyter, Anthropic)
- 14-tool catalog spanning knowledge retrieval, data analysis, unit validation, and output generation
- All new API endpoints (`/api/v1/analysis/*`, `/api/v1/datasets/*`)
- Jupyter kernel lifecycle management
- Physical validation via `pint`

**Out-of-Scope:**
- Authentication / OAuth (out of current milestone)
- Multi-user tenancy and RBAC
- Cloud deployment infrastructure (Kubernetes, Terraform)
- The existing MVP endpoints (`/api/v1/chat`, `/api/v1/documents`) — these are untouched

---

## 2. Level 1: Context Diagram

The highest-level view showing the system and its external actors.

```
╔══════════════════════════════════════════════════════════════════════╗
║                        SYSTEM CONTEXT                               ║
╚══════════════════════════════════════════════════════════════════════╝

        ┌─────────────────────┐
        │      Data            │
        │    Scientist         │  (Human User)
        │      User            │  Asks data analysis questions
        └────────┬────────────┘  Uploads CSV / Excel datasets
                 │ HTTP/REST (JSON)
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │               Data Scientist AI-Agent System              │
  │                                                           │
  │  Conversational AI backend that:                          │
  │  • Understands natural-language analysis requests         │
  │  • Executes Python code against uploaded datasets         │
  │  • Validates physical units and magnitudes                │
  │  • Generates Jupyter notebooks as output                  │
  │  • Maintains per-session conversation state               │
  └───────┬────────────────┬────────────────┬────────────────┘
          │                │                │
          │ HTTPS/REST      │ File I/O        │ IPC / HTTP
          ▼                ▼                ▼
  ┌───────────────┐  ┌─────────────┐  ┌──────────────────────┐
  │  Anthropic    │  │   Local     │  │   Jupyter Kernel     │
  │  Claude API   │  │ Filesystem  │  │   (Optional)         │
  │               │  │             │  │                      │
  │  claude-sonnet │  │  /datasets/ │  │  Python 3.x process  │
  │  -4-6 family   │  │  /docs/     │  │  running per-session │
  │               │  │  /figures/  │  │  kernel via          │
  │  messages API │  │  /notebooks/│  │  jupyter_client      │
  └───────────────┘  └─────────────┘  └──────────────────────┘
```

**External System Descriptions:**

| External System | Protocol | Purpose |
|---|---|---|
| Anthropic Claude API | HTTPS REST | LLM reasoning, text generation, (optionally) hosted code execution |
| Local Filesystem | POSIX file I/O | Dataset storage, domain docs, figure cache, notebook output |
| Jupyter Kernel | ZMQ (via jupyter_client) | Stateful Python REPL for code execution with true variable persistence |

---

## 3. Level 2: Container Diagram

The Data Scientist AI-Agent System decomposed into its runtime containers.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                  DATA SCIENTIST AI-AGENT SYSTEM (Containers)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Browser / API Client                                                │
  │  (React UI  OR  curl / Postman / SDK)                               │
  │                                                                      │
  │  • Sends POST /api/v1/analysis/chat with message + session_id        │
  │  • Renders figures from GET /api/v1/analysis/{sid}/figures/{fid}    │
  │  • Downloads notebook from GET /api/v1/analysis/{sid}/notebook      │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │ HTTP/JSON
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  FastAPI Backend  [Python 3.12 / Uvicorn]                            │
  │                                                                      │
  │  ┌─────────────────────┐   ┌────────────────────────────────────┐   │
  │  │   MVP Routes         │   │   New Analysis Routes              │   │
  │  │  POST /api/v1/chat  │   │  POST /api/v1/analysis/chat        │   │
  │  │  POST /api/v1/docs  │   │  GET  /api/v1/analysis/{sid}/...   │   │
  │  │  (UNCHANGED)         │   │  GET  /api/v1/datasets             │   │
  │  └─────────────────────┘   └────────────────────────────────────┘   │
  │                                          │                           │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │  DataScienceAgentService  (ReAct Loop)                        │   │
  │  │  • Builds system prompt with domain context                   │   │
  │  │  • Calls Claude API for Thought/Action parsing                │   │
  │  │  • Dispatches to ToolRegistry                                  │   │
  │  │  • Appends Observation blocks                                 │   │
  │  │  • Validates physical units before final return               │   │
  │  └──────────────────────────────────────────────────────────────┘   │
  │                                          │                           │
  │  ┌──────────────────┐  ┌──────────────────────────────────────┐     │
  │  │  ToolRegistry     │  │  Infrastructure Layer                │     │
  │  │  knowledge_tools  │  │  ┌──────────────────────────────┐   │     │
  │  │  data_tools       │  │  │  SubprocessCodeRunner        │   │     │
  │  │  validation_tools │  │  │  JupyterKernelManager        │   │     │
  │  │  output_tools     │  │  │  AnthropicCodeExecRunner     │   │     │
  │  └──────────────────┘  │  │  UnitRegistry (pint)         │   │     │
  │                         │  └──────────────────────────────┘   │     │
  │  ┌──────────────────┐  └──────────────────────────────────────┘     │
  │  │  SessionStore     │                                               │
  │  │  InMemory         │  (AnalysisSession keyed by session_id)        │
  │  └──────────────────┘                                               │
  └──────────────────────────────────────────────────────────────────────┘
                   │                           │
          HTTPS    │                  ZMQ/IPC  │
                   ▼                           ▼
  ┌────────────────────────┐    ┌────────────────────────────────┐
  │   Anthropic Claude API  │    │   Jupyter Kernel Process       │
  │   claude-sonnet-4-6     │    │   (one per AnalysisSession)    │
  │   messages.create()     │    │   IPykernel / Python 3.12      │
  └────────────────────────┘    └────────────────────────────────┘
```

**Container Responsibilities:**

| Container | Technology | Key Responsibility |
|---|---|---|
| FastAPI Backend | Python 3.12, Uvicorn | All HTTP request handling, routing, session management |
| DataScienceAgentService | Pure Python | ReAct protocol, tool dispatch, physical validation gate |
| ToolRegistry | Pure Python dicts | Maps tool names → functions, validates inputs |
| SessionStore | In-memory dict | Holds `AnalysisSession` objects keyed by `session_id` |
| Infrastructure Layer | subprocess / jupyter_client / anthropic | Code execution, kernel management, unit parsing |
| React UI (optional) | React 19, TypeScript, Vite | Frontend interface for data scientists |

---

## 4. Level 3: Component Diagram (FastAPI Backend)

Internal components of the FastAPI backend.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              FASTAPI BACKEND — COMPONENT DIAGRAM                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌───────────────────────────────────────────────────────────────────────┐
  │  api/v1/analysis.py                                                    │
  │  ┌──────────────────────────────────────────────────────────────────┐ │
  │  │  POST /api/v1/analysis/chat                                       │ │
  │  │  GET  /api/v1/analysis/{session_id}/figures/{figure_id}           │ │
  │  │  GET  /api/v1/analysis/{session_id}/notebook                      │ │
  │  └──────────────────────┬───────────────────────────────────────────┘ │
  └─────────────────────────┼─────────────────────────────────────────────┘
                            │ calls
  ┌─────────────────────────▼─────────────────────────────────────────────┐
  │  services/data_agent.py   DataScienceAgentService                      │
  │                                                                         │
  │  ┌────────────────────┐   ┌──────────────────────┐                     │
  │  │ PhysicalContext     │   │ ReActParser           │                     │
  │  │ Injector           │   │ • parse_response()    │                     │
  │  │ • load_domain_docs │   │ • extract_action()    │                     │
  │  │ • build_prompt()   │   │ • extract_thought()   │                     │
  │  └────────────────────┘   └──────────┬───────────┘                     │
  │           │                          │                                  │
  │  ┌────────▼──────────────────────────▼──────────────────────────────┐ │
  │  │  run(session: AnalysisSession) → str                              │ │
  │  │  ReAct loop (max 20 iterations):                                  │ │
  │  │    1. Build prompt with context                                   │ │
  │  │    2. Call Claude API → parse Thought/Action                      │ │
  │  │    3. Dispatch action → ToolRegistry                              │ │
  │  │    4. Append Observation                                          │ │
  │  │    5. Detect "Final Answer:" → validation gate → return           │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  └────────────────────────┬──────────────────────────────────────────────┘
                           │ invokes tools via
  ┌────────────────────────▼──────────────────────────────────────────────┐
  │  services/knowledge_tools.py     services/data_tools.py               │
  │                                                                         │
  │  list_domain_documents()         execute_python_code(code)             │
  │  read_domain_document(name)      get_execution_variables()             │
  │  search_domain_knowledge(query)  get_figure(figure_id)                 │
  │  list_datasets()                 list_figures()                        │
  │  inspect_dataset(name)                                                 │
  │  describe_columns(name, cols)                                           │
  └───────────┬─────────────────────────────┬─────────────────────────────┘
              │                             │ delegates
  ┌───────────▼──────────────┐  ┌──────────▼──────────────────────────────┐
  │  services/tools.py        │  │  infrastructure/                         │
  │  (EXISTING — unchanged)   │  │  code_runner.py  → SubprocessCodeRunner  │
  │  list_local_documents()   │  │  jupyter_bridge.py → JupyterKernelMgr   │
  │  read_local_document()    │  │  anthropic_code_exec.py → AnthropicExec  │
  └───────────────────────────┘  │  unit_registry.py → pint.UnitRegistry   │
                                  └──────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  domain/                                                                  │
  │  models.py (EXISTING)          analysis_models.py (NEW)                  │
  │  AgentSession                  AnalysisSession, DatasetMeta,             │
  │  AgentError, AgentLoopError    AnalysisResult, PhysicalUnit              │
  │  (exceptions.py EXTENDED)      ReActLoopError, CodeExecutionError, etc.  │
  └──────────────────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  core/config.py  (EXTENDED)                                               │
  │  code_execution_backend: str = "subprocess"  # | "jupyter" | "anthropic" │
  │  datasets_dir: Path, domain_docs_dir: Path                               │
  │  max_react_iterations: int = 20                                           │
  │  max_code_output_bytes: int = 102_400                                     │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Key Design Principles

### Principle 1: Backward Compatibility is Inviolable
The existing MVP (`/api/v1/chat`, `AgentService`, `AgentSession`, `tools.py`) is not modified. All new capabilities live in separate modules. The new system extends, never replaces.

### Principle 2: ReAct over Pure Tool-Use
Rather than relying on Claude's native tool-use stop reason, the Data Scientist Agent uses an explicit text-based ReAct protocol (Thought → Action → Observation). This makes the reasoning trace auditable, exportable, and framework-agnostic.

### Principle 3: Infrastructure is Pluggable
Code execution is abstracted behind a `CodeRunner` interface. Switching from subprocess isolation to a full Jupyter kernel to Anthropic-hosted execution requires only a configuration change, not a code change.

### Principle 4: Physical Reality as a Hard Constraint
Every final answer is passed through a physical validation gate before being returned to the user. A result claiming "efficiency = 450%" is rejected. This is non-optional.

### Principle 5: Domain as First-Class Context
Domain knowledge documents (PDFs, Markdown) are not just retrieval targets — they are parsed at startup to build a structured physical context that is injected into every system prompt.

---

## 6. Data Flow Walkthrough

**Request:** `"Analyze power_plant_data.csv and compute thermal efficiency"`

```
Step 1: HTTP Ingress
  POST /api/v1/analysis/chat
  Body: { "message": "Analyze power_plant_data.csv ...", "session_id": null }
  → api/v1/analysis.py handler receives request
  → SessionStore.get_or_create(session_id) → new AnalysisSession

Step 2: Context Injection
  DataScienceAgentService.run(session)
  → PhysicalContextInjector.build_system_prompt()
    → reads /docs/thermodynamics.md, /docs/power_plants.md
    → extracts unit definitions (MW, °C, bar, η)
    → builds: "You are a data scientist expert. Domain context: ..."
  → session.add_user_message("Analyze power_plant_data.csv ...")

Step 3: ReAct Iteration 1 — List available datasets
  → Claude API called with system prompt + message history
  ← Claude returns:
      "Thought: First I should confirm the dataset exists and understand its schema.
       Action: inspect_dataset
       Action Input: {"file_name": "power_plant_data.csv"}"
  → ReActParser.parse_response() extracts thought, action, action_input
  → ToolRegistry.dispatch("inspect_dataset", {"file_name": "power_plant_data.csv"})
  → knowledge_tools.inspect_dataset("power_plant_data.csv")
     → loads first 5 rows with pandas (nrows=5)
     → computes shape, dtypes, null_counts, sample_rows
     → returns JSON string of DatasetMeta
  → session.react_trace.append({thought, action, observation})
  → Append "Observation: {DatasetMeta JSON}\n" to messages

Step 4: ReAct Iteration 2 — Execute computation
  → Claude API called
  ← Claude returns:
      "Thought: Dataset has columns heat_input_MW, net_output_MW. 
               Thermal efficiency = net_output / heat_input.
       Action: execute_python_code
       Action Input: {"code": "import pandas as pd\ndf = pd.read_csv('...')\ndf['efficiency'] = df['net_output_MW'] / df['heat_input_MW']\nprint(df['efficiency'].describe())"}"
  → data_tools.execute_python_code(code)
  → CodeRunnerFactory.get() → SubprocessCodeRunner (from settings)
  → SubprocessCodeRunner.execute(code)
     → AST analysis: no banned imports
     → subprocess.run(["python", "-c", code], timeout=30, capture_output=True)
     → captures stdout, stderr, figure IDs
     → pickles variable state for next call
  → returns AnalysisResult JSON
  → Append Observation

Step 5: ReAct Iteration 3 — Validate results
  → Claude returns:
      "Thought: Efficiency values look high. Let me validate.
       Action: validate_physical_units
       Action Input: {"quantity": "thermal_efficiency", "value": 0.42, "unit": "%", ...}"
  → validation_tools.validate_physical_units(...)
  → UnitRegistry.validate() → stage 1: unit parsing OK
                             → stage 2: dimensional check OK (dimensionless)
                             → stage 3: magnitude check: 42% within [10%, 60%] → VALID
  → Returns PhysicalUnit JSON

Step 6: Final Answer
  → Claude returns:
      "Final Answer: The thermal efficiency averages 42.3% (range: 38%–47%).
                     This is within expected bounds for a coal-fired plant."
  → ReActParser detects "Final Answer:" prefix
  → Physical validation gate: scans final answer for numeric values → all valid
  → AnalysisResult appended to session
  → return "The thermal efficiency averages 42.3% ..."

Step 7: HTTP Response
  → AnalysisResponse built:
    { "response": "...", "session_id": "abc-123",
      "react_trace": [{...}, {...}, {...}],
      "figures": [],
      "notebook_available": false }
  → 200 OK returned to client
```

---

## 7. Backward Compatibility

The following is an explicit guarantee:

**`/api/v1/chat`** — endpoint unchanged. Same request/response schema. Same `AgentService`. Same `AgentSession`. Same tool set (`list_local_documents`, `read_local_document`). Adding new modules does not affect this path.

**`AgentSession`** — untouched dataclass in `domain/models.py`. `AnalysisSession` is a subclass that extends it with optional new fields, maintaining Liskov substitution.

**`tools.py`** — unchanged. New tools live in `knowledge_tools.py` and `data_tools.py`.

**`config.py`** — extended with new fields, all with defaults. Existing field names and defaults preserved.

**Test suite for MVP** — all existing tests continue passing. New tests target new modules only.

---

## 8. Deployment View

### Local Development

```
Developer Machine
├── uvicorn app.main:app --reload --port 8000
├── Jupyter kernel: launched on demand by JupyterKernelManager
├── /datasets/    ← user uploads via API
├── /docs/        ← domain knowledge files (checked in)
└── .env          ← ANTHROPIC_API_KEY, CODE_EXECUTION_BACKEND=subprocess
```

### Production (Single Server)

```
Production Server (8-core, 32GB RAM)
├── systemd: uvicorn app.main:app --workers 4 --port 8000
├── nginx: reverse proxy, static file serving for figures
├── /data/datasets/    ← persistent volume
├── /data/docs/        ← persistent volume  
├── /data/figures/     ← ephemeral, cleared on restart
└── ENV: CODE_EXECUTION_BACKEND=jupyter  (full kernel per session)
     ANTHROPIC_API_KEY=sk-ant-...
     MAX_SESSIONS=50
     JUPYTER_KERNEL_TIMEOUT=3600
```

**What changes between environments:**

| Setting | Local Dev | Production |
|---|---|---|
| `code_execution_backend` | `subprocess` | `jupyter` |
| `workers` | 1 (--reload) | 4 |
| `max_react_iterations` | 20 | 20 |
| `datasets_dir` | `./datasets` | `/data/datasets` |
| `jupyter_kernel_timeout` | 600s | 3600s |
| Nginx | No | Yes |

---

## 9. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.12 | All backend code |
| Web Framework | FastAPI | 0.115.x | HTTP routing, request/response handling, OpenAPI generation |
| ASGI Server | Uvicorn | 0.30.x | Async server, multi-worker support |
| Data Validation | Pydantic v2 | 2.7.x | API schemas, settings, response models |
| LLM Client | anthropic | 0.30.x | Claude API access, messages.create() |
| Data Library | pandas | 2.2.x | Dataset loading, inspection, statistics |
| Numeric | numpy | 1.26.x | Array operations in user code |
| Plot Library | matplotlib | 3.9.x | Figure generation in executed code |
| Unit Validation | pint | 0.23.x | Physical unit parsing, conversion, dimensional analysis |
| Jupyter Client | jupyter_client | 8.6.x | Kernel management, ZMQ communication |
| Notebook Format | nbformat | 5.10.x | .ipynb creation, cell management |
| Code Security | ast (stdlib) | 3.12 | AST-based code analysis before execution |
| Settings | pydantic-settings | 2.x | .env loading, environment variable binding |
| HTTP Client | httpx | 0.27.x | Async HTTP for any external calls |
| Testing | pytest + pytest-asyncio | 8.x | Unit and integration tests |
| Frontend (optional) | React 19 + Vite + TypeScript | 19 / 5.x / 5.x | Data scientist UI |
| Frontend Styling | Tailwind CSS v4 | 4.x | Component styling |
