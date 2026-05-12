# Data Scientist AI Agent

> **Domain-aware AI agent for data science** — FastAPI + Anthropic Claude + ReAct reasoning loop + `pint` physical unit validation + Jupyter notebook export.

---

## Key Differentiators

| Feature | Description |
|---|---|
| **ReAct Reasoning Trace** | Full `Thought → Action → Observation` loop visible in API response |
| **Physical Validation** | `pint`-backed unit registry enforces thermodynamic constraints (efficiency ≤ 100%, T > 0 K, etc.) |
| **15 Tools** | 6 knowledge + 6 execution + 3 validation tools |
| **Notebook Export** | Every session exportable as `.ipynb` with code cells and reasoning |
| **Multi-backend Code Execution** | Subprocess (safe, default) or Jupyter kernel (stateful) |
| **Domain Knowledge** | Reads and searches `.md` domain docs before analysing data |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                     │
│         app/api/v1/analysis.py · datasets.py            │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                  Application Layer                       │
│   app/services/data_agent.py   ← ReAct Loop             │
│   app/services/knowledge_tools.py  (Group A, 6 tools)   │
│   app/services/data_tools.py       (Group B, 6 tools)   │
│   app/services/tool_definitions.py (15 tool schemas)    │
│   app/services/memory.py           (session store)      │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│               Infrastructure Layer                       │
│   app/infrastructure/code_runner.py    (subprocess/jup) │
│   app/infrastructure/unit_registry.py (pint, Group C)   │
│   app/infrastructure/notebook_exporter.py (nbformat)    │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│                    Domain Layer                          │
│   app/domain/analysis_models.py   (AnalysisSession, …)  │
│   app/domain/models.py            (AgentSession)        │
│   app/domain/exceptions.py        (AgentError, …)       │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
cd ai-agent-data-scientist
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY
```

### 3. Generate sample data

```bash
python scripts/create_sample_data.py
```

### 4. Verify installation

```bash
python scripts/verify_install.py
```

### 5. Run the server

```bash
uv run uvicorn app.main:app --reload --port 8001
```

Open http://localhost:8001/docs for the interactive API documentation.

---

## API Reference

### POST `/api/v1/analysis/chat`

Submit a natural language analysis request.

**Request body:**
```json
{
  "message": "Analyze power_plant_data.csv and compute the mean thermal efficiency",
  "session_id": null
}
```

**Response:**
```json
{
  "response": "The mean thermal efficiency is 36.73% ...",
  "session_id": "a1b2c3d4-...",
  "react_trace": [
    {
      "thought": "I should first read domain documents ...",
      "action": "list_domain_documents({})",
      "observation": "[{\"file_name\": \"power_plant_thermodynamics.md\", ...}]"
    }
  ],
  "figures": [
    {
      "figure_id": "fig_000",
      "retrieval_url": "/api/v1/analysis/<session_id>/figures/fig_000"
    }
  ],
  "notebook_available": false,
  "unit_validations": [],
  "iterations_used": 8,
  "model": "claude-sonnet-4-6",
  "status": "completed"
}
```

### GET `/api/v1/analysis/{session_id}/figures/{figure_id}`

Returns the figure as a PNG image (`image/png`).

### GET `/api/v1/analysis/{session_id}/notebook`

Downloads the exported Jupyter notebook (`.ipynb`). Requires `export_notebook` to be called first.

### GET `/api/v1/datasets`

List all available datasets.

### GET `/api/v1/datasets/{name}/schema`

Get dataset schema, statistics, and sample rows.

### GET `/health`

Liveness probe — returns model and backend info.

---

## Tool Catalog

### Group A — Knowledge Tools (6)

| Tool | Description |
|---|---|
| `list_domain_documents` | List available `.md` knowledge files |
| `read_domain_document` | Read full content of a document |
| `search_domain_knowledge` | Keyword search across all docs |
| `list_datasets` | List available dataset files |
| `inspect_dataset` | Schema, stats, and 5 sample rows |
| `describe_columns` | Detailed per-column statistics |

### Group B — Execution Tools (6)

| Tool | Description |
|---|---|
| `execute_python_code` | Run Python (pandas/numpy/matplotlib pre-imported) |
| `get_execution_variables` | Snapshot of last execution's variables |
| `get_figure` | Retrieve figure as base64 PNG |
| `list_figures` | List all session figure IDs |
| `export_notebook` | Export session as `.ipynb` |
| `save_figure` | Save figure to `outputs/figures/` |

### Group C — Physical Validation Tools (3)

| Tool | Description |
|---|---|
| `validate_physical_units` | Check unit + magnitude against domain ranges |
| `convert_units` | Unit conversion via `pint` |
| `check_magnitude` | Plausibility check against domain ranges |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `MAX_TOKENS` | `8192` | Max tokens per API call |
| `MAX_RETRIES` | `2` | API retry count |
| `CODE_EXECUTION_BACKEND` | `subprocess` | `subprocess` or `jupyter` |
| `CODE_EXECUTION_TIMEOUT` | `30` | Seconds before code execution timeout |
| `MAX_REACT_ITERATIONS` | `20` | Max ReAct loop iterations |
| `DATASETS_DIR` | `data/datasets` | Dataset search path |
| `DOMAIN_DOCS_DIR` | `data/domain_docs` | Domain knowledge docs path |
| `FIGURES_DIR` | `outputs/figures` | Figure output directory |
| `NOTEBOOKS_DIR` | `outputs/notebooks` | Notebook output directory |

---

## Example curl Commands

```bash
# Start a new analysis
curl -X POST http://localhost:8001/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the average thermal efficiency in power_plant_data.csv?"}'

# Continue a session
curl -X POST http://localhost:8001/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Now plot efficiency over time", "session_id": "<session_id>"}'

# Download a figure
curl http://localhost:8001/api/v1/analysis/<session_id>/figures/fig_000 --output plot.png

# List datasets
curl http://localhost:8001/api/v1/datasets

# Get dataset schema
curl http://localhost:8001/api/v1/datasets/power_plant_data.csv/schema

# Health check
curl http://localhost:8001/health
```

---

## ReAct Protocol

The agent follows the **Thought → Action → Observation** loop:

```
Thought: I need to understand the domain before analysing data.
Action: list_domain_documents
Action Input: {}

Observation: [{"file_name": "power_plant_thermodynamics.md", "size_bytes": 1842}]

Thought: Let me read the thermodynamics document first.
Action: read_domain_document
Action Input: {"file_name": "power_plant_thermodynamics.md"}

Observation: # Power Plant Thermodynamics ...

...

Thought: I have all the information needed.
Final Answer: The mean thermal efficiency is 36.73% (range: 33.1%–40.2%).
```

The full trace is returned in `react_trace` so you can inspect every reasoning step.

---

## Project Structure

```
ai-agent-data-scientist/
├── app/
│   ├── api/v1/
│   │   ├── analysis.py        # POST /chat, GET figures/notebook
│   │   └── datasets.py        # GET /datasets
│   ├── core/
│   │   └── config.py          # pydantic-settings
│   ├── domain/
│   │   ├── analysis_models.py # AnalysisSession, DatasetMeta, etc.
│   │   ├── exceptions.py      # AgentError hierarchy
│   │   └── models.py          # AgentSession
│   ├── infrastructure/
│   │   ├── code_runner.py     # Subprocess + Jupyter backends
│   │   ├── notebook_exporter.py
│   │   └── unit_registry.py   # pint validation
│   ├── schemas/
│   │   ├── analysis.py        # Request/response Pydantic models
│   │   └── datasets.py
│   ├── services/
│   │   ├── data_agent.py      # ReAct loop
│   │   ├── data_tools.py      # Group B tools
│   │   ├── knowledge_tools.py # Group A tools
│   │   ├── memory.py          # Session stores
│   │   └── tool_definitions.py # 15 tool JSON schemas
│   └── main.py                # FastAPI app factory
├── data/
│   ├── datasets/              # CSV, Parquet, Excel files
│   └── domain_docs/           # Markdown knowledge files
├── outputs/
│   ├── figures/               # Saved PNG plots
│   └── notebooks/             # Exported .ipynb files
├── scripts/
│   ├── create_sample_data.py  # Generate sample datasets
│   └── verify_install.py      # Dependency check
├── tests/
├── .env.example
├── CLAUDE.md
└── pyproject.toml
```
