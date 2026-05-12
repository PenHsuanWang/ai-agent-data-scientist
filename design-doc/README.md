# Data Scientist AI-Agent — From Scratch

> **A domain-aware agentic system that reasons about data with physical meaning**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Anthropic SDK](https://img.shields.io/badge/anthropic--sdk-0.50%2B-blueviolet.svg)](https://github.com/anthropic/anthropic-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project teaches you how to build a **production-quality AI agent for data science** from absolute first principles — no LangChain, no AutoGen, no magic. Starting from a minimal FastAPI + Anthropic SDK skeleton, you extend it step-by-step into a domain-aware data scientist agent capable of loading datasets, executing Python code, validating physical units, and exporting reproducible Jupyter notebooks.

The central thesis is that **data science results must be physically meaningful**, not just numerically plausible. An agent that computes "thermal efficiency = 112.5%" has made a catastrophic error — it violates the first law of thermodynamics — yet most AI systems have no mechanism to catch it. This agent does, by pairing code execution with a physical validation layer backed by domain documents and the `pint` unit library.

The guide is structured as a progressive learning path: you begin with fundamentals (ReAct, Clean Architecture, physical modeling), move through the Anthropic SDK capabilities that power the system, implement each component incrementally, and finish with a reference section and worked examples you can run against real datasets.

---

## Key Differentiators

- **Explicit ReAct loop** — Every reasoning step is visible as `Thought → Action → Observation` text, giving full auditability instead of black-box tool-chaining
- **Physical unit validation** — The `pint` library enforces dimensional correctness; domain-specific range checks flag results that violate physics (efficiency > 100%, negative absolute temperatures, etc.)
- **Domain knowledge injection** — Before any analysis, the agent reads relevant domain documents and extracts unit definitions, typical operating ranges, and physical laws relevant to the dataset
- **Three code execution backends** — Local subprocess, Jupyter kernel (persistent state), or Anthropic's cloud sandbox (`code_execution_20260120`) — switchable via config
- **Reproducible notebook export** — Every code cell and output accumulated during a session is exported as a `.ipynb` file you can run, share, or version-control
- **Native Anthropic SDK** — Zero third-party orchestration; you see exactly how tool definitions, `tool_use` blocks, and `tool_result` blocks work at the API level

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│   FastAPI  ─  POST /api/v1/analysis/chat                            │
│              GET  /api/v1/analysis/{id}/notebook                    │
│              GET  /api/v1/datasets                                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────────────┐
│                       APPLICATION LAYER                             │
│   DataScienceAgentService  (ReAct loop orchestration)               │
│   ├── ToolExecutor          (dispatches 14 tools)                   │
│   ├── PhysicalValidator     (pint + domain ranges)                  │
│   └── NotebookExporter      (nbformat cell accumulation)            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                            │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│   │  Code Execution  │  │  Document Store   │  │  Dataset Store   │  │
│   │  ─────────────  │  │  ─────────────── │  │  ─────────────── │  │
│   │  Subprocess      │  │  LocalFileStore  │  │  PandasLoader    │  │
│   │  JupyterKernel   │  │  (markdown/pdf)  │  │  (csv/xls/h5/   │  │
│   │  AnthropicCloud  │  │                  │  │   parquet/json)  │  │
│   └─────────────────┘  └──────────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────────────┐
│                         DOMAIN LAYER                                │
│   AnalysisSession · DatasetMeta · AnalysisResult · PhysicalUnit     │
│   DomainProfile · ExecutionResult · NotebookCell                    │
│   (Pure Python dataclasses — zero framework dependencies)           │
└─────────────────────────────────────────────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────────────────┐
│                      ANTHROPIC CLAUDE API                           │
│   claude-opus-4-7 / claude-sonnet-4-6 / claude-haiku-4-5           │
│   Tools: user-defined (14) + server (code_execution_20260120)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

### Fundamentals
1. [01_fundamentals/01_react_pattern.md](01_fundamentals/01_react_pattern.md) — ReAct: Reason + Act loop, parsing, pitfalls
2. [01_fundamentals/02_clean_architecture.md](01_fundamentals/02_clean_architecture.md) — Four-layer architecture, SOLID, DI
3. [01_fundamentals/03_physical_domain_modeling.md](01_fundamentals/03_physical_domain_modeling.md) — pint, dimensional analysis, domain profiles
4. [01_fundamentals/04_data_science_workflow.md](01_fundamentals/04_data_science_workflow.md) — CRISP-DM + AI augmentation, EDA checklist

### Anthropic SDK Reference
5. [02_anthropic_sdk_reference/01_messages_api.md](02_anthropic_sdk_reference/01_messages_api.md) — Messages API deep-dive, stop reasons, content blocks
6. [02_anthropic_sdk_reference/02_tool_use_client.md](02_anthropic_sdk_reference/02_tool_use_client.md) — Client-side tool-use loop and dispatch
7. [02_anthropic_sdk_reference/03_tool_use_server.md](02_anthropic_sdk_reference/03_tool_use_server.md) — Server tools, pause_turn flow, orchestration
8. [02_anthropic_sdk_reference/04_code_execution_tool.md](02_anthropic_sdk_reference/04_code_execution_tool.md) — Anthropic code execution tool details
9. [02_anthropic_sdk_reference/05_files_api.md](02_anthropic_sdk_reference/05_files_api.md) — Files API upload/use patterns
10. [02_anthropic_sdk_reference/06_models_reference.md](02_anthropic_sdk_reference/06_models_reference.md) — Model selection and cost/capability matrix
11. [02_anthropic_sdk_reference/07_programmatic_tool_calling.md](02_anthropic_sdk_reference/07_programmatic_tool_calling.md) — Programmatic tool calling and allowed_callers

### Software Design
12. [03_software_design/01_system_overview.md](03_software_design/01_system_overview.md) — C4 context/container/component architecture
13. [03_software_design/02_domain_layer_design.md](03_software_design/02_domain_layer_design.md) — Domain entities and value objects
14. [03_software_design/03_react_service_design.md](03_software_design/03_react_service_design.md) — ReAct engine and parser design
15. [03_software_design/04_tool_system_design.md](03_software_design/04_tool_system_design.md) — 14-tool taxonomy, schemas, dispatch model
16. [03_software_design/05_code_execution_design.md](03_software_design/05_code_execution_design.md) — Subprocess/Jupyter/Anthropic backends
17. [03_software_design/06_physical_validation_design.md](03_software_design/06_physical_validation_design.md) — Unit/range/law validation pipeline
18. [03_software_design/07_jupyter_bridge_design.md](03_software_design/07_jupyter_bridge_design.md) — Kernel lifecycle and notebook generation
19. [03_software_design/08_api_design.md](03_software_design/08_api_design.md) — API contracts and route behavior

### Implementation Guide
20. [04_implementation_guide/01_project_setup.md](04_implementation_guide/01_project_setup.md) — Setup, dependencies, project structure
21. [04_implementation_guide/02_phase1_foundation.md](04_implementation_guide/02_phase1_foundation.md) — Domain + config foundation
22. [04_implementation_guide/03_phase2_tools.md](04_implementation_guide/03_phase2_tools.md) — Tool implementations and registry
23. [04_implementation_guide/04_phase3_react_engine.md](04_implementation_guide/04_phase3_react_engine.md) — Parser/context/service loop
24. [04_implementation_guide/05_phase4_validation.md](04_implementation_guide/05_phase4_validation.md) — Physical validation layer
25. [04_implementation_guide/06_phase5_jupyter.md](04_implementation_guide/06_phase5_jupyter.md) — Jupyter bridge integration
26. [04_implementation_guide/07_phase6_api.md](04_implementation_guide/07_phase6_api.md) — FastAPI integration endpoints

### Reference
27. [05_reference/01_sdk_cheatsheet.md](05_reference/01_sdk_cheatsheet.md) — Anthropic SDK quick-reference
28. [05_reference/02_tool_definitions_catalog.md](05_reference/02_tool_definitions_catalog.md) — Full 14-tool catalog and JSON schemas
29. [05_reference/03_react_prompt_templates.md](05_reference/03_react_prompt_templates.md) — ReAct prompt templates by domain
30. [05_reference/04_physical_constants.md](05_reference/04_physical_constants.md) — Physical constants and domain ranges
31. [05_reference/05_troubleshooting.md](05_reference/05_troubleshooting.md) — Operational troubleshooting guide

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/claude_python_sdk.git
cd ai-agent-from-scratch/data-scientist-agent
uv sync

# 2. Set API key and start server
export ANTHROPIC_API_KEY=sk-ant-...
uv run uvicorn app.main:app --reload --port 8001

# 3. Send an analysis request
curl -X POST http://localhost:8001/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Analyze the turbine_data.csv dataset and validate all efficiency values", "session_id": null}'
```

---

## Document Structure

| File | Layer | Purpose |
|------|-------|---------|
| `README.md` | — | Project overview and navigation |
| `00_motivation_and_goals.md` | — | Why this system, design goals, alternatives |
| `01_fundamentals/01_react_pattern.md` | Concept | ReAct loop theory and implementation |
| `01_fundamentals/02_clean_architecture.md` | Concept | Architecture layers and SOLID |
| `01_fundamentals/03_physical_domain_modeling.md` | Concept | pint, units, domain validation |
| `01_fundamentals/04_data_science_workflow.md` | Concept | CRISP-DM + AI augmentation |
| `02_anthropic_sdk_reference/01_messages_api.md` | SDK | Messages API parameters |
| `02_anthropic_sdk_reference/02_tool_use_client.md` | SDK | Tool use loop (client tools) |
| `02_anthropic_sdk_reference/03_tool_use_server.md` | SDK | Server tool orchestration |
| `02_anthropic_sdk_reference/04_code_execution_tool.md` | SDK | Code execution tool behavior |
| `02_anthropic_sdk_reference/05_files_api.md` | SDK | Files API patterns |
| `02_anthropic_sdk_reference/06_models_reference.md` | SDK | Model choice and pricing |
| `02_anthropic_sdk_reference/07_programmatic_tool_calling.md` | SDK | Programmatic tool calling |
| `03_software_design/01_system_overview.md` | Architecture | System-level design and C4 diagrams |
| `03_software_design/02_domain_layer_design.md` | Domain | Dataclass/entity design |
| `03_software_design/03_react_service_design.md` | Application | ReAct orchestration design |
| `03_software_design/04_tool_system_design.md` | Application | Tool system design |
| `03_software_design/05_code_execution_design.md` | Infrastructure | Three execution backends |
| `03_software_design/06_physical_validation_design.md` | Infrastructure | Validation architecture |
| `03_software_design/07_jupyter_bridge_design.md` | Infrastructure | Jupyter kernel bridge |
| `03_software_design/08_api_design.md` | Presentation | API routes |
| `04_implementation_guide/01_project_setup.md` | — | Setup steps |
| `04_implementation_guide/02_phase1_foundation.md` | Domain | Implement domain and config foundations |
| `04_implementation_guide/03_phase2_tools.md` | Application | Implement tool set |
| `04_implementation_guide/04_phase3_react_engine.md` | Application | Implement ReAct loop |
| `04_implementation_guide/05_phase4_validation.md` | Infrastructure | Physical validation implementation |
| `04_implementation_guide/06_phase5_jupyter.md` | Infrastructure | Jupyter integration |
| `04_implementation_guide/07_phase6_api.md` | Presentation | FastAPI routes and schemas |
| `05_reference/01_sdk_cheatsheet.md` | — | SDK quick reference |
| `05_reference/02_tool_definitions_catalog.md` | — | Tool schema catalog |
| `05_reference/03_react_prompt_templates.md` | — | Prompt templates |
| `05_reference/04_physical_constants.md` | — | Constants and ranges |
| `05_reference/05_troubleshooting.md` | — | Error guide |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | `python --version` |
| uv | latest | `pip install uv` |
| Anthropic API key | — | [console.anthropic.com](https://console.anthropic.com) |
| Git | 2.x+ | for cloning |
| `jupyter_client` | 8.x+ | for Jupyter backend (optional) |
| `pint` | 0.24+ | for unit validation |
| `nbformat` | 5.x+ | for notebook export |
| `pandas` | 2.x+ | for dataset loading |

You do **not** need a local Jupyter server — the `JupyterKernelManager` backend launches a kernel in-process. All other code execution options work without any local Jupyter installation.

---

## Learning Path

This guide is designed to be read sequentially, but each section is self-contained:

```
New to AI agents?           → Start at 01_fundamentals/
Know the basics?            → Jump to 02_anthropic_sdk_reference/
Building right now?         → Use 04_implementation_guide/
Debugging an issue?         → See 05_reference/03_troubleshooting.md
Want the full picture fast? → Read 05_reference/04_worked_example.md
```

---

## License

MIT — see `LICENSE` file for details.
