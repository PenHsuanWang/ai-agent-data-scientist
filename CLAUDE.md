# Data Scientist AI Agent

## Project Overview

Domain-aware AI agent for data science built on FastAPI + Anthropic Claude SDK.
Uses a **ReAct (Reason + Act) reasoning loop** with 15 tools, `pint` physical unit validation,
Python code execution, and Jupyter notebook export.

## Architecture

Four-layer Clean Architecture:
- **Domain** (`app/domain/`): Zero-dependency dataclasses — `AgentSession`, `AnalysisSession`, `DatasetMeta`, `AnalysisResult`, `PhysicalUnit`
- **Application** (`app/services/`): ReAct loop (`data_agent.py`), 15 tools across 3 groups
- **Infrastructure** (`app/infrastructure/`): `code_runner.py` (subprocess/jupyter), `unit_registry.py` (pint), `notebook_exporter.py` (nbformat)
- **Presentation** (`app/api/v1/`): `analysis.py`, `datasets.py`

## Key Design Principles

1. **ReAct Protocol**: Claude outputs `Thought: / Action: / Action Input:` or `Final Answer:` — the service parses and dispatches
2. **Physical Validation**: All efficiency/temperature/pressure/power values validated against domain ranges via `pint`
3. **Tools Never Raise**: All 15 tools return `"Error: ..."` strings; Claude self-corrects
4. **Stateless Tools**: Session state lives in `AnalysisSession` (in-memory, swappable to Redis)
5. **Separation of Concerns**: Tools retrieve data; Claude reasons about it

## Development Commands

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8001
python scripts/create_sample_data.py
python scripts/verify_install.py
```

## Coding Standards

- Python 3.12+ generics: `list[str]`, `dict[str, Any]`, `int | None`
- `SecretStr` for ANTHROPIC_API_KEY — never log it
- Tools return `str` always — no exceptions propagate from tools
- Domain layer has zero external imports

## Adding a New Tool

1. Implement function in `app/services/knowledge_tools.py` or `data_tools.py` (returns `str`)
2. Add JSON schema to `app/services/tool_definitions.py`
3. Register in `data_agent.py` → `_build_tool_registry()`
4. Nothing else needs to change
