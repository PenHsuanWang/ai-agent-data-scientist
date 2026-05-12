# Tool Definitions Catalog — All 14 JSON Schemas

Complete tool definitions as a Python module. Import and use directly.

---

## Full Module: `app/services/tool_definitions.py`

```python
# app/services/tool_definitions.py
"""
All 14 tool definitions for the Data Scientist AI Agent.

Groups:
  Group A — Knowledge Tools (6): read-only access to domain docs + datasets
  Group B — Execution Tools (6): Python code execution, figures, notebooks
  Group C — Validation Tools (3): physical unit checks via pint

Usage:
  from app.services.tool_definitions import TOOL_DEFINITIONS, TOOL_REGISTRY
  # Pass TOOL_DEFINITIONS to anthropic client only when using tool_use protocol.
  # For text-based ReAct, pass them only in the system prompt description.
"""
from __future__ import annotations
from typing import Any


# ============================================================================
# GROUP A: KNOWLEDGE TOOLS
# Safe for programmatic calling — read-only, no side effects.
# ============================================================================

KNOWLEDGE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_domain_documents",
        "description": (
            "Return a JSON list of available domain knowledge documents (Markdown files) "
            "in the domain_docs/ directory. Always call this first to discover what "
            "background knowledge is available before reading any document. "
            "Returns: JSON array of filenames, e.g. [\"thermodynamics.md\", \"units.md\"]"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_domain_document",
        "description": (
            "Return the full text content of a domain knowledge Markdown document. "
            "Use this to understand physical constraints, unit definitions, expected "
            "value ranges, and domain-specific terminology before analysing data. "
            "Returns: full Markdown text of the document, or an error string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": (
                        "Basename of the .md file to read. "
                        "Example: \"power_plant_thermodynamics.md\""
                    ),
                },
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "search_domain_knowledge",
        "description": (
            "Keyword search across all domain knowledge documents. "
            "Returns ranked snippets from the most relevant documents. "
            "Use when you need to find specific information (e.g. a unit definition, "
            "a physical constraint) without reading every document in full. "
            "Returns: JSON array [{\"file\": str, \"score\": float, \"snippet\": str}]"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language or keyword query. "
                        "Examples: \"turbine isentropic efficiency range\", "
                        "\"CO2 emission units\", \"steam pressure MPa definition\""
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 3, max: 10.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_datasets",
        "description": (
            "Return a JSON list of available dataset files in the datasets/ directory "
            "with their format and file size. Call this before inspect_dataset to "
            "discover which files are available. "
            "Returns: JSON array [{\"file_name\": str, \"format\": str, \"size_bytes\": int}]"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "inspect_dataset",
        "description": (
            "Load a dataset file and return its schema, shape, numeric statistics, "
            "and first 5 sample rows as JSON. Supports CSV, Parquet, Excel (.xlsx), "
            "and HDF5 formats. Use this before execute_python_code to understand "
            "column names, data types, and value ranges. "
            "Returns: JSON with keys: file_name, format, rows, columns, column_names, "
            "dtypes, numeric_stats (min/max/mean/std per numeric column), sample_rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": (
                        "Basename of the dataset file. "
                        "Example: \"power_plant_data.csv\""
                    ),
                },
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "describe_columns",
        "description": (
            "Return detailed per-column statistics for a list of specified columns. "
            "For numeric columns: min, max, mean, median, std, Q25, Q75, skewness. "
            "For categorical columns: top 10 value counts. "
            "For datetime columns: min/max date range. "
            "Use when inspect_dataset's summary statistics are insufficient. "
            "Returns: JSON dict {column_name: {stats...}}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Basename of the dataset file.",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of column names to describe in detail. "
                        "Example: [\"efficiency_pct\", \"gross_power_MW\"]"
                    ),
                    "minItems": 1,
                },
            },
            "required": ["file_name", "columns"],
        },
    },
]


# ============================================================================
# GROUP B: EXECUTION TOOLS
# execute_python_code has a timeout guard. Others are read-only.
# ============================================================================

EXECUTION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "execute_python_code",
        "description": (
            "Execute Python code for data analysis, statistical computation, "
            "and visualisation. The following are pre-imported: "
            "pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns. "
            "Load datasets with: pd.read_csv('data/datasets/<filename>') "
            "Any call to plt.show() will capture the current figure as a PNG. "
            "Print values you want to see in the Observation. "
            "IMPORTANT: Each call runs in an isolated subprocess — variables do NOT "
            "persist between calls unless you use the Jupyter backend. "
            "Always print your results (do not rely on expression evaluation). "
            "Returns: combined stdout output and list of captured figure IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete, self-contained Python code to execute. "
                        "Must include all imports needed beyond the pre-imported stack. "
                        "Use print() for all output you want to see."
                    ),
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_execution_variables",
        "description": (
            "Return a JSON snapshot of Python variables from the last code execution. "
            "Only JSON-serialisable values (numbers, strings, lists, dicts) are included. "
            "DataFrames and complex objects are represented as their repr() string. "
            "Returns: JSON dict {variable_name: value}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_figure",
        "description": (
            "Return the base64-encoded PNG data for a specific figure generated in this session. "
            "Use list_figures first to discover available figure IDs. "
            "Returns: JSON with figure_id, format (\"png\"), encoding (\"base64\"), and data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {
                    "type": "string",
                    "description": "Figure ID as returned by execute_python_code. Example: \"fig_000\"",
                },
            },
            "required": ["figure_id"],
        },
    },
    {
        "name": "list_figures",
        "description": (
            "Return a JSON list of all figure IDs generated so far in this session. "
            "Returns: JSON with count and figure_ids list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "export_notebook",
        "description": (
            "Export the entire analysis session as a Jupyter notebook (.ipynb). "
            "Each Thought from the ReAct trace becomes a Markdown cell. "
            "Each execute_python_code call becomes a Code cell with its output. "
            "Non-code actions (list_datasets, inspect_dataset, etc.) become "
            "Markdown cells showing the action and observation. "
            "The notebook is saved to outputs/notebooks/ and can be downloaded "
            "via GET /api/v1/analysis/sessions/{session_id}/notebook. "
            "Returns: JSON with notebook_path, cell count, and download URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Descriptive title for the notebook. "
                        "Example: \"Power Plant Efficiency Analysis — January 2024\""
                    ),
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "save_figure",
        "description": (
            "Save a session figure to disk as a PNG file in the outputs/figures/ directory. "
            "Returns: JSON with saved_to path, figure_id, and file size in bytes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {
                    "type": "string",
                    "description": "Figure ID to save. Example: \"fig_000\"",
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Output filename without extension. Will be saved as <filename>.png. "
                        "Example: \"efficiency_vs_power_plot\""
                    ),
                },
            },
            "required": ["figure_id", "filename"],
        },
    },
]


# ============================================================================
# GROUP C: VALIDATION TOOLS
# All safe for programmatic calling — no side effects.
# ============================================================================

VALIDATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "validate_physical_units",
        "description": (
            "Validate that a computed physical quantity has correct units and a "
            "plausible magnitude for the given domain. Performs 3-stage validation: "
            "(1) Parse unit string with pint to catch typos and unknown units. "
            "(2) Dimensional check — verify the unit has the right physical dimensions "
            "for the quantity type (e.g. temperature must not have pressure dimensions). "
            "(3) Range check — compare magnitude against known engineering ranges "
            "from DOMAIN_RANGES (e.g. thermal efficiency must be 25-50%). "
            "ALWAYS call this before Final Answer for key numerical results. "
            "Returns: JSON with is_valid (bool), warning (str), value (in canonical unit), "
            "unit (canonical unit string), expected_range ([min, max])."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity_name": {
                    "type": "string",
                    "description": (
                        "Human-readable name of the quantity for error messages. "
                        "Example: \"mean thermal efficiency\", \"HP steam temperature\""
                    ),
                },
                "value": {
                    "type": "number",
                    "description": "Numeric magnitude of the quantity.",
                },
                "unit": {
                    "type": "string",
                    "description": (
                        "pint-compatible unit string. Examples: "
                        "\"percent\", \"degC\", \"MPa\", \"MW\", \"kg/s\", \"kJ/kWh\", "
                        "\"dimensionless\", \"g/kWh\""
                    ),
                },
                "domain_key": {
                    "type": "string",
                    "description": (
                        "Key in DOMAIN_RANGES for range and dimension check. "
                        "If omitted, only unit parsing is validated (no range check). "
                        "Available keys: thermal_efficiency, steam_temperature_hp, "
                        "steam_temperature_lp, flue_gas_temperature, condenser_temperature, "
                        "ambient_temperature, steam_pressure_hp, steam_pressure_lp, "
                        "condenser_pressure, boiler_pressure, gross_power_output, "
                        "net_power_output, auxiliary_power, heat_rate, "
                        "turbine_isentropic_efficiency, boiler_efficiency, pump_efficiency, "
                        "generator_efficiency, steam_flow_rate, feedwater_flow_rate, "
                        "fuel_flow_rate, co2_emission_intensity, nox_emission_intensity, "
                        "so2_emission_intensity, probability, correlation_coefficient"
                    ),
                },
            },
            "required": ["quantity_name", "value", "unit"],
        },
    },
    {
        "name": "convert_units",
        "description": (
            "Convert a numeric value from one physical unit to another using dimensional analysis. "
            "Handles all SI and common engineering units including temperature scales "
            "(degC, degF, K), pressure units (Pa, kPa, MPa, bar, psi), "
            "power units (W, kW, MW, GW), and compound units (kJ/kWh, g/kWh). "
            "Returns: JSON with original ({value, unit}) and converted ({value, unit}), "
            "or {\"error\": str} if conversion is impossible (incompatible dimensions)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "Numeric value to convert.",
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit string. Examples: \"degC\", \"MPa\", \"kW\"",
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit string. Examples: \"degF\", \"bar\", \"MW\"",
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "check_magnitude",
        "description": (
            "Quick check: is a numeric value physically plausible for a known domain quantity? "
            "Less verbose than validate_physical_units — just returns plausible: true/false "
            "with the expected range and a short message. "
            "Use for rapid sanity checks during analysis before the final validation step. "
            "Returns: JSON with plausible (bool), domain_key, range ([min, max]), "
            "value_in_canonical_unit, canonical_unit, message, description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "Numeric value to check.",
                },
                "unit": {
                    "type": "string",
                    "description": "Unit of the value. Examples: \"percent\", \"MW\", \"degC\"",
                },
                "domain_key": {
                    "type": "string",
                    "description": "Domain quantity key from DOMAIN_RANGES (see validate_physical_units for list).",
                },
            },
            "required": ["value", "unit", "domain_key"],
        },
    },
]


# ============================================================================
# COMBINED EXPORTS
# ============================================================================

TOOL_DEFINITIONS: list[dict[str, Any]] = (
    KNOWLEDGE_TOOLS
    + EXECUTION_TOOLS
    + VALIDATION_TOOLS
)

assert len(TOOL_DEFINITIONS) == 15, f"Expected 15 tools, got {len(TOOL_DEFINITIONS)}"

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    # Maps tool name -> registry info
    # allowed_callers: "agent" = only Claude can invoke; "api" = can also be called directly
    "list_domain_documents": {
        "module": "knowledge_tools",
        "function": "list_domain_documents",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "read_domain_document": {
        "module": "knowledge_tools",
        "function": "read_domain_document",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "search_domain_knowledge": {
        "module": "knowledge_tools",
        "function": "search_domain_knowledge",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "list_datasets": {
        "module": "knowledge_tools",
        "function": "list_datasets",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "inspect_dataset": {
        "module": "knowledge_tools",
        "function": "inspect_dataset",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "describe_columns": {
        "module": "knowledge_tools",
        "function": "describe_columns",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "execute_python_code": {
        "module": "data_tools",
        "function": "execute_python_code",
        "allowed_callers": ["agent"],  # only Claude — code execution is guarded
        "has_side_effects": True,      # mutates session.figures + spawns subprocess
        "timeout_seconds": 30,
    },
    "get_execution_variables": {
        "module": "data_tools",
        "function": "get_execution_variables",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "get_figure": {
        "module": "data_tools",
        "function": "get_figure",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "list_figures": {
        "module": "data_tools",
        "function": "list_figures",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "export_notebook": {
        "module": "data_tools",
        "function": "export_notebook",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": True,   # writes .ipynb to disk
    },
    "save_figure": {
        "module": "data_tools",
        "function": "save_figure",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": True,   # writes .png to disk
    },
    "validate_physical_units": {
        "module": "data_tools",
        "function": "validate_physical_units_tool",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,  # reads-only, appends to session.validated_units
    },
    "convert_units": {
        "module": "data_tools",
        "function": "convert_units_tool",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
    "check_magnitude": {
        "module": "data_tools",
        "function": "check_magnitude_tool",
        "allowed_callers": ["agent", "api"],
        "has_side_effects": False,
    },
}

# Convenience: tool names by group
KNOWLEDGE_TOOL_NAMES = [t["name"] for t in KNOWLEDGE_TOOLS]
EXECUTION_TOOL_NAMES = [t["name"] for t in EXECUTION_TOOLS]
VALIDATION_TOOL_NAMES = [t["name"] for t in VALIDATION_TOOLS]
ALL_TOOL_NAMES = list(TOOL_REGISTRY.keys())
SAFE_TOOL_NAMES = [
    name for name, info in TOOL_REGISTRY.items()
    if not info["has_side_effects"]
]
```

---

## Tool Summary Table

| # | Name | Group | Safe? | Side Effects |
|---|---|---|---|---|
| 1 | list_domain_documents | Knowledge | Yes | None |
| 2 | read_domain_document | Knowledge | Yes | None |
| 3 | search_domain_knowledge | Knowledge | Yes | None |
| 4 | list_datasets | Knowledge | Yes | None |
| 5 | inspect_dataset | Knowledge | Yes | None |
| 6 | describe_columns | Knowledge | Yes | None |
| 7 | execute_python_code | Execution | Guarded | Subprocess + figures |
| 8 | get_execution_variables | Execution | Yes | None |
| 9 | get_figure | Execution | Yes | None |
| 10 | list_figures | Execution | Yes | None |
| 11 | export_notebook | Execution | Yes | Writes .ipynb |
| 12 | save_figure | Execution | Yes | Writes .png |
| 13 | validate_physical_units | Validation | Yes | Appends to session |
| 14 | convert_units | Validation | Yes | None |
| 15 | check_magnitude | Validation | Yes | None |

---

## Using TOOL_DEFINITIONS with the Anthropic API

When using the **tool_use protocol** (not text-based ReAct), pass the definitions directly:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=TOOL_DEFINITIONS,     # pass all 15 definitions
    tool_choice={"type": "auto"},
    messages=messages,
)
```

When using **text-based ReAct** (recommended for this agent), do NOT pass `tools=`.
Instead, include the tool list in the system prompt (see `context_injector.py`).

---

## Adding a New Tool

1. Implement the function in `knowledge_tools.py` or `data_tools.py`
2. Add the JSON schema to the appropriate `*_TOOLS` list
3. Add an entry to `TOOL_REGISTRY`
4. Add a `case "tool_name":` branch in `DataScienceAgentService._dispatch()`
5. Update the `TOOL_LIST_TEXT` string in `context_injector.py`
6. Write a test in `tests/test_phase2_tools.py`
