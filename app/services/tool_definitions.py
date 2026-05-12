"""All 15 tool JSON schemas and the unified TOOL_DEFINITIONS list.

Groups:
  Group A — Knowledge Tools (7): list/read/search domain docs + list/inspect/describe datasets
             + get_coding_standards
  Group B — Execution Tools (6): code execution, figures, notebook export
  Group C — Validation Tools (3): physical unit validation via pint

Usage in DataScienceAgentService:
  from app.services.tool_definitions import TOOL_DEFINITIONS
  # Pass in the system prompt description block (text-based ReAct)
  # OR pass directly to messages.create(tools=TOOL_DEFINITIONS) for native tool-use
"""
from __future__ import annotations

from typing import Any

# ──────────────────────────────────────────────────────────────────── #
# Group A: Knowledge Tools                                              #
# ──────────────────────────────────────────────────────────────────── #

KNOWLEDGE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_domain_documents",
        "description": (
            "Return a JSON list of available domain knowledge documents (Markdown files). "
            "Always call this first to discover what background knowledge is available. "
            "Returns: JSON array [{\"file_name\": str, \"size_bytes\": int}]"
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_domain_document",
        "description": (
            "Return the full text content of a domain knowledge document. "
            "Use this to understand physical constraints, unit definitions, and domain-specific "
            "terminology before analysing data. "
            "Returns: full Markdown text, or an error string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Basename of the .md file. E.g. 'power_plant_thermodynamics.md'",
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
            "Returns: JSON array [{\"file\": str, \"score\": int, \"snippet\": str}]"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or keyword query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return. Default 3, max 10.",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_datasets",
        "description": (
            "Return a JSON list of available dataset files with format and size. "
            "Returns: JSON array [{\"file_name\": str, \"format\": str, \"size_bytes\": int}]"
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "inspect_dataset",
        "description": (
            "Load a dataset and return its schema, shape, numeric statistics, and 5 sample rows. "
            "Supports CSV, Parquet, Excel (.xlsx), HDF5, JSON. "
            "Returns: JSON with file_name, format, rows, columns, column_names, dtypes, "
            "numeric_stats, sample_rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Basename of the dataset. E.g. 'power_plant_data.csv'",
                },
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "describe_columns",
        "description": (
            "Return detailed per-column statistics for specified columns. "
            "Numeric: min/max/mean/median/std/Q25/Q75/skew. "
            "Categorical: top 10 value counts. Datetime: date range. "
            "Returns: JSON dict {column_name: {stats}}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string", "description": "Dataset basename."},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of column names to describe.",
                    "minItems": 1,
                },
            },
            "required": ["file_name", "columns"],
        },
    },
    {
        "name": "get_coding_standards",
        "description": (
            "Return the project's coding style guide, analysis strategy playbook, "
            "and visualization format specification. "
            "ALWAYS call this at the start of any analysis task to learn: "
            "(1) Python coding conventions (PEP8, pandas patterns, numpy best practices), "
            "(2) The 7-step analysis workflow (EDA → clean → compute → validate → visualize → conclude), "
            "(3) Pre-configured plot helpers (COLORS, PALETTE, label_bars, add_reference_line, "
            "engineering_plot, format_axis_units) and mandatory plot elements (axis labels with units, "
            "title, legend, tight_layout, plt.show()), "
            "(4) Plot type recipes with copy-paste code for time-series, histograms, "
            "correlation heatmaps, bar charts, scatter plots, and multi-panel dashboards. "
            "Returns: full Markdown specification document."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# ──────────────────────────────────────────────────────────────────── #
# Group B: Execution Tools                                              #
# ──────────────────────────────────────────────────────────────────── #

EXECUTION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "execute_python_code",
        "description": (
            "Execute Python code for data analysis, statistics, and visualisation. "
            "Pre-imported: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns. "
            "Load datasets with: pd.read_csv('data/datasets/<filename>') "
            "Any plt.show() call captures the figure as a PNG. "
            "ALWAYS use print() — do not rely on expression evaluation. "
            "Returns: JSON with stdout, figures list, and success flag."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete, self-contained Python code. "
                        "Include all imports beyond the pre-imported stack. "
                        "Use print() for all output."
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
            "Returns: JSON dict {variable_name: type_name}"
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_figure",
        "description": (
            "Return base64-encoded PNG data for a figure generated in this session. "
            "Returns: JSON with figure_id, format ('png'), encoding ('base64'), and data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {
                    "type": "string",
                    "description": "Figure ID from execute_python_code output. E.g. 'fig_000'",
                },
            },
            "required": ["figure_id"],
        },
    },
    {
        "name": "list_figures",
        "description": (
            "Return JSON list of all figure IDs generated so far in this session. "
            "Returns: JSON with count and figure_ids list."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "export_notebook",
        "description": (
            "Export the entire analysis session as a Jupyter notebook (.ipynb). "
            "Each Thought becomes a Markdown cell. "
            "Each execute_python_code call becomes a Code cell with its output. "
            "Returns: JSON with notebook_path, cell_count, and download_url."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Descriptive title for the notebook.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "save_figure",
        "description": (
            "Save a session figure to disk as PNG in outputs/figures/. "
            "Returns: JSON with saved_to path, figure_id, and size_bytes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {"type": "string", "description": "Figure ID to save."},
                "filename": {
                    "type": "string",
                    "description": "Output filename without extension. E.g. 'efficiency_plot'",
                },
            },
            "required": ["figure_id", "filename"],
        },
    },
]

# ──────────────────────────────────────────────────────────────────── #
# Group C: Physical Validation Tools                                    #
# ──────────────────────────────────────────────────────────────────── #

VALIDATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "validate_physical_units",
        "description": (
            "Validate that a physical quantity has sensible units and magnitude. "
            "Checks unit parseability, dimensional correctness, and domain-specific ranges. "
            "Always call this after computing efficiency, temperature, pressure, or power values. "
            "Returns: JSON PhysicalUnit with is_valid flag and human-readable message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity": {
                    "type": "string",
                    "description": "Name of the physical quantity. E.g. 'thermal_efficiency'",
                },
                "value": {
                    "type": "number",
                    "description": "Numeric value to validate.",
                },
                "unit": {
                    "type": "string",
                    "description": "Unit string parseable by pint. E.g. '%', 'MW', 'degC', 'MPa'",
                },
            },
            "required": ["quantity", "value", "unit"],
        },
    },
    {
        "name": "convert_units",
        "description": (
            "Convert a value from one unit to another using pint. "
            "Returns: JSON with original_value, original_unit, converted_value, converted_unit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Value to convert."},
                "from_unit": {"type": "string", "description": "Source unit. E.g. 'degF'"},
                "to_unit": {"type": "string", "description": "Target unit. E.g. 'degC'"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "check_magnitude",
        "description": (
            "Check whether a value is physically plausible for the given quantity "
            "based on domain knowledge ranges. "
            "Returns: JSON with is_plausible flag, expected_range, and diagnostic message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity": {"type": "string", "description": "Physical quantity name."},
                "value": {"type": "number", "description": "Value to check."},
                "unit": {"type": "string", "description": "Unit of the value."},
            },
            "required": ["quantity", "value", "unit"],
        },
    },
]

# ──────────────────────────────────────────────────────────────────── #
# Combined list (all 15 tools)                                          #
# ──────────────────────────────────────────────────────────────────── #

TOOL_DEFINITIONS: list[dict[str, Any]] = KNOWLEDGE_TOOLS + EXECUTION_TOOLS + VALIDATION_TOOLS
