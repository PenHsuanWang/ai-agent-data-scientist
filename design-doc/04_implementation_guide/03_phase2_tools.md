# Phase 2 — Building the Tool Suite

## Overview

Phase 2 builds the 14 tools that the ReAct agent can call. Tools are organised into three groups:

| Group | File | Tools |
|---|---|---|
| Knowledge | `services/knowledge_tools.py` | list_domain_documents, read_domain_document, search_domain_knowledge, list_datasets, inspect_dataset, describe_columns |
| Physical | `infrastructure/unit_registry.py` + `services/data_tools.py` | validate_physical_units, convert_units, check_magnitude |
| Execution | `infrastructure/code_runner.py` + `services/data_tools.py` | execute_python_code, get_execution_variables, get_figure, list_figures, export_notebook, save_figure |

---

## 1. Create `app/services/knowledge_tools.py`

```python
# app/services/knowledge_tools.py
"""
Knowledge retrieval tools for the ReAct agent.

These tools let Claude discover and read domain documents and datasets
without executing arbitrary code. All I/O is read-only.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.domain.analysis_models import DatasetMeta
from app.domain.exceptions import DatasetNotFoundError


# ── Helpers ────────────────────────────────────────────────────────────────

SUPPORTED_DATASET_EXTENSIONS = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
}


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    base = Path(data_dir) if data_dir else settings.data_dir
    return base


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path, nrows=5_000)  # cap for inspection
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path, nrows=5_000)
    elif ext in (".h5", ".hdf5"):
        return pd.read_hdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ── Tool 1: list_domain_documents ─────────────────────────────────────────

def list_domain_documents(data_dir: str | Path | None = None) -> str:
    """
    Return a JSON list of markdown files in the domain_docs/ directory.

    Returns:
        JSON string: list of file names (not full paths)

    Example return value:
        '["power_plant_thermodynamics.md", "unit_definitions.md"]'
    """
    docs_dir = _resolve_data_dir(data_dir) / "domain_docs"
    if not docs_dir.exists():
        return json.dumps([])
    files = sorted(p.name for p in docs_dir.glob("*.md"))
    return json.dumps(files)


# ── Tool 2: read_domain_document ──────────────────────────────────────────

def read_domain_document(
    file_name: str,
    data_dir: str | Path | None = None,
) -> str:
    """
    Return the full text content of a domain document.

    Args:
        file_name: Basename of the .md file (e.g. "power_plant_thermodynamics.md")

    Returns:
        str: Full markdown text content, or an error message string.
    """
    docs_dir = _resolve_data_dir(data_dir) / "domain_docs"
    file_path = docs_dir / file_name

    if not file_path.exists():
        available = sorted(p.name for p in docs_dir.glob("*.md"))
        return (
            f"ERROR: File '{file_name}' not found in domain_docs/.\n"
            f"Available files: {', '.join(available) or 'none'}"
        )

    # Security: restrict to domain_docs dir (no path traversal)
    try:
        file_path.resolve().relative_to(docs_dir.resolve())
    except ValueError:
        return "ERROR: Path traversal detected — access denied."

    return file_path.read_text(encoding="utf-8")


# ── Tool 3: search_domain_knowledge ───────────────────────────────────────

def _tf_idf_score(query_terms: list[str], doc_text: str) -> float:
    """Simplified TF score (no corpus-wide IDF — small doc set)."""
    text_lower = doc_text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    if not words:
        return 0.0
    total = len(words)
    score = 0.0
    for term in query_terms:
        count = words.count(term.lower())
        if count > 0:
            score += (count / total) * (1 + math.log(count))
    return score


def search_domain_knowledge(
    query: str,
    data_dir: str | Path | None = None,
    top_k: int = 3,
    max_snippet_chars: int = 800,
) -> str:
    """
    Keyword search across all domain documents, returning ranked snippets.

    Args:
        query: Natural language or keyword query.
        top_k: Maximum number of document snippets to return.
        max_snippet_chars: Max characters per snippet.

    Returns:
        JSON string with ranked results:
        [{"file": "...", "score": 0.42, "snippet": "..."}]
    """
    docs_dir = _resolve_data_dir(data_dir) / "domain_docs"
    if not docs_dir.exists():
        return json.dumps([])

    query_terms = re.findall(r"\b\w{3,}\b", query.lower())
    if not query_terms:
        return json.dumps({"error": "Query too short — provide at least one 3-character word."})

    results = []
    for doc_path in sorted(docs_dir.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        score = _tf_idf_score(query_terms, text)
        if score == 0.0:
            continue

        # Find best matching paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        best_para = max(paragraphs, key=lambda p: _tf_idf_score(query_terms, p), default="")
        snippet = best_para[:max_snippet_chars]
        if len(best_para) > max_snippet_chars:
            snippet += "..."

        results.append({
            "file": doc_path.name,
            "score": round(score, 4),
            "snippet": snippet,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return json.dumps(results[:top_k], indent=2)


# ── Tool 4: list_datasets ─────────────────────────────────────────────────

def list_datasets(data_dir: str | Path | None = None) -> str:
    """
    Return a JSON list of available dataset files with basic metadata.

    Returns:
        JSON string: list of {file_name, format, size_bytes}
    """
    datasets_dir = _resolve_data_dir(data_dir) / "datasets"
    if not datasets_dir.exists():
        return json.dumps([])

    files = []
    for path in sorted(datasets_dir.iterdir()):
        ext = path.suffix.lower()
        if ext not in SUPPORTED_DATASET_EXTENSIONS:
            continue
        files.append({
            "file_name": path.name,
            "format": SUPPORTED_DATASET_EXTENSIONS[ext],
            "size_bytes": path.stat().st_size,
        })
    return json.dumps(files, indent=2)


# ── Tool 5: inspect_dataset ───────────────────────────────────────────────

def inspect_dataset(
    file_name: str,
    data_dir: str | Path | None = None,
) -> str:
    """
    Load a dataset and return its schema, shape, and first 5 rows as JSON.

    Args:
        file_name: Basename of the dataset file.

    Returns:
        JSON string with schema and sample data, or error string.
    """
    datasets_dir = _resolve_data_dir(data_dir) / "datasets"
    file_path = datasets_dir / file_name

    if not file_path.exists():
        available = [p.name for p in datasets_dir.glob("*")
                     if p.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS]
        raise DatasetNotFoundError(file_name, str(datasets_dir), available)

    try:
        df = _load_dataframe(file_path)
    except Exception as e:
        return json.dumps({"error": f"Failed to load '{file_name}': {e}"})

    # Build stats for numeric columns
    numeric_stats: dict[str, dict] = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        numeric_stats[col] = {
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "null_count": int(df[col].isna().sum()),
        }

    result = {
        "file_name": file_name,
        "format": SUPPORTED_DATASET_EXTENSIONS.get(file_path.suffix.lower(), "unknown"),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_stats": numeric_stats,
        "sample_rows": df.head(5).to_dict(orient="records"),
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
    }
    return json.dumps(result, indent=2, default=str)


# ── Tool 6: describe_columns ──────────────────────────────────────────────

def describe_columns(
    file_name: str,
    columns: list[str],
    data_dir: str | Path | None = None,
) -> str:
    """
    Return detailed per-column statistics for specified columns.

    Args:
        file_name: Basename of the dataset file.
        columns: List of column names to describe.

    Returns:
        JSON string with per-column statistics.
    """
    datasets_dir = _resolve_data_dir(data_dir) / "datasets"
    file_path = datasets_dir / file_name

    if not file_path.exists():
        return json.dumps({"error": f"Dataset '{file_name}' not found."})

    try:
        df = _load_dataframe(file_path)
    except Exception as e:
        return json.dumps({"error": str(e)})

    missing = [c for c in columns if c not in df.columns]
    if missing:
        return json.dumps({
            "error": f"Columns not found: {missing}",
            "available_columns": list(df.columns),
        })

    result: dict[str, Any] = {}
    for col in columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "unique_count": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            s = series.dropna()
            col_info.update({
                "min": round(float(s.min()), 6),
                "max": round(float(s.max()), 6),
                "mean": round(float(s.mean()), 6),
                "median": round(float(s.median()), 6),
                "std": round(float(s.std()), 6),
                "q25": round(float(s.quantile(0.25)), 6),
                "q75": round(float(s.quantile(0.75)), 6),
                "skew": round(float(s.skew()), 4),
            })
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            top_values = series.value_counts().head(10).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["min_date"] = str(series.min())
            col_info["max_date"] = str(series.max())
        result[col] = col_info

    return json.dumps(result, indent=2, default=str)
```

---

## 2. Create `app/infrastructure/unit_registry.py`

```python
# app/infrastructure/unit_registry.py
"""
Physical unit validation using pint.

Provides:
- ureg: singleton UnitRegistry (one instance per process)
- DOMAIN_RANGES: expected ranges for 25+ physical quantities
- validate_physical_units(): 3-stage pipeline
- convert_units(): safe unit conversion
- check_magnitude(): range check for a quantity
"""
from __future__ import annotations

import threading
from typing import Any

import pint

from app.domain.analysis_models import PhysicalUnit
from app.domain.exceptions import UnitValidationError


# ── Singleton UnitRegistry ─────────────────────────────────────────────────
# pint registries are NOT thread-safe to construct concurrently.
# Create once at module load time.
_registry_lock = threading.Lock()
_ureg: pint.UnitRegistry | None = None


def get_ureg() -> pint.UnitRegistry:
    global _ureg
    if _ureg is None:
        with _registry_lock:
            if _ureg is None:
                _ureg = pint.UnitRegistry()
                # Register common aliases not in default registry
                _ureg.define("percent = 0.01 = pct")
                _ureg.define("ppm = 1e-6")
    return _ureg


# Convenience alias
ureg = get_ureg()


# ── DOMAIN_RANGES ──────────────────────────────────────────────────────────
# Format: quantity_key -> {
#   "unit": pint-compatible string,
#   "min": float,
#   "max": float,
#   "description": str,
# }

DOMAIN_RANGES: dict[str, dict[str, Any]] = {
    # ── Temperature ───────────────────────────────────────────────────────
    "steam_temperature_hp": {
        "unit": "degC",
        "min": 400.0, "max": 650.0,
        "description": "HP steam temperature in coal/gas power plants",
    },
    "steam_temperature_lp": {
        "unit": "degC",
        "min": 150.0, "max": 350.0,
        "description": "LP steam temperature after IP turbine",
    },
    "flue_gas_temperature": {
        "unit": "degC",
        "min": 80.0, "max": 200.0,
        "description": "Flue gas exit temperature after air heater",
    },
    "condenser_temperature": {
        "unit": "degC",
        "min": 20.0, "max": 55.0,
        "description": "Condenser saturation temperature",
    },
    "ambient_temperature": {
        "unit": "degC",
        "min": -30.0, "max": 55.0,
        "description": "Outdoor dry-bulb temperature",
    },
    "cooling_water_temperature": {
        "unit": "degC",
        "min": 5.0, "max": 40.0,
        "description": "Cooling water inlet temperature",
    },

    # ── Pressure ──────────────────────────────────────────────────────────
    "steam_pressure_hp": {
        "unit": "MPa",
        "min": 10.0, "max": 30.0,
        "description": "HP turbine inlet pressure (supercritical: >22 MPa)",
    },
    "steam_pressure_lp": {
        "unit": "kPa",
        "min": 3.0, "max": 20.0,
        "description": "LP turbine exhaust pressure (condenser inlet)",
    },
    "condenser_pressure": {
        "unit": "kPa",
        "min": 3.0, "max": 15.0,
        "description": "Condenser vacuum pressure",
    },
    "boiler_pressure": {
        "unit": "MPa",
        "min": 8.0, "max": 35.0,
        "description": "Boiler drum or once-through pressure",
    },

    # ── Power and Energy ──────────────────────────────────────────────────
    "gross_power_output": {
        "unit": "MW",
        "min": 50.0, "max": 1500.0,
        "description": "Generator gross electrical output",
    },
    "net_power_output": {
        "unit": "MW",
        "min": 40.0, "max": 1450.0,
        "description": "Net power after auxiliary consumption",
    },
    "auxiliary_power": {
        "unit": "MW",
        "min": 5.0, "max": 80.0,
        "description": "Internal auxiliary power consumption",
    },
    "heat_rate": {
        "unit": "kJ/kWh",
        "min": 7000.0, "max": 12000.0,
        "description": "Plant heat rate (lower is better: ideal ~3600 kJ/kWh)",
    },

    # ── Efficiency ────────────────────────────────────────────────────────
    "thermal_efficiency": {
        "unit": "percent",
        "min": 25.0, "max": 50.0,
        "description": "Gross thermal efficiency; modern plants: 40–48%",
    },
    "turbine_isentropic_efficiency": {
        "unit": "percent",
        "min": 75.0, "max": 95.0,
        "description": "Turbine stage isentropic efficiency",
    },
    "boiler_efficiency": {
        "unit": "percent",
        "min": 80.0, "max": 95.0,
        "description": "Boiler combustion efficiency",
    },
    "pump_efficiency": {
        "unit": "percent",
        "min": 60.0, "max": 90.0,
        "description": "Feed pump or condensate pump efficiency",
    },
    "generator_efficiency": {
        "unit": "percent",
        "min": 97.0, "max": 99.5,
        "description": "Generator mechanical-to-electrical efficiency",
    },

    # ── Mass Flow ─────────────────────────────────────────────────────────
    "steam_flow_rate": {
        "unit": "kg/s",
        "min": 50.0, "max": 800.0,
        "description": "Main steam mass flow rate",
    },
    "feedwater_flow_rate": {
        "unit": "kg/s",
        "min": 50.0, "max": 800.0,
        "description": "Feedwater pump flow rate",
    },
    "fuel_flow_rate": {
        "unit": "kg/s",
        "min": 5.0, "max": 200.0,
        "description": "Coal/gas fuel mass flow rate",
    },

    # ── Emissions ─────────────────────────────────────────────────────────
    "co2_emission_intensity": {
        "unit": "g/kWh",
        "min": 350.0, "max": 1100.0,
        "description": "CO2 specific emission (gas: 400–500, coal: 800–1000 g/kWh)",
    },
    "nox_emission_intensity": {
        "unit": "mg/Nm3",
        "min": 50.0, "max": 500.0,
        "description": "NOx emission concentration at stack",
    },
    "so2_emission_intensity": {
        "unit": "mg/Nm3",
        "min": 10.0, "max": 200.0,
        "description": "SO2 after FGD system",
    },

    # ── Probability / Statistics ──────────────────────────────────────────
    "probability": {
        "unit": "dimensionless",
        "min": 0.0, "max": 1.0,
        "description": "Any probability or normalized score",
    },
    "correlation_coefficient": {
        "unit": "dimensionless",
        "min": -1.0, "max": 1.0,
        "description": "Pearson/Spearman correlation",
    },
}


# ── validate_physical_units() ──────────────────────────────────────────────

def validate_physical_units(
    quantity_name: str,
    value: float,
    unit: str,
    domain_key: str | None = None,
    raise_on_error: bool = False,
) -> PhysicalUnit:
    """
    3-stage physical validation pipeline.

    Stage 1: Parse the unit string with pint (catches typos/unknown units).
    Stage 2: Dimensionality check if domain_key is provided.
    Stage 3: Range check against DOMAIN_RANGES[domain_key].

    Args:
        quantity_name: Human-readable name for error messages.
        value: Numeric magnitude to validate.
        unit: pint-compatible unit string (e.g. "degC", "MPa", "percent").
        domain_key: Key into DOMAIN_RANGES for range + dimension check.
        raise_on_error: If True, raise UnitValidationError on failure.

    Returns:
        PhysicalUnit with is_valid, warning populated.
    """
    reg = get_ureg()
    warning = ""
    is_valid = True

    # Stage 1: Parse unit
    try:
        quantity = value * reg.parse_expression(unit)
    except pint.errors.UndefinedUnitError as e:
        if raise_on_error:
            raise UnitValidationError(quantity_name, value, unit, f"Unknown unit: {e}")
        return PhysicalUnit(
            name=quantity_name, value=value, unit=unit,
            is_valid=False, warning=f"Unknown unit: {e}",
        )
    except Exception as e:
        if raise_on_error:
            raise UnitValidationError(quantity_name, value, unit, str(e))
        return PhysicalUnit(
            name=quantity_name, value=value, unit=unit,
            is_valid=False, warning=str(e),
        )

    if domain_key is None or domain_key not in DOMAIN_RANGES:
        return PhysicalUnit(
            name=quantity_name, value=value, unit=unit, is_valid=True
        )

    spec = DOMAIN_RANGES[domain_key]
    expected_unit = spec["unit"]
    lo, hi = spec["min"], spec["max"]

    # Stage 2: Dimensionality check (convert to expected unit)
    try:
        converted = quantity.to(reg.parse_expression(expected_unit))
        check_value = converted.magnitude
    except pint.errors.DimensionalityError as e:
        if raise_on_error:
            raise UnitValidationError(
                quantity_name, value, unit,
                f"Wrong dimensions — expected {expected_unit}: {e}",
                dimensionality_error=True,
            )
        return PhysicalUnit(
            name=quantity_name, value=value, unit=unit,
            is_valid=False,
            warning=f"Dimensionality mismatch: expected {expected_unit}",
        )

    # Stage 3: Range check
    if not (lo <= check_value <= hi):
        msg = (
            f"Value {check_value:.4g} {expected_unit} is outside expected range "
            f"[{lo}, {hi}] for '{domain_key}'"
        )
        if raise_on_error:
            raise UnitValidationError(
                quantity_name, value, unit, msg, expected_range=(lo, hi)
            )
        is_valid = False
        warning = msg

    return PhysicalUnit(
        name=quantity_name,
        value=check_value,
        unit=expected_unit,
        is_valid=is_valid,
        warning=warning,
        expected_range=(lo, hi),
    )


# ── convert_units() ────────────────────────────────────────────────────────

def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
) -> dict[str, Any]:
    """
    Convert a value from one unit to another using pint.

    Returns:
        {"original": {"value": v, "unit": u}, "converted": {"value": v2, "unit": u2}}
        or {"error": "..."} on failure.
    """
    reg = get_ureg()
    try:
        q = value * reg.parse_expression(from_unit)
        converted = q.to(reg.parse_expression(to_unit))
        return {
            "original": {"value": value, "unit": from_unit},
            "converted": {
                "value": round(float(converted.magnitude), 6),
                "unit": to_unit,
            },
        }
    except pint.errors.DimensionalityError:
        return {"error": f"Cannot convert {from_unit} to {to_unit}: incompatible dimensions"}
    except pint.errors.UndefinedUnitError as e:
        return {"error": f"Unknown unit: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ── check_magnitude() ──────────────────────────────────────────────────────

def check_magnitude(
    value: float,
    unit: str,
    domain_key: str,
) -> dict[str, Any]:
    """
    Quick range check: is this value physically plausible for domain_key?

    Returns:
        {"plausible": bool, "domain_key": str, "range": [min, max],
         "value_in_canonical_unit": float, "canonical_unit": str, "message": str}
    """
    result = validate_physical_units(
        quantity_name=domain_key.replace("_", " "),
        value=value,
        unit=unit,
        domain_key=domain_key,
        raise_on_error=False,
    )
    spec = DOMAIN_RANGES.get(domain_key, {})
    return {
        "plausible": result.is_valid,
        "domain_key": domain_key,
        "range": [spec.get("min"), spec.get("max")],
        "value_in_canonical_unit": result.value,
        "canonical_unit": result.unit,
        "message": result.warning or "Value is within expected range.",
        "description": spec.get("description", ""),
    }
```

---

## 3. Create `app/infrastructure/code_runner.py`

```python
# app/infrastructure/code_runner.py
"""
SubprocessCodeRunner: safe, isolated Python code execution.

Each execute() call spawns a fresh subprocess with a complete preamble
(pandas, numpy, matplotlib Agg, figure capture). The subprocess prints
a JSON envelope containing stdout, figures (base64 PNGs), and variables.

State persistence: variables are NOT shared between calls (subprocess
is isolated). For stateful sessions, use JupyterKernelManager instead.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from app.domain.analysis_models import AnalysisResult
from app.domain.exceptions import CodeExecutionError


# ── PREAMBLE injected before every user script ────────────────────────────

PREAMBLE = r'''
import sys
import io
import json
import base64
import traceback

# Data stack
import pandas as pd
import numpy as np

# Matplotlib — non-interactive Agg backend (no X display needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Figure capture infrastructure
_figures: dict[str, str] = {}
_fig_counter: list[int] = [0]
_captured_vars: dict = {}

# Monkey-patch plt.show() to auto-capture figures
_original_show = plt.show
def _patched_show(*args, **kwargs):
    _capture_current_figure()
plt.show = _patched_show

def _capture_current_figure() -> str | None:
    """Capture the current matplotlib figure to base64 PNG."""
    fig = plt.gcf()
    if not fig.get_axes():
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    fid = f"fig_{_fig_counter[0]:03d}"
    _fig_counter[0] += 1
    _figures[fid] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return fid

# Stdout capture
_stdout_lines: list[str] = []
_original_stdout = sys.stdout

class _CaptureStdout(io.StringIO):
    def write(self, s):
        _stdout_lines.append(s)
        return super().write(s)

sys.stdout = _CaptureStdout()
'''

POSTAMBLE = r'''
# Capture any figure that wasn't explicitly shown
_capture_current_figure()

# Restore stdout
sys.stdout = _original_stdout

# Collect simple serialisable variables from global scope
_skip_keys = set(dir(__builtins__)) | {
    '__builtins__', '__doc__', '__loader__', '__name__',
    '__package__', '__spec__', '_figures', '_fig_counter',
    '_captured_vars', '_stdout_lines', '_original_stdout',
    '_CaptureStdout', '_capture_current_figure', '_patched_show',
    '_original_show', 'pd', 'np', 'plt', 'sns',
    'sys', 'io', 'json', 'base64', 'traceback', 'matplotlib',
}
for _k, _v in list(globals().items()):
    if _k.startswith('_') or _k in _skip_keys:
        continue
    try:
        json.dumps(_v)  # only include JSON-serialisable values
        _captured_vars[_k] = _v
    except (TypeError, ValueError):
        _captured_vars[_k] = repr(_v)

_envelope = {
    "stdout": "".join(_stdout_lines),
    "figures": _figures,
    "variables": _captured_vars,
    "error": "",
}
print(json.dumps(_envelope), file=_original_stdout)
'''

PREAMBLE_ERROR_WRAPPER_START = r'''
try:
'''
PREAMBLE_ERROR_WRAPPER_END = r'''
except Exception as _exc:
    import traceback as _tb
    _envelope = {
        "stdout": "".join(_stdout_lines),
        "figures": {},
        "variables": {},
        "error": f"{type(_exc).__name__}: {_exc}",
        "traceback": _tb.format_exc(),
    }
    sys.stdout = _original_stdout
    print(json.dumps(_envelope), file=_original_stdout)
    sys.exit(0)
'''


def _indent_user_code(code: str, indent: str = "    ") -> str:
    return "\n".join(indent + line for line in code.splitlines())


# ── SubprocessCodeRunner ──────────────────────────────────────────────────

class SubprocessCodeRunner:
    """
    Executes Python code in an isolated subprocess.

    Each call is completely isolated: no shared state between executions.
    This is the safest backend but cannot persist DataFrame objects across
    tool calls. For stateful sessions, prefer JupyterKernelManager.

    Usage:
        runner = SubprocessCodeRunner(timeout=30)
        result = runner.execute("print(2 + 2)")
        print(result.stdout)   # "4\n"
    """

    def __init__(self, timeout: int = 30, data_dir: Path | None = None):
        self.timeout = timeout
        self.data_dir = data_dir or Path("data")
        self._last_variables: dict[str, Any] = {}

    def execute(self, code: str) -> AnalysisResult:
        """
        Run code in a subprocess and return an AnalysisResult.

        The code is wrapped in:
        1. PREAMBLE (imports, figure capture, stdout capture)
        2. User code (indented inside try/except)
        3. POSTAMBLE (print JSON envelope to stdout)

        Returns:
            AnalysisResult with stdout, figures, variables, error populated.
        """
        indented_code = _indent_user_code(code)
        full_script = (
            PREAMBLE
            + PREAMBLE_ERROR_WRAPPER_START
            + indented_code
            + PREAMBLE_ERROR_WRAPPER_END
            + POSTAMBLE
        )

        start_ms = int(time.monotonic() * 1000)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", full_script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.data_dir.parent),  # run from project root
            )
        except subprocess.TimeoutExpired:
            elapsed = int(time.monotonic() * 1000) - start_ms
            return AnalysisResult(
                error=f"Execution timed out after {self.timeout}s",
                timed_out=True,
                execution_time_ms=elapsed,
            )

        elapsed = int(time.monotonic() * 1000) - start_ms

        stderr = proc.stderr.strip()

        # Parse the JSON envelope from stdout
        stdout_text = proc.stdout.strip()
        if not stdout_text:
            return AnalysisResult(
                stderr=stderr,
                error="No output from subprocess — possible crash.",
                execution_time_ms=elapsed,
            )

        # The last line should be the JSON envelope
        lines = stdout_text.splitlines()
        json_line = lines[-1]
        try:
            envelope = json.loads(json_line)
        except json.JSONDecodeError:
            # Subprocess may have printed non-JSON (e.g. crash traceback)
            return AnalysisResult(
                stdout=stdout_text,
                stderr=stderr,
                error="Failed to parse subprocess output as JSON.",
                execution_time_ms=elapsed,
            )

        self._last_variables = envelope.get("variables", {})

        return AnalysisResult(
            stdout=envelope.get("stdout", ""),
            stderr=stderr + "\n" + envelope.get("traceback", ""),
            figures=envelope.get("figures", {}),
            variables=envelope.get("variables", {}),
            error=envelope.get("error", ""),
            timed_out=False,
            execution_time_ms=elapsed,
        )

    def get_state(self) -> dict[str, Any]:
        """Return variables from the last successful execution."""
        return dict(self._last_variables)
```

---

## 4. Create `app/services/data_tools.py`

```python
# app/services/data_tools.py
"""
Data science tools for the ReAct agent.

These tools wrap the infrastructure layer (code runners, unit registry)
and provide the interface that data_agent.py calls via _dispatch().
All functions return strings (JSON or plain text) suitable for injection
as Observation: blocks in the ReAct loop.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import nbformat
import nbformat.v4 as nbv4

from app.core.config import settings
from app.domain.analysis_models import AnalysisResult, AnalysisSession
from app.infrastructure.code_runner import SubprocessCodeRunner
from app.infrastructure.unit_registry import (
    check_magnitude,
    convert_units,
    validate_physical_units,
)

# Runner cache: one runner per session (subprocess = no state, but we cache
# for consistency; swap for JupyterKernelManager when that backend is active)
_runners: dict[str, SubprocessCodeRunner] = {}


def _get_runner(session: AnalysisSession) -> SubprocessCodeRunner:
    if session.session_id not in _runners:
        _runners[session.session_id] = SubprocessCodeRunner(
            timeout=settings.code_execution_timeout,
            data_dir=settings.data_dir,
        )
    return _runners[session.session_id]


# ── Tool: execute_python_code ─────────────────────────────────────────────

def execute_python_code(code: str, session: AnalysisSession) -> str:
    """
    Execute Python code via the configured backend and return observation string.

    The session's figures dict is updated with any generated plots.
    """
    backend = settings.code_execution_backend

    if backend == "subprocess":
        runner = _get_runner(session)
        result = runner.execute(code)
    elif backend == "jupyter":
        from app.infrastructure.jupyter_bridge import jupyter_manager
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            jupyter_manager.execute_cell(session.session_id, code)
        )
    else:
        result = AnalysisResult(error=f"Unknown backend: {backend}")

    # Merge figures into session
    if result.figures:
        session.merge_figures(result.figures)

    return result.to_observation_string()


# ── Tool: get_execution_variables ─────────────────────────────────────────

def get_execution_variables(session: AnalysisSession) -> str:
    """
    Return a JSON summary of variables from the last code execution.
    """
    runner = _get_runner(session)
    state = runner.get_state()
    if not state:
        return json.dumps({"message": "No variables captured yet — run execute_python_code first."})
    return json.dumps(state, indent=2, default=str)


# ── Tool: get_figure ──────────────────────────────────────────────────────

def get_figure(figure_id: str, session: AnalysisSession) -> str:
    """
    Return the base64-encoded PNG for a specific figure ID.
    """
    if figure_id not in session.figures:
        return json.dumps({
            "error": f"Figure '{figure_id}' not found.",
            "available": session.figure_ids,
        })
    return json.dumps({
        "figure_id": figure_id,
        "format": "png",
        "encoding": "base64",
        "data_length": len(session.figures[figure_id]),
        "data": session.figures[figure_id],
    })


# ── Tool: list_figures ────────────────────────────────────────────────────

def list_figures(session: AnalysisSession) -> str:
    """
    Return a JSON list of all figure IDs generated in this session.
    """
    return json.dumps({
        "count": len(session.figures),
        "figure_ids": session.figure_ids,
    })


# ── Tool: validate_physical_units_tool ───────────────────────────────────

def validate_physical_units_tool(
    quantity_name: str,
    value: float,
    unit: str,
    domain_key: str | None = None,
    session: AnalysisSession | None = None,
) -> str:
    """
    Validate that a physical quantity has correct units and plausible magnitude.
    Returns JSON result string.
    """
    result = validate_physical_units(
        quantity_name=quantity_name,
        value=value,
        unit=unit,
        domain_key=domain_key,
        raise_on_error=False,
    )
    if session is not None:
        session.add_validated_unit(result)

    return json.dumps(result.to_dict(), indent=2)


# ── Tool: convert_units_tool ──────────────────────────────────────────────

def convert_units_tool(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units. Returns JSON result string."""
    result = convert_units(value, from_unit, to_unit)
    return json.dumps(result, indent=2)


# ── Tool: check_magnitude_tool ────────────────────────────────────────────

def check_magnitude_tool(value: float, unit: str, domain_key: str) -> str:
    """Check if a value is within the expected range for a domain quantity."""
    result = check_magnitude(value, unit, domain_key)
    return json.dumps(result, indent=2)


# ── Tool: export_notebook ─────────────────────────────────────────────────

def export_notebook(title: str, session: AnalysisSession) -> str:
    """
    Export the session's ReAct trace as a Jupyter notebook (.ipynb).

    Each code execution step becomes a Code cell.
    Each Thought becomes a Markdown cell.
    Figures are embedded as markdown image references.

    Returns JSON with the notebook path.
    """
    nb = nbv4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    # Title cell
    nb.cells.append(nbv4.new_markdown_cell(f"# {title}\n\nGenerated by Data Scientist Agent"))

    # Standard imports cell
    nb.cells.append(nbv4.new_code_cell(
        "import pandas as pd\nimport numpy as np\n"
        "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
        "%matplotlib inline"
    ))

    # Extract code cells from react_trace
    for i, step in enumerate(session.react_trace):
        # Thought as markdown
        if step.get("thought"):
            nb.cells.append(nbv4.new_markdown_cell(
                f"### Step {i+1} — Thought\n\n{step['thought']}"
            ))

        action = step.get("action", "")
        action_input = step.get("action_input", {})

        if action == "execute_python_code" and "code" in action_input:
            code = action_input["code"]
            output = step.get("observation", "")
            cell = nbv4.new_code_cell(code)
            if output and not output.startswith("Figures"):
                cell.outputs = [
                    nbv4.new_output(
                        output_type="stream",
                        name="stdout",
                        text=output[:2000],
                    )
                ]
            nb.cells.append(cell)
        elif action:
            # Non-code action as markdown
            nb.cells.append(nbv4.new_markdown_cell(
                f"**Action:** `{action}`\n\n"
                f"**Input:** `{json.dumps(action_input)}`\n\n"
                f"**Observation:** {step.get('observation', '')[:500]}"
            ))

    # Final answer
    if session.final_answer:
        nb.cells.append(nbv4.new_markdown_cell(
            f"## Final Answer\n\n{session.final_answer}"
        ))

    # Save notebook
    settings.notebooks_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
    safe_title = safe_title.replace(" ", "_")[:50]
    nb_path = settings.notebooks_dir / f"{safe_title}_{session.session_id[:8]}.ipynb"

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    session.notebook_path = nb_path

    return json.dumps({
        "notebook_path": str(nb_path),
        "cells": len(nb.cells),
        "message": f"Notebook exported to {nb_path}",
    })


# ── Tool: save_figure ────────────────────────────────────────────────────

def save_figure(
    figure_id: str,
    filename: str,
    session: AnalysisSession,
) -> str:
    """
    Save a session figure to disk as a PNG file.

    Args:
        figure_id: e.g. "fig_000"
        filename: Output filename (basename only, saved under figures_dir)
    """
    if figure_id not in session.figures:
        return json.dumps({
            "error": f"Figure '{figure_id}' not found.",
            "available": session.figure_ids,
        })

    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".png"):
        filename += ".png"
    out_path = settings.figures_dir / filename

    b64_data = session.figures[figure_id]
    png_bytes = base64.b64decode(b64_data)
    out_path.write_bytes(png_bytes)

    return json.dumps({
        "saved_to": str(out_path),
        "figure_id": figure_id,
        "size_bytes": len(png_bytes),
    })
```

---

## 5. Tool Definitions (JSON Schemas)

```python
# app/services/tool_definitions.py
"""
All 14 tool definitions as Anthropic tool_use JSON schemas.
Organised by group for readability.
"""
from typing import Any

# ── Group 1: Knowledge tools (6) ─────────────────────────────────────────

KNOWLEDGE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_domain_documents",
        "description": (
            "Return a JSON list of available domain knowledge documents (Markdown files). "
            "Always call this first before reading any document."
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
            "Return the full text content of a domain knowledge document. "
            "Use to understand physical constraints, unit definitions, and domain context "
            "before analysing data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Basename of the .md file, e.g. 'power_plant_thermodynamics.md'",
                },
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "search_domain_knowledge",
        "description": (
            "Keyword search across all domain documents. Returns ranked snippets from matching docs. "
            "Use when you need to find information about a specific concept without reading all docs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or keyword query, e.g. 'turbine efficiency range'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_datasets",
        "description": (
            "Return a JSON list of available dataset files with format and size metadata. "
            "Use before inspect_dataset to discover what data is available."
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
            "Load a dataset and return its schema (column names, dtypes), shape, "
            "numeric statistics (min/max/mean/std), and first 5 rows as JSON. "
            "Use this before execute_python_code to understand the data structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Basename of the dataset file, e.g. 'power_plant_data.csv'",
                },
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "describe_columns",
        "description": (
            "Return detailed statistics for specific columns: quartiles, skew, "
            "top value counts for categoricals, date range for timestamps. "
            "Use when you need deeper stats than inspect_dataset provides."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Basename of the dataset file",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of column names to describe",
                },
            },
            "required": ["file_name", "columns"],
        },
    },
]

# ── Group 2: Execution tools (6) ──────────────────────────────────────────

EXECUTION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "execute_python_code",
        "description": (
            "Execute Python code for data analysis and visualisation. "
            "pandas, numpy, matplotlib, and seaborn are pre-imported. "
            "Load data with pd.read_csv('data/datasets/<filename>'). "
            "Any plt.show() call or figure saved before show() will be captured. "
            "Print values you want to see in the Observation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete, self-contained Python code to execute",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_execution_variables",
        "description": (
            "Return a JSON object of Python variables from the last code execution. "
            "Only JSON-serialisable scalars and simple collections are included."
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
            "Return the base64-encoded PNG data for a specific figure. "
            "Use list_figures first to discover available figure IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {
                    "type": "string",
                    "description": "Figure ID, e.g. 'fig_000'",
                },
            },
            "required": ["figure_id"],
        },
    },
    {
        "name": "list_figures",
        "description": "Return a JSON list of all figure IDs generated in this session.",
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
            "Thoughts become Markdown cells; code executions become Code cells. "
            "Call at the end of an analysis to provide a reproducible artefact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the notebook, e.g. 'Power Plant Efficiency Analysis'",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "save_figure",
        "description": "Save a session figure to disk as a PNG file in the outputs/figures/ directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {"type": "string", "description": "Figure ID, e.g. 'fig_000'"},
                "filename": {"type": "string", "description": "Output filename without extension"},
            },
            "required": ["figure_id", "filename"],
        },
    },
]

# ── Group 3: Physical validation tools (3) ───────────────────────────────

VALIDATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "validate_physical_units",
        "description": (
            "Validate that a computed physical quantity has the correct units and "
            "a plausible magnitude for the given domain. "
            "Returns validation result with is_valid, warning, and expected_range. "
            "Use when you've computed a key result before including it in the Final Answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity_name": {
                    "type": "string",
                    "description": "Human-readable quantity name, e.g. 'thermal efficiency'",
                },
                "value": {"type": "number", "description": "Numeric magnitude"},
                "unit": {"type": "string", "description": "pint unit string, e.g. 'percent', 'degC', 'MPa'"},
                "domain_key": {
                    "type": "string",
                    "description": "Key in DOMAIN_RANGES for range check, e.g. 'thermal_efficiency'",
                },
            },
            "required": ["quantity_name", "value", "unit"],
        },
    },
    {
        "name": "convert_units",
        "description": "Convert a value from one physical unit to another using dimensional analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'degC'"},
                "to_unit": {"type": "string", "description": "Target unit, e.g. 'degF'"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "check_magnitude",
        "description": (
            "Quick check: is this value physically plausible for a known domain quantity? "
            "Returns plausible: true/false with the expected range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string"},
                "domain_key": {
                    "type": "string",
                    "description": "Domain key, e.g. 'thermal_efficiency', 'steam_pressure_hp'",
                },
            },
            "required": ["value", "unit", "domain_key"],
        },
    },
]

# ── Combined list + registry ───────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = (
    KNOWLEDGE_TOOLS + EXECUTION_TOOLS + VALIDATION_TOOLS
)

TOOL_REGISTRY: dict[str, str] = {
    # Maps tool name -> module.function for dynamic dispatch
    # "safe" = can be called programmatically without user confirmation
    "list_domain_documents": "knowledge_tools.list_domain_documents",      # safe
    "read_domain_document": "knowledge_tools.read_domain_document",        # safe
    "search_domain_knowledge": "knowledge_tools.search_domain_knowledge",  # safe
    "list_datasets": "knowledge_tools.list_datasets",                      # safe
    "inspect_dataset": "knowledge_tools.inspect_dataset",                  # safe
    "describe_columns": "knowledge_tools.describe_columns",                # safe
    "execute_python_code": "data_tools.execute_python_code",               # guarded (timeout)
    "get_execution_variables": "data_tools.get_execution_variables",       # safe
    "get_figure": "data_tools.get_figure",                                 # safe
    "list_figures": "data_tools.list_figures",                             # safe
    "export_notebook": "data_tools.export_notebook",                       # safe
    "save_figure": "data_tools.save_figure",                               # safe
    "validate_physical_units": "data_tools.validate_physical_units_tool",  # safe
    "convert_units": "data_tools.convert_units_tool",                      # safe
    "check_magnitude": "data_tools.check_magnitude_tool",                  # safe
}
```

---

## 6. Testing Phase 2

```python
# tests/test_phase2_tools.py
"""Tests for Phase 2 tool implementations."""
import json
import pytest
from pathlib import Path
from app.domain.analysis_models import AnalysisSession
from app.services.knowledge_tools import (
    list_domain_documents, read_domain_document,
    search_domain_knowledge, list_datasets, inspect_dataset,
)
from app.infrastructure.unit_registry import (
    validate_physical_units, convert_units, check_magnitude
)
from app.infrastructure.code_runner import SubprocessCodeRunner


class TestKnowledgeTools:
    def test_list_domain_documents(self, tmp_path):
        (tmp_path / "domain_docs").mkdir()
        (tmp_path / "domain_docs" / "thermo.md").write_text("# Thermo")
        result = json.loads(list_domain_documents(data_dir=tmp_path))
        assert "thermo.md" in result

    def test_read_missing_document(self, tmp_path):
        (tmp_path / "domain_docs").mkdir()
        result = read_domain_document("missing.md", data_dir=tmp_path)
        assert "ERROR" in result

    def test_search_returns_ranked(self, tmp_path):
        (tmp_path / "domain_docs").mkdir()
        (tmp_path / "domain_docs" / "a.md").write_text("efficiency thermal rankine cycle")
        (tmp_path / "domain_docs" / "b.md").write_text("emissions CO2 flue gas")
        results = json.loads(search_domain_knowledge("efficiency", data_dir=tmp_path))
        assert len(results) >= 1
        assert results[0]["file"] == "a.md"


class TestUnitRegistry:
    def test_valid_temperature(self):
        pu = validate_physical_units("steam temp", 540.0, "degC",
                                      domain_key="steam_temperature_hp")
        assert pu.is_valid

    def test_invalid_temperature_too_high(self):
        pu = validate_physical_units("steam temp", 800.0, "degC",
                                      domain_key="steam_temperature_hp")
        assert not pu.is_valid
        assert "outside expected range" in pu.warning

    def test_efficiency_over_100(self):
        pu = validate_physical_units("efficiency", 150.0, "percent",
                                      domain_key="thermal_efficiency")
        assert not pu.is_valid

    def test_convert_celsius_to_fahrenheit(self):
        result = convert_units(100.0, "degC", "degF")
        assert "converted" in result
        assert abs(result["converted"]["value"] - 212.0) < 0.1

    def test_convert_incompatible_units(self):
        result = convert_units(100.0, "degC", "MPa")
        assert "error" in result

    def test_check_magnitude_plausible(self):
        result = check_magnitude(37.0, "percent", "thermal_efficiency")
        assert result["plausible"] is True

    def test_check_magnitude_implausible(self):
        result = check_magnitude(200.0, "percent", "thermal_efficiency")
        assert result["plausible"] is False


class TestSubprocessRunner:
    def test_simple_print(self):
        runner = SubprocessCodeRunner(timeout=10)
        result = runner.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    def test_numpy_operation(self):
        runner = SubprocessCodeRunner(timeout=10)
        result = runner.execute("import numpy as np\nprint(np.mean([1,2,3,4,5]))")
        assert result.success
        assert "3.0" in result.stdout

    def test_syntax_error(self):
        runner = SubprocessCodeRunner(timeout=10)
        result = runner.execute("def broken(:\n    pass")
        assert not result.success
        assert result.error

    def test_figure_capture(self):
        runner = SubprocessCodeRunner(timeout=15)
        result = runner.execute(
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1,2,3])\n"
            "plt.show()"
        )
        assert result.success
        assert len(result.figures) == 1
```

```bash
pytest tests/test_phase2_tools.py -v
```

---

## Checkpoint

After Phase 2, your project should have:

```
✅ app/services/knowledge_tools.py      — 6 knowledge retrieval tools
✅ app/infrastructure/unit_registry.py  — pint singleton + 25 DOMAIN_RANGES
✅ app/infrastructure/code_runner.py    — SubprocessCodeRunner with figure capture
✅ app/services/data_tools.py           — 9 execution + validation tools
✅ app/services/tool_definitions.py     — 14 JSON schemas + TOOL_REGISTRY
✅ tests/test_phase2_tools.py           — Tool tests passing
```

→ **Next**: [04_phase3_react_engine.md](04_phase3_react_engine.md) — The ReAct loop
