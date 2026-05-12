# Tool System Design — 14 Tools Across 4 Groups

**Document Version:** 1.0  
**Status:** Approved  
**Bounded Context:** Tool Execution — Knowledge, Data, Validation, Output  

---

## 1. Design Principles

### Principle 1: Tools Never Raise
Every tool function returns a `str`. On error, it returns `"Error: <human-readable reason>"`. The ReAct loop records this as an Observation and Claude can self-correct. No exception propagates from a tool to the service layer.

### Principle 2: Path Safety is Mandatory
Every tool that takes a `file_name` parameter validates it:
- Reject strings containing `..` (directory traversal)
- Reject absolute paths (starting with `/`)
- Resolve against a known safe base directory
- Reject resolved paths that escape the base directory

### Principle 3: Deterministic Output Format
Each tool always returns the same structure for the same input. Knowledge tools return plain text or JSON. Data tools return JSON-serialized `AnalysisResult`. Validation tools return JSON-serialized `PhysicalUnit`. This allows Claude to reliably parse Observations.

### Principle 4: Size Limits on All I/O
- Max file content returned: 50KB per document
- Max code stdout/stderr: 100KB combined
- Max figure base64: 5MB
- Max dataset sample rows: 5

### Principle 5: Idempotent Where Possible
`inspect_dataset`, `describe_columns`, `list_datasets` have no side effects. `execute_python_code` is intentionally stateful (by design of the code execution backend).

---

## 2. Tool Groups Architecture

```
services/
├── knowledge_tools.py     # 6 tools: list_domain_documents, read_domain_document,
│                          #          search_domain_knowledge, list_datasets,
│                          #          inspect_dataset, describe_columns
├── data_tools.py          # 4 tools: execute_python_code, get_execution_variables,
│                          #          get_figure, list_figures
└── (validation + output tools live in infrastructure/unit_registry.py)
    #                      # 3 tools: validate_physical_units, convert_units, check_magnitude
    #                      # 2 tools: export_notebook, save_figure
```

Each group file is self-contained: it imports from `domain/` and `infrastructure/` but not from other `services/` files (no circular deps).

---

## 3. Extended TOOL_REGISTRY Pattern

The unified registry is assembled in `services/data_agent.py` (the only place that needs all 14 tools). This avoids a monolithic `tools.py` and keeps group files independently testable.

```python
# Pattern for building the registry
from app.services.knowledge_tools import (
    list_domain_documents, read_domain_document, search_domain_knowledge,
    list_datasets, inspect_dataset, describe_columns,
)
from app.services.data_tools import (
    execute_python_code, get_execution_variables, get_figure, list_figures,
)
from app.infrastructure.unit_registry import (
    validate_physical_units, convert_units, check_magnitude,
    export_notebook, save_figure,
)

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    # ── Knowledge group ───────────────────────────────
    "list_domain_documents":   list_domain_documents,
    "read_domain_document":    read_domain_document,
    "search_domain_knowledge": search_domain_knowledge,
    "list_datasets":           list_datasets,
    "inspect_dataset":         inspect_dataset,
    "describe_columns":        describe_columns,
    # ── Data/Code group ───────────────────────────────
    "execute_python_code":     execute_python_code,
    "get_execution_variables": get_execution_variables,
    "get_figure":              get_figure,
    "list_figures":            list_figures,
    # ── Physical Validation group ─────────────────────
    "validate_physical_units": validate_physical_units,
    "convert_units":           convert_units,
    "check_magnitude":         check_magnitude,
    # ── Output group ──────────────────────────────────
    "export_notebook":         export_notebook,
    "save_figure":             save_figure,
}
```

---

## 4. Knowledge Tools Deep Dive

### 4.1 list_domain_documents()

**Purpose:** Returns a formatted list of available domain knowledge files.

**Path Safety:** Lists only within `settings.domain_docs_dir`. No user-controlled path.

**Output Format:**
```
Available domain documents:
- thermodynamics.md (12,450 bytes)
- power_plant_operations.md (8,230 bytes)
- heat_transfer.md (6,100 bytes)
```

```python
def list_domain_documents() -> str:
    """Lists all domain knowledge documents available for reading."""
    docs_dir = settings.domain_docs_dir
    if not docs_dir.exists():
        return "Error: Domain documents directory not found."
    
    files = sorted(docs_dir.glob("*"))
    readable = [
        f for f in files
        if f.is_file() and f.suffix.lower() in {".md", ".txt", ".rst", ".pdf"}
    ]
    if not readable:
        return "No domain documents found."
    
    lines = ["Available domain documents:"]
    for f in readable:
        size = f.stat().st_size
        lines.append(f"- {f.name} ({size:,} bytes)")
    return "\n".join(lines)
```

### 4.2 read_domain_document(file_name: str)

**Purpose:** Returns the full content of a specific document.

**Path Safety:** Validates `file_name`, resolves against `domain_docs_dir`, rejects traversal.

**Encoding:** Reads with UTF-8, falls back to `errors="replace"` for binary/corrupt files.

**Size Limit:** Truncates at 50KB with a notice.

```python
def read_domain_document(file_name: str) -> str:
    """Reads and returns the content of a domain document."""
    MAX_BYTES = 51_200  # 50KB
    
    # Path safety
    if ".." in file_name or file_name.startswith("/"):
        return f"Error: Invalid file_name '{file_name}'."
    
    docs_dir = settings.domain_docs_dir
    resolved = (docs_dir / file_name).resolve()
    
    if not str(resolved).startswith(str(docs_dir.resolve())):
        return f"Error: Access denied — path escapes documents directory."
    
    if not resolved.exists():
        return f"Error: Document '{file_name}' not found."
    
    if not resolved.is_file():
        return f"Error: '{file_name}' is not a file."
    
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error: Cannot read '{file_name}': {e}"
    
    if len(content) > MAX_BYTES:
        content = content[:MAX_BYTES]
        content += f"\n\n[File truncated at {MAX_BYTES} bytes]"
    
    return content
```

### 4.3 search_domain_knowledge(query: str)

**Purpose:** Keyword search across all domain documents. Returns snippets.

**Implementation:** Simple inverted-index full-text search (no external search library).

**Algorithm:**
1. Tokenize `query` → set of lowercase tokens
2. For each document, scan lines for token matches
3. Score lines by token hit count
4. Return top-5 matching snippets with document name and line number

```python
def search_domain_knowledge(query: str) -> str:
    """Searches domain documents for lines matching the query."""
    if not query.strip():
        return "Error: Query cannot be empty."
    
    tokens = set(query.lower().split())
    docs_dir = settings.domain_docs_dir
    
    if not docs_dir.exists():
        return "Error: Domain documents directory not found."
    
    results: list[tuple[int, str, str, int]] = []  # (score, doc_name, line, lineno)
    
    for doc in sorted(docs_dir.glob("*")):
        if not doc.is_file() or doc.suffix.lower() not in {".md", ".txt", ".rst"}:
            continue
        try:
            lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        
        for lineno, line in enumerate(lines, 1):
            line_lower = line.lower()
            score = sum(1 for t in tokens if t in line_lower)
            if score > 0:
                results.append((score, doc.name, line.strip(), lineno))
    
    if not results:
        return f"No results found for query: '{query}'"
    
    # Sort by score descending, take top 5
    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:5]
    
    lines_out = [f"Search results for '{query}':"]
    for score, doc_name, line, lineno in top:
        lines_out.append(f"\n[{doc_name}:{lineno}] (score={score})\n  {line}")
    
    return "\n".join(lines_out)
```

### 4.4 list_datasets()

**Purpose:** Lists all available datasets with basic metadata.

**Supported Formats:** `.csv`, `.xlsx`, `.xls`, `.parquet`, `.hdf5`, `.h5`, `.json`

**Output Format:** JSON array of {file_name, format, file_size_bytes}

```python
def list_datasets() -> str:
    """Lists all available dataset files."""
    import json
    
    datasets_dir = settings.datasets_dir
    if not datasets_dir.exists():
        return "Error: Datasets directory not found."
    
    SUPPORTED = {".csv", ".xlsx", ".xls", ".parquet", ".hdf5", ".h5", ".json"}
    
    datasets = []
    for f in sorted(datasets_dir.glob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            datasets.append({
                "file_name": f.name,
                "format": f.suffix.lower().lstrip("."),
                "file_size_bytes": f.stat().st_size,
            })
    
    if not datasets:
        return "No datasets found. Upload a dataset via POST /api/v1/datasets."
    
    return json.dumps(datasets, indent=2)
```

### 4.5 inspect_dataset(file_name: str)

**Purpose:** Loads the first 5 rows and computes metadata without loading the full dataset into memory.

**Memory efficiency:** Uses `nrows=5` for CSV, `nrows=5` for Excel, `pd.read_parquet(...).head(5)` for Parquet. Total shape is computed separately with a lightweight count operation.

**Returns:** JSON-serialized `DatasetMeta`

```python
def inspect_dataset(file_name: str) -> str:
    """Inspects a dataset and returns its schema and sample rows as JSON."""
    import json
    import pandas as pd
    from app.domain.analysis_models import DatasetMeta
    
    # Path safety
    if ".." in file_name or file_name.startswith("/"):
        return f"Error: Invalid file_name '{file_name}'."
    
    datasets_dir = settings.datasets_dir
    path = (datasets_dir / file_name).resolve()
    
    if not str(path).startswith(str(datasets_dir.resolve())):
        return "Error: Access denied."
    
    if not path.exists():
        return f"Error: Dataset '{file_name}' not found. Use list_datasets() to see available files."
    
    try:
        ext = path.suffix.lower()
        
        if ext == ".csv":
            df_head = pd.read_csv(path, nrows=5)
            # Count total rows efficiently
            with open(path, "rb") as f:
                total_rows = sum(1 for _ in f) - 1  # subtract header
        elif ext in {".xlsx", ".xls"}:
            df_head = pd.read_excel(path, nrows=5)
            df_full = pd.read_excel(path)
            total_rows = len(df_full)
        elif ext == ".parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            total_rows = pf.metadata.num_rows
            df_head = pf.read_row_group(0).to_pandas().head(5)
        elif ext in {".hdf5", ".h5"}:
            df_head = pd.read_hdf(path, stop=5)
            df_full = pd.read_hdf(path)
            total_rows = len(df_full)
        elif ext == ".json":
            df_head = pd.read_json(path, lines=True, nrows=5)
            df_full = pd.read_json(path, lines=True)
            total_rows = len(df_full)
        else:
            return f"Error: Unsupported file format '{ext}'."
        
        total_cols = len(df_head.columns)
        
        meta = DatasetMeta(
            file_name=file_name,
            path=path,
            shape=(total_rows, total_cols),
            dtypes={col: str(dtype) for col, dtype in df_head.dtypes.items()},
            sample_rows=df_head.head(5).fillna("").to_dict(orient="records"),
            null_counts={col: int(df_head[col].isna().sum()) for col in df_head.columns},
            file_size_bytes=path.stat().st_size,
        )
        return json.dumps(meta.to_json_dict(), indent=2)
    
    except Exception as e:
        return f"Error: Failed to inspect '{file_name}': {e}"
```

### 4.6 describe_columns(file_name: str, columns: list[str])

**Purpose:** Per-column statistics for specified columns. Handles numeric, categorical, and datetime types differently.

**Returns:** JSON with per-column stats

```python
def describe_columns(file_name: str, columns: list[str]) -> str:
    """Computes descriptive statistics for specified columns."""
    import json
    import pandas as pd
    
    if ".." in file_name or file_name.startswith("/"):
        return f"Error: Invalid file_name."
    
    datasets_dir = settings.datasets_dir
    path = (datasets_dir / file_name).resolve()
    
    if not path.exists():
        return f"Error: Dataset '{file_name}' not found."
    
    try:
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return f"Error: Columns not found: {missing}. Available: {list(df.columns)}"
        
        stats: dict[str, dict] = {}
        for col in columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                stats[col] = {
                    "type": "numeric",
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "25%": float(series.quantile(0.25)),
                    "50%": float(series.median()),
                    "75%": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "null_count": int(series.isna().sum()),
                }
            elif pd.api.types.is_datetime64_any_dtype(series):
                stats[col] = {
                    "type": "datetime",
                    "count": int(series.count()),
                    "min": str(series.min()),
                    "max": str(series.max()),
                    "null_count": int(series.isna().sum()),
                }
            else:
                stats[col] = {
                    "type": "categorical",
                    "count": int(series.count()),
                    "unique": int(series.nunique()),
                    "top_values": series.value_counts().head(5).to_dict(),
                    "null_count": int(series.isna().sum()),
                }
        
        return json.dumps(stats, indent=2, default=str)
    
    except Exception as e:
        return f"Error: Failed to describe columns: {e}"
```

---

## 5. Data Tools Deep Dive

### 5.1 execute_python_code(session, code)

**Purpose:** Executes Python code in the configured backend. Returns `AnalysisResult` JSON.

**Backend Selection:** Reads `settings.code_execution_backend` → factory creates the runner once per session (stored in `session.code_runner_state["runner"]`).

```python
def execute_python_code(session: "AnalysisSession", code: str) -> str:
    """Executes Python code and returns stdout, stderr, figures, and variable list."""
    import json
    from app.infrastructure.code_runner import CodeRunnerFactory
    
    if not code.strip():
        return "Error: Code cannot be empty."
    
    # Get or create runner for this session
    if "runner" not in session.code_runner_state:
        session.code_runner_state["runner"] = CodeRunnerFactory.create(
            backend=settings.code_execution_backend,
            session_id=session.session_id,
        )
    
    runner = session.code_runner_state["runner"]
    
    result = runner.execute(code)  # returns AnalysisResult
    
    # Register any new figures in the session
    for fig_id in result.figures:
        b64 = runner.get_figure_b64(fig_id)
        if b64:
            session.add_figure(fig_id, b64)
    
    return json.dumps(result.to_json_dict(), indent=2)
```

### 5.2 get_execution_variables(session)

**Purpose:** Returns the names and types of all variables currently defined in the execution state.

```python
def get_execution_variables(session: "AnalysisSession") -> str:
    """Returns a list of variables defined in the current execution state."""
    import json
    
    if "runner" not in session.code_runner_state:
        return "No code has been executed yet in this session."
    
    runner = session.code_runner_state["runner"]
    variables = runner.get_state()  # dict[str, str] (name → type name)
    
    if not variables:
        return "No variables defined."
    
    return json.dumps(variables, indent=2)
```

### 5.3 get_figure(session, figure_id)

**Purpose:** Returns the base64 PNG for a named figure.

```python
def get_figure(session: "AnalysisSession", figure_id: str) -> str:
    """Returns the base64-encoded PNG for the given figure_id."""
    if figure_id not in session.figures:
        available = ", ".join(session.figures.keys()) or "none"
        return f"Error: Figure '{figure_id}' not found. Available: {available}"
    return session.figures[figure_id]
```

### 5.4 list_figures(session)

**Purpose:** Lists all figures generated in this session.

```python
def list_figures(session: "AnalysisSession") -> str:
    """Lists all figure IDs generated in this session."""
    import json
    
    if not session.figures:
        return "No figures generated yet."
    
    return json.dumps(list(session.figures.keys()), indent=2)
```

---

## 6. Physical Validation Tools

(Full implementations in `infrastructure/unit_registry.py` — see File 6)

### 6.1 validate_physical_units(quantity, value, unit, domain_context)

Returns JSON-serialized `PhysicalUnit`. Runs the 3-stage pipeline:
1. **Parse unit string** with pint — fails fast on unparseable units
2. **Dimensional check** — ensures quantity has expected dimensions (e.g., efficiency is dimensionless)
3. **Magnitude check** — compares value against `DOMAIN_RANGES[quantity]`

### 6.2 convert_units(value, from_unit, to_unit)

```python
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Converts a numeric value between units using pint."""
    try:
        import pint
        ureg = _get_unit_registry()
        quantity = value * ureg(from_unit)
        converted = quantity.to(to_unit)
        return f"{converted.magnitude:.6g} {to_unit}"
    except Exception as e:
        return f"Error: Cannot convert {value} {from_unit!r} to {to_unit!r}: {e}"
```

### 6.3 check_magnitude(quantity, value, unit)

```python
def check_magnitude(quantity: str, value: float, unit: str) -> str:
    """Checks if a value is within the expected range for a physical quantity."""
    import json
    
    # Fuzzy match quantity name in DOMAIN_RANGES
    matched_key = _fuzzy_match_quantity(quantity)
    if not matched_key:
        return f"Warning: Quantity '{quantity}' not in domain range database. Cannot validate magnitude."
    
    range_entry = DOMAIN_RANGES[matched_key]
    min_val, max_val, canonical_unit, description = range_entry
    
    # Convert to canonical unit for comparison
    try:
        ureg = _get_unit_registry()
        q = (value * ureg(unit)).to(canonical_unit)
        val_canonical = q.magnitude
    except Exception:
        return f"Error: Cannot convert '{unit}' to canonical unit '{canonical_unit}'."
    
    in_range = min_val <= val_canonical <= max_val
    return json.dumps({
        "quantity": quantity,
        "value": value,
        "unit": unit,
        "canonical_unit": canonical_unit,
        "value_canonical": val_canonical,
        "expected_range": [min_val, max_val],
        "in_range": in_range,
        "description": description,
    }, indent=2)
```

---

## 7. Output Tools

### 7.1 export_notebook(session, title)

Builds a `.ipynb` file from `session.jupyter_cells` using `nbformat`. Returns the path.

```python
def export_notebook(session: "AnalysisSession", title: str = "Analysis") -> str:
    """Exports the session's execution history as a Jupyter notebook (.ipynb)."""
    import nbformat
    import json
    from pathlib import Path
    
    if not session.jupyter_cells:
        return "Error: No code cells to export. Execute some code first."
    
    nb = nbformat.v4.new_notebook()
    nb.metadata["title"] = title
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    
    # Add title markdown cell
    nb.cells.append(nbformat.v4.new_markdown_cell(f"# {title}\n\nGenerated by Data Scientist AI Agent"))
    
    # Add each executed code cell
    for cell_data in session.jupyter_cells:
        cell = nbformat.v4.new_code_cell(source=cell_data["source"])
        cell.outputs = cell_data.get("outputs", [])
        cell.execution_count = cell_data.get("execution_count")
        nb.cells.append(cell)
    
    # Save to notebooks directory
    notebooks_dir = settings.notebooks_dir
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in title)
    out_path = notebooks_dir / f"{session.session_id}_{safe_title}.ipynb"
    
    nbformat.write(nb, str(out_path))
    return str(out_path)
```

### 7.2 save_figure(session, figure_id, filename)

```python
def save_figure(session: "AnalysisSession", figure_id: str, filename: str) -> str:
    """Saves a figure to disk and returns the file path."""
    import base64
    
    if figure_id not in session.figures:
        return f"Error: Figure '{figure_id}' not found."
    
    if ".." in filename or filename.startswith("/"):
        return "Error: Invalid filename."
    
    figures_dir = settings.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = figures_dir / filename
    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")
    
    try:
        img_data = base64.b64decode(session.figures[figure_id])
        out_path.write_bytes(img_data)
        return str(out_path)
    except Exception as e:
        return f"Error: Failed to save figure: {e}"
```

---

## 8. Complete Tool Definitions (JSON Schemas for messages.create())

```python
# Tool definitions for Claude's `tools` parameter
TOOL_DEFINITIONS = [
    {
        "name": "list_domain_documents",
        "description": "Lists all available domain knowledge documents (engineering manuals, specifications, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_domain_document",
        "description": "Reads the full content of a domain knowledge document.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "The filename of the document to read (e.g., 'thermodynamics.md')",
                },
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "search_domain_knowledge",
        "description": "Searches all domain documents for lines matching a keyword query. Returns top matching snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (space-separated keywords)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_datasets",
        "description": "Lists all available dataset files (CSV, Excel, Parquet, HDF5) with size information.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "inspect_dataset",
        "description": "Inspects a dataset: returns shape, column names, data types, sample rows, and null counts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Dataset filename (e.g., 'power_plant_data.csv')",
                },
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "describe_columns",
        "description": "Returns detailed statistics for specified columns: mean, std, min, max, percentiles for numeric; unique values for categorical.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Dataset filename",
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
    {
        "name": "execute_python_code",
        "description": "Executes Python code in an isolated environment. Returns stdout, stderr, figure IDs, and variable names. State persists across calls in the same session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Import pandas as pd, numpy as np, matplotlib.pyplot as plt — these are pre-available.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_execution_variables",
        "description": "Returns a dict of variable names → type names currently defined in the execution state.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_figure",
        "description": "Returns the base64-encoded PNG data for a figure generated during code execution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {
                    "type": "string",
                    "description": "Figure ID as returned in the execute_python_code result figures list",
                },
            },
            "required": ["figure_id"],
        },
    },
    {
        "name": "list_figures",
        "description": "Lists all figure IDs generated in the current session.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "validate_physical_units",
        "description": "Validates that a physical quantity value is dimensionally correct and within a reasonable range for the domain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity": {
                    "type": "string",
                    "description": "Physical quantity name (e.g., 'thermal_efficiency', 'temperature', 'power_output')",
                },
                "value": {
                    "type": "number",
                    "description": "The numeric value to validate",
                },
                "unit": {
                    "type": "string",
                    "description": "Unit string (e.g., '%', 'MW', '°C', 'bar')",
                },
                "domain_context": {
                    "type": "string",
                    "description": "Context describing what system this value comes from (e.g., 'coal power plant', 'HVAC system')",
                },
            },
            "required": ["quantity", "value", "unit", "domain_context"],
        },
    },
    {
        "name": "convert_units",
        "description": "Converts a numeric value from one physical unit to another.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The numeric value to convert",
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit (e.g., 'MW', '°C', 'bar')",
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit (e.g., 'GW', 'K', 'kPa')",
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "check_magnitude",
        "description": "Checks if a value's magnitude is within the expected range for a named physical quantity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity": {
                    "type": "string",
                    "description": "Physical quantity name to look up in domain ranges database",
                },
                "value": {
                    "type": "number",
                    "description": "The value to check",
                },
                "unit": {
                    "type": "string",
                    "description": "Unit of the value",
                },
            },
            "required": ["quantity", "value", "unit"],
        },
    },
    {
        "name": "export_notebook",
        "description": "Exports all executed code cells in this session as a Jupyter notebook (.ipynb) file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the notebook (used as filename and header)",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "save_figure",
        "description": "Saves a figure from the session to disk as a PNG file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_id": {
                    "type": "string",
                    "description": "Figure ID to save",
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g., 'efficiency_plot.png')",
                },
            },
            "required": ["figure_id", "filename"],
        },
    },
]
```

---

## 9. Adding a New Tool — 4-Step Guide

### Step 1: Implement the Function

```python
# In the appropriate group file (knowledge_tools.py, data_tools.py, etc.)

def my_new_tool(param1: str, param2: int = 10) -> str:
    """
    One-line description of what this tool does.
    
    Returns:
        JSON string on success, "Error: ..." string on failure.
    """
    try:
        result = _do_something(param1, param2)
        return json.dumps(result)
    except SomeSpecificError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Unexpected failure in my_new_tool: {e}"
```

**Rules:**
- Never raise exceptions
- Return `"Error: ..."` for all failure cases
- Path-check any file_name parameters
- Keep side effects minimal and documented

### Step 2: Write the JSON Schema

```python
# Add to TOOL_DEFINITIONS list in data_agent.py

{
    "name": "my_new_tool",
    "description": "One sentence: what it does and when the agent should use it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Clear description with example values",
            },
            "param2": {
                "type": "integer",
                "description": "Description with default value noted",
                "default": 10,
            },
        },
        "required": ["param1"],
    },
}
```

### Step 3: Register in TOOL_REGISTRY

```python
# In services/data_agent.py, add to TOOL_REGISTRY dict:
from app.services.my_group import my_new_tool

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    ...
    "my_new_tool": my_new_tool,   # Add here
}

# If the tool needs session context:
_SESSION_TOOLS = frozenset({
    ...,
    "my_new_tool",  # Add here if it takes session= parameter
})
```

### Step 4: Write Tests

```python
# tests/services/test_my_new_tool.py

def test_my_new_tool_success():
    result = my_new_tool("valid_param")
    data = json.loads(result)
    assert "expected_key" in data

def test_my_new_tool_invalid_param():
    result = my_new_tool("../bad/path")
    assert result.startswith("Error:")

def test_my_new_tool_empty_param():
    result = my_new_tool("")
    assert result.startswith("Error:")
```

**Tests must not:**
- Hit the real filesystem (use `tmp_path` fixture for file-based tools)
- Call Claude API
- Spin up a Jupyter kernel
