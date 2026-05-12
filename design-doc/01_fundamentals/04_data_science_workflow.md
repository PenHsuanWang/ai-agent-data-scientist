# The AI-Augmented Data Science Workflow

> *"Data science is 80% data preparation and 20% complaining about data preparation."*
> — Anonymous

---

## Table of Contents

1. [Traditional DS Workflow (CRISP-DM)](#1-traditional-ds-workflow-crisp-dm)
2. [AI Agent Augmentation](#2-ai-agent-augmentation)
3. [The 4-Phase Analysis Loop](#3-the-4-phase-analysis-loop)
4. [Data Formats Supported](#4-data-formats-supported)
5. [Exploratory Data Analysis Pattern](#5-exploratory-data-analysis-pattern)
6. [Code Generation Best Practices](#6-code-generation-best-practices)
7. [From Analysis to Notebook](#7-from-analysis-to-notebook)
8. [The Physical Interpretation Step](#8-the-physical-interpretation-step)

---

## 1. Traditional DS Workflow (CRISP-DM)

### Overview

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) was published in 1999 and remains the dominant framework for organizing data science projects. It defines 6 phases in a cycle — not a linear sequence, because real projects iterate.

```
                    ┌──────────────────────────┐
                    │                          │
              ┌─────▼─────┐             ┌──────▼──────┐
              │ Business   │             │ Data        │
              │ Understanding◄──────────►Understanding │
              └─────┬─────┘             └──────┬──────┘
                    │                          │
              ┌─────▼─────┐             ┌──────▼──────┐
              │ Deployment │             │ Data        │
              │            ◄──────────── Preparation  │
              └─────▲─────┘             └──────┬──────┘
                    │                          │
              ┌─────┴─────┐             ┌──────▼──────┐
              │ Evaluation │◄────────── │ Modeling    │
              └───────────┘             └─────────────┘
```

### Phase 1: Business Understanding

Translate the stakeholder's question into a precise analytical objective. This is the hardest step, because stakeholders often know what they want but not what they're asking for.

**Typical inputs**: Meeting notes, requirements documents, informal conversation
**Typical outputs**: Problem statement, success criteria, analytical approach

**Example**:
- What stakeholder said: *"We want to know if our plant is running efficiently"*
- What the analysis objective is: *"Compute hourly net thermal efficiency η_th for Q4 2024 and identify hours where η_th deviates more than 2σ from the rolling 7-day mean, correlating deviations with meteorological and operational covariates."*

### Phase 2: Data Understanding

Discover what data is available, its quality, and its relevance to the objective.

**Typical activities**:
- Inventory available data sources
- Load sample of each dataset
- Examine column names, types, ranges, missing values
- Check for obvious data quality issues (negative values in positive-only quantities, impossible timestamps, constant columns)
- Identify join keys if multiple tables exist

**Typical outputs**: Data inventory, data quality report, variable dictionary

### Phase 3: Data Preparation

The most time-consuming phase. Transforms raw data into a form suitable for analysis.

**Typical activities**:
- Handling missing values (imputation vs. dropping vs. flagging)
- Unit standardization (ensure all temperatures in same unit, etc.)
- Feature engineering (compute derived quantities: efficiency, ΔT, rolling averages)
- Outlier detection and handling
- Train/test splitting (if modeling)
- Documentation of all transformations

### Phase 4: Modeling

Apply statistical or machine learning methods to the prepared data.

**Note**: For exploratory/diagnostic data science (which is this agent's primary use case), "modeling" often means computing descriptive statistics, correlation analysis, and time-series decomposition — not necessarily ML.

### Phase 5: Evaluation

Assess whether the model/analysis answers the original business question. This is where physical validation belongs — a model that predicts efficiency > 100% is not merely "inaccurate"; it is physically impossible, and the evaluation step must catch this.

### Phase 6: Deployment

In the context of this agent: deployment means producing a notebook, report, or API endpoint that communicates the findings in a usable form. The `export_notebook` tool is the primary deployment mechanism.

---

## 2. AI Agent Augmentation

### Where Claude Adds Value at Each Phase

| CRISP-DM Phase | Without AI | With AI Agent |
|----------------|-----------|---------------|
| **Business Understanding** | Human writes specification | Agent helps translate vague questions into precise analytical objectives; surfaces edge cases |
| **Data Understanding** | Human manually explores each column | Agent calls `inspect_dataset` and `describe_dataset`; identifies unit mismatches, flag suspicious patterns |
| **Data Preparation** | Human writes boilerplate cleaning code | Agent generates idiomatic pandas code with explicit unit handling; explains each transformation |
| **Modeling** | Human writes analysis code | Agent generates, runs, and iterates on analysis code in a single turn |
| **Evaluation** | Human checks results mentally | Agent calls `validate_units` and `check_magnitude` on every quantitative result |
| **Deployment** | Human assembles notebook manually | Agent accumulates cells and calls `export_notebook` |

### What the Agent Cannot Replace

- **Domain expertise**: The agent knows general physics and engineering, but it does not know your specific plant's quirks, maintenance history, or design tolerances. This must come from domain documents.
- **Stakeholder communication**: The agent produces technical analysis; translating that into a business recommendation requires human judgment.
- **Creative hypothesis generation**: The agent explores hypotheses it can express as tool calls. Novel hypotheses that require domain intuition still come from humans.
- **Data acquisition**: The agent works with datasets that already exist. Getting the right data is a human responsibility.

---

## 3. The 4-Phase Analysis Loop

The data scientist agent's workflow maps directly onto CRISP-DM but is structured around the 4 tool groups:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: KNOW  (Knowledge Tools)                                   │
│                                                                     │
│  list_local_documents() → read_local_document() → search_documents()│
│  list_datasets() → inspect_dataset()                                │
│                                                                     │
│  Goal: Understand the domain and available data BEFORE touching it  │
│  Duration: Typically 3–6 tool calls                                 │
│  Output: Domain profile (unit conventions, typical ranges,          │
│          physical laws, dataset schema)                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 2: EXPLORE  (Data Tools)                                     │
│                                                                     │
│  load_dataset() → describe_dataset() → execute_python(EDA code)    │
│                                                                     │
│  Goal: Understand the actual data — distributions, outliers,        │
│        missing values, correlations, time patterns                  │
│  Duration: Typically 4–8 tool calls                                 │
│  Output: EDA summary, identification of analysis-ready variables    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 3: ANALYZE  (Data Tools)                                     │
│                                                                     │
│  execute_python() [loop] → get_figure() → save_figure()            │
│                                                                     │
│  Goal: Compute the quantities that answer the business question      │
│  Duration: Typically 5–12 tool calls                                │
│  Output: Computed results (efficiency, anomalies, predictions)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 4: VALIDATE  (Physical Validation Tools)                     │
│                                                                     │
│  validate_units() → check_magnitude() → convert_units()            │
│                                                                     │
│  Goal: Verify every quantitative result is physically plausible     │
│  Duration: Typically 1 call per result (3–8 total)                  │
│  Output: Validated results OR flagged discrepancies + diagnosis     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 5: EXPORT  (Output Tools)                                    │
│                                                                     │
│  export_notebook() → (return final answer to user)                  │
│                                                                     │
│  Goal: Package all cells and findings into a reproducible notebook  │
│  Duration: 1–2 tool calls                                           │
│  Output: .ipynb file + structured final answer                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase Transition Logic

The agent should not move to Phase 2 until Phase 1 is complete (at least one domain document read, dataset schema inspected). This is enforced via the system prompt:

```python
REACT_SYSTEM_PROMPT_PHASES = """
## Analysis Phase Protocol

You MUST follow this phase order. Do not skip phases.

### Phase 1: KNOW (Complete before any analysis)
Before touching the dataset:
- Call list_local_documents() to see available domain documents
- Call read_local_document() on at least one relevant document
- Call list_datasets() to see available datasets
- Call inspect_dataset() to see the schema of the target dataset
You must know: unit conventions, expected value ranges, physical laws applicable.

### Phase 2: EXPLORE (Complete before computing derived quantities)
Load the dataset and run basic EDA:
- describe_dataset() for statistical summary
- execute_python() for distribution plots and correlation analysis
You must know: missing value percentage, outlier presence, temporal gaps.

### Phase 3: ANALYZE
Compute the quantities requested by the user.
Apply explicit unit conversions in every calculation.
Store results in a dictionary — never print raw DataFrames.

### Phase 4: VALIDATE
For every numeric result with a unit:
- Call validate_units() or check_magnitude()
- If validation fails: investigate before continuing

### Phase 5: EXPORT
Call export_notebook() and provide a structured final answer.
"""
```

---

## 4. Data Formats Supported

The `LocalDatasetStore` and `load_dataset` tool support 6 file formats:

### CSV (Comma-Separated Values)

```python
import pandas as pd

def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file with sensible defaults for engineering datasets.
    
    Common kwargs:
        sep: column separator (default ",", use "\t" for TSV)
        encoding: file encoding (default "utf-8", try "latin-1" if errors)
        parse_dates: list of columns to parse as datetime
        index_col: column to use as row index
        dtype: dict of column → dtype (use for large integer IDs)
        na_values: additional strings to treat as NaN
    """
    return pd.read_csv(
        path,
        encoding=kwargs.pop("encoding", "utf-8"),
        parse_dates=kwargs.pop("parse_dates", False),
        na_values=kwargs.pop("na_values", ["", "NA", "N/A", "nan", "NaN", "NULL", "#N/A"]),
        **kwargs,
    )

# Example: loading a sensor data CSV with timestamps
df = load_csv(
    "plant_sensors_2024_Q4.csv",
    parse_dates=["timestamp"],
    index_col="timestamp",
)
```

**Common pitfalls**:
- Comma in numeric values (European locale): use `sep=";"` and `decimal=","`
- Encoding issues with special characters: try `encoding="latin-1"` or `encoding="cp1252"`
- Mixed timestamp formats in a single column: use `pd.to_datetime(df['ts'], infer_datetime_format=True)`

### Excel (xlsx/xls)

```python
def load_excel(path: str, **kwargs) -> pd.DataFrame:
    """
    Load an Excel file. Defaults to first sheet.
    
    Common kwargs:
        sheet_name: sheet name (str) or index (int), default 0
        header: row number for header (default 0)
        skiprows: number of rows to skip at top (common in reports with title rows)
        usecols: list of column letters or names to load (for wide files)
    """
    return pd.read_excel(
        path,
        sheet_name=kwargs.pop("sheet_name", 0),
        header=kwargs.pop("header", 0),
        **kwargs,
    )

# Example: loading the second sheet of a multi-sheet report
df_q3 = load_excel("plant_report_2024.xlsx", sheet_name="Q3_Data", skiprows=3)
```

**Common pitfalls**:
- Files with merged cells in headers: use `header=None` and manually set column names
- Date columns stored as Excel serial numbers: `pd.to_datetime(df['date'], unit='D', origin='1899-12-30')`
- Large `.xlsx` files are slow: consider converting to CSV or Parquet first

### HDF5

```python
def load_hdf5(path: str, key: str = "/data", **kwargs) -> pd.DataFrame:
    """
    Load a dataset from an HDF5 file.
    HDF5 supports hierarchical data — specify the key to the target dataset.
    
    Use h5py to explore keys first:
        import h5py
        with h5py.File(path, 'r') as f:
            f.visit(print)  # Print all keys
    """
    return pd.read_hdf(path, key=key, **kwargs)

# Example: loading time-series from HDF5 archive
df = load_hdf5("sensor_archive.h5", key="/plant3/sensors/2024/Q4")
```

**Common pitfalls**:
- Wrong key: results in `KeyError` — always inspect with `h5py` first
- HDF5 format created by MATLAB vs pandas have different conventions
- Very large HDF5 files: use `start=` and `stop=` kwargs to load slices

### Parquet

```python
def load_parquet(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a Parquet file. Parquet is the preferred format for large datasets:
    - Column-oriented: fast for column subsets
    - Compressed: typically 3–10× smaller than CSV
    - Schema-aware: preserves dtypes including datetime
    
    Requires pyarrow or fastparquet: pip install pyarrow
    
    Common kwargs:
        columns: list of column names to load (column pruning)
        filters: list of filter tuples for row pruning
    """
    return pd.read_parquet(path, **kwargs)

# Example: loading only efficiency-related columns
df = load_parquet(
    "sensor_data_2024.parquet",
    columns=["timestamp", "gross_power_mw", "auxiliary_power_mw", "fuel_flow_kg_s"],
)
```

**Common pitfalls**:
- Mixed Parquet versions (written by Spark vs pandas): use `engine="pyarrow"` explicitly
- Partition directories (Hive format): pass the directory path, not a specific file

### JSON

```python
def load_json(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a JSON file or JSON Lines file.
    
    JSON can represent tabular data in several orientations:
        "records": [{"col1": val1, "col2": val2}, ...]  ← most common
        "split": {"columns": [...], "data": [[...], ...]}
        "columns": {"col1": {index: val, ...}, ...}
    
    For JSONL (one JSON object per line): use lines=True
    """
    return pd.read_json(
        path,
        orient=kwargs.pop("orient", "records"),
        lines=kwargs.pop("lines", False),
        **kwargs,
    )

# Example: loading from an API response saved as JSONL
df = load_json("api_export.jsonl", lines=True)
```

### Format Detection and Dispatch

```python
LOADERS = {
    ".csv": load_csv,
    ".tsv": lambda p, **kw: load_csv(p, sep="\t", **kw),
    ".parquet": load_parquet,
    ".h5": load_hdf5,
    ".hdf5": load_hdf5,
    ".xlsx": load_excel,
    ".xls": load_excel,
    ".json": load_json,
    ".jsonl": lambda p, **kw: load_json(p, lines=True, **kw),
}

def load_dataset(path: str, **kwargs) -> pd.DataFrame:
    """Auto-detect format and load dataset."""
    suffix = Path(path).suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {list(LOADERS.keys())}"
        )
    return loader(path, **kwargs)
```

---

## 5. Exploratory Data Analysis Pattern

### The 8-Step EDA Checklist

The agent follows this checklist in the EXPLORE phase. Each step generates a code cell in the accumulating notebook.

```python
# Step 1: Basic shape and schema
def eda_step1_schema(df: pd.DataFrame) -> dict:
    """What do we have?"""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
    }

# Step 2: Missing values
def eda_step2_missing(df: pd.DataFrame) -> dict:
    """How much data is missing?"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    return {
        "missing_counts": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "columns_all_null": list(df.columns[df.isnull().all()]),
        "columns_any_null": list(df.columns[df.isnull().any()]),
    }

# Step 3: Numeric summary statistics
def eda_step3_statistics(df: pd.DataFrame) -> dict:
    """What are the distributions?"""
    numeric_df = df.select_dtypes(include="number")
    stats = numeric_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    # Flag potentially wrong units: check for values that seem too large or too small
    suspicious = {}
    for col in numeric_df.columns:
        col_stats = stats.loc[col]
        if col_stats['max'] > 1e10 or (col_stats['min'] < 0 and col.endswith(('_pct', '_eff', '_ratio'))):
            suspicious[col] = {
                "min": col_stats["min"],
                "max": col_stats["max"],
                "flag": "SUSPICIOUS: may have unit or sign issue",
            }
    return {
        "statistics": stats.to_dict(orient="index"),
        "suspicious_columns": suspicious,
    }

# Step 4: Duplicate rows
def eda_step4_duplicates(df: pd.DataFrame) -> dict:
    """Are there duplicate records?"""
    n_dups = df.duplicated().sum()
    return {
        "duplicate_rows": int(n_dups),
        "duplicate_pct": round(n_dups / len(df) * 100, 2),
        "recommendation": (
            "Remove duplicates before analysis" if n_dups > 0
            else "No duplicates found"
        ),
    }

# Step 5: Constant or near-constant columns
def eda_step5_constant_columns(df: pd.DataFrame, threshold: float = 0.99) -> dict:
    """Which columns add no information?"""
    constant_cols = []
    near_constant_cols = []
    for col in df.select_dtypes(include="number").columns:
        value_counts = df[col].value_counts(normalize=True)
        if len(value_counts) == 1:
            constant_cols.append(col)
        elif value_counts.iloc[0] >= threshold:
            near_constant_cols.append({
                "column": col,
                "dominant_value": value_counts.index[0],
                "frequency": float(value_counts.iloc[0]),
            })
    return {
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
    }

# Step 6: Correlation analysis
def eda_step6_correlations(df: pd.DataFrame, target_col: str = None) -> dict:
    """Which variables are related?"""
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()
    
    # Find high correlations (excluding self-correlations)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    "col1": corr_matrix.columns[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": round(float(corr_val), 3),
                })
    
    result = {"high_correlations": high_corr_pairs}
    
    # If target column specified, show correlations with target
    if target_col and target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(
            key=abs, ascending=False
        )
        result["correlations_with_target"] = target_corr.head(10).round(3).to_dict()
    
    return result

# Step 7: Time-series specific checks (if datetime index)
def eda_step7_temporal(df: pd.DataFrame) -> dict:
    """Is the time series complete and regular?"""
    if not isinstance(df.index, pd.DatetimeIndex):
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if not datetime_cols:
            return {"skipped": "No datetime index or columns found"}
        df = df.set_index(datetime_cols[0])
    
    time_diffs = df.index.to_series().diff().dropna()
    most_common_freq = time_diffs.mode()[0]
    gaps = time_diffs[time_diffs > most_common_freq * 2]
    
    return {
        "index_range": [str(df.index.min()), str(df.index.max())],
        "inferred_frequency": str(most_common_freq),
        "total_periods": len(df),
        "gap_count": int(len(gaps)),
        "largest_gap": str(gaps.max()) if len(gaps) > 0 else "None",
        "gap_timestamps": gaps.index[:5].tolist() if len(gaps) > 0 else [],
    }

# Step 8: Outlier detection
def eda_step8_outliers(df: pd.DataFrame, method: str = "iqr") -> dict:
    """Which values are statistically extreme?"""
    outlier_summary = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        elif method == "zscore":
            mean, std = series.mean(), series.std()
            lower, upper = mean - 4 * std, mean + 4 * std
        
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) > 0:
            outlier_summary[col] = {
                "count": int(len(outliers)),
                "pct": round(len(outliers) / len(series) * 100, 2),
                "min_outlier": float(outliers.min()),
                "max_outlier": float(outliers.max()),
                "expected_range": [round(float(lower), 3), round(float(upper), 3)],
            }
    return outlier_summary
```

### Generating the EDA Report

```python
def run_full_eda(df: pd.DataFrame, dataset_name: str) -> dict:
    """Run all 8 EDA steps and return a consolidated report."""
    report = {
        "dataset": dataset_name,
        "step_1_schema": eda_step1_schema(df),
        "step_2_missing": eda_step2_missing(df),
        "step_3_statistics": eda_step3_statistics(df),
        "step_4_duplicates": eda_step4_duplicates(df),
        "step_5_constant": eda_step5_constant_columns(df),
        "step_6_correlations": eda_step6_correlations(df),
        "step_7_temporal": eda_step7_temporal(df),
        "step_8_outliers": eda_step8_outliers(df),
    }
    
    # Generate human-readable summary
    issues = []
    if report["step_2_missing"]["columns_any_null"]:
        n = len(report["step_2_missing"]["columns_any_null"])
        issues.append(f"{n} columns have missing values")
    if report["step_4_duplicates"]["duplicate_rows"] > 0:
        issues.append(f"{report['step_4_duplicates']['duplicate_rows']} duplicate rows")
    if report["step_5_constant"]["constant_columns"]:
        issues.append(f"{len(report['step_5_constant']['constant_columns'])} constant columns")
    if report["step_8_outliers"]:
        issues.append(f"{len(report['step_8_outliers'])} columns with outliers")
    if report["step_3_statistics"]["suspicious_columns"]:
        issues.append(f"{len(report['step_3_statistics']['suspicious_columns'])} suspicious columns (check units)")
    
    report["issues_summary"] = issues if issues else ["No significant data quality issues found"]
    return report
```

---

## 6. Code Generation Best Practices

### How to Prompt Claude to Write Good Analysis Code

The system prompt for the data scientist agent includes specific instructions on code style:

```python
CODE_GENERATION_GUIDELINES = """
## Code Generation Standards

When writing Python code for analysis, always follow these conventions:

### 1. Explicit unit annotations in variable names
BAD:  flow = df['fuel_flow'] * df['lhv']
GOOD: heat_input_mw = (df['fuel_flow_kg_min'] / 60.0) * df['lhv_mj_kg']
      # fuel_flow_kg_min / 60 → kg/s; kg/s × MJ/kg = MW

### 2. Comments for every unit conversion
Always annotate:
  # unit_from → unit_to (conversion factor)
  # e.g.: kg/min → kg/s (÷60)
  # e.g.: BTU/hr → W (×0.293071)

### 3. Results in structured dicts
BAD:  print(efficiency.describe())
GOOD: result = {
          "mean_efficiency_pct": float(efficiency.mean()),
          "std_efficiency_pct": float(efficiency.std()),
          "min_pct": float(efficiency.min()),
          "max_pct": float(efficiency.max()),
          "n_samples": int(len(efficiency)),
      }
      print(result)

### 4. Figure saving
When creating a figure, save it AND call get_figure():
  fig, ax = plt.subplots(figsize=(12, 5))
  ax.plot(df.index, efficiency_pct)
  ax.set_xlabel("Time")
  ax.set_ylabel("Thermal Efficiency (%)")
  ax.set_title("Q4 2024 Plant Thermal Efficiency")
  ax.axhline(y=48.0, color='g', linestyle='--', label='Design (48%)')
  ax.axhline(y=45.0, color='r', linestyle='--', label='Alarm threshold (45%)')
  ax.legend()
  fig.savefig("efficiency_q4.png", dpi=150, bbox_inches='tight')
  plt.close(fig)

### 5. Defensive column access
BAD:  df['efficiency'] = ...
GOOD: required_cols = ['fuel_flow_kg_min', 'fuel_lhv_mj_kg', 'gross_power_mw']
      missing = [c for c in required_cols if c not in df.columns]
      if missing:
          print(f"Error: missing columns: {missing}. Available: {list(df.columns)}")
      else:
          # proceed with computation

### 6. Timestamp handling
Always localize or drop timezone info explicitly:
  df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
  # OR for naive timestamps:
  df.index = pd.to_datetime(df.index)
"""
```

### Anti-Patterns to Avoid

```python
# ANTI-PATTERN 1: Magic numbers without explanation
heat_loss = power * 0.124  # What is 0.124? Where does it come from?

# BETTER: Named constant with source
AUXILIARY_LOSS_FRACTION = 0.124  # From design doc: section 4.2 "Auxiliary loads"
heat_loss = power * AUXILIARY_LOSS_FRACTION

# ANTI-PATTERN 2: Printing DataFrames directly
print(df.head(20))  # Too verbose for tool observations — will be truncated

# BETTER: Extract relevant information
summary = {
    "shape": df.shape,
    "first_row": df.iloc[0].to_dict(),
    "columns_with_nulls": df.columns[df.isnull().any()].tolist(),
}
print(summary)

# ANTI-PATTERN 3: Modifying the original DataFrame
df['efficiency'] = df['power'] / df['heat_input']  # Modifies df in place

# BETTER: Assign to a new variable
df_analyzed = df.copy()
df_analyzed['efficiency_pct'] = (df['net_power_mw'] / df['heat_input_mw']) * 100.0

# ANTI-PATTERN 4: Silently ignoring NaN
efficiency_mean = df['efficiency'].mean()  # NaN in column → NaN result, no warning

# BETTER: Handle NaN explicitly
n_valid = df['efficiency'].notna().sum()
n_total = len(df)
efficiency_mean = df['efficiency'].mean()  # pandas mean() skips NaN by default
print({
    "mean_efficiency": float(efficiency_mean),
    "valid_samples": int(n_valid),
    "null_samples": int(n_total - n_valid),
    "null_pct": round((n_total - n_valid) / n_total * 100, 1),
})
```

---

## 7. From Analysis to Notebook

### Cell Accumulation Strategy

Every `execute_python` call produces a code cell. Every `Thought` produces a markdown cell. The notebook accumulates these in order:

```
Notebook Cell Sequence:
  [Markdown] Session header (title, date, user request)
  [Markdown] Phase 1: Domain Knowledge
  [Code]     list_local_documents() call result
  [Code]     read_local_document() — domain reference
  [Markdown] Phase 2: Exploratory Data Analysis  
  [Code]     EDA Step 1: Schema inspection
  [Code]     EDA Step 2: Missing values
  ...
  [Markdown] Phase 3: Analysis
  [Markdown] Thought: "I will compute efficiency with explicit unit handling..."
  [Code]     efficiency computation code
  [Markdown] Phase 4: Validation
  [Code]     validate_units() results
  [Markdown] Findings and Conclusions
```

### Cell Output Capture

When using the Subprocess or Jupyter backends, cell outputs need to be captured and stored:

```python
from dataclasses import dataclass, field

@dataclass
class NotebookCell:
    cell_type: str   # "code" or "markdown"
    source: str
    outputs: list[dict] = field(default_factory=list)
    execution_count: int | None = None
    metadata: dict = field(default_factory=dict)

def execution_result_to_outputs(result: ExecutionResult) -> list[dict]:
    """Convert an ExecutionResult to nbformat output dicts."""
    outputs = []
    
    if result.stdout:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": result.stdout,
        })
    
    if result.stderr:
        outputs.append({
            "output_type": "stream",
            "name": "stderr",
            "text": result.stderr,
        })
    
    for figure_b64 in result.figures:
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": figure_b64, "text/plain": "<Figure>"},
            "metadata": {"image/png": {"width": 800, "height": 400}},
        })
    
    return outputs
```

### Exporting to .ipynb

```python
import nbformat
from pathlib import Path

def export_to_notebook(
    cells: list[NotebookCell],
    output_path: str,
    kernel_name: str = "python3",
) -> str:
    """
    Export accumulated cells to a Jupyter notebook file.
    
    Returns the path to the created file.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": kernel_name,
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.12.0",
    }
    
    nbformat_cells = []
    for cell in cells:
        if cell.cell_type == "code":
            nb_cell = nbformat.v4.new_code_cell(source=cell.source)
            nb_cell.outputs = cell.outputs
            if cell.execution_count is not None:
                nb_cell.execution_count = cell.execution_count
        else:  # markdown
            nb_cell = nbformat.v4.new_markdown_cell(source=cell.source)
        
        nb_cell.metadata.update(cell.metadata)
        nbformat_cells.append(nb_cell)
    
    nb.cells = nbformat_cells
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    return str(output_path)
```

---

## 8. The Physical Interpretation Step

### Why Every Number Needs a Triad

A well-formed analytical finding has three parts: **value + unit + context**. None of the three is optional.

| Component | Example | What happens without it |
|-----------|---------|------------------------|
| **Value** | 36.2 | Useless — no quantity |
| **Unit** | % | Ambiguous — 36.2 what? |
| **Context** | "...compared to design spec of 38% and Q3 average of 37.4%; a 3.2% relative decrease consistent with 8% fuel increase" | Just a number — no insight |

### The Interpretation Template

The agent's final answer should follow this template for every quantitative finding:

```
[QUANTITY NAME]: [VALUE] [UNIT]
  Status: [NORMAL / ANOMALOUS / IMPOSSIBLE / DEGRADED]
  Baseline: [design spec or historical average for comparison]
  Change: [absolute and relative change from baseline]
  Physical check: [validation result]
  Implication: [what this means for the system/process]
  Recommendation: [specific next step if action is warranted]
```

**Example**:
```
Thermal Efficiency (Q4 2024 average): 36.2%
  Status: DEGRADED (below design range 38–42%)
  Baseline: Design spec = 40.0%, Q3 2024 average = 37.4%
  Change: -1.2 percentage points from Q3; -3.8 from design (-9.5% relative)
  Physical check: PASSED — 36.2% is within thermodynamic bounds [0%, 66.1% Carnot limit]
  Implication: The 3.8 point efficiency gap corresponds to approximately 8% additional
               fuel consumption at constant output — consistent with the reported fuel
               increase. This is a performance degradation, not a measurement error.
  Recommendation: Schedule boiler tube inspection and cleaning. Check condenser vacuum.
                  Compare efficiency trend curve against planned maintenance dates.
```

### Implementing the Interpretation Step

```python
def format_physical_finding(
    quantity_name: str,
    value: float,
    unit: str,
    validation_result: dict,
    baseline: dict | None = None,
    domain_context: str | None = None,
) -> str:
    """
    Format a physical finding with full context for the final answer.
    
    Args:
        quantity_name: Human-readable name of the quantity
        value: Computed value
        unit: Unit string
        validation_result: Output from validate_against_range()
        baseline: Optional {"design": float, "previous": float, "previous_label": str}
        domain_context: Optional interpretation from domain knowledge
    """
    lines = [f"**{quantity_name}**: {value:.3g} {unit}"]
    
    # Validation status
    if validation_result.get("valid") is False:
        lines.append(f"  ⚠️ Status: VALIDATION FAILED")
        for issue in validation_result.get("issues", []):
            lines.append(f"  - {issue}")
    elif validation_result.get("warnings"):
        lines.append(f"  ⚠️ Status: ANOMALOUS (within physical limits but unusual)")
        for warning in validation_result.get("warnings", []):
            lines.append(f"  - {warning}")
    else:
        lines.append(f"  ✓ Status: WITHIN NORMAL RANGE")
    
    # Baseline comparison
    if baseline:
        if "design" in baseline:
            delta = value - baseline["design"]
            delta_pct = (delta / baseline["design"]) * 100 if baseline["design"] != 0 else float("inf")
            lines.append(
                f"  Baseline: Design = {baseline['design']:.3g} {unit} "
                f"(Δ = {delta:+.3g} {unit}, {delta_pct:+.1f}%)"
            )
        if "previous" in baseline:
            delta = value - baseline["previous"]
            label = baseline.get("previous_label", "previous period")
            lines.append(
                f"  vs. {label}: {baseline['previous']:.3g} {unit} "
                f"(Δ = {delta:+.3g} {unit})"
            )
    
    # Domain context
    if domain_context:
        lines.append(f"  Context: {domain_context}")
    
    return "\n".join(lines)
```

---

*Congratulations on completing the Fundamentals section. You now understand:*
- *The ReAct loop and why it produces auditable, reliable analysis*
- *Clean Architecture and how it makes the system extensible and testable*
- *Physical domain modeling and why units are first-class citizens*
- *The structured data science workflow the agent follows*

*Continue to [02_anthropic_sdk_reference/](../02_anthropic_sdk_reference/) to learn the SDK mechanics that power the agent loop.*
