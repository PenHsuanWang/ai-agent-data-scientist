# Code Execution Tool — Deep Dive for Data Analysis

## 1. Why Code Execution Tool for Data Science

The `code_execution_20260120` server tool transforms Claude from a text-based assistant into an actual computational engine. For data science workflows, this unlocks:

- **Verified arithmetic**: No hallucinated statistics — all numbers are computed
- **Reproducible analysis**: The exact code Claude runs is visible in the response
- **No local environment setup**: No conda environments, no dependency hell, no version conflicts
- **Pre-installed scientific stack**: pandas, numpy, scipy, sklearn, matplotlib, seaborn, plotly all ready to use
- **Figures returned directly**: Matplotlib/plotly charts come back as image data
- **Stateful sessions**: Variables persist across multiple tool calls in the same conversation

**The key shift in paradigm**: Instead of Claude *describing* how to analyse data, Claude *performs* the analysis and reports verified results.

---

## 2. Enabling the Tool

### Basic Setup

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",          # Supports code execution
    max_tokens=8192,                     # Higher limit for analysis output
    tools=[
        {
            "type": "code_execution_20260120",
            "name": "code_execution"
        }
    ],
    messages=[{
        "role": "user",
        "content": "Calculate the mean and standard deviation of [1, 5, 3, 8, 4, 2, 7, 6]."
    }]
)
```

### With Files API (for uploading data)

```python
import anthropic

client = anthropic.Anthropic()

# First, upload your data file
with open("dataset.csv", "rb") as f:
    uploaded = client.beta.files.upload(
        file=("dataset.csv", f, "text/csv")
    )
file_id = uploaded.id

# Reference it in the message
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    tools=[
        {"type": "code_execution_20260120", "name": "code_execution"}
    ],
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Perform a full exploratory data analysis on this dataset:"},
            {
                "type": "container_upload",
                "file_id": file_id,
                "filename": "dataset.csv"   # Available at /files/dataset.csv
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
```

### Model Requirements

| Model | Code Execution | Programmatic Tool Calling |
|-------|:---:|:---:|
| `claude-opus-4-7` | ✅ | ✅ |
| `claude-sonnet-4-6` | ✅ | ✅ |
| `claude-haiku-4-5-20251001` | ✅ | ❌ (basic only) |

For programmatic tool calling (Section 8), use Sonnet 4.6 or Opus 4.7.

---

## 3. Pre-Installed Package Catalog

The container comes with these packages ready to import (no `pip install` needed):

### Data Manipulation
```python
import pandas as pd          # DataFrame operations, CSV/Excel/JSON I/O
import numpy as np           # Array operations, linear algebra, random
```

### Scientific Computing
```python
import scipy                 # Statistical tests, optimisation, signal processing
from scipy import stats      # t-tests, chi-square, correlation, distributions
from scipy import optimize   # Curve fitting, minimisation
from scipy import signal     # FFT, filtering, convolution
import sympy                 # Symbolic mathematics, calculus, algebra
import pint                  # Physical unit handling and conversion
```

### Machine Learning
```python
import sklearn               # scikit-learn: classification, regression, clustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
```

### Visualisation
```python
import matplotlib.pyplot as plt    # Core plotting
import matplotlib as mpl           # Configuration
import seaborn as sns              # Statistical visualisation
import plotly.express as px        # Interactive charts
import plotly.graph_objects as go  # Low-level plotly
```

### Other Available Packages
- `json`, `csv`, `pathlib`, `os`, `sys`, `re`, `math`, `statistics`, `collections`, `itertools` (standard library)
- `datetime`, `time`, `calendar`

### NOT Available
- `tensorflow`, `torch`, `keras` — deep learning frameworks are NOT installed
- `xgboost`, `lightgbm`, `catboost` — gradient boosting libraries NOT available
- `sqlalchemy`, `psycopg2` — database connectors NOT available
- `requests`, `httpx`, `aiohttp` — network access NOT available (no internet)
- Any package not listed above — `pip install` will fail

---

## 4. Container Environment

### Hardware Specs

```
CPU:    1 core (shared)
RAM:    5 GiB
Disk:   5 GiB (for files and generated outputs)
```

### Python Environment

```python
# Python 3.11+ (exact minor version may vary)
import sys
print(sys.version)  # 3.11.x
print(sys.path)     # Standard Python paths + /files directory
```

### Filesystem Structure

```
/
├── files/          ← Your uploaded files land here (via container_upload)
│   ├── dataset.csv
│   └── config.json
├── tmp/            ← Temporary files during execution
└── (standard Python paths)
```

Files Claude saves to disk (e.g., generated charts) persist within the session and can be downloaded.

### Network Access

**None.** The container has no internet access. You cannot:
- Make HTTP requests
- Connect to external databases
- Download data

All data must be provided via the Files API or directly in the prompt.

---

## 5. Session Persistence

The container persists state across tool calls within the **same conversation session**, subject to the idle timeout.

```
Conversation Turn 1: Claude imports pandas, loads CSV, defines df
                     → df now exists in container memory

Conversation Turn 2: Claude references df directly
                     → Works! No reload needed

[4.5 minutes of idle time pass]

Conversation Turn 3: Claude tries to reference df
                     → NameError: df not defined (container reset)
```

### Practical Implications

```python
# Turn 1: Load data (Claude does this)
import pandas as pd
import numpy as np

df = pd.read_csv('/files/sales_data.csv')
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(df.dtypes)

# Turn 2: Use loaded data (still in memory if < 4.5 min idle)
# Claude can write this directly
revenue_by_month = df.groupby('month')['revenue'].sum()
print(revenue_by_month)

# Turn 3: More analysis (state persists)
df['rolling_avg'] = df['revenue'].rolling(window=7).mean()
```

### Handling Container Resets

In long-running agents, guard against resets:

```python
system_prompt = (
    "If you encounter a NameError or see that previously-loaded data is not available, "
    "reload the data from /files/ before continuing. "
    "The uploaded files are always available at /files/[filename]."
)
```

---

## 6. Working with Data

### Uploading a CSV and Getting Results

```python
import anthropic
import base64
import json

client = anthropic.Anthropic()

def analyse_csv(csv_path: str, question: str) -> dict:
    """Upload a CSV and ask Claude to analyse it."""
    # Step 1: Upload file
    with open(csv_path, "rb") as f:
        uploaded = client.beta.files.upload(
            file=(csv_path.split("/")[-1], f, "text/csv")
        )

    # Step 2: Request analysis
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "container_upload",
                    "file_id": uploaded.id,
                    "filename": csv_path.split("/")[-1]
                }
            ]
        }],
        betas=["files-api-2025-04-14"]
    )

    # Step 3: Extract results
    result = {
        "analysis": "",
        "code_executed": [],
        "figures": [],
        "stop_reason": response.stop_reason,
    }

    for block in response.content:
        if block.type == "text":
            result["analysis"] = block.text
        elif block.type == "server_tool_use" and block.name == "code_execution":
            result["code_executed"].append(block.input.get("code", ""))
        elif block.type == "tool_result":
            # Check for generated images in tool results
            if isinstance(block.content, list):
                for item in block.content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        result["figures"].append(item)

    # Clean up uploaded file
    client.beta.files.delete(uploaded.id)

    return result


# Usage
result = analyse_csv(
    "sales_2025.csv",
    "Perform exploratory data analysis: distributions, correlations, and trends. "
    "Create visualisations for the key insights."
)
print(result["analysis"])
```

### Providing Data Inline (Small Datasets)

For small datasets (<100 rows), embed directly in the prompt:

```python
import pandas as pd
import io

df = pd.DataFrame({"month": range(1, 13), "revenue": [45000, 52000, 48000, 61000, 58000, 72000, 69000, 75000, 71000, 83000, 88000, 95000]})
csv_str = df.to_csv(index=False)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
    messages=[{
        "role": "user",
        "content": f"Analyse this monthly revenue data and forecast Q1 next year:\n\n```csv\n{csv_str}\n```"
    }]
)
```

---

## 7. Figure Generation

Claude can generate matplotlib and seaborn figures, which come back as base64-encoded image data in the response.

### How Figures Come Back

When Claude's code creates a matplotlib figure and calls `plt.show()` or `plt.savefig()`, the image is captured and returned as an image content block.

```python
# Claude generates this code:
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
ax1.hist(df['revenue'], bins=30, color='steelblue', edgecolor='white')
ax1.set_title('Revenue Distribution')
ax1.set_xlabel('Revenue ($)')

# Time series
ax2.plot(df['date'], df['revenue'], color='coral', linewidth=2)
ax2.set_title('Revenue Over Time')
ax2.set_xlabel('Date')

plt.tight_layout()
plt.show()  # This triggers image capture
```

### Extracting Figure Data

```python
import base64

def extract_figures(response) -> list[bytes]:
    """Extract all generated figure images from a code execution response."""
    figures = []
    for block in response.content:
        # Images may appear in tool_result blocks
        if hasattr(block, "content") and isinstance(block.content, list):
            for item in block.content:
                if isinstance(item, dict) and item.get("type") == "image":
                    source = item.get("source", {})
                    if source.get("type") == "base64":
                        img_bytes = base64.b64decode(source["data"])
                        figures.append(img_bytes)
    return figures

# Save figures to disk
figures = extract_figures(response)
for i, fig_bytes in enumerate(figures):
    with open(f"figure_{i}.png", "wb") as f:
        f.write(fig_bytes)
    print(f"Saved figure_{i}.png ({len(fig_bytes)} bytes)")
```

### Requesting Specific Chart Types

Guide Claude with explicit charting instructions:

```python
question = (
    "Create the following visualisations:\n"
    "1. A histogram of revenue distribution with a KDE overlay\n"
    "2. A correlation heatmap of all numeric columns\n"
    "3. A time-series plot with a 30-day rolling average\n"
    "Use seaborn for charts 1 and 2, matplotlib for chart 3. "
    "Use a consistent colour palette and add titles to all charts."
)
```

---

## 8. Programmatic Tool Calling

This is the most powerful feature of `code_execution_20260120`: Claude can call your custom Python functions from within its sandboxed code, without requiring extra API round-trips.

### The Execution Flow

```
Your process                     Anthropic sandbox (code_execution)
─────────────────                ─────────────────────────────────
1. Define tool with              3. Claude's generated Python:
   allowed_callers                  result = await validate_units(
                                        value=25.4, unit="km/h",
                                        expected_dimension="velocity"
                                    )
2. Send API request    →→→

                                 4. `await validate_units(...)` triggers
                                    call back to YOUR Python function

5. Your function runs ←←←        
   in your process
   Returns result     →→→        5. Result injected into sandbox

                                 6. Claude's code continues with result
                       ←←←
6. Final response                
   with all results
```

### Complete Setup Example

```python
import anthropic
import json

client = anthropic.Anthropic()

# --- Your custom tool implementations ---

def validate_physical_units(value: float, unit: str, expected_dimension: str) -> str:
    """Your domain tool — runs in YOUR process, not the sandbox."""
    DIMENSION_MAP = {
        "length": ["m", "km", "cm", "mm", "ft", "in", "miles"],
        "mass": ["kg", "g", "mg", "lb", "oz"],
        "time": ["s", "ms", "min", "hr", "h"],
        "temperature": ["K", "°C", "°F", "C", "F"],
        "energy": ["J", "kJ", "cal", "kcal", "eV", "kWh"],
        "velocity": ["m/s", "km/h", "mph", "knots"],
    }
    valid_units = DIMENSION_MAP.get(expected_dimension, [])
    if unit in valid_units:
        return json.dumps({"status": "valid", "value": value, "unit": unit})
    else:
        return json.dumps({
            "status": "invalid",
            "error": f"Unit '{unit}' is not valid for dimension '{expected_dimension}'",
            "valid_units": valid_units
        })


def lookup_conversion_factor(from_unit: str, to_unit: str) -> str:
    """Returns the multiplication factor to convert from_unit to to_unit."""
    FACTORS = {
        ("km", "m"): 1000,
        ("m", "km"): 0.001,
        ("km/h", "m/s"): 0.27778,
        ("m/s", "km/h"): 3.6,
        ("°C", "K"): None,  # Requires offset — not multiplicative
    }
    factor = FACTORS.get((from_unit, to_unit))
    if factor is None:
        return json.dumps({"error": f"Conversion from {from_unit} to {to_unit} not in registry"})
    return json.dumps({"from": from_unit, "to": to_unit, "factor": factor})


# --- Tool definitions with allowed_callers ---

TOOLS = [
    {
        "type": "code_execution_20260120",
        "name": "code_execution"
    },
    {
        "name": "validate_physical_units",
        "description": (
            "Validates that a numeric measurement uses the correct units for its physical dimension. "
            "Call this in a loop from code_execution to validate many measurements efficiently. "
            "Returns 'valid' or an error with the list of accepted units."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric value"},
                "unit": {"type": "string", "description": "The unit string, e.g. 'km/h', 'kg'"},
                "expected_dimension": {
                    "type": "string",
                    "enum": ["length", "mass", "time", "temperature", "energy", "velocity"],
                    "description": "The expected physical dimension"
                }
            },
            "required": ["value", "unit", "expected_dimension"]
        },
        "allowed_callers": ["code_execution_20260120"]   # ← Key field
    },
    {
        "name": "lookup_conversion_factor",
        "description": "Returns the multiplication factor to convert from one unit to another.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"}
            },
            "required": ["from_unit", "to_unit"]
        },
        "allowed_callers": ["code_execution_20260120"]
    }
]

# --- Client tool registry ---
CLIENT_TOOLS = {
    "validate_physical_units": validate_physical_units,
    "lookup_conversion_factor": lookup_conversion_factor,
}


def run_with_programmatic_tools(user_message: str) -> str:
    """Agentic loop supporting programmatic tool calling."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "tool_use":
            # Client tool calls (non-programmatic ones that Claude calls directly)
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = CLIENT_TOOLS.get(block.name)
                    result = fn(**block.input) if fn else f"Error: Unknown tool '{block.name}'"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "pause_turn":
            messages.append({"role": "user", "content": "Please continue."})

        else:
            raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")


# Example
measurements = [
    {"id": 1, "value": 25.4, "unit": "km/h", "dimension": "velocity"},
    {"id": 2, "value": 100.0, "unit": "kg", "dimension": "mass"},
    {"id": 3, "value": 9.8, "unit": "m/s2", "dimension": "velocity"},  # Wrong units
    {"id": 4, "value": 1000.0, "unit": "m", "dimension": "length"},
]

result = run_with_programmatic_tools(
    f"Validate the units in this measurement dataset: {json.dumps(measurements)}. "
    "Report which measurements have invalid units and why."
)
print(result)
```

---

## 9. Limitations

Understanding what the code execution tool **cannot** do is as important as knowing what it can:

| Limitation | Details |
|-----------|---------|
| No internet | Cannot make HTTP requests, no pip install |
| No persistence beyond timeout | 4.5-minute idle timeout; restart resets state |
| No arbitrary packages | Only pre-installed packages available |
| No Jupyter | No IPython magic, no `%matplotlib inline` |
| Not on Bedrock/Vertex | Anthropic API and Azure only |
| Not ZDR-eligible | Data may be retained |
| 1 CPU | No parallel computation within container |
| 5 GiB RAM | Large datasets may cause OOM errors |
| No GPU | No CUDA, no hardware acceleration |

---

## 10. Comparison with Local Execution Alternatives

| Feature | Cloud Code Execution | Local SubprocessRunner | JupyterBridge |
|---------|:--------------------:|:----------------------:|:-------------:|
| Setup required | None | Python + subprocess | Jupyter server |
| Internet access | ❌ | ✅ | ✅ |
| Custom packages | ❌ | ✅ | ✅ |
| State persistence | Session-scoped | Process-scoped | Kernel-scoped |
| Security isolation | ✅ Sandboxed | ❌ Host process | ⚠️ Partial |
| GPU support | ❌ | ✅ (if available) | ✅ (if available) |
| Reproducibility | ✅ Fixed env | ⚠️ Depends on env | ⚠️ Depends on env |
| Cross-platform | ✅ Always | ✅ | ✅ |
| Production safety | ✅ | ⚠️ Risk of harm | ⚠️ Risk of harm |
| ZDR eligible | ❌ | ✅ | ✅ |
| Cost | Token-based | Compute-based | Compute-based |

**Decision guide:**
- Use **Cloud Code Execution** for: quick analyses, demos, when environment consistency matters, when security is critical
- Use **Local SubprocessRunner** for: custom packages, GPU workloads, internet-required tasks, ZDR requirements
- Use **JupyterBridge** for: interactive development workflows, when you want to inspect intermediate state

---

## 11. Production Integration Pattern

```python
import anthropic
import base64
import os
import time
from pathlib import Path

class DataAnalysisSession:
    """
    Manages a code execution session for data analysis.
    Handles file uploads, session state, and result extraction.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages = []
        self.uploaded_file_ids = []
        self.session_start = time.time()

    def upload_file(self, local_path: str) -> str:
        """Upload a file and track its ID for cleanup."""
        path = Path(local_path)
        mime_type = "text/csv" if path.suffix == ".csv" else "application/octet-stream"

        with open(local_path, "rb") as f:
            uploaded = self.client.beta.files.upload(
                file=(path.name, f, mime_type)
            )
        self.uploaded_file_ids.append(uploaded.id)
        return uploaded.id

    def add_file_to_conversation(self, file_id: str, filename: str):
        """Stage a file for the next message."""
        if not hasattr(self, "_pending_files"):
            self._pending_files = []
        self._pending_files.append({"file_id": file_id, "filename": filename})

    def send(self, user_text: str, tools: list = None) -> dict:
        """Send a message and run the agentic loop."""
        content = [{"type": "text", "text": user_text}]

        # Attach any pending files
        if hasattr(self, "_pending_files"):
            for pf in self._pending_files:
                content.append({
                    "type": "container_upload",
                    "file_id": pf["file_id"],
                    "filename": pf["filename"]
                })
            self._pending_files = []

        self.messages.append({"role": "user", "content": content})

        default_tools = [{"type": "code_execution_20260120", "name": "code_execution"}]
        active_tools = tools or default_tools

        # Run loop
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                tools=active_tools,
                messages=self.messages,
                betas=["files-api-2025-04-14"],
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return self._extract_result(response)

            elif response.stop_reason == "pause_turn":
                self.messages.append({"role": "user", "content": "Please continue."})

            else:
                raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")

    def _extract_result(self, response) -> dict:
        result = {"text": "", "figures": [], "code": []}
        for block in response.content:
            if block.type == "text":
                result["text"] = block.text
            elif block.type == "server_tool_use" and block.name == "code_execution":
                result["code"].append(block.input.get("code", ""))
        return result

    def cleanup(self):
        """Delete uploaded files."""
        for file_id in self.uploaded_file_ids:
            try:
                self.client.beta.files.delete(file_id)
            except Exception:
                pass
        self.uploaded_file_ids = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
```

---

## 12. Complete Working Example

Full end-to-end: upload CSV → analyse → get figure → interpret results.

```python
import anthropic
import base64
import csv
import io
import random

# --- Generate sample data ---
def generate_sample_csv() -> str:
    """Generate a synthetic sales CSV as a string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "product", "region", "revenue", "units"])
    random.seed(42)

    products = ["Widget A", "Widget B", "Gadget X"]
    regions = ["North", "South", "East", "West"]

    for month in range(1, 13):
        for _ in range(50):
            revenue = random.normalvariate(5000, 1500) * (1 + month * 0.05)
            writer.writerow([
                f"2025-{month:02d}-{random.randint(1,28):02d}",
                random.choice(products),
                random.choice(regions),
                round(max(0, revenue), 2),
                random.randint(1, 100)
            ])
    return output.getvalue()


def full_analysis_workflow():
    client = anthropic.Anthropic()

    # Step 1: Prepare data
    csv_data = generate_sample_csv()
    csv_bytes = csv_data.encode("utf-8")

    # Step 2: Upload file
    uploaded = client.beta.files.upload(
        file=("sales_2025.csv", io.BytesIO(csv_bytes), "text/csv")
    )
    print(f"Uploaded file: {uploaded.id}")

    # Step 3: Run analysis
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Perform a comprehensive analysis of this sales dataset:\n"
                        "1. Load the CSV and display shape and dtypes\n"
                        "2. Compute monthly revenue totals and show the growth trend\n"
                        "3. Compare revenue by product and by region using groupby\n"
                        "4. Create a 2x2 figure: (a) monthly revenue bar chart, "
                        "(b) product revenue pie chart, (c) regional revenue heatmap, "
                        "(d) revenue distribution histogram\n"
                        "5. Identify the top 3 product-region combinations by revenue\n"
                        "6. Write a brief executive summary of the key findings"
                    )
                },
                {
                    "type": "container_upload",
                    "file_id": uploaded.id,
                    "filename": "sales_2025.csv"
                }
            ]
        }],
        betas=["files-api-2025-04-14"]
    )

    # Step 4: Extract results
    analysis_text = ""
    code_executed = []
    figures_base64 = []

    for block in response.content:
        if block.type == "text":
            analysis_text = block.text
        elif block.type == "server_tool_use" and block.name == "code_execution":
            code_executed.append(block.input.get("code", ""))

    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(analysis_text)

    print(f"\nCode blocks executed: {len(code_executed)}")
    for i, code in enumerate(code_executed):
        print(f"\n--- Code block {i+1} ---")
        print(code[:200] + ("..." if len(code) > 200 else ""))

    # Step 5: Cleanup
    client.beta.files.delete(uploaded.id)
    print(f"\nCleaned up file {uploaded.id}")

    print(f"\nStop reason: {response.stop_reason}")
    print(f"Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


if __name__ == "__main__":
    full_analysis_workflow()
```
