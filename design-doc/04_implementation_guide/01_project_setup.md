# Project Setup — Extending ai-agent-mvp

## Overview

This guide extends the existing `ai-agent-mvp` FastAPI application into a full **Data Scientist AI-Agent** with:
- Multi-step ReAct reasoning loop
- Safe Python code execution (subprocess, Jupyter, or Anthropic sandbox)
- Physical unit validation via `pint`
- Domain knowledge retrieval
- Figure generation and notebook export

### Alignment with `antropic_python_sdk_build_from_scratch_tutorial`

This implementation stays on top of the base tutorial architecture:
- Keep existing MVP flow untouched: `POST /api/v1/chat` + `AgentService` + `tools.py`
- Extend session handling in `app/services/memory.py` (do not introduce a parallel session store module)
- Add new capability as parallel modules: `data_agent.py`, analysis/datasets routers, and infrastructure backends
- Preserve dependency direction from Chapter 03: Presentation → Application → Domain, Infrastructure → Domain

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | `python --version` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Anthropic API Key | — | `https://console.anthropic.com` |
| Jupyter (optional) | 7+ | Only needed for `CODE_EXECUTION_BACKEND=jupyter` |
| Git | 2.40+ | — |

Check your environment:

```bash
python --version          # Python 3.12.x
uv --version              # uv 0.4.x
echo $ANTHROPIC_API_KEY   # sk-ant-...
```

---

## 2. Clone and Setup

```bash
# Clone the MVP repository (or use your existing checkout)
git clone https://github.com/your-org/ai-agent-mvp.git
cd ai-agent-mvp

# Install all dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Verify FastAPI + Anthropic SDK are present
python -c "import fastapi, anthropic; print('Base dependencies OK')"
```

---

## 3. New Dependencies

Edit `pyproject.toml` to add the data-science stack. The full `[project]` section should look like:

```toml
[project]
name = "data-scientist-agent"
version = "0.2.0"
description = "AI Data Scientist Agent built on FastAPI + Anthropic SDK"
requires-python = ">=3.12"
dependencies = [
    # --- Existing MVP dependencies ---
    "fastapi>=0.115",
    "anthropic>=0.40.0",
    "pydantic-settings>=2.0",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",

    # --- Phase 1: Physical unit validation ---
    "pint>=0.24",              # Quantity + unit registry, dimensional analysis

    # --- Phase 2: Code execution backends ---
    "jupyter_client>=8.0",     # Jupyter kernel bridge (optional backend)
    "nbformat>=5.10",          # Build and export .ipynb notebook files

    # --- Phase 3: Data loading ---
    "pandas>=2.0",             # DataFrame API, CSV/Excel/Parquet/HDF5 I/O
    "numpy>=1.26",             # Numerical arrays, vectorised operations

    # --- Phase 4: Visualisation ---
    "matplotlib>=3.8",         # Figure generation (Agg backend, no display needed)
    "seaborn>=0.13",           # Statistical plots on top of matplotlib

    # --- Phase 5: File format support ---
    "openpyxl>=3.1",           # pandas Excel (.xlsx) read/write engine
    "pyarrow>=14.0",           # pandas Parquet engine
    "tables>=3.9",             # pandas HDF5 engine (PyTables)
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",             # FastAPI TestClient async support
    "ipykernel>=6.29",         # Register Python 3 Jupyter kernel for tests
]
```

Install the new dependencies:

```bash
uv sync
# uv resolves + installs everything listed above in one pass

# For Jupyter kernel support (optional):
uv add --dev ipykernel
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

---

## 4. Directory Structure

Create all new directories and `__init__.py` files:

```bash
# Inside the project root (where app/ lives)
mkdir -p app/infrastructure
mkdir -p data/domain_docs
mkdir -p data/datasets
mkdir -p outputs/figures
mkdir -p outputs/notebooks

# __init__.py files so Python treats them as packages
touch app/infrastructure/__init__.py
touch app/services/__init__.py    # may already exist

# New source files (empty — filled in later phases)
touch app/domain/analysis_models.py
touch app/services/data_agent.py
touch app/services/knowledge_tools.py
touch app/services/data_tools.py
touch app/infrastructure/code_runner.py
touch app/infrastructure/jupyter_bridge.py
touch app/infrastructure/anthropic_code_exec.py
touch app/infrastructure/unit_registry.py
touch app/api/v1/analysis.py
touch app/api/v1/datasets.py
```

Final tree (new items marked `★`):

```
project-root/
├── app/
│   ├── domain/
│   │   ├── models.py                   # EXISTING — AgentSession
│   │   ├── analysis_models.py          # ★ NEW
│   │   └── exceptions.py              # EXTEND
│   ├── services/
│   │   ├── agent.py                    # EXISTING — untouched
│   │   ├── data_agent.py              # ★ NEW
│   │   ├── tools.py                    # EXISTING — untouched
│   │   ├── knowledge_tools.py         # ★ NEW
│   │   └── data_tools.py              # ★ NEW
│   ├── infrastructure/                # ★ NEW directory
│   │   ├── __init__.py
│   │   ├── code_runner.py
│   │   ├── jupyter_bridge.py
│   │   ├── anthropic_code_exec.py
│   │   └── unit_registry.py
│   ├── api/v1/
│   │   ├── chat.py                     # EXISTING — untouched
│   │   ├── analysis.py                # ★ NEW
│   │   └── datasets.py                # ★ NEW
│   └── core/
│       └── config.py                  # EXTEND
├── data/
│   ├── domain_docs/                   # ★ Markdown knowledge base
│   └── datasets/                      # ★ CSV/Excel/Parquet/HDF5 files
├── outputs/
│   ├── figures/                       # ★ Saved PNG exports
│   └── notebooks/                     # ★ Exported .ipynb files
├── tests/
├── pyproject.toml
└── .env
```

---

## 5. Environment Variables

Create or extend `.env` in the project root:

```bash
# ── Existing MVP settings ──────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
APP_NAME="Data Scientist Agent"
DEBUG=false

# ── Model configuration ────────────────────────────────────────────────────
# Use the latest Claude Sonnet model
CLAUDE_MODEL=claude-sonnet-4-6

# Maximum tokens per Claude response (8192 allows detailed analyses)
MAX_TOKENS=8192

# ── Data directories ───────────────────────────────────────────────────────
# Root directory where datasets and domain documents are stored
DATA_DIR=data

# ── Code execution backend ────────────────────────────────────────────────
# subprocess: safe, isolated, no state persistence between calls
# jupyter:    stateful kernel, figures via display_data, requires ipykernel
# anthropic:  Anthropic-hosted sandbox (requires beta header)
CODE_EXECUTION_BACKEND=subprocess

# Seconds before a code execution is killed (subprocess/jupyter)
CODE_EXECUTION_TIMEOUT=30

# Enable the Jupyter kernel bridge (ignored when backend != jupyter)
ENABLE_JUPYTER_BRIDGE=false

# ── ReAct loop controls ───────────────────────────────────────────────────
# Hard limit on Thought/Action/Observation iterations before giving up
MAX_REACT_ITERATIONS=20

# ── Output directories ────────────────────────────────────────────────────
FIGURES_DIR=outputs/figures
NOTEBOOKS_DIR=outputs/notebooks
```

Load settings in `app/core/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Existing
    anthropic_api_key: str
    app_name: str = "Data Scientist Agent"
    debug: bool = False

    # Model
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192

    # Data
    data_dir: Path = Path("data")

    # Code execution
    code_execution_backend: str = "subprocess"   # subprocess | jupyter | anthropic
    code_execution_timeout: int = 30
    enable_jupyter_bridge: bool = False

    # ReAct
    max_react_iterations: int = 20

    # Outputs
    figures_dir: Path = Path("outputs/figures")
    notebooks_dir: Path = Path("outputs/notebooks")

settings = Settings()
```

---

## 6. Verify Installation

Create `scripts/verify_install.py` and run it:

```python
#!/usr/bin/env python3
"""Verify all data-scientist-agent dependencies are importable."""

import sys
import importlib

REQUIRED = [
    ("fastapi", "FastAPI web framework"),
    ("anthropic", "Anthropic Python SDK"),
    ("pydantic_settings", "Pydantic Settings v2"),
    ("uvicorn", "ASGI server"),
    ("pint", "Physical unit validation"),
    ("pandas", "DataFrame library"),
    ("numpy", "Numerical computing"),
    ("matplotlib", "Figure generation"),
    ("seaborn", "Statistical plots"),
    ("openpyxl", "Excel file support"),
    ("pyarrow", "Parquet file support"),
    ("tables", "HDF5 file support"),
    ("nbformat", "Jupyter notebook format"),
]

OPTIONAL = [
    ("jupyter_client", "Jupyter kernel bridge"),
    ("ipykernel", "Python3 Jupyter kernel"),
]

def check(module_name: str, description: str, required: bool = True) -> bool:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "?")
        print(f"  ✅ {module_name:<22} {version:<12} — {description}")
        return True
    except ImportError as e:
        mark = "❌" if required else "⚠️ "
        label = "MISSING" if required else "optional"
        print(f"  {mark} {module_name:<22} {label:<12} — {description}")
        if required:
            print(f"     Fix: uv add {module_name}")
        return not required

print("\n=== Data Scientist Agent — Dependency Check ===\n")
print("Required:")
ok = all(check(m, d) for m, d in REQUIRED)
print("\nOptional (Jupyter bridge):")
for m, d in OPTIONAL:
    check(m, d, required=False)

# Quick pint sanity check
try:
    import pint
    ureg = pint.UnitRegistry()
    q = 57.3 * ureg.degC
    q_f = q.to("degF")
    print(f"\n  ✅ pint unit conversion: {q} → {q_f:.2f}")
except Exception as e:
    print(f"\n  ❌ pint sanity check failed: {e}")

# Quick matplotlib non-display check
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    plt.close(fig)
    print("  ✅ matplotlib Agg backend works (no display needed)")
except Exception as e:
    print(f"  ❌ matplotlib check failed: {e}")

print()
if ok:
    print("✅ All required dependencies satisfied. Ready to implement!\n")
    sys.exit(0)
else:
    print("❌ Some required dependencies are missing. Run: uv sync\n")
    sys.exit(1)
```

```bash
python scripts/verify_install.py
```

Expected output:
```
=== Data Scientist Agent — Dependency Check ===

Required:
  ✅ fastapi               0.115.x      — FastAPI web framework
  ✅ anthropic             0.40.x       — Anthropic Python SDK
  ...
  ✅ pint                  0.24.x       — Physical unit validation

Optional (Jupyter bridge):
  ⚠️  jupyter_client        optional     — Jupyter kernel bridge

  ✅ pint unit conversion: 57.3 degree_Celsius → 135.14 degree_Fahrenheit
  ✅ matplotlib Agg backend works (no display needed)

✅ All required dependencies satisfied. Ready to implement!
```

---

## 7. Sample Data Setup

Run this script once to populate `data/` with fixtures for development and testing:

```python
#!/usr/bin/env python3
"""Generate sample datasets and domain documents for development."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data")
(BASE / "domain_docs").mkdir(parents=True, exist_ok=True)
(BASE / "datasets").mkdir(parents=True, exist_ok=True)
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/notebooks").mkdir(parents=True, exist_ok=True)

# ── 1. power_plant_data.csv ───────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 500
timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
df = pd.DataFrame({
    "timestamp": timestamps,
    "steam_temp_C": rng.normal(540, 5, n).round(2),
    "steam_pressure_MPa": rng.normal(16.5, 0.3, n).round(3),
    "gross_power_MW": rng.normal(620, 15, n).round(2),
    "heat_rate_kJ_kWh": rng.normal(9800, 120, n).round(1),
    "flue_gas_temp_C": rng.normal(135, 8, n).round(2),
    "co2_emission_g_kWh": rng.normal(820, 25, n).round(1),
    "efficiency_pct": (3600 / rng.normal(9800, 120, n) * 100).round(3),
})
df.to_csv(BASE / "datasets/power_plant_data.csv", index=False)
print(f"✅ Created power_plant_data.csv ({len(df)} rows)")

# ── 2. turbine_efficiency.parquet ─────────────────────────────────────────
df2 = pd.DataFrame({
    "load_pct": np.linspace(40, 100, 100),
    "isentropic_efficiency": 0.88 - 0.0008 * (np.linspace(40, 100, 100) - 100) ** 2 / 100,
    "mechanical_efficiency": rng.uniform(0.985, 0.995, 100).round(4),
    "stage": rng.choice(["HP", "IP", "LP"], 100),
})
df2.to_parquet(BASE / "datasets/turbine_efficiency.parquet", index=False)
print("✅ Created turbine_efficiency.parquet")

# ── 3. sensor_readings.xlsx ───────────────────────────────────────────────
df3 = pd.DataFrame({
    "sensor_id": [f"S{i:04d}" for i in range(1, 51)],
    "location": rng.choice(["boiler", "turbine", "condenser", "cooling_tower"], 50),
    "temp_K": rng.normal(800, 50, 50).round(1),
    "pressure_kPa": rng.normal(1500, 100, 50).round(1),
    "flow_kg_s": rng.normal(200, 20, 50).round(2),
})
df3.to_excel(BASE / "datasets/sensor_readings.xlsx", index=False)
print("✅ Created sensor_readings.xlsx")

# ── 4. domain_docs/power_plant_thermodynamics.md ─────────────────────────
thermo_doc = """# Power Plant Thermodynamics

## Rankine Cycle Overview

A steam power plant converts heat energy to electrical energy via the Rankine cycle.
The four main processes are:
1. **Pump** (1→2): Isentropic compression of liquid water
2. **Boiler** (2→3): Constant-pressure heat addition, water → steam
3. **Turbine** (3→4): Isentropic expansion producing shaft work
4. **Condenser** (4→1): Constant-pressure heat rejection

## Key Performance Indicators

| Quantity | Symbol | Typical Range | Unit |
|---|---|---|---|
| Steam temperature (HP) | T_s | 520–600 | °C |
| Steam pressure (HP) | P_s | 14–20 | MPa |
| Gross power output | P | 500–1200 | MW |
| Thermal efficiency | η | 35–48 | % |
| Heat rate | HR | 7500–10500 | kJ/kWh |
| Flue gas temperature | T_fg | 110–160 | °C |

## Efficiency Calculations

Thermal efficiency: η = W_net / Q_in = 3600 / HR

Where:
- W_net: net electrical output [kJ/kWh]
- Q_in: fuel heat input [kJ/kWh]
- HR: heat rate [kJ/kWh] — lower is better

## Physical Constraints

- Carnot limit sets maximum theoretical efficiency
- HP steam temperature bounded by metallurgy: 620°C max for conventional steel
- Condenser pressure typically 4–10 kPa (saturation temperature 28–46°C)
- CO₂ emissions: 750–900 g/kWh for modern coal plants

## Unit Definitions

- 1 MW = 1000 kW = 1,000,000 W
- 1 MPa = 10 bar = 145.04 psi
- η [%] = 3600 / HR [kJ/kWh] × 100
"""
(BASE / "domain_docs/power_plant_thermodynamics.md").write_text(thermo_doc)
print("✅ Created power_plant_thermodynamics.md")

# ── 5. domain_docs/unit_definitions.md ───────────────────────────────────
units_doc = """# Unit Definitions for Power Plant Analysis

## Temperature
- °C (Celsius): operational temperature scale; 0°C = 273.15 K
- K (Kelvin): thermodynamic temperature; T[K] = T[°C] + 273.15
- °F (Fahrenheit): T[°F] = T[°C] × 9/5 + 32

## Pressure
- Pa (Pascal): SI base unit of pressure; 1 Pa = 1 N/m²
- kPa: 1000 Pa; condenser pressure typically 4–10 kPa
- MPa: 10⁶ Pa; HP steam pressure typically 14–20 MPa
- bar: 100 kPa; 1 bar ≈ 1 atm

## Power and Energy
- W (Watt): 1 J/s
- kW, MW, GW: 10³, 10⁶, 10⁹ W
- kWh: energy = power × time; 1 kWh = 3600 kJ
- kJ/kWh: heat rate units (fuel input per electrical output)

## Mass Flow
- kg/s: steam/water mass flow rate
- t/h: tonnes per hour; 1 t/h = 1000/3600 kg/s ≈ 0.2778 kg/s

## Efficiency
- Dimensionless (0–1) or percentage (0–100%)
- Isentropic efficiency: ratio of actual to ideal work
- Thermal efficiency: ratio of net work to heat input
"""
(BASE / "domain_docs/unit_definitions.md").write_text(units_doc)
print("✅ Created unit_definitions.md")

print("\n✅ Sample data setup complete.")
print(f"   Domain docs: {list((BASE / 'domain_docs').glob('*.md'))}")
print(f"   Datasets:    {list((BASE / 'datasets').glob('*'))}")
```

```bash
python scripts/create_sample_data.py
```

---

## 8. Running the Agent

### Start the server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### First API call — health check

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### First analysis request

```bash
curl -s -X POST http://localhost:8000/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Load the power plant dataset and compute the mean thermal efficiency.",
    "data_dir": "data"
  }' | python3 -m json.tool
```

Expected response shape:

```json
{
  "response": "The mean thermal efficiency of the power plant is 36.73%...",
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "completed",
  "react_trace": [
    {
      "thought": "I need to load the power plant dataset first.",
      "action": "list_datasets",
      "action_input": {},
      "observation": "[\"power_plant_data.csv\", ...]"
    }
  ],
  "figures": [],
  "notebook_available": false
}
```

---

## Next Steps

→ **Phase 1**: [02_phase1_foundation.md](02_phase1_foundation.md) — Domain models and configuration  
→ **Phase 2**: [03_phase2_tools.md](03_phase2_tools.md) — Tool suite implementation  
→ **Phase 3**: [04_phase3_react_engine.md](04_phase3_react_engine.md) — ReAct loop engine  
