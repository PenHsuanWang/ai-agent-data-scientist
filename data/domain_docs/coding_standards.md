# Data Scientist AI Agent — Coding Standards & Analysis Playbook

This document defines the mandatory coding conventions, analysis workflow, and
visualization format specification that the agent must follow for every task.

---

## 1. Python Coding Style

### 1.1 General Conventions

| Rule | Standard |
|------|----------|
| Style guide | PEP 8 strict |
| Line length | 100 characters max |
| Indentation | 4 spaces (no tabs) |
| Naming — variables, functions | `snake_case` |
| Naming — constants | `UPPER_SNAKE_CASE` |
| Naming — classes | `PascalCase` |
| String literals | f-strings preferred; no `%` formatting |
| Type hints | Required on all function signatures |
| Docstrings | Google style for functions with > 2 params |

### 1.2 Import Order

```python
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third-party (alphabetical)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 3. Local application
from app.core.config import settings
```

### 1.3 Pandas Best Practices

```python
# ✅ Preferred — explicit, readable
df = (
    df
    .dropna(subset=['efficiency_pct'])
    .assign(efficiency_frac=lambda x: x['efficiency_pct'] / 100)
    .query('efficiency_frac < 1.0')               # First Law check
    .rename(columns={'gross_power_MW': 'P_gross'})
)

# ✅ Use .loc for label-based indexing
df.loc[df['temperature_C'] > 600, 'status'] = 'over_limit'

# ✅ Name intermediate results — no single-letter variables for dataframes
power_df = df[['timestamp', 'gross_power_MW', 'net_power_MW']].copy()
eff_by_load = df.groupby('load_pct', observed=True)['efficiency_pct'].mean()

# ❌ Avoid — ambiguous chaining without parens, magic numbers
df2 = df[df['x'] > 0.35]['y']   # unclear intent
```

### 1.4 NumPy Best Practices

```python
# ✅ Use descriptive variable names for physics quantities
heat_rate = np.array(df['heat_rate_kJ_kWh'])
efficiency_pct = 3600 / heat_rate * 100          # Rankine efficiency formula

# ✅ Named constants for physical limits
EFFICIENCY_MAX_PCT = 100.0      # First Law upper bound
TEMP_ABSOLUTE_ZERO_C = -273.15  # Third Law lower bound
COAL_HEAT_RATE_MIN = 7_000      # kJ/kWh — modern supercritical coal plant

# ✅ Validate before reporting
assert efficiency_pct.max() <= EFFICIENCY_MAX_PCT, (
    f"Efficiency {efficiency_pct.max():.1f}% exceeds 100% — check unit conversion"
)
```

### 1.5 Print Formatting

Always use `print()` for output — never rely on expression evaluation.

```python
# ✅ Structured print output
print(f"Dataset shape: {df.shape}")
print(f"Mean efficiency: {df['efficiency_pct'].mean():.2f} %")
print(f"Peak power: {df['gross_power_MW'].max():.1f} MW")
print(f"\nDescriptive statistics:\n{df.describe().round(2)}")

# ✅ Section headers for readability
print("=" * 60)
print("THERMAL EFFICIENCY ANALYSIS")
print("=" * 60)
```

---

## 2. Analysis Strategy (7-Step Workflow)

Every analysis task MUST follow this ordered workflow. Do not skip steps.

### Step 1 — Understand the Question
- Identify the target quantity (efficiency? power? emissions?)
- Identify required columns and units
- Read domain documents to understand physical context

### Step 2 — Explore the Data (EDA)
```python
print(f"Shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nSummary statistics:\n{df.describe().round(3)}")
```

### Step 3 — Clean and Preprocess
```python
# Handle nulls
df = df.dropna(subset=['critical_column'])    # or fillna() with domain justification

# Remove physical impossibilities
df = df[df['efficiency_pct'].between(0, 100)]
df = df[df['temperature_C'] >= -273.15]

# Fix dtypes
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['load_pct'] = pd.to_numeric(df['load_pct'], errors='coerce')
```

### Step 4 — Compute and Analyse
- Perform aggregations, correlations, groupbys
- Use descriptive variable names for intermediate results
- Print each computed metric with its unit

### Step 5 — Validate Physical Units
- Call `validate_physical_units` for EVERY computed efficiency, temperature, pressure, or power
- If `is_valid = false`, investigate before continuing
- Call `check_magnitude` if a result seems unusual

### Step 6 — Visualise (see Section 3 for full spec)
- Create at least one plot per analysis
- Always call `plt.show()` to capture the figure

### Step 7 — Interpret and Conclude
- State numeric findings with units and uncertainty
- Cross-reference with domain document ranges
- Flag any anomalies or potential data quality issues
- End with a plain-language summary

---

## 3. Visualization Format Specification

### 3.1 Pre-Configured Helpers (auto-injected)

The following are available in every execution without importing:

| Helper | Purpose |
|--------|---------|
| `COLORS` | Named color dict — `COLORS['blue']`, `COLORS['orange']`, etc. |
| `PALETTE` | List of 8 colors — used by seaborn automatically |
| `C_GOOD` / `C_WARN` / `C_BAD` / `C_NEUTRAL` | Semantic status colors |
| `label_bars(ax)` | Add value labels on bar charts |
| `add_reference_line(ax, value, label)` | Draw horizontal spec/limit line |
| `format_axis_units(ax, xlabel, ylabel, title)` | Set labels and call tight_layout |
| `engineering_plot(nrows, ncols, title)` | Pre-styled figure factory |

### 3.2 Mandatory Plot Elements

Every plot MUST include:

```python
# ✅ Always required
ax.set_xlabel('Variable Name [unit]')      # unit in square brackets
ax.set_ylabel('Variable Name [unit]')
ax.set_title('Descriptive Title — Dataset Name')

# ✅ Required for multi-series plots
ax.legend(loc='best')

# ✅ Required at end — enables figure capture
plt.tight_layout()
plt.show()
```

### 3.3 Plot Type Recipes

#### Time Series
```python
fig, ax = engineering_plot(title='Power Output Over Time')
ax.plot(df['timestamp'], df['gross_power_MW'],
        color=COLORS['blue'], label='Gross Power')
ax.plot(df['timestamp'], df['net_power_MW'],
        color=COLORS['orange'], label='Net Power', linestyle='--')
ax.fill_between(df['timestamp'],
                df['net_power_MW'], df['gross_power_MW'],
                alpha=0.15, color=COLORS['bluegrey'], label='Aux. Load')
format_axis_units(ax, xlabel='Time', ylabel='Power [MW]',
                  title='Gross vs Net Power Output')
ax.legend()
plt.show()
```

#### Distribution (Histogram + KDE)
```python
fig, ax = engineering_plot(figsize=(10, 6))
ax.hist(df['efficiency_pct'], bins=30,
        color=COLORS['blue'], alpha=0.7, edgecolor='white', label='Frequency')
ax.axvline(df['efficiency_pct'].mean(), color=C_WARN,
           linestyle='--', linewidth=2, label=f"Mean: {df['efficiency_pct'].mean():.1f}%")
add_reference_line(ax, 100, label='First Law limit (100%)', color=C_BAD)
format_axis_units(ax, xlabel='Thermal Efficiency [%]', ylabel='Count',
                  title='Efficiency Distribution')
ax.legend()
plt.show()
```

#### Correlation Heatmap
```python
fig, ax = engineering_plot(figsize=(10, 8))
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()
sns.heatmap(
    corr, annot=True, fmt='.2f', cmap='coolwarm',
    center=0, linewidths=0.5, ax=ax,
    annot_kws={'size': 9}
)
ax.set_title('Correlation Matrix — Numeric Variables')
plt.tight_layout()
plt.show()
```

#### Bar Chart (Grouped Comparison)
```python
fig, ax = engineering_plot(figsize=(10, 6))
categories = df.groupby('stage')['isentropic_efficiency'].mean()
bars = ax.bar(categories.index, categories.values,
              color=PALETTE[:len(categories)], edgecolor='white', linewidth=0.8)
label_bars(ax, fmt='{:.2f}')
format_axis_units(ax, xlabel='Turbine Stage',
                  ylabel='Isentropic Efficiency [-]',
                  title='Mean Isentropic Efficiency by Stage')
plt.show()
```

#### Scatter with Regression
```python
fig, ax = engineering_plot(figsize=(10, 7))
sns.regplot(data=df, x='heat_rate_kJ_kWh', y='efficiency_pct',
            ax=ax, scatter_kws={'alpha': 0.4, 's': 20, 'color': COLORS['blue']},
            line_kws={'color': COLORS['orange'], 'linewidth': 2})
format_axis_units(ax, xlabel='Heat Rate [kJ/kWh]',
                  ylabel='Thermal Efficiency [%]',
                  title='Efficiency vs Heat Rate — Rankine Cycle')
plt.show()
```

#### Subplots Dashboard
```python
fig, axes = engineering_plot(nrows=2, ncols=2,
                              title='Power Plant Performance Dashboard')
# top-left
axes[0, 0].plot(df['timestamp'], df['gross_power_MW'], color=COLORS['blue'])
format_axis_units(axes[0, 0], ylabel='Power [MW]', title='Gross Output')

# top-right
axes[0, 1].hist(df['efficiency_pct'], bins=25, color=COLORS['green'], edgecolor='white')
format_axis_units(axes[0, 1], xlabel='Efficiency [%]', title='Efficiency Distribution')

# bottom-left
sns.boxplot(data=df, x='stage', y='isentropic_efficiency',
            palette=PALETTE, ax=axes[1, 0])
format_axis_units(axes[1, 0], ylabel='η_is [-]', title='Efficiency by Stage')

# bottom-right
axes[1, 1].scatter(df['steam_temp_C'], df['efficiency_pct'],
                    alpha=0.3, s=12, color=COLORS['purple'])
format_axis_units(axes[1, 1], xlabel='Steam Temp [°C]',
                  ylabel='Efficiency [%]', title='Temp vs Efficiency')

plt.tight_layout()
plt.show()
```

### 3.4 Figure Sizing Guidelines

| Plot type | `figsize` |
|-----------|-----------|
| Single time-series or scatter | `(12, 7)` ← default |
| Histogram or distribution | `(10, 6)` |
| Correlation heatmap | `(10, 8)` |
| 2×2 dashboard | `(14, 10)` |
| 1×2 side-by-side comparison | `(14, 6)` |
| 3×1 tall multi-panel | `(10, 14)` |

### 3.5 Anti-Patterns to Avoid

```python
# ❌ No axis labels or title
ax.plot(x, y)
plt.show()

# ❌ No units in axis labels
ax.set_xlabel('Temperature')       # missing unit

# ❌ Default matplotlib blue for everything — use COLORS dict
ax.plot(x1, y1)
ax.plot(x2, y2)

# ❌ Relying on Jupyter display instead of plt.show()
df.head()                          # use print(df.head()) instead

# ❌ fig.savefig() — the runner captures via plt.show() automatically
fig.savefig('output.png')          # do NOT do this; use plt.show()
```

---

## 4. Reporting Standards

### 4.1 Numeric Precision

| Quantity | Decimal places |
|----------|---------------|
| Efficiency (%) | 2 — e.g. `38.45 %` |
| Temperature (°C) | 1 — e.g. `541.2 °C` |
| Pressure (MPa) | 3 — e.g. `16.450 MPa` |
| Power (MW) | 1 — e.g. `621.3 MW` |
| Heat rate (kJ/kWh) | 0 — e.g. `9 812 kJ/kWh` |
| Correlation coefficient | 3 — e.g. `r = −0.847` |
| p-value | 4 — e.g. `p = 0.0031` |

### 4.2 Summary Statement Template

End every analysis with a structured summary:

```
## Summary

- **Dataset**: power_plant_data.csv (500 rows, 10 columns)
- **Key finding**: Mean thermal efficiency = 36.73 % (range: 33.1 %–40.5 %)
- **Physical validation**: All efficiency values within [0, 100 %] ✅
- **Anomaly detected**: 3 rows with efficiency > 39 % — above typical design spec
- **Recommendation**: Investigate operating conditions for the 3 high-efficiency periods
```
