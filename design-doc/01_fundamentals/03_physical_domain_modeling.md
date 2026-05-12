# Physical Domain Modeling for AI Agents

> *"Without the unit, a number is just a rumour."*

---

## Table of Contents

1. [Why Physical Meaning Matters](#1-why-physical-meaning-matters)
2. [Dimensional Analysis Fundamentals](#2-dimensional-analysis-fundamentals)
3. [The pint Library](#3-the-pint-library)
4. [Domain Range Validation](#4-domain-range-validation)
5. [Physical Laws as Constraints](#5-physical-laws-as-constraints)
6. [Domain Document Context Injection](#6-domain-document-context-injection)
7. [Integration with Claude](#7-integration-with-claude)
8. [Building a Domain Profile](#8-building-a-domain-profile)

---

## 1. Why Physical Meaning Matters

### Numbers vs. Measurements

In pure mathematics, `57.3` is a number. In the physical world, `57.3` is meaningless without a unit:
- `57.3 °C` — dangerously high CPU temperature (normal: 40–70°C under load)
- `57.3 °F` — a mild spring afternoon
- `57.3 kPa` — low blood pressure (normal: 80–120 kPa systolic)
- `57.3 MPa` — moderate hydraulic pressure in industrial equipment
- `57.3 kg` — a lightweight adult human
- `57.3 kg/s` — a massive flow rate (roughly filling a bathtub every 3 seconds)

The same raw number `57.3` carries completely different physical implications depending on what quantity it represents. An AI agent that loses track of units — or never tracks them at all — is producing numbers without meaning, regardless of how sophisticated its reasoning appears.

### The Compounding Error Problem

Unit errors are especially dangerous because they compound:

1. Dataset has `fuel_flow` column — agent assumes kg/s, but it's actually kg/min
2. Agent computes `heat_input = fuel_flow × LHV` → result is 60× too large
3. Agent computes `efficiency = power / heat_input` → result is 60× too small
4. Agent reports: "efficiency = 0.6%, far below design spec of 38%"
5. Engineer wastes weeks investigating a catastrophic efficiency problem that doesn't exist

Or the opposite direction:
1. Agent divides when it should multiply during unit conversion
2. `efficiency = 112.5%` — impossible, but presented confidently
3. Engineer concludes fuel billing is incorrect (the only way to explain >100% efficiency)

Neither error is caught by syntax checking, type checking, or unit testing of the code itself. Only **dimensional analysis** can catch them.

### Why AI Agents Are Particularly Vulnerable

Human engineers have physical intuition. When a calculation returns 112.5% efficiency, a trained engineer's immediate reaction is "that's wrong — check the units." They have internalized years of practical experience that tells them what numbers are plausible.

Language models, by contrast, are optimized to produce text that looks correct, not to reason about physical plausibility. Claude knows the *facts* about thermodynamics (it was trained on physics textbooks), but it does not automatically apply this knowledge as a filter on computed results. Without an explicit mechanism to trigger plausibility checking, it will report whatever the calculation produces.

This is not a limitation of Claude's intelligence — it is a design gap. The fix is architectural: add a formal physical validation step to the agent's workflow.

---

## 2. Dimensional Analysis Fundamentals

### SI Base Units

The International System of Units (SI) defines 7 base units from which all physical quantities can be derived:

| Base Quantity | Unit | Symbol | Dimension |
|--------------|------|--------|-----------|
| Length | metre | m | L |
| Mass | kilogram | kg | M |
| Time | second | s | T |
| Electric current | ampere | A | I |
| Temperature | kelvin | K | Θ |
| Amount of substance | mole | mol | N |
| Luminous intensity | candela | cd | J |

### Derived Units (Examples)

| Quantity | Unit | Symbol | Dimension |
|----------|------|--------|-----------|
| Force | newton | N | ML/T² |
| Energy | joule | J | ML²/T² |
| Power | watt | W | ML²/T³ |
| Pressure | pascal | Pa | M/LT² |
| Frequency | hertz | Hz | 1/T |
| Velocity | metres/second | m/s | L/T |
| Heat flux | watts/metre² | W/m² | M/T³ |
| Specific heat capacity | joules/(kg·K) | J/(kg·K) | L²/T²Θ |

### Dimensional Homogeneity

An equation is **dimensionally homogeneous** if both sides have the same dimensions. Checking dimensional homogeneity is a necessary (but not sufficient) condition for a correct equation.

**Example**: Thermal power equation

```
Q = ṁ × cp × ΔT

Where:
  Q  = heat transfer rate [W] = [ML²/T³]
  ṁ  = mass flow rate [kg/s] = [M/T]
  cp = specific heat capacity [J/(kg·K)] = [L²/T²Θ]
  ΔT = temperature difference [K] = [Θ]

Dimensional check:
  [M/T] × [L²/T²Θ] × [Θ] = [ML²/T³] ✓
```

**Example of a dimensional error**:
```
η = W_net / Q_in    (correct — both in Watts → dimensionless)
η = W_net / ṁ       (wrong — Watts / (kg/s) = J/kg, not dimensionless!)
```

This error would silently return a number (with units J/kg, not percent) that looks plausible if the values are in a similar range to the expected efficiency.

### Unit Conversion Factors

Common conversions relevant to engineering datasets:

```python
UNIT_CONVERSIONS = {
    # Temperature
    ("celsius", "kelvin"): lambda x: x + 273.15,
    ("fahrenheit", "kelvin"): lambda x: (x + 459.67) * 5/9,
    ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    
    # Flow rates
    ("kg/min", "kg/s"): lambda x: x / 60.0,
    ("tonne/hr", "kg/s"): lambda x: x * 1000 / 3600,
    ("lb/min", "kg/s"): lambda x: x * 0.453592 / 60.0,
    
    # Power/Energy
    ("BTU/hr", "W"): lambda x: x * 0.293071,
    ("kW", "MW"): lambda x: x / 1000.0,
    ("hp", "kW"): lambda x: x * 0.745700,
    
    # Pressure
    ("psi", "Pa"): lambda x: x * 6894.76,
    ("bar", "Pa"): lambda x: x * 1e5,
    ("atm", "Pa"): lambda x: x * 101325,
}
```

---

## 3. The pint Library

### Installation

```bash
pip install pint
# or with uv:
uv add pint
```

### UnitRegistry: The Foundation

The `UnitRegistry` is the central object in `pint`. It knows all SI units and most engineering units, and it enforces dimensional consistency:

```python
from pint import UnitRegistry, DimensionalityError

ureg = UnitRegistry()
Q_ = ureg.Quantity  # Convenience alias
```

### Creating Quantities

```python
# Simple quantities
temperature = Q_(100, "degC")      # 100 degrees Celsius
flow_rate = Q_(2.5, "kg/s")        # 2.5 kg/s
power = Q_(500, "kW")              # 500 kW
pressure = Q_(101325, "Pa")        # 1 atmosphere

# From strings (useful for parsing user input or column names)
temp2 = ureg.parse_expression("100 degC")
flow2 = ureg.parse_expression("150 kg/min")
```

### Unit Conversion

```python
# Convert between compatible units
flow_rate_kgmin = Q_(2.5, "kg/s").to("kg/min")
print(flow_rate_kgmin)  # 150.0 kilogram / minute

temp_fahrenheit = Q_(100, "degC").to("degF")
print(temp_fahrenheit)  # 212.0 degree_Fahrenheit

power_watts = Q_(500, "kW").to("W")
print(power_watts)  # 500000.0 watt

# Offset units (temperature) need special treatment
temp_kelvin = Q_(0, "degC").to("K")
print(temp_kelvin)  # 273.15 kelvin
```

### Dimensional Checking

```python
# Arithmetic that is dimensionally correct
heat_input = Q_(150, "kg/min").to("kg/s") * Q_(43.5, "MJ/kg")
print(heat_input)         # 108.75 MW (correct: kg/s × MJ/kg = MW)

# Arithmetic that is dimensionally wrong — pint raises DimensionalityError
try:
    invalid = Q_(500, "kW") / Q_(150, "kg/min")
    # This gives kW/(kg/min) = kW·min/kg ≠ efficiency (dimensionless)
    efficiency = invalid.to("dimensionless")  # Will raise!
except DimensionalityError as e:
    print(f"Caught dimensional error: {e}")
```

### A Practical Unit Validation Function

```python
from pint import UnitRegistry, DimensionalityError, UndefinedUnitError
from typing import Optional

ureg = UnitRegistry()

def validate_unit_conversion(
    value: float,
    from_unit: str,
    to_unit: str,
) -> tuple[float, str]:
    """
    Convert a value between units, raising informative errors on failure.
    
    Returns:
        (converted_value, message)
    
    Examples:
        >>> validate_unit_conversion(150.0, "kg/min", "kg/s")
        (2.5, "Converted 150.0 kg/min → 2.5 kg/s")
        
        >>> validate_unit_conversion(100.0, "kW", "kg/s")  # incompatible
        raises DimensionalityError
    """
    try:
        quantity = ureg.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
        return (
            float(converted.magnitude),
            f"Converted {value} {from_unit} → {float(converted.magnitude):.6g} {to_unit}"
        )
    except UndefinedUnitError as e:
        raise ValueError(f"Unknown unit: {e}")
    except DimensionalityError as e:
        raise DimensionalityError(
            f"Cannot convert from '{from_unit}' to '{to_unit}': "
            f"they measure different physical quantities. {e}"
        )


def check_dimensional_consistency(
    formula_components: dict[str, tuple[float, str]]
) -> dict[str, str]:
    """
    Check that formula components have correct dimensional relationships.
    
    Args:
        formula_components: {name: (value, unit_string)}
    
    Returns:
        dict with 'status' ('ok' or 'error') and 'message'
    
    Example:
        check_dimensional_consistency({
            "fuel_flow": (2.5, "kg/s"),
            "LHV": (43.5, "MJ/kg"),
            "expected_result_unit": (0, "MW"),  # what heat_input should be in
        })
    """
    try:
        quantities = {
            name: ureg.Quantity(val, unit)
            for name, (val, unit) in formula_components.items()
        }
        return {"status": "ok", "quantities": {
            name: str(q) for name, q in quantities.items()
        }}
    except (UndefinedUnitError, DimensionalityError) as e:
        return {"status": "error", "message": str(e)}
```

### Handling Dimensionless Quantities

Efficiency, dimensionless ratios, and percentages require special treatment:

```python
# Efficiency is dimensionless — pint can express it as "dimensionless" or "percent"
efficiency_decimal = Q_(0.362, "dimensionless")
efficiency_pct = efficiency_decimal.to("percent")
print(efficiency_pct)  # 36.2 percent

# Check if a quantity is dimensionless
def is_dimensionless(value: float, unit_str: str) -> bool:
    try:
        q = ureg.Quantity(value, unit_str)
        # Dimensionless quantities have all zero exponents in their dimensionality
        return q.dimensionality == ureg.dimensionless.dimensionality
    except Exception:
        return False

# Examples
print(is_dimensionless(36.2, "percent"))      # True
print(is_dimensionless(36.2, "kW"))           # False
print(is_dimensionless(1.0, "m/m"))           # True (length ratio)
print(is_dimensionless(1.0, "kg/kg"))         # True (mass ratio)
```

---

## 4. Domain Range Validation

Dimensional analysis tells you if a result has the *right kind* of unit. Range validation tells you if it has a *plausible value* for that unit in a given domain.

### The Range Registry

```python
# application/validation_engine.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class PhysicalRange:
    """Defines the acceptable range for a physical quantity in a domain."""
    quantity: str
    unit: str
    min_value: Optional[float]  # None means no lower bound
    max_value: Optional[float]  # None means no upper bound
    typical_min: Optional[float] = None  # Typical operating range (tighter)
    typical_max: Optional[float] = None
    absolute_min: Optional[float] = None  # Hard physical limit (e.g., 0 K)
    absolute_max: Optional[float] = None  # Hard physical limit (e.g., η ≤ 1.0)
    notes: str = ""

DOMAIN_RANGES: dict[str, list[PhysicalRange]] = {
    
    "thermodynamics": [
        PhysicalRange(
            quantity="thermal_efficiency",
            unit="dimensionless",
            min_value=0.0,
            max_value=1.0,           # First Law: efficiency ≤ 100%
            typical_min=0.30,
            typical_max=0.55,
            absolute_min=0.0,
            absolute_max=1.0,
            notes="Rankine cycle: 30–45%. Combined cycle: 50–62%. Carnot sets theoretical max.",
        ),
        PhysicalRange(
            quantity="temperature",
            unit="kelvin",
            min_value=0.0,           # Third Law: absolute zero is lower bound
            max_value=None,
            typical_min=250.0,       # Typical industrial process temperatures
            typical_max=1500.0,
            absolute_min=0.0,
            notes="Absolute zero is 0 K. Steam turbine inlet: 773–873 K (500–600°C).",
        ),
        PhysicalRange(
            quantity="heat_transfer_rate",
            unit="W",
            min_value=None,
            max_value=None,
            typical_min=1e3,
            typical_max=1e9,
            notes="Industrial heat exchangers: kW to GW range.",
        ),
        PhysicalRange(
            quantity="specific_heat_ratio",
            unit="dimensionless",
            min_value=1.0,           # γ = cp/cv ≥ 1 always
            max_value=None,
            typical_min=1.1,
            typical_max=1.7,
            absolute_min=1.0,
            notes="Dry air: 1.4. Monatomic ideal gases: 1.67. Steam: ~1.13.",
        ),
    ],
    
    "fluid_dynamics": [
        PhysicalRange(
            quantity="pressure",
            unit="Pa",
            min_value=0.0,
            max_value=None,
            typical_min=1e3,
            typical_max=1e8,
            absolute_min=0.0,
            notes="Absolute pressure cannot be negative. Gauge pressure can be.",
        ),
        PhysicalRange(
            quantity="reynolds_number",
            unit="dimensionless",
            min_value=0.0,
            max_value=None,
            typical_min=1.0,
            typical_max=1e8,
            notes="Re < 2300: laminar. Re > 4000: turbulent. 2300–4000: transitional.",
        ),
        PhysicalRange(
            quantity="mach_number",
            unit="dimensionless",
            min_value=0.0,
            max_value=None,
            typical_min=0.0,
            typical_max=25.0,        # Approximate atmospheric re-entry speeds
            absolute_min=0.0,
            notes="Subsonic: Ma < 1. Transonic: 0.8–1.2. Supersonic: Ma > 1.",
        ),
    ],
    
    "electrochemistry": [
        PhysicalRange(
            quantity="pH",
            unit="dimensionless",
            min_value=0.0,
            max_value=14.0,
            typical_min=0.0,
            typical_max=14.0,
            absolute_min=0.0,
            absolute_max=14.0,
            notes="pH scale is 0–14 for aqueous solutions at 25°C. Values outside this range indicate "
                  "extreme concentrations or measurement errors in most industrial contexts.",
        ),
        PhysicalRange(
            quantity="cell_voltage",
            unit="V",
            min_value=0.0,
            max_value=None,
            typical_min=0.5,
            typical_max=5.0,
            notes="Lithium-ion cell: 2.5–4.2 V. Fuel cell: 0.5–1.23 V. Lead-acid: 1.8–2.1 V/cell.",
        ),
    ],
    
    "mechanical": [
        PhysicalRange(
            quantity="stress",
            unit="Pa",
            min_value=None,        # Compression is negative stress
            max_value=None,
            typical_min=-1e9,
            typical_max=1e9,
            notes="Steel yield: ~250 MPa. Compressive stress is negative by convention.",
        ),
        PhysicalRange(
            quantity="coefficient_of_performance",
            unit="dimensionless",
            min_value=0.0,
            max_value=None,        # COP CAN exceed 1 (heat pumps move more heat than work done)
            typical_min=2.0,
            typical_max=6.0,
            absolute_min=0.0,
            notes="Heat pump COP = Q_delivered / W_input > 1. COP_carnot = T_hot/(T_hot - T_cold).",
        ),
    ],
}


def lookup_range(quantity: str, domain: str) -> Optional[PhysicalRange]:
    """Find the PhysicalRange for a quantity in a given domain."""
    ranges = DOMAIN_RANGES.get(domain, [])
    for r in ranges:
        if r.quantity.lower() == quantity.lower():
            return r
    # Try all domains if specific one not found
    for domain_ranges in DOMAIN_RANGES.values():
        for r in domain_ranges:
            if r.quantity.lower() == quantity.lower():
                return r
    return None
```

### Using the Range Registry

```python
def validate_against_range(
    quantity_name: str,
    value: float,
    unit: str,
    domain: str = "thermodynamics",
) -> dict:
    """
    Validate a physical quantity against known domain ranges.
    
    Returns a result dict suitable for returning from a tool call.
    """
    range_def = lookup_range(quantity_name, domain)
    
    if range_def is None:
        return {
            "valid": None,  # Unknown — no range defined
            "message": f"No range definition found for '{quantity_name}' in domain '{domain}'. "
                       f"Cannot validate. Consider checking the domain documentation manually.",
        }
    
    issues = []
    warnings = []
    
    # Check absolute physical limits first
    if range_def.absolute_min is not None and value < range_def.absolute_min:
        issues.append(
            f"PHYSICAL IMPOSSIBILITY: {quantity_name} = {value} {unit} is below the "
            f"absolute minimum of {range_def.absolute_min} {range_def.unit}. "
            f"This violates a fundamental physical law. {range_def.notes}"
        )
    
    if range_def.absolute_max is not None and value > range_def.absolute_max:
        issues.append(
            f"PHYSICAL IMPOSSIBILITY: {quantity_name} = {value} {unit} exceeds the "
            f"absolute maximum of {range_def.absolute_max} {range_def.unit}. "
            f"This violates a fundamental physical law. {range_def.notes}"
        )
    
    # Check operational range
    if range_def.min_value is not None and value < range_def.min_value:
        issues.append(
            f"BELOW VALID RANGE: {quantity_name} = {value} {unit} is below the "
            f"minimum operational value of {range_def.min_value}."
        )
    
    if range_def.max_value is not None and value > range_def.max_value:
        issues.append(
            f"ABOVE VALID RANGE: {quantity_name} = {value} {unit} exceeds the "
            f"maximum operational value of {range_def.max_value}."
        )
    
    # Check typical range (softer warning)
    if range_def.typical_min is not None and value < range_def.typical_min:
        warnings.append(
            f"UNUSUAL (low): {value} {unit} is below the typical range "
            f"[{range_def.typical_min}, {range_def.typical_max}]. "
            f"Verify units. {range_def.notes}"
        )
    
    if range_def.typical_max is not None and value > range_def.typical_max:
        warnings.append(
            f"UNUSUAL (high): {value} {unit} is above the typical range "
            f"[{range_def.typical_min}, {range_def.typical_max}]. "
            f"Verify units. {range_def.notes}"
        )
    
    return {
        "valid": len(issues) == 0,
        "quantity": quantity_name,
        "value": value,
        "unit": unit,
        "domain": domain,
        "issues": issues,
        "warnings": warnings,
        "range_notes": range_def.notes,
    }
```

---

## 5. Physical Laws as Constraints

### First Law of Thermodynamics (Energy Conservation)

The most commonly violated physical law in data analysis:

```python
def check_energy_conservation(
    work_output: float,      # W or any energy unit
    heat_input: float,       # same unit as work_output
    losses: float = 0.0,     # known losses (friction, radiation, etc.)
) -> dict:
    """
    Check that work_output ≤ heat_input (First Law of Thermodynamics).
    Efficiency must be ≤ 100% for any real system.
    """
    if heat_input <= 0:
        return {"valid": False, "error": "heat_input must be positive"}
    
    efficiency = work_output / heat_input
    
    if efficiency > 1.0:
        return {
            "valid": False,
            "efficiency": efficiency,
            "violation": "FIRST LAW VIOLATION",
            "message": (
                f"Computed efficiency = {efficiency:.1%} > 100%. "
                f"This violates conservation of energy. "
                f"Likely cause: unit mismatch (e.g., work in MW but heat in GJ/hr). "
                f"Check: 1 MW = 3.6 GJ/hr. Also verify flow rate units (kg/s vs kg/min)."
            ),
        }
    
    total_accounted = work_output + losses
    unaccounted_loss_fraction = (heat_input - total_accounted) / heat_input
    
    return {
        "valid": True,
        "efficiency": efficiency,
        "efficiency_pct": efficiency * 100,
        "unaccounted_loss_fraction": unaccounted_loss_fraction,
        "message": f"Energy conservation satisfied. Efficiency = {efficiency:.1%}.",
    }
```

### Carnot Efficiency Limit

```python
def carnot_efficiency(
    T_hot_kelvin: float,
    T_cold_kelvin: float,
) -> dict:
    """
    Compute the Carnot efficiency limit for a heat engine.
    
    No real heat engine can exceed this limit (Second Law of Thermodynamics).
    
    Args:
        T_hot_kelvin: Temperature of the hot reservoir in Kelvin
        T_cold_kelvin: Temperature of the cold reservoir in Kelvin
    """
    if T_cold_kelvin <= 0 or T_hot_kelvin <= 0:
        return {"error": "Temperatures must be positive (in Kelvin)"}
    
    if T_cold_kelvin >= T_hot_kelvin:
        return {
            "error": "T_hot must be greater than T_cold for a heat engine to produce work",
            "T_hot_K": T_hot_kelvin,
            "T_cold_K": T_cold_kelvin,
        }
    
    eta_carnot = 1.0 - (T_cold_kelvin / T_hot_kelvin)
    
    return {
        "carnot_efficiency": eta_carnot,
        "carnot_efficiency_pct": eta_carnot * 100,
        "T_hot_K": T_hot_kelvin,
        "T_cold_K": T_cold_kelvin,
        "interpretation": (
            f"Maximum possible efficiency for a heat engine between "
            f"{T_hot_kelvin:.1f} K and {T_cold_kelvin:.1f} K is "
            f"{eta_carnot:.1%}. Any computed efficiency above this indicates "
            f"a measurement or calculation error."
        ),
    }
```

### Mass Balance

```python
def check_mass_balance(
    inflows: dict[str, float],    # {stream_name: flow_rate_kg_s}
    outflows: dict[str, float],   # {stream_name: flow_rate_kg_s}
    tolerance_fraction: float = 0.01,  # 1% tolerance for measurement error
) -> dict:
    """
    Check that mass in = mass out (Conservation of Mass).
    
    For steady-state systems without chemical reactions,
    total inflow must equal total outflow within measurement tolerance.
    """
    total_in = sum(inflows.values())
    total_out = sum(outflows.values())
    
    if total_in == 0:
        return {"error": "Total inflow is zero — cannot compute mass balance"}
    
    imbalance_fraction = abs(total_in - total_out) / total_in
    
    balanced = imbalance_fraction <= tolerance_fraction
    
    return {
        "balanced": balanced,
        "total_inflow_kg_s": total_in,
        "total_outflow_kg_s": total_out,
        "imbalance_fraction": imbalance_fraction,
        "imbalance_kg_s": total_in - total_out,
        "message": (
            f"Mass balance {'satisfied' if balanced else 'VIOLATED'}. "
            f"Imbalance: {imbalance_fraction:.1%} "
            f"({'within' if balanced else 'exceeds'} {tolerance_fraction:.1%} tolerance). "
            + ("" if balanced else
               "Check for: unmeasured streams, sensor calibration errors, or leaks.")
        ),
    }
```

---

## 6. Domain Document Context Injection

### How It Works

The physical validation engine cannot know the specific operational parameters of every dataset without external context. A turbine efficiency of 0.35 might be perfectly normal for an older plant and alarming for a brand-new one. A flow rate of 150 kg/s might be design-nominal for one system and a dangerous exceedance for another.

The solution is **domain document context injection**: before any analysis, the agent reads domain-specific documents that contain:
1. The specific equipment/system being analyzed
2. Design parameters and nominal operating ranges
3. Known failure modes and their signatures
4. Any non-standard unit conventions used in this dataset

This context is then used to parameterize the validation engine with system-specific rather than generic ranges.

### Document Format

Domain documents should be structured markdown with machine-readable sections:

```markdown
# Power Plant Thermal Analysis — Domain Reference

## System Identification
- Plant: Riverside Combined Cycle Station, Unit 3
- Technology: Siemens SGT-800 gas turbine + HRSG + steam turbine
- Design capacity: 250 MW_net
- Commissioning date: 2018-03-15

## Key Performance Parameters

### Thermal Efficiency
- Design efficiency: 52.3% (lower heating value basis)
- Typical operating range: 48–55% (load-dependent)
- Alarm threshold: < 45% (investigate) or > 58% (sensor fault likely)
- Carnot limit at design conditions: 64.7% (T_hot=1300°C, T_cold=15°C)
- IMPORTANT: Efficiency > 56% indicates EITHER exceptional performance OR a measurement error

### Temperature
- Gas turbine inlet: 1250–1350°C (normal operation)
- Gas turbine exhaust: 540–580°C
- HP steam temperature: 560–580°C
- Alarm: GT exhaust > 600°C (cooling system fault)
- IMPORTANT: Values in Kelvin in this dataset (not Celsius)

### Flow Rates
- Fuel gas flow: 5.2–7.8 kg/s (at 250 MW load)
- Air flow: 520–680 kg/s
- Steam flow (HP): 80–105 kg/s
- UNIT WARNING: fuel_flow_raw column is in kg/min, NOT kg/s (historical data format)
```

### Parsing Context from Documents

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentContext:
    """Physical context extracted from a domain document."""
    system_name: Optional[str] = None
    efficiency_range: Optional[tuple[float, float]] = None
    temperature_unit: str = "celsius"           # May be overridden by doc
    flow_rate_unit: str = "kg/s"               # May be overridden by doc
    unit_overrides: dict[str, str] = None       # column_name → actual_unit
    design_parameters: dict[str, float] = None  # parameter → design_value
    alarm_thresholds: dict[str, tuple[float, float]] = None  # parameter → (low, high)
    warnings: list[str] = None                  # Explicit warnings from doc

    def __post_init__(self):
        if self.unit_overrides is None:
            self.unit_overrides = {}
        if self.design_parameters is None:
            self.design_parameters = {}
        if self.alarm_thresholds is None:
            self.alarm_thresholds = {}
        if self.warnings is None:
            self.warnings = []


def extract_context_from_document(document_text: str) -> DocumentContext:
    """
    Extract machine-readable context from a domain document.
    
    This is intentionally simple — in production, you'd use a more
    sophisticated extraction approach (or ask Claude to extract it).
    """
    ctx = DocumentContext()
    
    # Extract system name
    name_match = re.search(r"^# (.+)", document_text, re.MULTILINE)
    if name_match:
        ctx.system_name = name_match.group(1).strip()
    
    # Extract efficiency range
    eff_match = re.search(
        r"efficiency.*?:\s*([\d.]+)[–\-]([\d.]+)\s*%",
        document_text, re.IGNORECASE
    )
    if eff_match:
        ctx.efficiency_range = (
            float(eff_match.group(1)) / 100,
            float(eff_match.group(2)) / 100,
        )
    
    # Extract UNIT WARNING annotations
    unit_warnings = re.findall(
        r"UNIT WARNING: (\w+) (?:column )?is in ([^,\n]+), NOT ([^\n]+)",
        document_text, re.IGNORECASE
    )
    for col, actual_unit, _stated_unit in unit_warnings:
        ctx.unit_overrides[col] = actual_unit.strip()
        ctx.warnings.append(
            f"Column '{col}' uses unit '{actual_unit.strip()}' "
            f"(not the default — apply conversion before use)"
        )
    
    return ctx
```

---

## 7. Integration with Claude

### Structuring Tool Results for Physical Reasoning

When a validation tool returns a result, the format should make physical reasoning easy for Claude:

```python
def format_validation_result_for_claude(
    validation: dict,
    quantity_name: str,
    value: float,
    unit: str,
) -> str:
    """
    Format a validation result as a clear, Claude-readable observation.
    
    Claude should be able to understand:
    1. Is this result physically valid?
    2. What is the specific violation (if any)?
    3. What are likely causes of the violation?
    4. What should be done next?
    """
    lines = []
    
    # Status line — most important info first
    if validation.get("valid") is False:
        lines.append(f"⚠️  VALIDATION FAILED: {quantity_name} = {value} {unit}")
    elif validation.get("valid") is True:
        lines.append(f"✓  VALIDATION PASSED: {quantity_name} = {value} {unit}")
    else:
        lines.append(f"?  VALIDATION UNKNOWN: {quantity_name} = {value} {unit}")
    
    # Issues (hard violations)
    for issue in validation.get("issues", []):
        lines.append(f"  ERROR: {issue}")
    
    # Warnings (soft violations)
    for warning in validation.get("warnings", []):
        lines.append(f"  WARNING: {warning}")
    
    # Contextual notes
    if validation.get("range_notes"):
        lines.append(f"  Context: {validation['range_notes']}")
    
    # Recommendations
    if validation.get("valid") is False:
        lines.append(
            "  Recommended next steps: "
            "1) Check unit conversion in source data. "
            "2) Verify sensor calibration. "
            "3) Review computation formula. "
            "4) Compare with domain document expected ranges."
        )
    
    return "\n".join(lines)
```

### System Prompt Instructions for Physical Validation

```python
PHYSICAL_VALIDATION_INSTRUCTIONS = """
## Physical Validation Protocol

After computing ANY numeric result that represents a physical quantity, you MUST:

1. Call validate_units() with the quantity name, computed value, and units.
2. If validate_units() returns valid=False, DO NOT include that result in your final answer.
   Instead, investigate the likely cause (unit mismatch, formula error) before proceeding.
3. If validate_units() returns valid=True but warns of unusual values, note this in
   your Thought and consider whether the anomaly needs further investigation.
4. For efficiency calculations specifically:
   - If computed efficiency > 100%: STOP. Check unit consistency before continuing.
   - If computed efficiency > Carnot limit: STOP. Check temperature units.
   - If computed efficiency < 10%: Verify this is not a fraction (0.37) vs percent (37%).

## Physical Constants (for sanity checking)
- 1 MW = 1 MJ/s = 1000 kJ/s = 3.6 GJ/hr = 3.412 × 10⁶ BTU/hr
- 0°C = 273.15 K = 32°F
- 1 bar = 100 kPa = 0.987 atm = 14.50 psi
- 1 kg/s = 60 kg/min = 3600 kg/hr = 2.205 lb/s
"""
```

---

## 8. Building a Domain Profile

A **Domain Profile** encapsulates all the physical knowledge needed to validate a specific analysis. It combines:
- Quantity ranges from the domain range registry
- Document-extracted context (system-specific parameters)
- Physical law constraints applicable to the domain

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass  
class DomainProfile:
    """
    All physical knowledge needed to validate analysis of a specific dataset/domain.
    
    Built by the agent before beginning analysis by reading domain documents
    and combining with registry defaults.
    """
    name: str                           # e.g., "Riverside CCGT Unit 3"
    domain: str                         # e.g., "thermodynamics"
    
    # System-specific parameters from domain document
    design_efficiency: Optional[float] = None        # e.g., 0.523
    efficiency_alarm_low: Optional[float] = None     # e.g., 0.45
    efficiency_alarm_high: Optional[float] = None    # e.g., 0.58
    
    # Unit overrides (column name → actual unit)
    unit_overrides: dict[str, str] = field(default_factory=dict)
    
    # Design parameters for reference
    design_parameters: dict[str, float] = field(default_factory=dict)
    
    # Physical laws that apply
    applicable_laws: list[str] = field(default_factory=list)
    
    # Free-form notes from domain document
    notes: list[str] = field(default_factory=list)
    
    def get_effective_range(self, quantity: str) -> Optional[PhysicalRange]:
        """Get the applicable range for a quantity, preferring system-specific over generic."""
        # First check system-specific overrides
        if quantity == "thermal_efficiency" and self.design_efficiency is not None:
            # Widen the range slightly to allow for measurement uncertainty
            low = (self.efficiency_alarm_low or 0.0)
            high = (self.efficiency_alarm_high or 1.0)
            return PhysicalRange(
                quantity="thermal_efficiency",
                unit="dimensionless",
                min_value=low,
                max_value=1.0,
                typical_min=low,
                typical_max=high,
                absolute_min=0.0,
                absolute_max=1.0,
                notes=f"System-specific range from domain profile: {self.name}",
            )
        
        # Fall back to domain registry
        return lookup_range(quantity, self.domain)
    
    @classmethod
    def from_document_context(
        cls,
        domain: str,
        ctx: DocumentContext,
    ) -> "DomainProfile":
        """Build a DomainProfile from a parsed document context."""
        profile = cls(
            name=ctx.system_name or "Unknown System",
            domain=domain,
            unit_overrides=ctx.unit_overrides,
            design_parameters=ctx.design_parameters,
            notes=ctx.warnings,
        )
        
        if ctx.efficiency_range:
            profile.efficiency_alarm_low = ctx.efficiency_range[0]
            profile.efficiency_alarm_high = ctx.efficiency_range[1]
        
        # Add applicable physical laws based on domain
        if domain == "thermodynamics":
            profile.applicable_laws = [
                "first_law_energy_conservation",
                "second_law_carnot_limit",
                "mass_conservation",
            ]
        elif domain == "fluid_dynamics":
            profile.applicable_laws = [
                "mass_conservation",
                "momentum_conservation",
                "bernoulli_equation",
            ]
        
        return profile


# Example: Building the profile for the power plant scenario
def build_power_plant_profile(document_text: str) -> DomainProfile:
    ctx = extract_context_from_document(document_text)
    return DomainProfile.from_document_context("thermodynamics", ctx)
```

### Using the Profile in Analysis

```python
def validate_analysis_result(
    profile: DomainProfile,
    quantity_name: str,
    value: float,
    unit: str,
) -> str:
    """
    Validate a computed result against the domain profile.
    Returns a formatted string suitable for a tool observation.
    """
    # Check unit overrides
    if quantity_name in profile.unit_overrides:
        declared_unit = profile.unit_overrides[quantity_name]
        if declared_unit != unit:
            return (
                f"UNIT CONFLICT: The domain profile says '{quantity_name}' "
                f"is in {declared_unit}, but you computed with {unit}. "
                f"Apply conversion before using this value."
            )
    
    # Get effective range (system-specific or registry default)
    range_def = profile.get_effective_range(quantity_name)
    
    if range_def is None:
        return f"No range validation available for '{quantity_name}' in profile '{profile.name}'"
    
    # Run validation
    validation = validate_against_range(quantity_name, value, unit, profile.domain)
    return format_validation_result_for_claude(validation, quantity_name, value, unit)
```

---

*Next: [04_data_science_workflow.md](04_data_science_workflow.md) — How the AI agent augments the CRISP-DM workflow.*
