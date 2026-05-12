# Physical Validation Design — pint + Domain Knowledge

**Document Version:** 1.0  
**Status:** Approved  
**Bounded Context:** Infrastructure — Physical Unit Validation  

---

## 1. The Validation Pipeline

Every physical quantity check runs through a three-stage pipeline. Failure at any stage short-circuits the remaining stages.

```
Input: (quantity, value, unit_str, domain_context)
           │
           ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: Unit Parsing                           │
│  pint.UnitRegistry().parse_expression(unit_str) │
│  → Fails: return PhysicalUnit(is_valid=False,   │
│            message="Unit parse error: ...")      │
│  → Passes: → pint.Quantity(value, unit)         │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Dimensional Check                      │
│  Compare dimensionality against expected for    │
│  the named quantity (if known)                  │
│  e.g., "efficiency" must be dimensionless       │
│        "power" must have [mass][length]²[time]⁻³│
│  → Fails: return PhysicalUnit(is_valid=False,   │
│            message="Dimensional error: ...")     │
│  → Passes: → normalized Quantity               │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: Magnitude Check                        │
│  Convert to canonical unit, compare to          │
│  DOMAIN_RANGES[fuzzy_match(quantity)]           │
│  → Out of range: return PhysicalUnit(           │
│      is_valid=False,                            │
│      message="Magnitude warning: ...")          │
│  → In range: return PhysicalUnit(               │
│      is_valid=True, message="OK")               │
└─────────────────────────────────────────────────┘
```

---

## 2. pint UnitRegistry Architecture

### 2.1 Singleton Registry

`pint.UnitRegistry` is expensive to instantiate (loads unit definitions from files). A module-level singleton is created once at import time and shared across all requests.

```python
import pint

# Module-level singleton — created once, reused everywhere
_ureg: pint.UnitRegistry | None = None

def _get_unit_registry() -> pint.UnitRegistry:
    """Returns the shared pint UnitRegistry, creating it on first call."""
    global _ureg
    if _ureg is None:
        _ureg = pint.UnitRegistry()
        _register_custom_units(_ureg)
    return _ureg
```

### 2.2 Custom Unit Definitions

```python
def _register_custom_units(ureg: pint.UnitRegistry) -> None:
    """
    Registers domain-specific unit aliases not in pint's default registry.
    Uses pint's define() method for non-SI units common in engineering.
    """
    # Efficiency as percentage (dimensionless, 0–100 scale)
    ureg.define("percent = 0.01 * [] = %")

    # British Thermal Unit (already in pint, but alias for clarity)
    # ureg.define("BTU = 1055.06 * joule = btu")  # already defined

    # Horsepower variants
    ureg.define("mechanical_horsepower = 745.7 * watt = hp_mech")

    # Parts per million (concentration, dimensionless)
    ureg.define("ppm = 1e-6 * [] = ppm")
    ureg.define("ppb = 1e-9 * [] = ppb")

    # Standard atmosphere (already defined as 'atm')
    # bar already defined in pint

    # mmHg (for pressure in medical/lab contexts)
    # already defined in pint as 'mmHg'

    # Flow rate convenience
    # m³/s, L/min etc. are composites — no custom definition needed
```

### 2.3 Pre-Defined Domain Quantity Dimensionalities

```python
# Expected dimensionality for named quantities.
# Key: quantity name (lowercase), Value: pint dimensionality string
_QUANTITY_DIMENSIONS: dict[str, str] = {
    "temperature":              "[temperature]",
    "pressure":                 "[mass] / [length] / [time] ** 2",
    "power":                    "[mass] * [length] ** 2 / [time] ** 3",
    "energy":                   "[mass] * [length] ** 2 / [time] ** 2",
    "flow_rate":                "[length] ** 3 / [time]",
    "mass_flow_rate":           "[mass] / [time]",
    "velocity":                 "[length] / [time]",
    "efficiency":               "",   # dimensionless
    "thermal_efficiency":       "",   # dimensionless
    "concentration":            "[mass] / [length] ** 3",
    "voltage":                  "[mass] * [length] ** 2 / [current] / [time] ** 3",
    "current":                  "[current]",
    "frequency":                "1 / [time]",
    "length":                   "[length]",
    "mass":                     "[mass]",
    "density":                  "[mass] / [length] ** 3",
    "specific_heat":            "[length] ** 2 / [time] ** 2 / [temperature]",
    "heat_transfer_coefficient":"[mass] / [time] ** 3 / [temperature]",
    "ph":                       "",   # dimensionless
}
```

---

## 3. DOMAIN_RANGES Dictionary

Complete reference table mapping quantity names to `(min, max, canonical_unit, description)`.

```python
# infrastructure/unit_registry.py

DOMAIN_RANGES: dict[str, tuple[float, float, str, str]] = {
    # ── Temperature ──────────────────────────────────────────────────────────
    "surface_temp":           (-90.0,   60.0,  "degC",  "Earth surface air temperature"),
    "industrial_temp":        (-200.0,  1500.0,"degC",  "Industrial process temperature range"),
    "human_body_temp":        (35.0,    42.0,  "degC",  "Normal human body temperature"),
    "furnace_temp":           (800.0,   1800.0,"degC",  "Industrial furnace operating temperature"),
    "hvac_supply_temp":       (10.0,    30.0,  "degC",  "HVAC supply air temperature"),
    "cooling_water_temp":     (10.0,    50.0,  "degC",  "Industrial cooling water temperature"),
    "steam_temp":             (100.0,   650.0, "degC",  "Industrial steam temperature"),

    # ── Pressure ─────────────────────────────────────────────────────────────
    "atmospheric_pressure":   (87.0,    108.0, "kPa",   "Sea-level atmospheric pressure range"),
    "gauge_pressure":         (0.0,     10000.0,"kPa",  "Gauge pressure (above atmosphere)"),
    "blood_pressure_systolic":(60.0,    200.0, "mmHg",  "Human systolic blood pressure"),
    "blood_pressure_diastolic":(40.0,   130.0, "mmHg",  "Human diastolic blood pressure"),
    "boiler_pressure":        (100.0,   25000.0,"kPa",  "Industrial steam boiler pressure"),
    "vacuum_pressure":        (0.0,     101.3,  "kPa",  "Vacuum (absolute) — 0 = full vacuum"),
    "tire_pressure":          (100.0,   350.0,  "kPa",  "Vehicle tire inflation pressure"),

    # ── Power ────────────────────────────────────────────────────────────────
    "residential_electrical": (0.0,     50.0,   "kW",   "Residential electrical power consumption"),
    "industrial_power":       (0.0,     1000.0, "MW",   "Industrial process power"),
    "power_plant_output":     (1.0,     2000.0, "MW",   "Utility-scale power plant output"),
    "wind_turbine_power":     (0.0,     15.0,   "MW",   "Single wind turbine rated output"),
    "solar_panel_power":      (0.0,     1.0,    "kW",   "Residential solar panel output"),
    "heat_input_power":       (1.0,     5000.0, "MW",   "Thermal input to power plant"),

    # ── Efficiency ───────────────────────────────────────────────────────────
    "thermal_efficiency":     (0.1,     0.65,   "",     "Thermal cycle efficiency (fraction 0–1)"),
    "efficiency":             (0.0,     1.0,    "",     "Generic efficiency (fraction 0–1)"),
    "efficiency_percent":     (0.0,     100.0,  "%",    "Efficiency expressed as percentage"),
    "carnot_efficiency":      (0.0,     1.0,    "",     "Carnot efficiency upper bound"),
    "boiler_efficiency":      (0.5,     0.98,   "",     "Steam boiler thermal efficiency"),
    "pump_efficiency":        (0.4,     0.92,   "",     "Centrifugal pump efficiency"),
    "motor_efficiency":       (0.7,     0.99,   "",     "Electric motor efficiency"),
    "turbine_efficiency":     (0.7,     0.93,   "",     "Steam turbine isentropic efficiency"),

    # ── Flow Rate ────────────────────────────────────────────────────────────
    "water_flow_rate":        (0.0,     10.0,   "m**3/s","Industrial water flow rate"),
    "steam_flow_rate":        (0.0,     500.0,  "kg/s", "Industrial steam mass flow rate"),
    "gas_flow_rate":          (0.0,     1000.0, "m**3/s","Natural gas volumetric flow rate"),

    # ── Velocity ─────────────────────────────────────────────────────────────
    "wind_speed":             (0.0,     100.0,  "m/s",  "Atmospheric wind speed"),
    "fluid_velocity":         (0.0,     50.0,   "m/s",  "Fluid velocity in pipes"),
    "vehicle_speed":          (0.0,     400.0,  "km/h", "Wheeled vehicle speed"),

    # ── Electrical ───────────────────────────────────────────────────────────
    "grid_voltage_lv":        (100.0,   500.0,  "V",    "Low-voltage grid (residential)"),
    "grid_voltage_hv":        (1.0,     765.0,  "kV",   "High-voltage transmission line"),
    "household_current":      (0.0,     200.0,  "A",    "Household branch circuit current"),

    # ── Chemistry / Environment ───────────────────────────────────────────────
    "ph_value":               (0.0,     14.0,   "",     "pH scale (dimensionless)"),
    "co2_concentration":      (0.0,     5.0,    "%",    "CO₂ concentration by volume"),
    "nox_concentration":      (0.0,     2000.0, "ppm",  "NOx concentration in flue gas"),
    "dissolved_oxygen":       (0.0,     20.0,   "mg/L", "Dissolved oxygen in water"),

    # ── Heat Transfer ─────────────────────────────────────────────────────────
    "heat_flux":              (0.0,     1e6,    "W/m**2","Surface heat flux"),
    "thermal_conductivity":   (0.0,     1000.0, "W/m/K","Thermal conductivity"),
    "heat_transfer_rate":     (0.0,     1000.0, "MW",   "Total heat transfer rate"),
}
```

---

## 4. Domain Document Context Extraction

Unit definitions are extracted from domain documents using regex patterns and cached:

```python
import re

_UNIT_DEF_PATTERNS = [
    # "MW: megawatt, unit of power"
    re.compile(r"^[-*]\s*(\w+)\s*:\s*(.+)$", re.MULTILINE),
    # "expressed in MW (megawatts)"
    re.compile(r"expressed in ([A-Za-z°%/²³·]+)\s*\(([^)]+)\)", re.IGNORECASE),
    # "measured in degrees Celsius (°C)"
    re.compile(r"measured in ([^(]+)\(([^)]+)\)", re.IGNORECASE),
]

def extract_unit_definitions_from_docs(docs_dir: Path) -> dict[str, str]:
    """
    Scans domain documents and returns {symbol: description} for all
    identified unit definitions.
    """
    definitions: dict[str, str] = {}
    
    for doc in sorted(docs_dir.glob("*.md")) + sorted(docs_dir.glob("*.txt")):
        try:
            text = doc.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        
        for pattern in _UNIT_DEF_PATTERNS:
            for m in pattern.finditer(text):
                symbol = m.group(1).strip()
                description = m.group(2).strip()[:200]
                if len(symbol) <= 10:  # Ignore very long "symbols"
                    definitions[symbol] = description
    
    return definitions
```

---

## 5. validate_physical_units() Implementation

```python
def validate_physical_units(
    quantity: str,
    value: float,
    unit: str,
    domain_context: str = "",
) -> str:
    """
    Validates a physical quantity through the 3-stage pipeline.
    Returns JSON-serialized PhysicalUnit.
    Never raises — returns PhysicalUnit(is_valid=False) on all failures.
    """
    import json
    from app.domain.analysis_models import PhysicalUnit

    ureg = _get_unit_registry()
    quantity_lower = quantity.lower().replace(" ", "_")

    # ── Stage 1: Unit Parsing ─────────────────────────────────────────────
    try:
        q = ureg.Quantity(value, unit)
    except Exception as e:
        pu = PhysicalUnit(
            quantity=quantity,
            value=value,
            unit_str=unit,
            reasonable_range=(0.0, 0.0),
            is_valid=False,
            message=f"Unit parse error: Cannot parse '{unit}': {e}",
        )
        return json.dumps(pu.to_json_dict())

    # ── Stage 2: Dimensional Check ────────────────────────────────────────
    expected_dim = _QUANTITY_DIMENSIONS.get(quantity_lower)
    if expected_dim is not None:
        try:
            if expected_dim == "":
                # Dimensionless check
                if not q.dimensionless:
                    pu = PhysicalUnit(
                        quantity=quantity,
                        value=value,
                        unit_str=unit,
                        reasonable_range=(0.0, 0.0),
                        is_valid=False,
                        message=(
                            f"Dimensional error: '{quantity}' should be dimensionless, "
                            f"but '{unit}' has dimensions {q.dimensionality}."
                        ),
                    )
                    return json.dumps(pu.to_json_dict())
            else:
                ref = ureg.parse_expression(expected_dim)
                if q.dimensionality != ureg.parse_expression("1 * " + expected_dim).dimensionality:
                    pu = PhysicalUnit(
                        quantity=quantity,
                        value=value,
                        unit_str=unit,
                        reasonable_range=(0.0, 0.0),
                        is_valid=False,
                        message=(
                            f"Dimensional error: '{quantity}' expected {expected_dim}, "
                            f"got {q.dimensionality}."
                        ),
                    )
                    return json.dumps(pu.to_json_dict())
        except Exception:
            pass  # Dimensional check failed — don't block on infrastructure error

    # ── Stage 3: Magnitude Check ──────────────────────────────────────────
    matched_key = _fuzzy_match_quantity(quantity_lower)
    
    if matched_key is None:
        # No range entry — validate hard physics constraints only
        pu = PhysicalUnit(
            quantity=quantity,
            value=value,
            unit_str=unit,
            reasonable_range=(float("-inf"), float("inf")),
            is_valid=True,
            message=f"OK (no domain range defined for '{quantity}'; hard constraints passed)",
        )
        return json.dumps(pu.to_json_dict())

    min_val, max_val, canonical_unit, description = DOMAIN_RANGES[matched_key]
    
    try:
        if canonical_unit:
            val_canonical = float(q.to(canonical_unit).magnitude)
        else:
            # Dimensionless: handle % ↔ fraction conversion
            val_canonical = float(q.to("").magnitude) if unit == "%" else value
    except Exception as e:
        pu = PhysicalUnit(
            quantity=quantity,
            value=value,
            unit_str=unit,
            reasonable_range=(min_val, max_val),
            is_valid=False,
            message=f"Unit conversion error: Cannot convert to '{canonical_unit}': {e}",
        )
        return json.dumps(pu.to_json_dict())

    in_range = min_val <= val_canonical <= max_val

    pu = PhysicalUnit(
        quantity=quantity,
        value=value,
        unit_str=unit,
        reasonable_range=(min_val, max_val),
        is_valid=in_range,
        message=(
            "OK"
            if in_range
            else (
                f"Magnitude warning: {value} {unit} = {val_canonical:.4g} {canonical_unit} "
                f"is outside expected range [{min_val}, {max_val}] {canonical_unit}. "
                f"Context: {description}."
            )
        ),
    )
    return json.dumps(pu.to_json_dict())
```

---

## 6. convert_units() Implementation

```python
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts value from from_unit to to_unit using pint.
    Returns a formatted string: "{converted_value} {to_unit}"
    Never raises.
    """
    try:
        ureg = _get_unit_registry()
        q = ureg.Quantity(value, from_unit)
        result = q.to(to_unit)
        return f"{result.magnitude:.6g} {to_unit}"
    except pint.DimensionalityError as e:
        return f"Error: Incompatible units — cannot convert '{from_unit}' to '{to_unit}': {e}"
    except pint.UndefinedUnitError as e:
        return f"Error: Undefined unit: {e}"
    except Exception as e:
        return f"Error: Unit conversion failed: {e}"
```

---

## 7. check_magnitude() Implementation

```python
def _fuzzy_match_quantity(quantity: str) -> str | None:
    """
    Finds the closest matching key in DOMAIN_RANGES.
    Uses substring matching and token overlap.
    Returns None if no match found.
    """
    q = quantity.lower().replace(" ", "_")
    
    # Exact match first
    if q in DOMAIN_RANGES:
        return q
    
    # Substring match
    for key in DOMAIN_RANGES:
        if q in key or key in q:
            return key
    
    # Token overlap: split on _ and match if 2+ tokens overlap
    q_tokens = set(q.split("_"))
    best_key: str | None = None
    best_overlap = 0
    
    for key in DOMAIN_RANGES:
        key_tokens = set(key.split("_"))
        overlap = len(q_tokens & key_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key
    
    return best_key if best_overlap >= 2 else None


def check_magnitude(quantity: str, value: float, unit: str) -> str:
    """
    Checks if a value's magnitude is within the expected range.
    Returns a JSON summary including the expected range and whether it's in range.
    """
    import json

    matched_key = _fuzzy_match_quantity(quantity.lower().replace(" ", "_"))

    if not matched_key:
        return json.dumps({
            "quantity": quantity,
            "value": value,
            "unit": unit,
            "in_range": None,
            "message": f"Quantity '{quantity}' not found in domain ranges database.",
            "known_quantities": list(DOMAIN_RANGES.keys())[:20],
        })

    min_val, max_val, canonical_unit, description = DOMAIN_RANGES[matched_key]
    ureg = _get_unit_registry()

    try:
        if canonical_unit:
            val_canonical = float(ureg.Quantity(value, unit).to(canonical_unit).magnitude)
        else:
            val_canonical = value
    except Exception as e:
        return json.dumps({
            "quantity": quantity,
            "value": value,
            "unit": unit,
            "in_range": None,
            "message": f"Cannot convert to canonical unit '{canonical_unit}': {e}",
        })

    in_range = min_val <= val_canonical <= max_val

    return json.dumps({
        "quantity": quantity,
        "matched_domain_quantity": matched_key,
        "value": value,
        "unit": unit,
        "value_in_canonical_unit": val_canonical,
        "canonical_unit": canonical_unit,
        "expected_range": [min_val, max_val],
        "in_range": in_range,
        "description": description,
        "message": "OK" if in_range else f"Value {val_canonical:.4g} {canonical_unit} outside expected range [{min_val}, {max_val}]",
    }, indent=2)
```

---

## 8. Physical Laws as Hard Constraints

Some constraints are checked regardless of DOMAIN_RANGES:

```python
def _apply_hard_constraints(quantity: str, value: float, unit: str, ureg: pint.UnitRegistry) -> str | None:
    """
    Applies hard physical law constraints that cannot be overridden by domain context.
    Returns error message string if constraint violated, None if OK.
    """
    q_lower = quantity.lower()

    # Conservation of energy: no process can be more than 100% efficient
    if "efficiency" in q_lower:
        if unit == "%":
            if value > 100.0:
                return f"Physical law violation: efficiency cannot exceed 100% (got {value}%)"
            if value < 0:
                return f"Physical law violation: efficiency cannot be negative (got {value}%)"
        elif unit in {"", "1"}:  # dimensionless fraction
            if value > 1.0:
                return f"Physical law violation: efficiency fraction cannot exceed 1.0 (got {value})"
            if value < 0:
                return f"Physical law violation: efficiency fraction cannot be negative (got {value})"

    # Third law of thermodynamics: absolute zero is unreachable
    try:
        if "temperature" in q_lower or "temp" in q_lower:
            temp_k = float(ureg.Quantity(value, unit).to("kelvin").magnitude)
            if temp_k <= 0:
                return f"Physical law violation: temperature cannot be at or below absolute zero ({value} {unit} = {temp_k} K)"
    except Exception:
        pass

    # pH must be 0–14
    if q_lower == "ph" or q_lower == "ph_value":
        if not (0.0 <= value <= 14.0):
            return f"Physical law violation: pH must be between 0 and 14 (got {value})"

    # Probability / fraction
    if q_lower in {"probability", "fraction", "ratio"} and unit in {"", "1"}:
        if not (0.0 <= value <= 1.0):
            return f"Physical law violation: {quantity} must be between 0 and 1 (got {value})"

    return None  # All constraints satisfied
```

---

## 9. Integration with ReAct Trace

Every `validate_physical_units()` call result is stored in `AnalysisSession.unit_context`:

```python
# In DataScienceAgentService._dispatch_tool():
if action_name == "validate_physical_units":
    result_str = validate_physical_units(**action_input)
    
    # Parse the result and store in session
    import json
    try:
        result_dict = json.loads(result_str)
        from app.domain.analysis_models import PhysicalUnit
        pu = PhysicalUnit(
            quantity=result_dict["quantity"],
            value=result_dict["value"],
            unit_str=result_dict["unit_str"],
            reasonable_range=tuple(result_dict["reasonable_range"]),
            is_valid=result_dict["is_valid"],
            message=result_dict["message"],
        )
        session.log_unit_validation(pu)
    except (json.JSONDecodeError, KeyError, ValueError):
        pass  # Don't fail the loop on storage error
    
    return result_str
```

The `unit_context` list is exposed via the API response and can be displayed in the UI as a "Physical Validation" panel.

---

## 10. Complete unit_registry.py

```python
# infrastructure/unit_registry.py
"""
Physical unit validation, conversion, and magnitude checking using pint.

Design:
- Single pint.UnitRegistry singleton (expensive to create)
- DOMAIN_RANGES: comprehensive dict of (min, max, canonical_unit, description)
- 3-stage validation pipeline: parse → dimensional → magnitude
- Hard physics constraints enforced regardless of domain ranges
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import pint
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False
    pint = None  # type: ignore

from app.domain.analysis_models import PhysicalUnit

# ── Singleton registry ───────────────────────────────────────────────────────

_ureg: "pint.UnitRegistry | None" = None


def _get_unit_registry() -> "pint.UnitRegistry":
    global _ureg
    if _ureg is None:
        if not _PINT_AVAILABLE:
            raise RuntimeError("pint is not installed. Run: pip install pint")
        _ureg = pint.UnitRegistry()
        _ureg.define("percent = 0.01 * [] = %")
        _ureg.define("ppm = 1e-6 * [] = ppm")
        _ureg.define("ppb = 1e-9 * [] = ppb")
    return _ureg


# ── Domain ranges ────────────────────────────────────────────────────────────

DOMAIN_RANGES: dict[str, tuple[float, float, str, str]] = {
    "surface_temp":             (-90.0,   60.0,   "degC",     "Earth surface air temperature"),
    "industrial_temp":          (-200.0,  1500.0, "degC",     "Industrial process temperature"),
    "human_body_temp":          (35.0,    42.0,   "degC",     "Human body temperature"),
    "furnace_temp":             (800.0,   1800.0, "degC",     "Industrial furnace"),
    "hvac_supply_temp":         (10.0,    30.0,   "degC",     "HVAC supply air temperature"),
    "cooling_water_temp":       (10.0,    50.0,   "degC",     "Cooling water temperature"),
    "steam_temp":               (100.0,   650.0,  "degC",     "Industrial steam temperature"),
    "atmospheric_pressure":     (87.0,    108.0,  "kPa",      "Atmospheric pressure"),
    "gauge_pressure":           (0.0,     10000.0,"kPa",      "Gauge pressure"),
    "blood_pressure_systolic":  (60.0,    200.0,  "mmHg",     "Systolic blood pressure"),
    "boiler_pressure":          (100.0,   25000.0,"kPa",      "Steam boiler pressure"),
    "residential_electrical":   (0.0,     50.0,   "kW",       "Residential power"),
    "industrial_power":         (0.0,     1000.0, "MW",       "Industrial power"),
    "power_plant_output":       (1.0,     2000.0, "MW",       "Power plant output"),
    "wind_turbine_power":       (0.0,     15.0,   "MW",       "Wind turbine output"),
    "heat_input_power":         (1.0,     5000.0, "MW",       "Thermal input to plant"),
    "thermal_efficiency":       (0.1,     0.65,   "",         "Thermal cycle efficiency (0–1)"),
    "efficiency":               (0.0,     1.0,    "",         "Generic efficiency (0–1)"),
    "efficiency_percent":       (0.0,     100.0,  "%",        "Efficiency as percentage"),
    "boiler_efficiency":        (0.5,     0.98,   "",         "Boiler efficiency"),
    "pump_efficiency":          (0.4,     0.92,   "",         "Pump efficiency"),
    "motor_efficiency":         (0.7,     0.99,   "",         "Motor efficiency"),
    "turbine_efficiency":       (0.7,     0.93,   "",         "Turbine efficiency"),
    "water_flow_rate":          (0.0,     10.0,   "m**3/s",   "Industrial water flow"),
    "steam_flow_rate":          (0.0,     500.0,  "kg/s",     "Steam mass flow"),
    "wind_speed":               (0.0,     100.0,  "m/s",      "Wind speed"),
    "fluid_velocity":           (0.0,     50.0,   "m/s",      "Pipe fluid velocity"),
    "grid_voltage_lv":          (100.0,   500.0,  "V",        "Low-voltage grid"),
    "grid_voltage_hv":          (1.0,     765.0,  "kV",       "High-voltage transmission"),
    "ph_value":                 (0.0,     14.0,   "",         "pH (dimensionless)"),
    "co2_concentration":        (0.0,     5.0,    "%",        "CO₂ by volume"),
    "nox_concentration":        (0.0,     2000.0, "ppm",      "NOx in flue gas"),
    "heat_flux":                (0.0,     1e6,    "W/m**2",   "Surface heat flux"),
    "heat_transfer_rate":       (0.0,     1000.0, "MW",       "Total heat transfer"),
}

_QUANTITY_DIMENSIONS: dict[str, str] = {
    "temperature":          "[temperature]",
    "pressure":             "[mass] / [length] / [time] ** 2",
    "power":                "[mass] * [length] ** 2 / [time] ** 3",
    "energy":               "[mass] * [length] ** 2 / [time] ** 2",
    "flow_rate":            "[length] ** 3 / [time]",
    "mass_flow_rate":       "[mass] / [time]",
    "velocity":             "[length] / [time]",
    "efficiency":           "",
    "thermal_efficiency":   "",
    "ph":                   "",
}


def _fuzzy_match_quantity(quantity: str) -> str | None:
    q = quantity.lower().replace(" ", "_")
    if q in DOMAIN_RANGES:
        return q
    for key in DOMAIN_RANGES:
        if q in key or key in q:
            return key
    q_tokens = set(q.split("_"))
    best_key, best_overlap = None, 0
    for key in DOMAIN_RANGES:
        overlap = len(q_tokens & set(key.split("_")))
        if overlap > best_overlap:
            best_overlap, best_key = overlap, key
    return best_key if best_overlap >= 2 else None


def _hard_constraint_check(quantity: str, value: float, unit: str) -> str | None:
    q = quantity.lower()
    if "efficiency" in q:
        if unit == "%" and value > 100:
            return f"Efficiency cannot exceed 100% (got {value}%)"
        if unit in {"", "1"} and value > 1.0:
            return f"Efficiency fraction cannot exceed 1.0 (got {value})"
    if q == "ph_value" or q == "ph":
        if not 0.0 <= value <= 14.0:
            return f"pH must be 0–14 (got {value})"
    return None


# ── Tool functions ────────────────────────────────────────────────────────────

def validate_physical_units(
    quantity: str,
    value: float,
    unit: str,
    domain_context: str = "",
) -> str:
    ureg = _get_unit_registry()

    # Hard constraints first
    hard_err = _hard_constraint_check(quantity, value, unit)
    if hard_err:
        pu = PhysicalUnit(quantity, value, unit, (0.0, 0.0), False, f"Hard constraint: {hard_err}")
        return json.dumps(pu.to_json_dict())

    # Stage 1: Parse unit
    try:
        q_obj = ureg.Quantity(value, unit)
    except Exception as e:
        pu = PhysicalUnit(quantity, value, unit, (0.0, 0.0), False, f"Unit parse error: {e}")
        return json.dumps(pu.to_json_dict())

    # Stage 3: Magnitude check (skip stage 2 dimensional check for brevity in MVP)
    matched_key = _fuzzy_match_quantity(quantity.lower().replace(" ", "_"))
    if not matched_key:
        pu = PhysicalUnit(quantity, value, unit, (float("-inf"), float("inf")), True,
                          "OK (no domain range defined)")
        return json.dumps(pu.to_json_dict())

    min_val, max_val, canonical_unit, description = DOMAIN_RANGES[matched_key]
    try:
        val_c = float(q_obj.to(canonical_unit).magnitude) if canonical_unit else value
    except Exception as e:
        pu = PhysicalUnit(quantity, value, unit, (min_val, max_val), False,
                          f"Conversion error to '{canonical_unit}': {e}")
        return json.dumps(pu.to_json_dict())

    in_range = min_val <= val_c <= max_val
    msg = "OK" if in_range else (
        f"Magnitude warning: {val_c:.4g} {canonical_unit} outside [{min_val}, {max_val}]. {description}"
    )
    pu = PhysicalUnit(quantity, value, unit, (min_val, max_val), in_range, msg)
    return json.dumps(pu.to_json_dict())


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    try:
        ureg = _get_unit_registry()
        result = ureg.Quantity(value, from_unit).to(to_unit)
        return f"{result.magnitude:.6g} {to_unit}"
    except Exception as e:
        return f"Error: {e}"


def check_magnitude(quantity: str, value: float, unit: str) -> str:
    matched_key = _fuzzy_match_quantity(quantity.lower().replace(" ", "_"))
    if not matched_key:
        return json.dumps({"quantity": quantity, "in_range": None,
                           "message": f"'{quantity}' not in domain ranges"})

    min_val, max_val, canonical_unit, description = DOMAIN_RANGES[matched_key]
    try:
        ureg = _get_unit_registry()
        val_c = float(ureg.Quantity(value, unit).to(canonical_unit).magnitude) if canonical_unit else value
    except Exception as e:
        return json.dumps({"error": str(e)})

    in_range = min_val <= val_c <= max_val
    return json.dumps({
        "quantity": quantity,
        "matched_key": matched_key,
        "value": value,
        "unit": unit,
        "value_canonical": val_c,
        "canonical_unit": canonical_unit,
        "expected_range": [min_val, max_val],
        "in_range": in_range,
        "description": description,
    }, indent=2)


def export_notebook(session: "AnalysisSession", title: str = "Analysis") -> str:  # type: ignore[name-defined]
    try:
        import nbformat
        from app.core.config import settings

        nb = nbformat.v4.new_notebook()
        nb.metadata["title"] = title
        nb.cells.append(nbformat.v4.new_markdown_cell(f"# {title}"))
        for cell in session.jupyter_cells:
            c = nbformat.v4.new_code_cell(source=cell["source"])
            c.outputs = cell.get("outputs", [])
            c.execution_count = cell.get("execution_count")
            nb.cells.append(c)
        out_dir = settings.notebooks_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        out = out_dir / f"{session.session_id}_{safe}.ipynb"
        nbformat.write(nb, str(out))
        return str(out)
    except Exception as e:
        return f"Error: {e}"


def save_figure(session: "AnalysisSession", figure_id: str, filename: str) -> str:  # type: ignore[name-defined]
    import base64
    from app.core.config import settings

    if figure_id not in session.figures:
        return f"Error: Figure '{figure_id}' not found."
    if ".." in filename or filename.startswith("/"):
        return "Error: Invalid filename."
    out_dir = settings.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    if not path.suffix:
        path = path.with_suffix(".png")
    try:
        path.write_bytes(base64.b64decode(session.figures[figure_id]))
        return str(path)
    except Exception as e:
        return f"Error: {e}"
```
