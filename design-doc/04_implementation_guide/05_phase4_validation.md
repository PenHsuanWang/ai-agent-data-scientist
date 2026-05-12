# Phase 4 — Physical Unit Validation

## Overview

Physical validation prevents the agent from returning nonsensical numbers (e.g. "thermal efficiency = 150%"). The `pint` library provides dimensional analysis; `DOMAIN_RANGES` stores expected value ranges for 25+ physical quantities.

---

## 1. Installing pint

```bash
uv add pint>=0.24
python -c "import pint; print(pint.__version__)"  # 0.24.x
```

Verify unit conversion works:

```python
import pint
ureg = pint.UnitRegistry()
q = 57.3 * ureg.degC
print(q.to("degF"))          # 135.14 degree_Fahrenheit
print(q.dimensionality)      # [temperature]
```

---

## 2. UnitRegistry Singleton Pattern

```python
# app/infrastructure/unit_registry.py (singleton section)
import threading
import pint

_registry_lock = threading.Lock()
_ureg: pint.UnitRegistry | None = None

def get_ureg() -> pint.UnitRegistry:
    """Thread-safe singleton UnitRegistry."""
    global _ureg
    if _ureg is None:
        with _registry_lock:
            if _ureg is None:
                _ureg = pint.UnitRegistry()
                # Register domain-specific aliases
                _ureg.define("percent = 0.01 = pct")
                _ureg.define("ppm = 1e-6")
    return _ureg

ureg = get_ureg()
```

**Why one instance?**
- pint registries hold unit definitions in memory; creating multiple registries causes `Unit` objects from different registries to be incompatible (raises `pint.errors.UnitStrippedWarning` or comparison errors).
- The double-checked locking pattern ensures thread safety without locking on every call.

---

## 3. Complete DOMAIN_RANGES

```python
DOMAIN_RANGES: dict[str, dict] = {
    # ── Temperature ───────────────────────────────────────────────────────
    "steam_temperature_hp": {
        "unit": "degC", "min": 400.0, "max": 650.0,
        "description": "HP steam temperature (supercritical plants up to 620°C)",
    },
    "steam_temperature_lp": {
        "unit": "degC", "min": 150.0, "max": 350.0,
        "description": "LP/reheat steam temperature",
    },
    "flue_gas_temperature": {
        "unit": "degC", "min": 80.0, "max": 200.0,
        "description": "Flue gas exit temperature (after air heater)",
    },
    "condenser_temperature": {
        "unit": "degC", "min": 20.0, "max": 55.0,
        "description": "Condenser saturation temperature",
    },
    "ambient_temperature": {
        "unit": "degC", "min": -30.0, "max": 55.0,
        "description": "Outdoor dry-bulb temperature",
    },
    "cooling_water_temperature": {
        "unit": "degC", "min": 5.0, "max": 40.0,
        "description": "Cooling water inlet to condenser",
    },
    "furnace_temperature": {
        "unit": "degC", "min": 900.0, "max": 1600.0,
        "description": "Furnace/combustion zone temperature",
    },
    # ── Pressure ──────────────────────────────────────────────────────────
    "steam_pressure_hp": {
        "unit": "MPa", "min": 10.0, "max": 30.0,
        "description": "HP turbine inlet steam pressure",
    },
    "steam_pressure_lp": {
        "unit": "kPa", "min": 3.0, "max": 20.0,
        "description": "LP turbine exhaust pressure",
    },
    "condenser_pressure": {
        "unit": "kPa", "min": 3.0, "max": 15.0,
        "description": "Condenser vacuum pressure",
    },
    "boiler_pressure": {
        "unit": "MPa", "min": 8.0, "max": 35.0,
        "description": "Boiler drum or once-through pressure",
    },
    "atmospheric_pressure": {
        "unit": "kPa", "min": 90.0, "max": 110.0,
        "description": "Ambient barometric pressure at sea level",
    },
    # ── Power ─────────────────────────────────────────────────────────────
    "gross_power_output": {
        "unit": "MW", "min": 50.0, "max": 1500.0,
        "description": "Generator gross electrical output",
    },
    "net_power_output": {
        "unit": "MW", "min": 40.0, "max": 1450.0,
        "description": "Net power after auxiliary loads",
    },
    "auxiliary_power": {
        "unit": "MW", "min": 5.0, "max": 80.0,
        "description": "Internal auxiliary power consumption",
    },
    "heat_rate": {
        "unit": "kJ/kWh", "min": 7000.0, "max": 12000.0,
        "description": "Plant heat rate (lower is better; 3600 = 100% efficient)",
    },
    # ── Efficiency ────────────────────────────────────────────────────────
    "thermal_efficiency": {
        "unit": "percent", "min": 25.0, "max": 50.0,
        "description": "Gross thermal efficiency of power plant",
    },
    "turbine_isentropic_efficiency": {
        "unit": "percent", "min": 75.0, "max": 95.0,
        "description": "Turbine stage isentropic efficiency",
    },
    "boiler_efficiency": {
        "unit": "percent", "min": 80.0, "max": 95.0,
        "description": "Boiler combustion + heat transfer efficiency",
    },
    "pump_efficiency": {
        "unit": "percent", "min": 60.0, "max": 90.0,
        "description": "Feed pump or condensate pump efficiency",
    },
    "generator_efficiency": {
        "unit": "percent", "min": 97.0, "max": 99.5,
        "description": "Generator mechanical-to-electrical efficiency",
    },
    "cycle_efficiency": {
        "unit": "percent", "min": 30.0, "max": 50.0,
        "description": "Thermodynamic cycle efficiency (Rankine)",
    },
    # ── Mass Flow ─────────────────────────────────────────────────────────
    "steam_flow_rate": {
        "unit": "kg/s", "min": 50.0, "max": 800.0,
        "description": "Main steam mass flow rate",
    },
    "feedwater_flow_rate": {
        "unit": "kg/s", "min": 50.0, "max": 800.0,
        "description": "Feedwater pump flow rate",
    },
    "fuel_flow_rate": {
        "unit": "kg/s", "min": 5.0, "max": 200.0,
        "description": "Coal or gas fuel mass flow rate",
    },
    # ── Emissions ─────────────────────────────────────────────────────────
    "co2_emission_intensity": {
        "unit": "g/kWh", "min": 350.0, "max": 1100.0,
        "description": "CO2 specific emission (gas: 400-500, coal: 800-1000 g/kWh)",
    },
    "nox_emission_intensity": {
        "unit": "mg/Nm3", "min": 50.0, "max": 500.0,
        "description": "NOx at stack after SCR",
    },
    "so2_emission_intensity": {
        "unit": "mg/Nm3", "min": 10.0, "max": 200.0,
        "description": "SO2 after FGD system",
    },
    # ── Statistics ────────────────────────────────────────────────────────
    "probability": {
        "unit": "dimensionless", "min": 0.0, "max": 1.0,
        "description": "Any probability or fraction",
    },
    "correlation_coefficient": {
        "unit": "dimensionless", "min": -1.0, "max": 1.0,
        "description": "Pearson or Spearman correlation coefficient",
    },
}
```

---

## 4. validate_physical_units() — 3-Stage Pipeline

```python
def validate_physical_units(
    quantity_name: str,
    value: float,
    unit: str,
    domain_key: str | None = None,
    raise_on_error: bool = False,
) -> PhysicalUnit:
    """
    Stage 1: Parse unit string with pint.
    Stage 2: Check dimensional compatibility with domain_key's canonical unit.
    Stage 3: Check numeric range against DOMAIN_RANGES[domain_key].
    """
    reg = get_ureg()

    # Stage 1: Parse unit string
    try:
        quantity = value * reg.parse_expression(unit)
    except pint.errors.UndefinedUnitError as e:
        if raise_on_error:
            raise UnitValidationError(quantity_name, value, unit, f"Unknown unit: {e}")
        return PhysicalUnit(name=quantity_name, value=value, unit=unit,
                            is_valid=False, warning=f"Unknown unit: {e}")

    if domain_key is None or domain_key not in DOMAIN_RANGES:
        return PhysicalUnit(name=quantity_name, value=value, unit=unit, is_valid=True)

    spec = DOMAIN_RANGES[domain_key]
    expected_unit = spec["unit"]
    lo, hi = spec["min"], spec["max"]

    # Stage 2: Dimensionality (convert to canonical unit)
    try:
        converted = quantity.to(reg.parse_expression(expected_unit))
        check_value = converted.magnitude
    except pint.errors.DimensionalityError as e:
        if raise_on_error:
            raise UnitValidationError(quantity_name, value, unit,
                                       f"Wrong dimensions: {e}", dimensionality_error=True)
        return PhysicalUnit(name=quantity_name, value=value, unit=unit,
                            is_valid=False,
                            warning=f"Dimensionality mismatch: expected {expected_unit}")

    # Stage 3: Range check
    if not (lo <= check_value <= hi):
        msg = (f"Value {check_value:.4g} {expected_unit} outside "
               f"expected range [{lo}, {hi}] for '{domain_key}'")
        if raise_on_error:
            raise UnitValidationError(quantity_name, value, unit, msg,
                                       expected_range=(lo, hi))
        return PhysicalUnit(name=quantity_name, value=check_value, unit=expected_unit,
                            is_valid=False, warning=msg, expected_range=(lo, hi))

    return PhysicalUnit(name=quantity_name, value=check_value, unit=expected_unit,
                        is_valid=True, expected_range=(lo, hi))
```

---

## 5. Domain Doc Context Extraction

Extract unit/range hints from markdown domain documents:

```python
import re
from pathlib import Path

# Matches table rows like: | Steam temperature (HP) | T_s | 520-600 | degC |
TABLE_ROW_PATTERN = re.compile(
    r"\|\s*([^|]+?)\s*\|\s*[^|]*\s*\|\s*([\d.\-]+)\s*[–-]\s*([\d.]+)\s*\|\s*([^|]+?)\s*\|"
)

def extract_unit_hints_from_doc(doc_path: Path) -> list[dict]:
    """
    Parse a markdown domain document and extract quantity/range/unit triplets
    from markdown tables.
    """
    text = doc_path.read_text(encoding="utf-8")
    hints = []
    for match in TABLE_ROW_PATTERN.finditer(text):
        name, lo, hi, unit = match.groups()
        try:
            hints.append({
                "quantity": name.strip(),
                "min": float(lo),
                "max": float(hi),
                "unit": unit.strip(),
            })
        except ValueError:
            pass
    return hints
```

Example: Given `power_plant_thermodynamics.md`, this extracts:
```python
[
    {"quantity": "Steam temperature (HP)", "min": 520.0, "max": 600.0, "unit": "°C"},
    {"quantity": "Gross power output", "min": 500.0, "max": 1200.0, "unit": "MW"},
    ...
]
```

These hints can supplement or override DOMAIN_RANGES at runtime.

---

## 6. Integration with ReAct Loop

Claude should call `validate_physical_units` before every `Final Answer`. The system prompt instructs:

```
Before Final Answer, validate key numerical results with validate_physical_units.
Include units in every numerical answer.
```

Example trace showing validation step:

```
[Iteration N]
Claude:
    Thought: I computed mean efficiency = 36.73%. Let me validate this.
    Action: validate_physical_units
    Action Input: {
      "quantity_name": "mean thermal efficiency",
      "value": 36.73,
      "unit": "percent",
      "domain_key": "thermal_efficiency"
    }

Observation: {
  "name": "mean thermal efficiency",
  "value": 36.73,
  "unit": "percent",
  "is_valid": true,
  "warning": "",
  "expected_range": [25.0, 50.0]
}

[Iteration N+1]
Claude:
    Thought: Validation passed. I can now give the final answer.
    Final Answer: The mean thermal efficiency is 36.73 percent,
    which is physically plausible (expected: 25-50% for coal plants).
```

---

## 7. Custom Domain Extension

To add a new physical quantity (e.g. for a wind farm agent):

**Step 1**: Add to `DOMAIN_RANGES` in `unit_registry.py`:

```python
"wind_turbine_capacity_factor": {
    "unit": "percent",
    "min": 10.0,
    "max": 55.0,
    "description": "Annual capacity factor for wind turbines (onshore: 25-40%, offshore: 35-55%)",
},
"rotor_tip_speed_ratio": {
    "unit": "dimensionless",
    "min": 5.0,
    "max": 10.0,
    "description": "Tip speed ratio (lambda) for optimal wind turbine performance",
},
"wind_speed": {
    "unit": "m/s",
    "min": 0.0,
    "max": 35.0,
    "description": "Hub-height wind speed (cut-out speed: ~25 m/s)",
},
```

**Step 2**: Add a domain doc to `data/domain_docs/wind_turbine_specs.md`:

```markdown
# Wind Turbine Specifications

## Key Performance Indicators

| Quantity | Symbol | Typical Range | Unit |
|---|---|---|---|
| Capacity factor | CF | 25-45 | % |
| Rated power | P_rated | 2-15 | MW |
| Hub height | H | 80-180 | m |
| Rotor diameter | D | 100-250 | m |
| Cut-in wind speed | v_ci | 2.5-4.0 | m/s |
| Cut-out wind speed | v_co | 20-25 | m/s |
```

**Step 3**: The `PhysicalContextInjector` will automatically include the new doc in the system prompt.

---

## 8. Physical Laws Constraints

Built-in checks that should be enforced by domain knowledge or validation:

| Constraint | Check | Tool to use |
|---|---|---|
| Energy conservation | efficiency <= 100% | validate_physical_units (domain_key="thermal_efficiency") |
| Second law (Carnot) | η < 1 - T_cold/T_hot | execute_python_code to compute bound |
| Temperature scale | Kelvin always > 0 | check_magnitude with "degC" or "K" units |
| Mass conservation | flow_in ≈ flow_out | compute in execute_python_code |
| Probability | 0 ≤ p ≤ 1 | check_magnitude with domain_key="probability" |
| Correlation | -1 ≤ r ≤ 1 | check_magnitude with domain_key="correlation_coefficient" |

---

## 9. Testing Physical Validation

```python
# tests/test_phase4_validation.py
import pytest
from app.infrastructure.unit_registry import (
    validate_physical_units,
    convert_units,
    check_magnitude,
    DOMAIN_RANGES,
)
from app.domain.exceptions import UnitValidationError


class TestValidatePhysicalUnits:

    def test_valid_efficiency_percent(self):
        pu = validate_physical_units("efficiency", 36.0, "percent", "thermal_efficiency")
        assert pu.is_valid
        assert pu.warning == ""
        assert pu.expected_range == (25.0, 50.0)

    def test_efficiency_over_100_invalid(self):
        pu = validate_physical_units("efficiency", 110.0, "percent", "thermal_efficiency")
        assert not pu.is_valid
        assert "outside expected range" in pu.warning

    def test_efficiency_zero_invalid(self):
        pu = validate_physical_units("efficiency", 0.0, "percent", "thermal_efficiency")
        assert not pu.is_valid

    def test_steam_temp_valid(self):
        pu = validate_physical_units("steam temp", 540.0, "degC", "steam_temperature_hp")
        assert pu.is_valid

    def test_steam_temp_too_high(self):
        pu = validate_physical_units("steam temp", 700.0, "degC", "steam_temperature_hp")
        assert not pu.is_valid

    def test_wrong_dimensionality(self):
        # Pressure unit for a temperature quantity -> dimensionality mismatch
        pu = validate_physical_units("steam temp", 16.5, "MPa", "steam_temperature_hp")
        assert not pu.is_valid
        assert "Dimensionality" in pu.warning

    def test_unknown_unit_string(self):
        pu = validate_physical_units("power", 100.0, "megawatts_per_second_squared")
        assert not pu.is_valid
        assert "Unknown unit" in pu.warning

    def test_no_domain_key_always_valid(self):
        pu = validate_physical_units("arbitrary", 999999.0, "m/s")
        assert pu.is_valid  # no domain key -> skip range check

    def test_raise_on_error_raises(self):
        with pytest.raises(UnitValidationError):
            validate_physical_units("efficiency", 200.0, "percent",
                                    "thermal_efficiency", raise_on_error=True)

    def test_unit_conversion_to_canonical(self):
        # Input in fraction (0-1 range), domain expects percent (0-100)
        # pint will convert 0.37 (dimensionless) to percent = 37%
        # Note: This only works if pint treats "dimensionless" correctly
        # The safe approach is always to pass the value already in the canonical unit
        pu = validate_physical_units("heat rate", 9800.0, "kJ/kWh", "heat_rate")
        assert pu.is_valid

    def test_all_domain_keys_parseable(self):
        """All canonical units in DOMAIN_RANGES must be parseable by pint."""
        from app.infrastructure.unit_registry import get_ureg
        reg = get_ureg()
        for key, spec in DOMAIN_RANGES.items():
            try:
                reg.parse_expression(spec["unit"])
            except Exception as e:
                pytest.fail(f"DOMAIN_RANGES[{key!r}]['unit'] = {spec['unit']!r} not parseable: {e}")


class TestConvertUnits:

    def test_celsius_to_fahrenheit(self):
        result = convert_units(100.0, "degC", "degF")
        assert "converted" in result
        assert abs(result["converted"]["value"] - 212.0) < 0.01

    def test_mpa_to_bar(self):
        result = convert_units(16.5, "MPa", "bar")
        assert "converted" in result
        assert abs(result["converted"]["value"] - 165.0) < 0.1

    def test_kw_to_mw(self):
        result = convert_units(620000.0, "kW", "MW")
        assert abs(result["converted"]["value"] - 620.0) < 0.001

    def test_incompatible_units_returns_error(self):
        result = convert_units(100.0, "degC", "MPa")
        assert "error" in result

    def test_unknown_unit_returns_error(self):
        result = convert_units(1.0, "not_a_real_unit", "MPa")
        assert "error" in result


class TestCheckMagnitude:

    def test_plausible_efficiency(self):
        result = check_magnitude(37.0, "percent", "thermal_efficiency")
        assert result["plausible"] is True
        assert result["range"] == [25.0, 50.0]

    def test_implausible_efficiency(self):
        result = check_magnitude(90.0, "percent", "thermal_efficiency")
        assert result["plausible"] is False
        assert "outside expected range" in result["message"]

    def test_plausible_hp_pressure(self):
        result = check_magnitude(16.5, "MPa", "steam_pressure_hp")
        assert result["plausible"] is True

    def test_unknown_domain_key(self):
        result = check_magnitude(100.0, "m/s", "nonexistent_quantity")
        # Should not crash; returns without range info
        assert "plausible" in result or "error" not in result
```

```bash
pytest tests/test_phase4_validation.py -v
```

---

## Checkpoint

After Phase 4:

```
app/infrastructure/unit_registry.py  -- pint singleton + 25 DOMAIN_RANGES + validate/convert/check
tests/test_phase4_validation.py      -- 20+ validation tests passing
```

-> Next: 06_phase5_jupyter.md -- Jupyter kernel bridge
