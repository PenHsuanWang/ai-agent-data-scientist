"""Physical unit validation tools using pint.

Provides three tool-callable functions:
  validate_physical_units(quantity, value, unit) → str (JSON PhysicalUnit)
  convert_units(value, from_unit, to_unit) → str (JSON result)
  check_magnitude(quantity, value, unit) → str (JSON PlausibilityCheck)

Also provides two output tools:
  export_notebook(session, title) → str (JSON path info)
  save_figure(session, figure_id, filename) → str (JSON path info)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, PhysicalUnit

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────── #
# pint UnitRegistry singleton                                           #
# ──────────────────────────────────────────────────────────────────── #

_ureg: Any = None


def _get_ureg() -> Any:
    global _ureg
    if _ureg is None:
        import pint
        _ureg = pint.UnitRegistry()
        _register_custom_units(_ureg)
    return _ureg


def _register_custom_units(ureg: Any) -> None:
    try:
        ureg.define("percent = 0.01 * [] = pct")
    except Exception:
        pass
    try:
        ureg.define("ppm = 1e-6 * [] = ppm")
    except Exception:
        pass
    try:
        ureg.define("ppb = 1e-9 * [] = ppb")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────── #
# Domain range registry (quantity → (min, max, canonical_unit))        #
# ──────────────────────────────────────────────────────────────────── #

DOMAIN_RANGES: dict[str, tuple[float, float, str]] = {
    # Thermodynamic efficiency (fraction or %)
    "thermal_efficiency": (0.0, 100.0, "percent"),
    "isentropic_efficiency": (0.0, 100.0, "percent"),
    "mechanical_efficiency": (0.0, 100.0, "percent"),
    "efficiency": (0.0, 100.0, "percent"),
    # Temperature (°C for operating ranges)
    "steam_temperature": (-50.0, 700.0, "degC"),
    "flue_gas_temperature": (50.0, 500.0, "degC"),
    "temperature": (-273.15, 5000.0, "degC"),
    # Pressure (MPa)
    "steam_pressure": (0.001, 35.0, "MPa"),
    "pressure": (0.0, 1000.0, "MPa"),
    # Power (MW)
    "gross_power": (0.0, 5000.0, "MW"),
    "net_power": (0.0, 5000.0, "MW"),
    "power": (-10000.0, 10000.0, "MW"),
    # Heat rate (kJ/kWh)
    "heat_rate": (2000.0, 20000.0, "kJ/kWh"),
    # CO2 (g/kWh)
    "co2_emission": (0.0, 2000.0, "g/kWh"),
    # Mass flow (kg/s)
    "mass_flow": (0.0, 10000.0, "kg/s"),
}


def _fuzzy_match_quantity(quantity: str) -> str | None:
    """Return the best-matching DOMAIN_RANGES key for a quantity name."""
    q_lower = quantity.lower()
    # Exact match
    if q_lower in DOMAIN_RANGES:
        return q_lower
    # Substring match
    for key in DOMAIN_RANGES:
        if key in q_lower or q_lower in key:
            return key
    return None


# ──────────────────────────────────────────────────────────────────── #
# Tool functions                                                        #
# ──────────────────────────────────────────────────────────────────── #


def validate_physical_units(quantity: str, value: float, unit: str) -> str:
    """Validate that a physical quantity has sensible units and magnitude.

    Returns JSON-serialised PhysicalUnit verdict.
    """
    try:
        ureg = _get_ureg()

        # Stage 1: Parse unit
        try:
            q = ureg.Quantity(value, unit)
        except Exception as e:
            result = PhysicalUnit(
                quantity=quantity,
                value=value,
                unit=unit,
                is_valid=False,
                message=f"Unit parse error: {e}",
            )
            return json.dumps(result.__dict__)

        # Stage 2: Magnitude check against domain ranges
        matched_key = _fuzzy_match_quantity(quantity)
        if matched_key:
            lo, hi, _ = DOMAIN_RANGES[matched_key]
            if not (lo <= value <= hi):
                result = PhysicalUnit(
                    quantity=quantity,
                    value=value,
                    unit=unit,
                    is_valid=False,
                    message=(
                        f"Magnitude {value} {unit} is outside expected range "
                        f"[{lo}, {hi}] for '{matched_key}'. "
                        f"This may indicate a unit mismatch or a data error."
                    ),
                )
                return json.dumps(result.__dict__)

        result = PhysicalUnit(
            quantity=quantity,
            value=value,
            unit=unit,
            is_valid=True,
            message=f"OK — {value} {unit} is physically plausible for '{quantity}'.",
        )
        return json.dumps(result.__dict__)

    except Exception as exc:
        logger.error("validate_physical_units error: %s", exc)
        return json.dumps({
            "quantity": quantity, "value": value, "unit": unit,
            "is_valid": False, "message": f"Error: {exc}",
        })


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value from one unit to another using pint.

    Returns JSON with original and converted values.
    """
    try:
        ureg = _get_ureg()
        q = ureg.Quantity(value, from_unit)
        converted = q.to(to_unit)
        return json.dumps({
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": round(float(converted.magnitude), 6),
            "converted_unit": to_unit,
            "success": True,
        })
    except Exception as exc:
        logger.error("convert_units error: %s", exc)
        return json.dumps({"success": False, "error": str(exc)})


def check_magnitude(quantity: str, value: float, unit: str) -> str:
    """Check whether a value is physically plausible for the given quantity.

    Returns JSON with is_plausible flag and domain range context.
    """
    matched_key = _fuzzy_match_quantity(quantity)
    if not matched_key:
        return json.dumps({
            "quantity": quantity,
            "value": value,
            "unit": unit,
            "is_plausible": True,
            "message": f"No domain range registered for '{quantity}'. Cannot verify.",
        })

    lo, hi, canonical = DOMAIN_RANGES[matched_key]
    is_plausible = lo <= value <= hi
    return json.dumps({
        "quantity": quantity,
        "matched_domain_key": matched_key,
        "value": value,
        "unit": unit,
        "expected_range": {"min": lo, "max": hi, "canonical_unit": canonical},
        "is_plausible": is_plausible,
        "message": (
            f"Value {value} {unit} is within the expected range [{lo}, {hi}] for '{matched_key}'."
            if is_plausible
            else (
                f"⚠ Value {value} {unit} is OUTSIDE the expected range [{lo}, {hi}] for '{matched_key}'. "
                f"Check for unit mismatch (e.g. fraction vs percentage) or data error."
            )
        ),
    })
