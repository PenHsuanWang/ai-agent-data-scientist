#!/usr/bin/env python3
"""Generate sample datasets and domain documents for development and testing."""
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data")
(BASE / "domain_docs").mkdir(parents=True, exist_ok=True)
(BASE / "datasets").mkdir(parents=True, exist_ok=True)
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/notebooks").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 500

# power_plant_data.csv
timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
heat_rates = rng.normal(9800, 120, n)
df = pd.DataFrame({
    "timestamp": timestamps,
    "steam_temp_C": rng.normal(540, 5, n).round(2),
    "steam_pressure_MPa": rng.normal(16.5, 0.3, n).round(3),
    "gross_power_MW": rng.normal(620, 15, n).round(2),
    "auxiliary_power_MW": rng.normal(18, 2, n).round(2),
    "heat_rate_kJ_kWh": heat_rates.round(1),
    "flue_gas_temp_C": rng.normal(135, 8, n).round(2),
    "co2_emission_g_kWh": rng.normal(820, 25, n).round(1),
    "efficiency_pct": (3600 / heat_rates * 100).round(3),
    "fuel_flow_kg_s": rng.normal(55, 3, n).round(3),
})
df.to_csv(BASE / "datasets/power_plant_data.csv", index=False)
print(f"✅ power_plant_data.csv ({len(df)} rows)")

# turbine_efficiency.parquet
load_pct = np.linspace(40, 100, 100)
df2 = pd.DataFrame({
    "load_pct": load_pct,
    "isentropic_efficiency": 0.88 - 0.0008 * (load_pct - 100)**2 / 100,
    "mechanical_efficiency": rng.uniform(0.985, 0.995, 100).round(4),
    "stage": rng.choice(["HP", "IP", "LP"], 100),
})
df2.to_parquet(BASE / "datasets/turbine_efficiency.parquet", index=False)
print("✅ turbine_efficiency.parquet")

# sensor_readings.xlsx
df3 = pd.DataFrame({
    "sensor_id": [f"S{i:04d}" for i in range(1, 51)],
    "location": rng.choice(["boiler", "turbine", "condenser", "cooling_tower"], 50),
    "temp_K": rng.normal(800, 50, 50).round(1),
    "pressure_kPa": rng.normal(1500, 100, 50).round(1),
    "flow_kg_s": rng.normal(200, 20, 50).round(2),
})
df3.to_excel(BASE / "datasets/sensor_readings.xlsx", index=False)
print("✅ sensor_readings.xlsx")

# domain_docs/power_plant_thermodynamics.md
Path(BASE / "domain_docs/power_plant_thermodynamics.md").write_text("""# Power Plant Thermodynamics

## Rankine Cycle Overview

A steam power plant converts heat to electrical energy via the Rankine cycle with four stages:
1. **Pump** (1→2): Isentropic compression of liquid water
2. **Boiler** (2→3): Constant-pressure heat addition, water → superheated steam
3. **Turbine** (3→4): Isentropic expansion producing shaft work
4. **Condenser** (4→1): Constant-pressure heat rejection

## Key Performance Indicators (KPIs)

| Quantity | Symbol | Typical Range | Unit |
|---|---|---|---|
| HP Steam temperature | T_s | 520–600 | °C |
| HP Steam pressure | P_s | 14–20 | MPa |
| Gross power output | P_gross | 500–1200 | MW |
| Thermal efficiency | η | 35–48 | % |
| Heat rate | HR | 7500–10500 | kJ/kWh |
| Flue gas temperature | T_fg | 110–160 | °C |
| CO₂ emissions | e_CO2 | 750–900 | g/kWh |

## Efficiency Formulas

Thermal efficiency: **η = 3600 / HR × 100 [%]**

Where HR is heat rate in kJ/kWh.

Net thermal efficiency: η_net = (P_gross − P_auxiliary) / Q_fuel_input

## Physical Constraints (Hard Limits)

- **First Law**: Efficiency > 100% is thermodynamically impossible
- HP steam temperature bounded by metallurgy: max 620°C for conventional steel
- Condenser pressure: 4–10 kPa (saturation temperature 28–46°C)
- Minimum heat rate for a modern coal plant: ~7,000 kJ/kWh

## Common Unit Conversions

- 1 MW = 1000 kW = 1,000,000 W
- 1 MPa = 10 bar = 145.04 psi
- Heat rate [kJ/kWh] = 3600 / efficiency [fraction]
- T[K] = T[°C] + 273.15
""")
print("✅ power_plant_thermodynamics.md")

# domain_docs/unit_definitions.md
Path(BASE / "domain_docs/unit_definitions.md").write_text("""# Unit Definitions for Power Plant Analysis

## Temperature
- °C (Celsius): T[K] = T[°C] + 273.15
- K (Kelvin): thermodynamic absolute temperature (never negative)
- °F (Fahrenheit): T[°F] = T[°C] × 9/5 + 32

## Pressure
- Pa: SI base unit (1 N/m²)
- kPa = 1000 Pa; condenser pressure 4–10 kPa
- MPa = 10⁶ Pa; HP steam pressure 14–20 MPa
- bar = 100 kPa ≈ 1 atm

## Power and Energy
- W (Watt) = 1 J/s
- kW = 10³ W; MW = 10⁶ W; GW = 10⁹ W
- kWh = 3600 kJ (energy)
- kJ/kWh = heat rate unit (fuel input per electrical output)

## Mass Flow
- kg/s (SI standard for mass flow rate)
- t/h (tonnes per hour): 1 t/h = 1000/3600 kg/s ≈ 0.2778 kg/s

## Efficiency
- Dimensionless fraction (0–1) OR percentage (0%–100%)
- Isentropic efficiency: actual work / ideal (isentropic) work
- Thermal efficiency: net electrical output / fuel heat input

## Emission Factors
- g/kWh: grams of pollutant per kilowatt-hour of electricity generated
- Typical CO₂ for coal: 820 g/kWh; natural gas: ~450 g/kWh
""")
print("✅ unit_definitions.md")

print("\n✅ Sample data setup complete.")
