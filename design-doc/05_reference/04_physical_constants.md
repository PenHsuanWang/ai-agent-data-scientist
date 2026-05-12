# Physical Constants and Domain Ranges Reference

---

## 1. SI Base Units

| Quantity | Unit Name | Symbol | Definition |
|---|---|---|---|
| Length | metre | m | Distance travelled by light in 1/299,792,458 s |
| Mass | kilogram | kg | Planck constant artefact (IPK since 2019) |
| Time | second | s | 9,192,631,770 hyperfine transitions of ¹³³Cs |
| Electric current | ampere | A | Defined via elementary charge e |
| Temperature | kelvin | K | Defined via Boltzmann constant k_B |
| Amount of substance | mole | mol | 6.02214076 × 10²³ elementary entities |
| Luminous intensity | candela | cd | Defined via luminous efficacy of 540 THz radiation |

---

## 2. Common Derived Units

| Quantity | Unit | Symbol | In SI Base |
|---|---|---|---|
| Force | newton | N | kg⋅m⋅s⁻² |
| Pressure | pascal | Pa | kg⋅m⁻¹⋅s⁻² = N/m² |
| Energy | joule | J | kg⋅m²⋅s⁻² = N⋅m |
| Power | watt | W | kg⋅m²⋅s⁻³ = J/s |
| Charge | coulomb | C | A⋅s |
| Voltage | volt | V | kg⋅m²⋅s⁻³⋅A⁻¹ = W/A |
| Resistance | ohm | Ω | kg⋅m²⋅s⁻³⋅A⁻² = V/A |
| Capacitance | farad | F | kg⁻¹⋅m⁻²⋅s⁴⋅A² = C/V |
| Inductance | henry | H | kg⋅m²⋅s⁻²⋅A⁻² = V⋅s/A |
| Magnetic flux | weber | Wb | kg⋅m²⋅s⁻²⋅A⁻¹ |
| Magnetic flux density | tesla | T | kg⋅s⁻²⋅A⁻¹ |
| Frequency | hertz | Hz | s⁻¹ |
| Luminous flux | lumen | lm | cd⋅sr |
| Illuminance | lux | lx | cd⋅sr⋅m⁻² |
| Radioactivity | becquerel | Bq | s⁻¹ |
| Absorbed dose | gray | Gy | m²⋅s⁻² |
| Catalytic activity | katal | kat | mol⋅s⁻¹ |
| Angle | radian | rad | dimensionless |
| Solid angle | steradian | sr | dimensionless |

### Non-SI Units Accepted by the SI

| Quantity | Unit | Symbol | SI Equivalent |
|---|---|---|---|
| Time | minute | min | 60 s |
| Time | hour | h | 3600 s |
| Time | day | d | 86400 s |
| Angle | degree | ° | π/180 rad |
| Volume | litre | L | 10⁻³ m³ |
| Mass | tonne | t | 10³ kg |
| Energy | kilowatt-hour | kWh | 3.6 × 10⁶ J |
| Pressure | bar | bar | 10⁵ Pa |
| Length | astronomical unit | au | 1.495978707 × 10¹¹ m |
| Area | hectare | ha | 10⁴ m² |

---

## 3. Fundamental Physical Constants (CODATA 2018)

| Constant | Symbol | Value | Unit |
|---|---|---|---|
| Speed of light (vacuum) | c | 299,792,458 | m⋅s⁻¹ |
| Planck constant | h | 6.62607015 × 10⁻³⁴ | J⋅s |
| Reduced Planck constant | ħ | 1.054571817 × 10⁻³⁴ | J⋅s |
| Elementary charge | e | 1.602176634 × 10⁻¹⁹ | C |
| Boltzmann constant | k_B | 1.380649 × 10⁻²³ | J⋅K⁻¹ |
| Avogadro constant | N_A | 6.02214076 × 10²³ | mol⁻¹ |
| Molar gas constant | R | 8.314462618 | J⋅mol⁻¹⋅K⁻¹ |
| Stefan-Boltzmann constant | σ | 5.670374419 × 10⁻⁸ | W⋅m⁻²⋅K⁻⁴ |
| Gravitational constant | G | 6.67430 × 10⁻¹¹ | m³⋅kg⁻¹⋅s⁻² |
| Standard gravity | g₀ | 9.80665 | m⋅s⁻² |
| Faraday constant | F | 96485.33212 | C⋅mol⁻¹ |
| Electron mass | mₑ | 9.1093837015 × 10⁻³¹ | kg |
| Proton mass | mₚ | 1.67262192369 × 10⁻²⁷ | kg |
| Neutron mass | mₙ | 1.67492749804 × 10⁻²⁷ | kg |
| Atomic mass unit | u | 1.66053906660 × 10⁻²⁷ | kg |
| Bohr radius | a₀ | 5.29177210903 × 10⁻¹¹ | m |
| Fine structure constant | α | 7.2973525693 × 10⁻³ | dimensionless |
| Rydberg constant | R∞ | 10,973,731.568160 | m⁻¹ |
| Vacuum permittivity | ε₀ | 8.8541878128 × 10⁻¹² | F⋅m⁻¹ |
| Vacuum permeability | μ₀ | 1.25663706212 × 10⁻⁶ | N⋅A⁻² |

---

## 4. DOMAIN_RANGES — All 25 Quantities

These are the exact ranges used in `unit_registry.py`. Each entry maps a
quantity name to `(min, max, canonical_pint_unit)`.

```python
DOMAIN_RANGES: dict[str, tuple[float, float, str]] = {
    # ----- Thermal / Power Plant -----
    "thermal_efficiency":          (0.20,   0.55,   "dimensionless"),
    "heat_rate":                   (6_000,  12_000, "kilojoule / kilowatt_hour"),
    "steam_temperature":           (200.0,  650.0,  "degC"),
    "steam_pressure":              (1.0,    35.0,   "megapascal"),
    "flue_gas_temperature":        (100.0,  450.0,  "degC"),
    "cooling_water_temperature":   (10.0,   50.0,   "degC"),
    "power_output":                (1.0,    2_000.0,"megawatt"),
    "auxiliary_power":             (0.5,    200.0,  "megawatt"),
    "co2_emission_rate":           (0.3,    1.2,    "tonne / megawatt_hour"),
    # ----- Process / Chemical -----
    "mass_flow_rate":              (0.1,    5_000.0,"kilogram / second"),
    "volumetric_flow_rate":        (1e-4,   100.0,  "meter ** 3 / second"),
    "concentration":               (0.0,    1_000.0,"gram / liter"),
    "reaction_temperature":        (-80.0,  400.0,  "degC"),
    "reaction_pressure":           (0.01,   20.0,   "megapascal"),
    "conversion":                  (0.0,    1.0,    "dimensionless"),
    # ----- Electrical / Grid -----
    "voltage":                     (0.1,    1_100.0,"kilovolt"),
    "current":                     (0.0,    50_000.0,"ampere"),
    "frequency":                   (45.0,   65.0,   "hertz"),
    "power_factor":                (0.0,    1.0,    "dimensionless"),
    # ----- Environmental -----
    "ambient_temperature":         (-60.0,  60.0,   "degC"),
    "relative_humidity":           (0.0,    100.0,  "percent"),
    "wind_speed":                  (0.0,    100.0,  "meter / second"),
    "solar_irradiance":            (0.0,    1_400.0,"watt / meter ** 2"),
    # ----- Statistical (always valid) -----
    "probability":                 (0.0,    1.0,    "dimensionless"),
    "correlation":                 (-1.0,   1.0,    "dimensionless"),
}
```

### Reading the Table

| Key | Min | Max | Canonical Unit | Notes |
|---|---|---|---|---|
| `thermal_efficiency` | 0.20 | 0.55 | dimensionless | fraction, not percent; Carnot sets upper bound |
| `heat_rate` | 6 000 | 12 000 | kJ/kWh | lower = more efficient |
| `steam_temperature` | 200 | 650 | °C | HP inlet; USC plants reach 620°C |
| `steam_pressure` | 1 | 35 | MPa | subcritical < 22.1 MPa; supercritical > 22.1 MPa |
| `flue_gas_temperature` | 100 | 450 | °C | below dewpoint risks acid corrosion |
| `cooling_water_temperature` | 10 | 50 | °C | determines condenser back-pressure |
| `power_output` | 1 | 2 000 | MW | net electrical output |
| `auxiliary_power` | 0.5 | 200 | MW | pumps, fans, controls |
| `co2_emission_rate` | 0.3 | 1.2 | t/MWh | gas 0.35–0.45; coal 0.75–1.05 |
| `mass_flow_rate` | 0.1 | 5 000 | kg/s | wide range covers pipe to pipeline |
| `volumetric_flow_rate` | 1×10⁻⁴ | 100 | m³/s | lab to large-scale reactor |
| `concentration` | 0 | 1 000 | g/L | covers trace to near-pure |
| `reaction_temperature` | −80 | 400 | °C | cryogenic to moderate-high temp |
| `reaction_pressure` | 0.01 | 20 | MPa | vacuum to high-pressure reactor |
| `conversion` | 0 | 1 | dimensionless | fraction; 1.0 = complete conversion |
| `voltage` | 0.1 | 1 100 | kV | low-voltage to ultra-high-voltage |
| `current` | 0 | 50 000 | A | electronic to power systems |
| `frequency` | 45 | 65 | Hz | allows for grid frequency deviations |
| `power_factor` | 0 | 1 | dimensionless | purely resistive = 1 |
| `ambient_temperature` | −60 | 60 | °C | Earth surface extremes |
| `relative_humidity` | 0 | 100 | % | |
| `wind_speed` | 0 | 100 | m/s | calm to extreme gust |
| `solar_irradiance` | 0 | 1 400 | W/m² | 1361 W/m² = solar constant |
| `probability` | 0 | 1 | dimensionless | axiom |
| `correlation` | −1 | 1 | dimensionless | Cauchy-Schwarz bound |

---

## 5. Unit Conversion Factors

### Energy

| From | To | Factor |
|---|---|---|
| 1 kWh | J | 3,600,000 |
| 1 kWh | kJ | 3,600 |
| 1 kWh | MJ | 3.6 |
| 1 BTU | J | 1,055.06 |
| 1 BTU | kJ | 1.05506 |
| 1 calorie | J | 4.184 |
| 1 kcal | kJ | 4.184 |
| 1 eV | J | 1.602176634 × 10⁻¹⁹ |
| 1 tonne of coal equivalent (tce) | GJ | 29.307 |
| 1 tonne of oil equivalent (toe) | GJ | 41.868 |

### Pressure

| From | To | Factor |
|---|---|---|
| 1 MPa | Pa | 1,000,000 |
| 1 MPa | bar | 10 |
| 1 MPa | kPa | 1,000 |
| 1 bar | Pa | 100,000 |
| 1 atm | Pa | 101,325 |
| 1 atm | bar | 1.01325 |
| 1 psi | Pa | 6,894.757 |
| 1 psi | MPa | 0.006895 |
| 1 mmHg (torr) | Pa | 133.322 |

### Temperature

| From | To | Formula |
|---|---|---|
| °C | K | K = °C + 273.15 |
| K | °C | °C = K − 273.15 |
| °C | °F | °F = °C × 9/5 + 32 |
| °F | °C | °C = (°F − 32) × 5/9 |
| °F | K | K = (°F + 459.67) × 5/9 |

### Power

| From | To | Factor |
|---|---|---|
| 1 MW | kW | 1,000 |
| 1 MW | W | 1,000,000 |
| 1 GW | MW | 1,000 |
| 1 horsepower (HP) | kW | 0.7457 |
| 1 kW | BTU/h | 3,412.14 |

### Mass and Flow

| From | To | Factor |
|---|---|---|
| 1 tonne | kg | 1,000 |
| 1 tonne | lb | 2,204.62 |
| 1 kg/s | t/h | 3.6 |
| 1 t/h | kg/s | 0.2778 |
| 1 kg/s | kg/h | 3,600 |

---

## 6. Pint Unit String Reference

These strings are accepted by `pint.UnitRegistry`. Use them in `Action Input:` JSON.

### Common Pint Unit Strings

```python
# Temperature (use these for pint — note degC vs celsius vs °C)
"degC"          # degrees Celsius  ✓  (preferred)
"celsius"       # also valid
"degF"          # degrees Fahrenheit
"kelvin"        # kelvin (= K)
"degR"          # degrees Rankine

# Pressure
"pascal"        # Pa
"kilopascal"    # kPa
"megapascal"    # MPa
"bar"           # bar
"atmosphere"    # atm
"psi"           # pounds per square inch

# Power
"watt"          # W
"kilowatt"      # kW
"megawatt"      # MW
"gigawatt"      # GW

# Energy
"joule"         # J
"kilojoule"     # kJ
"megajoule"     # MJ
"kilowatt_hour" # kWh  ✓  (underscore, not hyphen)
"kilowatthour"  # also valid

# Heat rate (compound)
"kilojoule / kilowatt_hour"   # kJ/kWh  ✓
"BTU / kilowatt_hour"         # BTU/kWh

# Concentration
"gram / liter"  # g/L
"kilogram / meter ** 3"  # kg/m³
"mole / liter"  # mol/L = M

# Emission rates
"tonne / megawatt_hour"   # t/MWh  ✓
"gram / kilowatt_hour"    # g/kWh  (use for CO2 in g)

# Dimensionless
"dimensionless"   # ✓  for fractions, efficiencies as fractions
"percent"         # %  (= 0.01 dimensionless in pint)
"ppm"             # = 1e-6 dimensionless (custom alias in unit_registry.py)

# Flow rates
"kilogram / second"        # kg/s
"meter ** 3 / second"      # m³/s
"liter / second"           # L/s

# Irradiance
"watt / meter ** 2"        # W/m²

# Frequency
"hertz"   # Hz
```

### Common Pint Gotchas

| Problem | Wrong | Right |
|---|---|---|
| Temperature offset vs scale | `ureg.Quantity(300, "degC").to("kelvin")` | Use `convert_units()` helper — handles offset correctly |
| kWh with hyphen | `"kilowatt-hour"` | `"kilowatt_hour"` |
| Percent as fraction | `0.36` labelled "percent" | Pass `36` with unit `"percent"`, or `0.36` with unit `"dimensionless"` |
| Compound unit with slash | `"kg/s"` | `"kilogram / second"` (explicit words safer) |
| Squared metre | `"m2"` | `"meter ** 2"` |
| Million units | `"MWh"` | `"megawatt_hour"` |

---

## 7. Thermodynamic Laws and Constraints

### First Law (Energy Conservation)

```
Q = ΔU + W
Q_in = W_net + Q_out
η = W_net / Q_in = 1 - Q_out / Q_in
```

### Second Law — Carnot Efficiency

```
η_Carnot = 1 - T_cold / T_hot       (temperatures in Kelvin)
η_real < η_Carnot                   (always)
```

**Example**: T_hot = 600°C = 873 K, T_cold = 30°C = 303 K  
η_Carnot = 1 − 303/873 = **65.3%**  
Real plant at similar conditions: **40–45%**

### Heat Rate ↔ Efficiency

```
HR [kJ/kWh] = 3600 / η [dimensionless]
η [%] = 3600 / HR [kJ/kWh] × 100
```

| Efficiency | Heat Rate |
|---|---|
| 25% | 14,400 kJ/kWh |
| 30% | 12,000 kJ/kWh |
| 35% | 10,286 kJ/kWh |
| 40% | 9,000 kJ/kWh |
| 45% | 8,000 kJ/kWh |
| 50% | 7,200 kJ/kWh |

### Rankine Cycle Identities

```
Net power = Gross power - Auxiliary power
Steam consumption [t/MWh] = Steam flow [t/h] / Net power [MW]
Condenser duty [MW] = Q_in [MW] - W_net [MW]
Make-up water [%] ≈ 1-3% of steam flow (typical)
```

### CO2 Emission Factors

| Fuel | kg CO2/GJ_thermal | Factor |
|---|---|---|
| Natural gas | 56.1 | |
| Coal (black) | 94.6 | |
| Coal (brown/lignite) | 101.0 | |
| Oil | 73.3 | |
| Biomass (wood pellets) | 112.0 | (biogenic — zero in national GHG accounts) |

```
CO2 [t/MWh_electric] = Emission_factor [kg/GJ_thermal]
                       × HR [GJ_thermal/MWh_electric]
                       / 1000          # kg → t
```

---

## 8. Dimensional Analysis Worked Examples

### Example 1: Is efficiency dimensionless or percentage?

```python
import pint
ureg = pint.UnitRegistry()

eff_fraction = ureg.Quantity(0.368, "dimensionless")
eff_percent  = ureg.Quantity(36.8,  "percent")

# Are they the same?
print(eff_fraction.to("percent"))    # 36.8 percent ✓
print(eff_percent.to("dimensionless"))  # 0.368 ✓
print(eff_fraction == eff_percent)  # True
```

**Rule**: Store efficiency as fraction (0.368) internally. Report as percent (36.8%) to users.

### Example 2: Convert heat rate to efficiency

```python
import pint
ureg = pint.UnitRegistry()

HR = ureg.Quantity(9800, "kilojoule / kilowatt_hour")

# 3600 kJ/kWh = 1 kWh/kWh (identity)
efficiency = (3600 * ureg.Quantity(1, "kilojoule / kilowatt_hour")) / HR
print(efficiency.to("dimensionless"))  # 0.3673...
print(efficiency.to("percent"))        # 36.73...%
```

### Example 3: CO2 emission rate from fuel data

```python
import pint
ureg = pint.UnitRegistry()

# Given: natural gas plant, HR = 8500 kJ/kWh
# Emission factor: 56.1 kg CO2 / GJ_thermal

HR = ureg.Quantity(8500, "kilojoule / kilowatt_hour")
ef = ureg.Quantity(56.1, "kilogram / gigajoule")

# HR in GJ/MWh: 8500 kJ/kWh × 1 MWh / 1000 kWh × 1 GJ / 1000 kJ
# = 8500/1000 = 8.5 GJ/MWh ... pint does this automatically:

HR_GJ_per_MWh = HR.to("gigajoule / megawatt_hour")
co2 = (ef * HR_GJ_per_MWh).to("kilogram / megawatt_hour")
print(co2)  # 476.85 kilogram / megawatt_hour = 0.477 t/MWh ✓
```

### Example 4: Check if a pressure is supercritical

```python
import pint
ureg = pint.UnitRegistry()

CRITICAL_PRESSURE = ureg.Quantity(22.064, "megapascal")  # water critical point

steam_P = ureg.Quantity(25.0, "megapascal")
is_supercritical = steam_P > CRITICAL_PRESSURE
print(f"Supercritical: {is_supercritical}")  # True ✓

# Convert psi to MPa for comparison
steam_P_psi = ureg.Quantity(3625, "psi")
print(steam_P_psi.to("megapascal"))  # 24.996... MPa ✓
```

### Example 5: Wind power density

```python
import pint
ureg = pint.UnitRegistry()

# P = 0.5 × ρ × A × v³
rho = ureg.Quantity(1.225, "kilogram / meter**3")  # air density at sea level
v   = ureg.Quantity(12.0, "meter / second")          # wind speed
A   = ureg.Quantity(1.0, "meter**2")                 # unit swept area

P_density = 0.5 * rho * v**3
print(P_density.to("watt / meter**2"))  # 1058.4 W/m² ✓

# For a turbine with 80m rotor diameter:
D = ureg.Quantity(80, "meter")
import math
A_turbine = math.pi * (D/2)**2
P_turbine = (0.5 * rho * A_turbine * v**3).to("kilowatt")
print(P_turbine)  # ~2660 kW at Cp=0.4 after applying Betz coefficient
```

---

## 9. Physical Plausibility Quick-Check Table

Use this as a mental filter before accepting an analysis result.

| Claim | Plausible? | Check |
|---|---|---|
| Thermal efficiency = 36.8% | ✅ Yes | Within 20–55% range |
| Thermal efficiency = 67% | ❌ No | Exceeds Carnot for typical conditions |
| Heat rate = 9800 kJ/kWh | ✅ Yes | Within 6000–12000 range |
| Heat rate = 2100 kJ/kWh | ❌ No | Below 3600 kJ/kWh (100% efficiency) |
| Steam temperature = 580°C | ✅ Yes | USC range |
| Steam temperature = 850°C | ❌ No | Exceeds all known plant designs |
| CO2 = 0.42 t/MWh | ✅ Yes | CCGT/gas range |
| CO2 = 2.5 t/MWh | ❌ No | 5× too high even for poorest coal plant |
| Power factor = 0.92 | ✅ Yes | Typical grid requirement |
| Power factor = 1.15 | ❌ No | Cannot exceed 1.0 |
| Relative humidity = 85% | ✅ Yes | High humidity conditions |
| Relative humidity = 115% | ❌ No | Cannot exceed 100% |
| Probability = 0.73 | ✅ Yes | Valid |
| Probability = −0.1 | ❌ No | Cannot be negative |
| Correlation = 0.87 | ✅ Yes | Strong positive correlation |
| Correlation = 1.4 | ❌ No | Cannot exceed 1.0 (Cauchy-Schwarz) |
