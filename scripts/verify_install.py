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
    ("nbformat", "Jupyter notebook format"),
]

OPTIONAL = [
    ("jupyter_client", "Jupyter kernel bridge"),
    ("ipykernel", "Python3 Jupyter kernel"),
]

def check(name, desc, required=True):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✅ {name:<22} {ver:<12} — {desc}")
        return True
    except ImportError:
        mark = "❌" if required else "⚠️ "
        label = "MISSING" if required else "optional"
        print(f"  {mark} {name:<22} {label:<12} — {desc}")
        return not required

print("\n=== Data Scientist Agent — Dependency Check ===\n")
print("Required:")
ok = all(check(m, d) for m, d in REQUIRED)
print("\nOptional:")
for m, d in OPTIONAL:
    check(m, d, required=False)

try:
    import pint
    ureg = pint.UnitRegistry()
    q = 57.3 * ureg.degC
    print(f"\n  ✅ pint sanity: {q} → {q.to('degF'):.2f}")
except Exception as e:
    print(f"\n  ❌ pint check failed: {e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1,2,3],[4,5,6])
    plt.close(fig)
    print("  ✅ matplotlib Agg backend works")
except Exception as e:
    print(f"  ❌ matplotlib check: {e}")

print()
if ok:
    print("✅ All required dependencies satisfied. Ready!\n")
    sys.exit(0)
else:
    print("❌ Missing dependencies. Run: uv sync\n")
    sys.exit(1)
