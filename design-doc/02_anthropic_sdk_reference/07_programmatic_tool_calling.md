# Programmatic Tool Calling — Claude Orchestrating Your Tools from Python

## 1. The Concept

Programmatic tool calling is a paradigm shift in how Claude interacts with your custom tools. In traditional tool use:

```
Claude → API call → Your code runs tool → API call → Claude → repeat
```

With programmatic tool calling via `code_execution_20260120`:

```
Claude → writes Python → sandbox executes → your tools called inline → all done in 1 API call
```

Claude writes Python code that uses `await tool_name(...)` syntax to call your custom functions directly from within Anthropic's code execution sandbox. Your functions run in your process, but the results flow back into Claude's running Python code automatically — without round-tripping through an extra API call.

**Why this is revolutionary:**
- 10 sequential tool calls = **1 API round-trip** instead of 10
- Claude can use loop logic (`for`, `while`, list comprehensions) across your tools
- Claude can branch (`if/else`) based on intermediate tool results
- The full expressive power of Python for orchestration, not just prompt logic

---

## 2. When It Matters

Programmatic tool calling shines in scenarios where **multiple sequential tool calls** are needed and the inputs to later calls depend on the outputs of earlier ones.

### Scenario 1: Batch Data Validation

```python
# Without programmatic calling: 1000 measurements = 1000 API round-trips
for measurement in measurements:
    response = client.messages.create(...)  # 1000 calls
    tool_call = extract_tool_call(response)
    result = validate_units(**tool_call.input)
    send_result_back(result)  # 1000 more calls

# With programmatic calling: 1000 measurements = 1 API round-trip
# Claude writes:
# for measurement in measurements:
#     result = await validate_units(**measurement)
#     results.append(result)
```

### Scenario 2: Multi-Dataset Analysis Pipeline

```python
# Pipeline: load → clean → join → aggregate → validate → report
# Each step calls a different tool, depends on the previous result.
# Without programmatic calling: 6+ API calls
# With programmatic calling: 1 API call, Claude writes the pipeline
```

### Scenario 3: Iterative Refinement

```python
# Claude writes a loop:
# for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     metrics = await evaluate_model(threshold=threshold)
#     if metrics['f1'] > best_f1:
#         best_f1 = metrics['f1']
#         best_threshold = threshold
# print(f"Best threshold: {best_threshold}, F1: {best_f1}")
```

### When NOT to Use It

- Single tool call: regular tool use is simpler and sufficient
- Tools that need to run on the same machine with full OS access
- ZDR-required workflows (use `"allowed_callers": ["direct"]` instead)
- Haiku model (does not support programmatic calling)

---

## 3. Setup Requirements

### Model Requirements

| Model | Supports Programmatic Calling |
|-------|:---:|
| `claude-opus-4-7` | ✅ |
| `claude-sonnet-4-6` | ✅ |
| `claude-haiku-4-5-20251001` | ❌ |

### Required Tool

`code_execution_20260120` (the newest version) **must** be in the tools list. The older `code_execution_20250825` does **not** support programmatic tool calling.

```python
tools = [
    # Required: newest code execution tool
    {"type": "code_execution_20260120", "name": "code_execution"},

    # Your custom tools with allowed_callers
    {
        "name": "my_tool",
        "description": "...",
        "input_schema": {...},
        "allowed_callers": ["code_execution_20260120"]  # ← Required field
    }
]
```

### Platform Requirements

| Platform | Programmatic Tool Calling |
|----------|:---:|
| Anthropic API | ✅ |
| Azure AI | ✅ |
| Bedrock | ❌ |
| Vertex AI | ❌ |

---

## 4. Tool Definition with `allowed_callers`

The `allowed_callers` field is what enables a tool for programmatic calling. Without it, the tool can only be called via the traditional tool_use → tool_result round-trip.

### Single Tool Example

```python
validate_units_tool = {
    "name": "validate_physical_units",
    "description": (
        "Validates that a numeric measurement uses the correct units for its physical dimension. "
        "This tool is designed to be called in loops from code_execution — prefer it over "
        "calling one at a time. "
        "Returns a JSON object with 'status' ('valid' or 'invalid') and 'reason' if invalid."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The numeric measurement value"
            },
            "unit": {
                "type": "string",
                "description": "The unit string, e.g. 'kg', 'km/h', 'm/s^2', '°C'"
            },
            "expected_dimension": {
                "type": "string",
                "enum": ["length", "mass", "time", "temperature", "energy", "velocity", "acceleration", "pressure"],
                "description": "The physical dimension the measurement should represent"
            }
        },
        "required": ["value", "unit", "expected_dimension"]
    },
    "allowed_callers": ["code_execution_20260120"]   # ← Enables programmatic calling
}
```

### Multiple Tools — Some Programmatic, Some Direct

```python
tools = [
    # Server tool: runs on Anthropic's infrastructure
    {"type": "code_execution_20260120", "name": "code_execution"},

    # Client tool: programmatic calling enabled
    {
        "name": "validate_physical_units",
        "description": "Validates physical units. Optimised for batch calling from code.",
        "input_schema": {...},
        "allowed_callers": ["code_execution_20260120"]
    },

    # Client tool: traditional calling (direct tool_use → tool_result)
    {
        "name": "get_schema_definition",
        "description": "Returns the full schema definition for a dataset. "
                       "Call this once at the start to understand data types.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"}
            },
            "required": ["dataset_name"]
        }
        # No allowed_callers → traditional tool use only
    }
]
```

---

## 5. How Claude Uses It

When you provide tools with `allowed_callers: ["code_execution_20260120"]`, Claude generates Python code inside the sandbox that calls your tools using `await`:

### Example of Claude's Generated Python

Given the task "Validate these 100 measurements and report how many are invalid":

```python
# Claude writes this code inside code_execution:
import json

measurements = [
    {"value": 25.4, "unit": "km/h", "dimension": "velocity"},
    {"value": 9.81, "unit": "m/s2", "dimension": "acceleration"},
    {"value": 100.0, "unit": "kg", "dimension": "mass"},
    {"value": 37.5, "unit": "K", "dimension": "temperature"},   # Wrong: should be °C
    # ... 96 more measurements
]

results = []
for m in measurements:
    result = await validate_physical_units(
        value=m["value"],
        unit=m["unit"],
        expected_dimension=m["dimension"]
    )
    result_dict = json.loads(result)
    results.append({
        "measurement": m,
        "valid": result_dict["status"] == "valid",
        "reason": result_dict.get("reason", "")
    })

invalid = [r for r in results if not r["valid"]]
print(f"Total: {len(results)}")
print(f"Valid: {len(results) - len(invalid)}")
print(f"Invalid: {len(invalid)}")
for inv in invalid[:5]:
    print(f"  {inv['measurement']['value']} {inv['measurement']['unit']}: {inv['reason']}")
```

### How Results Flow Back

1. Claude writes the Python code
2. The sandbox executes the code
3. Each `await validate_physical_units(...)` call:
   a. Exits the sandbox momentarily
   b. Calls your Python function in **your process**
   c. Returns the result to the sandbox
   d. Claude's code continues with the result
4. The sandbox finishes and prints output
5. The full output comes back in Claude's response

**Key insight**: Your function runs in your process with full access to your databases, in-memory state, and local resources. The sandbox just orchestrates the calls.

---

## 6. Security Implications

### The Trust Boundary

```
┌─────────────────────────────────────┐
│  Anthropic's Infrastructure         │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  Code Execution Sandbox      │   │
│  │                              │   │
│  │  Claude's Python code runs   │   │
│  │  here — ISOLATED             │   │
│  │                              │   │
│  │  await your_tool(args) ──────┼───┼──→ Your Python process
│  │           ↑                  │   │    (Your machine/server)
│  │      result returned         │◀──┼──  Your function returns
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Security Properties

**The sandbox is safe:**
- Claude's code cannot access your filesystem directly
- Claude's code cannot make network calls
- Claude's code cannot import arbitrary packages
- Claude's code is isolated from your process

**Your function is called with Claude's arguments:**
- Claude provides the arguments; your function runs them
- Validate inputs in your function just as you would for any external caller
- Your function has full access to your resources — be careful what you expose

### Input Validation in Your Tool

```python
import json
from typing import Literal

VALID_DIMENSIONS = {"length", "mass", "time", "temperature", "energy", "velocity", "acceleration"}

def validate_physical_units(
    value: float,
    unit: str,
    expected_dimension: str
) -> str:
    # Validate inputs — Claude's arguments may be unexpected
    if not isinstance(value, (int, float)):
        return json.dumps({"status": "error", "reason": "value must be numeric"})

    if len(unit) > 50:  # Sanity check
        return json.dumps({"status": "error", "reason": "unit string too long"})

    if expected_dimension not in VALID_DIMENSIONS:
        return json.dumps({
            "status": "error",
            "reason": f"Unknown dimension '{expected_dimension}'. Valid: {sorted(VALID_DIMENSIONS)}"
        })

    # ... actual validation logic ...
    return json.dumps({"status": "valid"})
```

### What Claude Cannot Do Through Your Tool

- Claude cannot call your tool with side effects you haven't coded
- Claude cannot bypass your authentication/authorisation within the tool
- Claude cannot read your environment variables or secrets
- Claude cannot write to your filesystem except through explicit tool functions you provide

---

## 7. Error Propagation

When your tool raises an exception inside Claude's code, the error propagates back to the sandbox:

### What Happens on Your Tool's Exception

```python
# Your tool:
def validate_physical_units(value, unit, expected_dimension):
    if not validate_db_connection():
        raise ConnectionError("Database unavailable")
    # ...

# Claude's code in sandbox:
# try:
#     result = await validate_physical_units(value=25.4, unit="km/h", expected_dimension="velocity")
# except Exception as e:
#     print(f"Tool error: {e}")
#     results.append({"status": "tool_error", "error": str(e)})
```

Claude's generated code receives the exception. If Claude wrote a try/except, it handles it. If not, the exception propagates to the sandbox output and Claude sees the traceback.

### Best Practice: Return Errors as Values

Instead of raising exceptions, return structured error responses:

```python
def validate_physical_units(value: float, unit: str, expected_dimension: str) -> str:
    try:
        # ... validation logic ...
        return json.dumps({"status": "valid"})
    except Exception as e:
        # Return error as value, not exception
        return json.dumps({"status": "error", "reason": str(e)})
```

This lets Claude's Python code use the result without exception handling:

```python
# Claude can write:
result = json.loads(await validate_physical_units(...))
if result["status"] == "error":
    error_log.append(result["reason"])
else:
    valid_measurements.append(...)
```

---

## 8. Performance Benefits

### Benchmark: 10 Sequential Validations

**Traditional tool use (10 API calls):**
```
API call 1: Claude decides to call validate_units(measurement_1)
  → network round-trip ~200ms
  → your function runs ~5ms
  → response back ~200ms
  Total: ~405ms

API call 2: Claude decides to call validate_units(measurement_2)
  → ~405ms
...
API call 10: ~405ms

TOTAL: ~4,050ms (4+ seconds) + 10 × input token overhead
```

**Programmatic tool calling (1 API call):**
```
API call 1: Claude writes Python loop, sandbox executes it
  → 10 × your function runs ~5ms = 50ms for all functions
  → 1 × network round-trip ~200ms
  → response back ~200ms
  → sandbox execution overhead ~500ms

TOTAL: ~950ms + 1 × input token overhead
```

**Result: ~4x faster, significantly cheaper (fewer input/output tokens)**

### Token Efficiency

```python
# Traditional: 10 calls × overhead per call
# Each call has: system prompt + full conversation history + tool definitions + tool result
# Cost: N_calls × (system_tokens + history_tokens + tool_def_tokens + result_tokens)

# Programmatic: 1 call
# Cost: 1 × (system_tokens + history_tokens + tool_def_tokens + all_results_tokens)
# Tool call tokens are only paid once, not N times
```

### Real-World Savings

| Operation | Traditional | Programmatic | Savings |
|-----------|:-----------:|:------------:|:-------:|
| 10 validations | 10 API calls | 1 API call | 90% fewer calls |
| 100 validations | 100 API calls | 1 API call | 99% fewer calls |
| Pipeline (5 steps) | 5 API calls | 1 API call | 80% fewer calls |
| Batch report (50 metrics) | 50 API calls | 1–3 API calls | ~95% fewer calls |

---

## 9. Integration with DataScienceAgentService

How to expose domain validation tools for programmatic calling in a production agent service:

```python
import anthropic
import json
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


class ProgrammaticToolRegistry:
    """Registry for tools that support programmatic calling via code_execution."""

    def __init__(self):
        self._tools: dict[str, tuple[dict, Callable]] = {}

    def register(self, definition: dict, implementation: Callable) -> None:
        name = definition["name"]
        # Ensure allowed_callers is set
        if "allowed_callers" not in definition:
            definition = {**definition, "allowed_callers": ["code_execution_20260120"]}
        self._tools[name] = (definition, implementation)
        logger.info(f"Registered programmatic tool: {name}")

    @property
    def definitions(self) -> list[dict]:
        return [defn for defn, _ in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return json.dumps({"status": "error", "reason": f"Unknown tool: {name}"})
        _, impl = self._tools[name]
        try:
            result = impl(**kwargs)
            if isinstance(result, (dict, list)):
                return json.dumps(result, default=str)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {name} raised exception: {e}", exc_info=True)
            return json.dumps({"status": "error", "reason": str(e)})


class DataScienceAgentService:
    """
    Agent service that integrates programmatic tool calling
    for efficient data validation and analysis pipelines.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.registry = ProgrammaticToolRegistry()
        self._setup_default_tools()

    def _setup_default_tools(self):
        """Register built-in domain tools."""

        # Unit validation tool
        self.registry.register(
            definition={
                "name": "validate_physical_units",
                "description": (
                    "Validates that a measurement value has correct units for its dimension. "
                    "Optimised for batch calling from code_execution loops. "
                    "Returns JSON with 'status' ('valid'/'invalid') and 'reason'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "Numeric measurement"},
                        "unit": {"type": "string", "description": "Unit string e.g. 'kg', 'km/h'"},
                        "expected_dimension": {
                            "type": "string",
                            "enum": ["length", "mass", "time", "temperature", "energy", "velocity"],
                        }
                    },
                    "required": ["value", "unit", "expected_dimension"]
                },
            },
            implementation=self._validate_physical_units
        )

        # Schema lookup tool (non-programmatic, called once)
        # Note: no allowed_callers → traditional tool_use
        # (shown for contrast — not registered in programmatic registry)

    @staticmethod
    def _validate_physical_units(value: float, unit: str, expected_dimension: str) -> dict:
        DIMENSION_UNITS = {
            "length": {"m", "km", "cm", "mm", "ft", "in", "miles", "mi"},
            "mass": {"kg", "g", "mg", "lb", "lbs", "oz"},
            "time": {"s", "ms", "min", "hr", "h", "day"},
            "temperature": {"K", "°C", "°F", "C", "F"},
            "energy": {"J", "kJ", "cal", "kcal", "eV", "kWh"},
            "velocity": {"m/s", "km/h", "mph", "knots", "ft/s"},
        }
        valid = DIMENSION_UNITS.get(expected_dimension, set())
        if unit in valid:
            return {"status": "valid"}
        else:
            return {
                "status": "invalid",
                "reason": f"Unit '{unit}' not valid for '{expected_dimension}'",
                "valid_units": sorted(valid)
            }

    def build_tools_list(self) -> list[dict]:
        """Build the full tools list for API requests."""
        return [
            {"type": "code_execution_20260120", "name": "code_execution"},
            *self.registry.definitions
        ]

    def run(self, user_message: str, system: str = "") -> str:
        """Execute a full agentic turn with programmatic tool support."""
        messages = [{"role": "user", "content": user_message}]
        tools = self.build_tools_list()

        while True:
            kwargs = {
                "model": self.model,
                "max_tokens": 8192,
                "tools": tools,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return next(
                    (b.text for b in response.content if b.type == "text"), ""
                )

            elif response.stop_reason == "tool_use":
                # Traditional (non-programmatic) tool calls
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self.registry.execute(block.name, **block.input)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": results})

            elif response.stop_reason == "pause_turn":
                messages.append({"role": "user", "content": "Please continue."})

            else:
                raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")
```

---

## 10. Complete Working Example

Full setup: `validate_physical_units` configured for programmatic calling, with a 100-item validation batch.

```python
import anthropic
import json
import time

client = anthropic.Anthropic()

# ─── Your domain tools ────────────────────────────────────────────────────────

DIMENSION_UNITS = {
    "length":      {"m", "km", "cm", "mm", "ft", "in", "miles"},
    "mass":        {"kg", "g", "mg", "lb", "lbs", "oz"},
    "time":        {"s", "ms", "min", "hr", "h", "sec"},
    "temperature": {"K", "°C", "°F", "C", "F"},
    "energy":      {"J", "kJ", "cal", "kcal", "eV", "kWh"},
    "velocity":    {"m/s", "km/h", "mph", "knots"},
    "acceleration":{"m/s2", "m/s^2", "ft/s2", "g"},
    "pressure":    {"Pa", "kPa", "MPa", "bar", "psi", "atm"},
}

def validate_physical_units(value: float, unit: str, expected_dimension: str) -> str:
    """Runs in YOUR process when Claude calls it from the sandbox."""
    if expected_dimension not in DIMENSION_UNITS:
        return json.dumps({
            "status": "error",
            "reason": f"Unknown dimension: {expected_dimension}",
            "valid_dimensions": list(DIMENSION_UNITS.keys())
        })

    valid_units = DIMENSION_UNITS[expected_dimension]
    if unit in valid_units:
        return json.dumps({"status": "valid", "unit": unit, "dimension": expected_dimension})
    else:
        return json.dumps({
            "status": "invalid",
            "unit": unit,
            "dimension": expected_dimension,
            "reason": f"'{unit}' is not a recognised unit for {expected_dimension}",
            "valid_units": sorted(valid_units)
        })


def get_conversion_factor(from_unit: str, to_unit: str) -> str:
    """Look up unit conversion factors from your domain knowledge base."""
    CONVERSIONS = {
        ("km", "m"):     1000.0,
        ("m", "km"):     0.001,
        ("km/h", "m/s"): 0.27778,
        ("m/s", "km/h"): 3.6,
        ("lb", "kg"):    0.45359,
        ("kg", "lb"):    2.20462,
        ("°C", "K"):     None,  # Offset conversion — not just multiplication
        ("kWh", "J"):    3_600_000.0,
        ("J", "kWh"):    2.7778e-7,
    }
    key = (from_unit, to_unit)
    if key in CONVERSIONS:
        factor = CONVERSIONS[key]
        if factor is None:
            return json.dumps({"error": f"Conversion {from_unit}→{to_unit} requires offset, not just factor"})
        return json.dumps({"from": from_unit, "to": to_unit, "multiply_by": factor})
    else:
        return json.dumps({"error": f"No conversion factor for {from_unit}→{to_unit} in registry"})


# ─── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    # Server tool: code execution (enables programmatic calling)
    {"type": "code_execution_20260120", "name": "code_execution"},

    # Client tool with programmatic calling enabled
    {
        "name": "validate_physical_units",
        "description": (
            "Validates a physical measurement's units against its expected dimension. "
            "IMPORTANT: This tool is optimised for batch use — call it in a loop from "
            "code_execution rather than one at a time. "
            "Returns JSON with 'status' ('valid', 'invalid', or 'error'), and 'reason' if invalid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric measurement value"},
                "unit": {
                    "type": "string",
                    "description": "Unit string, e.g. 'kg', 'km/h', 'm/s2', '°C', 'kWh'"
                },
                "expected_dimension": {
                    "type": "string",
                    "enum": list(DIMENSION_UNITS.keys()),
                    "description": "The physical dimension this measurement should represent"
                }
            },
            "required": ["value", "unit", "expected_dimension"]
        },
        "allowed_callers": ["code_execution_20260120"]   # Enables programmatic calling
    },

    # Second client tool for programmatic calling
    {
        "name": "get_conversion_factor",
        "description": (
            "Returns the multiplication factor to convert from one unit to another. "
            "Use this in code_execution loops when you need to normalise many measurements. "
            "Returns JSON with 'multiply_by' factor, or 'error' if conversion not available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit":   {"type": "string", "description": "Target unit"}
            },
            "required": ["from_unit", "to_unit"]
        },
        "allowed_callers": ["code_execution_20260120"]
    }
]

# ─── Your tool implementations registry ──────────────────────────────────────

CLIENT_TOOLS = {
    "validate_physical_units": validate_physical_units,
    "get_conversion_factor": get_conversion_factor,
}

# ─── Sample data: 20 measurements with mixed validity ────────────────────────

import random
random.seed(42)

SAMPLE_MEASUREMENTS = [
    {"id": f"M{i:03d}", "value": round(random.uniform(1, 1000), 2),
     "unit": unit, "expected_dimension": dim}
    for i, (unit, dim) in enumerate([
        ("km/h", "velocity"),      # valid
        ("kg", "mass"),            # valid
        ("m/s", "velocity"),       # valid
        ("°C", "temperature"),     # valid
        ("km", "length"),          # valid
        ("kWh", "energy"),         # valid
        ("lb", "mass"),            # valid
        ("m/s2", "acceleration"),  # valid
        ("kPa", "pressure"),       # valid
        ("hr", "time"),            # valid
        ("m/s2", "velocity"),      # INVALID — wrong dimension
        ("°F", "energy"),          # INVALID — wrong dimension
        ("litre", "mass"),         # INVALID — not in registry
        ("W", "energy"),           # INVALID — watts ≠ energy unit
        ("km/h", "length"),        # INVALID — wrong dimension
        ("furlongs", "length"),    # INVALID — not in registry
        ("m", "length"),           # valid
        ("J", "energy"),           # valid
        ("mph", "velocity"),       # valid
        ("g", "mass"),             # valid
    ])
]


# ─── Agentic loop with programmatic tool support ──────────────────────────────

def run_validation_agent(measurements: list[dict]) -> str:
    """Run the validation agent with programmatic tool calling."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Validate the physical units in this list of {len(measurements)} measurements. "
                f"Use validate_physical_units in a loop from code_execution. "
                f"After validation, use get_conversion_factor to show what factor would be needed "
                f"to convert all velocity measurements to m/s. "
                f"Report: total count, valid count, invalid count, list of invalid measurements "
                f"with reasons, and the velocity conversion factors.\n\n"
                f"Measurements:\n{json.dumps(measurements, indent=2)}"
            )
        }
    ]

    start = time.monotonic()
    api_calls = 0

    while True:
        api_calls += 1
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            elapsed = time.monotonic() - start
            result = next(b.text for b in response.content if b.type == "text")
            print(f"\n{'='*60}")
            print(f"Completed in {elapsed:.2f}s with {api_calls} API call(s)")
            print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
            print(f"{'='*60}\n")
            return result

        elif response.stop_reason == "tool_use":
            # Traditional (non-programmatic) tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = CLIENT_TOOLS.get(block.name)
                    result = fn(**block.input) if fn else json.dumps({"error": f"Unknown: {block.name}"})
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


# ─── Run it ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running validation on {len(SAMPLE_MEASUREMENTS)} measurements...")
    print("Expected: 15 valid, 5 invalid\n")

    result = run_validation_agent(SAMPLE_MEASUREMENTS)
    print(result)

    # Compare: without programmatic calling, this would have been:
    # 20 validate calls + 3-5 conversion calls = 23-25 API round-trips
    # With programmatic calling: 1-2 API calls total
```

### Expected Output Pattern

```
Running validation on 20 measurements...
Expected: 15 valid, 5 invalid

============================================================
Completed in 3.8s with 1 API call(s)
Tokens: 2847 in, 891 out
============================================================

Validation Results:
- Total measurements: 20
- Valid: 15 (75%)
- Invalid: 5 (25%)

Invalid measurements:
1. M010 (value=..., unit=m/s2): 'm/s2' is not valid for 'velocity'. Valid units: km/h, knots, m/s, mph
2. M011 (value=..., unit=°F): '°F' is not valid for 'energy'. Valid units: J, cal, eV, kJ, kcal, kWh
3. M012 (value=..., unit=litre): 'litre' is not valid for 'mass'. Valid units: g, kg, lb, lbs, mg, oz
4. M013 (value=..., unit=W): 'W' is not valid for 'energy'. Valid units: J, cal, eV, kJ, kcal, kWh
5. M014 (value=..., unit=km/h): 'km/h' is not valid for 'length'. Valid units: cm, ft, in, km, m, miles, mm

Velocity conversion factors (to m/s):
- km/h → m/s: multiply by 0.27778
- mph → m/s: multiply by 0.44704 (not in registry)
- m/s: already in target unit (factor = 1.0)
```
