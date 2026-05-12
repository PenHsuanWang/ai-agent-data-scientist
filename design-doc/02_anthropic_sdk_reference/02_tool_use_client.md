# Client Tool Use — User-Defined and Anthropic-Schema Tools

## 1. Concept

**Tool use** (also called function calling) lets Claude request that your code execute a function and return the result. This enables Claude to:

- Query live databases or APIs
- Execute calculations too complex to reason about in text
- Take actions in external systems (file writes, API calls, notifications)
- Access real-time data (prices, sensor readings, query results)

**The tool-use contract:**

1. You define tools (name, description, schema) in the request
2. Claude decides whether/which tools to call
3. Claude returns a `tool_use` content block with the tool name and arguments
4. **Your code** executes the function
5. You send the result back as a `tool_result` in the next user message
6. Claude incorporates the result and continues

This is "client-executed" because **your code** runs the function — Claude only decides to call it and provides the arguments. Contrast with server tools (Section 03) where Anthropic's infrastructure runs the function.

---

## 2. Tool Definition Schema

```python
tool_definition = {
    # Required fields
    "name": "get_dataset_statistics",    # ^[a-zA-Z0-9_-]{1,64}$  (regex enforced)
    "description": "...",                # Critical for performance (see Section 4)
    "input_schema": {                    # Standard JSON Schema object
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": "string",
                "description": "Name of the dataset to analyse, e.g. 'sales_q1_2025'"
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of column names to compute statistics for. "
                               "If omitted, statistics are computed for all numeric columns."
            },
            "include_percentiles": {
                "type": "boolean",
                "description": "Whether to include p5, p25, p75, p95 in results.",
                "default": False
            }
        },
        "required": ["dataset_name"]     # Fields Claude must always provide
    },

    # Optional fields
    "input_examples": [                  # See Section 5 — helps with complex inputs
        {
            "dataset_name": "sales_q1",
            "columns": ["revenue", "units"],
            "include_percentiles": True
        }
    ],
    "strict": True,                      # See Section 7 — guarantees schema conformance
    "allowed_callers": [                 # See File 07 — for programmatic tool calling
        "code_execution_20260120"
    ],
}
```

### Field Details

| Field | Type | Required | Constraint |
|-------|------|----------|-----------|
| `name` | `str` | ✅ | `^[a-zA-Z0-9_-]{1,64}$` — alphanumeric, dash, underscore only |
| `description` | `str` | ✅ (strongly) | No hard limit; aim for 2–4 sentences |
| `input_schema` | `dict` | ✅ | Must be JSON Schema `{"type": "object", ...}` |
| `input_examples` | `list[dict]` | ❌ | Each example must match the schema |
| `strict` | `bool` | ❌ | Default `False`; when `True`, output validated against schema |
| `allowed_callers` | `list[str]` | ❌ | Only `"code_execution_20260120"` is a valid value currently |

### JSON Schema Support

Claude supports a subset of JSON Schema draft 7:

```python
# Supported types
"type": "string" | "number" | "integer" | "boolean" | "array" | "object" | "null"

# Supported keywords
"enum": ["option1", "option2"]          # Enumerated values
"default": value                        # Default value (informational)
"minimum" / "maximum": number           # Numeric bounds
"minLength" / "maxLength": int          # String length bounds
"items": {...}                          # Array item schema
"properties": {...}                     # Object properties
"required": [...]                       # Required property names
"additionalProperties": false           # Disallow extra fields

# NOT supported: $ref, allOf, anyOf (use flat schemas)
```

---

## 3. The Agentic Loop

The core loop for tool-using agents. Claude may call multiple tools before giving a final answer.

```python
import anthropic
import json

client = anthropic.Anthropic()

# Example tool implementations
def get_dataset_statistics(dataset_name: str, columns: list = None, include_percentiles: bool = False) -> str:
    # In production: query your database/dataframe
    return json.dumps({
        "dataset": dataset_name,
        "rows": 10000,
        "mean": {"revenue": 542.3, "units": 14.2},
        "std": {"revenue": 123.4, "units": 5.1},
    })

def run_sql_query(query: str) -> str:
    # In production: execute against your DB
    return json.dumps([{"date": "2025-01", "revenue": 50000}])

# Tool registry
TOOLS = {
    "get_dataset_statistics": get_dataset_statistics,
    "run_sql_query": run_sql_query,
}

TOOL_DEFINITIONS = [
    {
        "name": "get_dataset_statistics",
        "description": "Computes descriptive statistics (mean, std, min, max, count) "
                       "for one or more columns in a named dataset. Use this when the user "
                       "asks about data distributions, central tendency, or spread. "
                       "Returns a JSON object with per-column statistics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "The dataset identifier"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to include. Omit for all numeric columns."
                },
                "include_percentiles": {
                    "type": "boolean",
                    "description": "Include p5/p25/p75/p95",
                    "default": False
                }
            },
            "required": ["dataset_name"]
        }
    },
    {
        "name": "run_sql_query",
        "description": "Executes a read-only SQL SELECT query against the data warehouse. "
                       "Use this for aggregations, joins, filtering, and time-series queries. "
                       "Do NOT use for mutations (INSERT, UPDATE, DELETE are blocked). "
                       "Returns results as a JSON array of row objects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid SQL SELECT statement. Use snake_case table/column names."
                }
            },
            "required": ["query"]
        }
    }
]


def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """Main agentic loop — runs until end_turn or max_iterations."""
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system="You are a data scientist. Use the available tools to answer questions about data.",
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        # Done — Claude gave a final text answer
        if response.stop_reason == "end_turn":
            # Find the text block
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Tool calls — execute each one
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_fn = TOOLS.get(block.name)
                if tool_fn is None:
                    result_content = f"Error: Unknown tool '{block.name}'"
                else:
                    try:
                        result_content = tool_fn(**block.input)
                    except Exception as e:
                        result_content = f"Error executing {block.name}: {str(e)}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"Agent did not complete within {max_iterations} iterations")


# Example usage
answer = run_agent("What are the statistics for the sales_q1 dataset, and what was the total revenue in January 2025?")
print(answer)
```

---

## 4. Building Effective Tool Descriptions

The description is the most important part of a tool definition. Claude uses it to decide **when** and **how** to call the tool.

### The 3-4 Sentence Guideline

Structure your description as:
1. **What it does** (one sentence)
2. **When to use it** (one sentence — be specific about triggers)
3. **What it returns** (one sentence)
4. **Important constraints or limitations** (optional, but critical)

### Good vs Poor Description Comparison

**❌ Poor description:**
```python
"description": "Gets statistics for a dataset."
```
Problems: Vague, no guidance on when to use it, no return format info.

**✅ Good description:**
```python
"description": (
    "Computes descriptive statistics (mean, std, min, max, count, quartiles) "
    "for numeric columns in a named dataset. "
    "Use this when the user asks about distributions, averages, variability, "
    "outliers, or any summary characteristics of a dataset — before writing "
    "SQL queries that aggregate data. "
    "Returns a JSON object mapping column names to their statistics. "
    "Does NOT support string/categorical columns; use run_sql_query for those."
)
```

**❌ Poor description:**
```python
"description": "Runs a query."
```

**✅ Good description:**
```python
"description": (
    "Executes a read-only SQL SELECT query against the production data warehouse "
    "(PostgreSQL 15, BigQuery dialect also supported). "
    "Use this for time-series queries, multi-table joins, custom aggregations, "
    "or when get_dataset_statistics cannot answer the question. "
    "Returns results as a JSON array of row objects, limited to 10,000 rows. "
    "SELECT only — INSERT, UPDATE, DELETE, DROP statements will be rejected."
)
```

### Parameter Descriptions Matter Too

```python
# ❌ Poor parameter description
"query": {"type": "string", "description": "SQL query"}

# ✅ Good parameter description
"query": {
    "type": "string",
    "description": (
        "A valid SQL SELECT statement. Use snake_case for table and column names. "
        "Available tables: sales_transactions, products, customers, campaigns. "
        "Example: SELECT product_id, SUM(revenue) FROM sales_transactions "
        "WHERE date >= '2025-01-01' GROUP BY product_id"
    )
}
```

---

## 5. Input Examples

The `input_examples` field provides concrete examples of valid tool inputs. This is especially helpful for tools with:

- Complex nested schemas
- Many optional fields
- Non-obvious parameter combinations
- Domain-specific formats

```python
{
    "name": "query_time_series",
    "description": "Queries time-series data with flexible aggregation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "start_date": {"type": "string", "description": "ISO 8601 date"},
            "end_date": {"type": "string"},
            "granularity": {"type": "string", "enum": ["hourly", "daily", "weekly", "monthly"]},
            "filters": {
                "type": "object",
                "description": "Key-value pairs for filtering, e.g. {'region': 'US'}"
            }
        },
        "required": ["metric", "start_date", "end_date"]
    },
    "input_examples": [
        # Simple example
        {
            "metric": "revenue",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "granularity": "monthly"
        },
        # Complex example with filters
        {
            "metric": "conversion_rate",
            "start_date": "2025-06-01",
            "end_date": "2025-06-30",
            "granularity": "daily",
            "filters": {"region": "EMEA", "product_line": "premium"}
        }
    ]
}
```

**When to use input_examples:**
- When the schema has 5+ properties
- When filter/options format is non-obvious
- When date/time formats are critical

**Token cost**: Each example adds ~20-50 tokens to the prompt. For large tool registries, omit examples from rarely-used tools.

---

## 6. Controlling `tool_choice`

### `auto` (default)

Claude decides whether to call a tool, call multiple tools, or respond with text.

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=TOOL_DEFINITIONS,
    tool_choice={"type": "auto"},  # This is the default; can be omitted
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
# Claude may answer in text without calling a tool
```

### `any` — Force at least one tool call

Claude must call at least one tool (its choice which one).

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=TOOL_DEFINITIONS,
    tool_choice={"type": "any"},
    messages=[{"role": "user", "content": "Analyse the data."}]
)
# Claude will always call at least one tool
```

**Use cases**: When you always need real data before answering; when text-only responses indicate a failure mode.

**Note**: NOT compatible with extended thinking.

### `tool` — Force a specific tool

Force Claude to call exactly the named tool.

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=TOOL_DEFINITIONS,
    tool_choice={"type": "tool", "name": "get_dataset_statistics"},
    messages=[{"role": "user", "content": "Tell me about the sales data."}]
)
# Claude will always call get_dataset_statistics
```

**Use cases**: Structured data extraction; form filling; when you know exactly which tool is needed.

**Note**: NOT compatible with extended thinking.

### `none` — Disable all tools

Claude cannot call any tools, even if tools are listed.

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=TOOL_DEFINITIONS,  # Still listed (for context), but disabled
    tool_choice={"type": "none"},
    messages=[{"role": "user", "content": "Summarise our previous findings."}]
)
# Claude responds in text only
```

**Use cases**: When you want Claude to reason about tools without calling them; when you're in the "synthesis" phase.

---

## 7. Strict Tool Use

When `"strict": True`, the SDK validates Claude's tool call output against the schema before returning it. This guarantees:

- All `required` fields are present
- Field types match the schema
- No extra fields (if `additionalProperties: false`)

```python
{
    "name": "create_report",
    "description": "Creates a structured analysis report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "key_findings": {
                "type": "array",
                "items": {"type": "string"}
            },
            "confidence_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["title", "summary", "key_findings", "confidence_score"],
        "additionalProperties": False
    },
    "strict": True,   # Guarantee structured output
}
```

**Combining strict with `tool_choice: "tool"`** is a powerful pattern for structured data extraction:

```python
# Force structured extraction with schema guarantees
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    tools=[extract_report_tool],
    tool_choice={"type": "tool", "name": "create_report"},
    messages=[{"role": "user", "content": f"Analyse this data: {data}"}]
)

# The tool call input is guaranteed to match the schema
report_data = response.content[0].input
```

---

## 8. Parallel Tool Calls

Claude may call multiple tools in a single response. All `tool_use` blocks in the response content must be executed and all results returned before Claude continues.

```python
def handle_parallel_tool_calls(response, tools: dict) -> list[dict]:
    """Execute all tool calls in a response and return results."""
    tool_results = []

    for block in response.content:
        if block.type != "tool_use":
            continue

        tool_fn = tools.get(block.name)
        if tool_fn is None:
            content = f"Error: Tool '{block.name}' not found"
        else:
            try:
                content = tool_fn(**block.input)
            except Exception as e:
                content = f"Error: {str(e)}"

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,  # Must match the block.id exactly
            "content": content,
        })

    return tool_results
```

**Example: Claude calls two tools in one shot**

```python
# Claude response may contain:
# [ToolUseBlock(id="tu_1", name="get_dataset_statistics", input={...}),
#  ToolUseBlock(id="tu_2", name="run_sql_query", input={...})]

# Execute both and return both results
tool_results = handle_parallel_tool_calls(response, TOOLS)
# tool_results = [
#   {"type": "tool_result", "tool_use_id": "tu_1", "content": "..."},
#   {"type": "tool_result", "tool_use_id": "tu_2", "content": "..."},
# ]

# For true parallelism with async
import asyncio

async def handle_parallel_async(response, async_tools: dict) -> list[dict]:
    tasks = []
    for block in response.content:
        if block.type == "tool_use":
            tool_fn = async_tools[block.name]
            tasks.append((block.id, tool_fn(**block.input)))

    results = []
    for tool_use_id, coro in tasks:
        result = await coro
        results.append({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
        })
    return results
```

---

## 9. Tool Results

### Result as String

```python
{
    "type": "tool_result",
    "tool_use_id": "toolu_01A09q90...",
    "content": "The query returned 3 rows: [{...}, {...}, {...}]"
}
```

### Result as List of Content Blocks

Use this for structured results (e.g., JSON with image):

```python
{
    "type": "tool_result",
    "tool_use_id": "toolu_01A09q90...",
    "content": [
        {"type": "text", "text": "Statistics computed successfully."},
        {"type": "text", "text": json.dumps({"mean": 42.0, "std": 5.2})}
    ]
}
```

### Error Results

Signal tool execution errors explicitly:

```python
{
    "type": "tool_result",
    "tool_use_id": "toolu_01A09q90...",
    "content": "Error: Dataset 'sales_q5' does not exist. Available datasets: sales_q1, sales_q2, sales_q3, sales_q4",
    "is_error": True   # Optional but useful — tells Claude the tool failed
}
```

When `is_error: True`, Claude will acknowledge the failure and may retry with different parameters or explain the problem to the user.

---

## 10. Anthropic-Schema Tools

Some tools have schemas baked into Claude's training, making them more reliable than custom tools. These use the `type` field in the tool definition:

```python
# Bash tool (computer use beta)
bash_tool = {"type": "bash_20250124", "name": "bash"}

# Text editor tool (computer use beta)
text_editor_tool = {"type": "text_editor_20250429", "name": "str_replace_based_edit_tool"}

# Computer tool (computer use beta)
computer_tool = {"type": "computer_20250124", "name": "computer"}
```

**Why they're more reliable:**
- The model is trained specifically on these schemas — less hallucination of parameters
- Anthropic optimises the tool calling patterns for them
- They have built-in safeguards and behaviours

**For data science**: You'll mostly use custom-defined tools. Anthropic-schema tools are used in computer-use scenarios.

---

## 11. Tool Registry Pattern

A `TOOL_REGISTRY` dict maps tool names to both their definition and their implementation, keeping them co-located:

```python
from typing import Callable, Any
import json

class Tool:
    def __init__(self, definition: dict, implementation: Callable):
        self.definition = definition
        self.implementation = implementation
        self.name = definition["name"]

    def execute(self, **kwargs) -> str:
        result = self.implementation(**kwargs)
        if isinstance(result, (dict, list)):
            return json.dumps(result, default=str)
        return str(result)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, definition: dict):
        """Decorator: @registry.register(definition)"""
        def decorator(fn: Callable) -> Callable:
            tool = Tool(definition, fn)
            self._tools[tool.name] = tool
            return fn
        return decorator

    @property
    def definitions(self) -> list[dict]:
        return [t.definition for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return f"Error: Unknown tool '{name}'"
        try:
            return self._tools[name].execute(**kwargs)
        except Exception as e:
            return f"Error: {str(e)}"


# Usage
registry = ToolRegistry()

@registry.register({
    "name": "get_dataset_statistics",
    "description": "Computes descriptive statistics for a dataset.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dataset_name": {"type": "string", "description": "Dataset identifier"}
        },
        "required": ["dataset_name"]
    }
})
def get_dataset_statistics(dataset_name: str) -> dict:
    # Real implementation here
    return {"mean": 42.0, "std": 5.2, "count": 1000}


# In the agent loop
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=registry.definitions,
    messages=messages,
)

for block in response.content:
    if block.type == "tool_use":
        result = registry.execute(block.name, **block.input)
```

---

## 12. Complete Working Example

A complete data analysis tool with request/response cycle:

```python
import anthropic
import json
import statistics

client = anthropic.Anthropic()

# --- Tool implementations ---

def calculate_statistics(numbers: list[float], include_outliers: bool = False) -> str:
    if not numbers:
        return json.dumps({"error": "Empty list provided"})
    result = {
        "count": len(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "std_dev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
        "min": min(numbers),
        "max": max(numbers),
    }
    if include_outliers:
        mean = result["mean"]
        std = result["std_dev"]
        result["outliers"] = [x for x in numbers if abs(x - mean) > 2 * std]
    return json.dumps(result, indent=2)

def linear_regression(x_values: list[float], y_values: list[float]) -> str:
    if len(x_values) != len(y_values):
        return json.dumps({"error": "x and y must have the same length"})
    n = len(x_values)
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        return json.dumps({"error": "All x values are the same"})
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return json.dumps({"slope": slope, "intercept": intercept, "equation": f"y = {slope:.4f}x + {intercept:.4f}"})

TOOLS_IMPL = {
    "calculate_statistics": calculate_statistics,
    "linear_regression": linear_regression,
}

TOOL_DEFINITIONS = [
    {
        "name": "calculate_statistics",
        "description": (
            "Computes descriptive statistics for a list of numbers, "
            "including mean, median, standard deviation, min, and max. "
            "Use this when you need to summarise or describe a numeric dataset. "
            "Optionally identifies statistical outliers (values beyond 2 standard deviations)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "The list of numeric values to analyse"
                },
                "include_outliers": {
                    "type": "boolean",
                    "description": "Whether to identify outliers. Default: false",
                    "default": False
                }
            },
            "required": ["numbers"]
        }
    },
    {
        "name": "linear_regression",
        "description": (
            "Fits a simple linear regression model (y = mx + b) to paired x and y values. "
            "Use this when the user asks about trends, relationships between variables, "
            "or wants to predict values. "
            "Returns the slope, intercept, and equation string. "
            "Does not handle multiple regression — use calculate_statistics for simple correlations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Independent variable values"
                },
                "y_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Dependent variable values (same length as x_values)"
                }
            },
            "required": ["x_values", "y_values"]
        }
    }
]

# --- Agentic loop ---

def run_data_analysis(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    system = (
        "You are a data scientist assistant with access to statistical tools. "
        "Always use the tools to compute exact values rather than estimating. "
        "After computing, explain the results clearly in plain language."
    )

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = TOOLS_IMPL.get(block.name)
                    result = fn(**block.input) if fn else f"Error: Unknown tool '{block.name}'"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

        else:
            raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")


# Test it
if __name__ == "__main__":
    data = [23.1, 45.2, 38.7, 29.4, 52.1, 41.8, 95.3, 36.2, 40.1, 43.7]
    result = run_data_analysis(
        f"Analyse this dataset: {data}. "
        "Are there any outliers? Also check if there's a linear trend "
        "by treating the index as x and the value as y."
    )
    print(result)
```
