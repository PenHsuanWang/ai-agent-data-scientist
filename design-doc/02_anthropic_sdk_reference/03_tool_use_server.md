# Server-Executed Tools — Web Search, Code Execution, Files API

## 1. Server Tool Model

Server tools are functions that **Anthropic's infrastructure** executes on Claude's behalf. Unlike client tools (where your code runs the function), with server tools:

- You declare the tool in the request (by name — no schema definition needed)
- Claude decides to call it and provides arguments
- **Anthropic's servers** execute the function
- Results come back as part of Claude's response automatically
- You do **not** need to send `tool_result` messages

**Comparison:**

| Aspect | Client Tools | Server Tools |
|--------|-------------|--------------|
| Who executes | Your code | Anthropic's servers |
| Tool definition | Full schema required | Just name + optional config |
| `tool_result` message | Required | Not needed |
| Examples | DB queries, custom APIs | Web search, code execution |
| Latency | Depends on your code | Managed by Anthropic |
| Cost | Tool execution is free | Some tools have usage costs |

### `server_tool_use` Block

When Claude calls a server tool, you see a `server_tool_use` content block in the response:

```python
# response.content may contain:
for block in response.content:
    if block.type == "server_tool_use":
        print(f"Server tool called: {block.name}")
        print(f"Input: {block.input}")
        # You do NOT need to execute this or send a tool_result
```

---

## 2. The Server-Side Loop

When server tools are involved, Anthropic runs an **internal iteration loop**:

```
Your request
    ↓
Claude decides to call web_search
    ↓
Anthropic executes web_search
    ↓
Result injected back to Claude
    ↓
Claude may call more tools or give final answer
    ↓
Final response returned to you
```

This loop runs on Anthropic's side. From your perspective, you make one API call and get back the final answer (which may include intermediate `server_tool_use` blocks for transparency).

### The `pause_turn` Stop Reason

If the server-side loop hits its iteration limit before Claude finishes, the response arrives with:

```python
response.stop_reason == "pause_turn"
```

This is not an error — it means Anthropic paused mid-task. To continue, simply re-send the conversation:

```python
def run_with_server_tools(messages: list, tools: list, max_resumes: int = 5) -> str:
    """Handle pause_turn by resuming automatically."""
    for _ in range(max_resumes):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "pause_turn":
            # Re-send conversation; Anthropic continues where it left off
            messages.append({"role": "user", "content": "Please continue."})
            continue

        if response.stop_reason == "tool_use":
            # Client tools also present — handle them
            return handle_client_tools(response, messages, tools)

    raise RuntimeError("Max resumes exceeded")
```

---

## 3. `web_search_20260209`

Performs a real-time web search and returns results. Claude uses this to answer questions requiring current information.

### Configuration

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[
        {
            "type": "web_search_20260209",
            "name": "web_search",
        }
    ],
    messages=[{
        "role": "user",
        "content": "What are the latest developments in transformer architectures as of 2025?"
    }]
)
```

### Domain Filtering

Restrict or block specific domains:

```python
{
    "type": "web_search_20260209",
    "name": "web_search",
    "allowed_domains": [
        "arxiv.org",
        "nature.com",
        "sciencedirect.com",
        "pubmed.ncbi.nlm.nih.gov"
    ]
}
```

```python
{
    "type": "web_search_20260209",
    "name": "web_search",
    "blocked_domains": [
        "reddit.com",
        "twitter.com",
        "facebook.com"
    ]
}
```

**Domain filtering rules:**
- `allowed_domains` and `blocked_domains` are mutually exclusive
- Subdomains are included automatically (e.g., `"nature.com"` covers `"www.nature.com"`)
- Use `allowed_domains` to restrict to authoritative sources (scientific search use case)
- Use `blocked_domains` to exclude noise sources while keeping the rest of the web

### Query Optimisation

Claude is trained to write good search queries, but you can guide it:

```python
system = (
    "When searching the web, always use specific, technical search terms. "
    "Prefer academic and official sources. "
    "For data-related questions, prioritise papers and technical documentation. "
    "Always cite your sources in the answer."
)
```

---

## 4. `web_fetch_20260209`

Fetches the content of a specific URL and returns it as text.

### Configuration

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
        }
    ],
    messages=[{
        "role": "user",
        "content": "Fetch and summarise this paper: https://arxiv.org/abs/1706.03762"
    }]
)
```

### Domain Restrictions

Same `allowed_domains` / `blocked_domains` options as `web_search_20260209`.

### When to Use vs `web_search`

| Situation | Use |
|-----------|-----|
| You know the exact URL | `web_fetch_20260209` |
| You need to find information | `web_search_20260209` |
| Reading API documentation | `web_fetch_20260209` |
| Researching a topic | `web_search_20260209` |
| Monitoring a specific page | `web_fetch_20260209` |

### Combining Both

```python
tools = [
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "web_fetch_20260209",  "name": "web_fetch"},
]
# Claude can search, then fetch the most relevant result
```

---

## 5. `code_execution_20260120`

Executes Python code in an Anthropic-managed sandboxed container. The newest and most capable code execution tool, supporting programmatic tool calling.

### Configuration

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    tools=[
        {"type": "code_execution_20260120", "name": "code_execution"}
    ],
    messages=[{
        "role": "user",
        "content": "Load the uploaded CSV file and show me summary statistics."
    }]
)
```

### Pre-Installed Python Packages

The container comes with the full data science stack pre-installed:

| Category | Packages |
|----------|---------|
| Data manipulation | `pandas`, `numpy` |
| Scientific computing | `scipy`, `sympy` |
| Machine learning | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Units & physics | `pint` |
| Statistics | `statsmodels` (via scipy) |

> **Note**: You cannot `pip install` additional packages. If you need a package not on this list, use client tools instead.

### Container Specifications

| Spec | Value |
|------|-------|
| CPU | 1 core |
| RAM | 5 GiB |
| Disk | 5 GiB |
| Python version | 3.11+ |
| Container lifetime | 30 days |
| Idle timeout | 4.5 minutes |
| Network access | None (isolated sandbox) |

### Session Persistence

Within a single conversation session, the container's state **persists across multiple code execution calls** (up to the 4.5-minute idle timeout):

```python
# Turn 1: Claude executes this code
"""
import pandas as pd
df = pd.read_csv('/files/sales.csv')
print(df.head())
"""

# Turn 2: Claude can reference df from Turn 1
"""
# df is still in memory!
mean_revenue = df['revenue'].mean()
print(f"Mean revenue: {mean_revenue}")
"""
```

After 4.5 minutes of idle time, the container resets and all variables are lost.

---

## 6. Files API

The Files API (beta) lets you upload files once and reference them by ID across multiple requests.

### Required Beta Header

```python
client = anthropic.Anthropic()
# Files API requires the beta header — the SDK handles this with betas parameter
```

### Uploading Files

```python
# Upload a CSV file
with open("sales_data.csv", "rb") as f:
    file_response = client.beta.files.upload(
        file=("sales_data.csv", f, "text/csv"),
    )

file_id = file_response.id  # "file_01ABC..."
print(f"Uploaded file ID: {file_id}")

# Upload an image
with open("chart.png", "rb") as f:
    image_file = client.beta.files.upload(
        file=("chart.png", f, "image/png"),
    )
```

### Using Files in Messages

Reference the file_id in message content:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyse the sales data in this file:"},
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id
                }
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
```

### Container Upload (for Code Execution)

To make files available inside the code execution container:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyse the uploaded CSV:"},
            {
                "type": "container_upload",
                "file_id": file_id,
                "filename": "sales_data.csv"  # Name inside the container
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
# Claude can now open '/files/sales_data.csv' in its Python code
```

---

## 7. Programmatic Tool Calling

Programmatic tool calling is an advanced feature of `code_execution_20260120` that lets Claude write Python code which **calls your custom tools directly**, without extra API round-trips.

### Setup

You declare your client tools with `"allowed_callers": ["code_execution_20260120"]`:

```python
validate_units_tool = {
    "name": "validate_physical_units",
    "description": (
        "Validates that a physical measurement has correct units for its type. "
        "Use this in loops to validate many measurements at once. "
        "Returns 'valid' or an error message explaining the unit mismatch."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "unit": {"type": "string"},
            "expected_dimension": {
                "type": "string",
                "enum": ["length", "mass", "time", "temperature", "energy", "velocity"]
            }
        },
        "required": ["value", "unit", "expected_dimension"]
    },
    "allowed_callers": ["code_execution_20260120"]  # Key field
}
```

### How Claude Uses It

Claude writes Python code like this inside the code execution environment:

```python
# Claude generates this code inside the sandbox
results = []
for measurement in measurements_list:
    # Claude calls your tool with `await` syntax
    result = await validate_physical_units(
        value=measurement["value"],
        unit=measurement["unit"],
        expected_dimension=measurement["dimension"]
    )
    results.append({"id": measurement["id"], "status": result})

valid_count = sum(1 for r in results if r["status"] == "valid")
print(f"Valid: {valid_count}/{len(results)}")
```

The `await` calls your actual Python function (running in your process), but from within Anthropic's sandbox.

---

## 8. ZDR and Privacy

Zero Data Retention (ZDR) means Anthropic does not store request or response data.

### ZDR Eligibility by Tool

| Tool | ZDR-Eligible |
|------|-------------|
| `web_search_20260209` | ✅ Yes |
| `web_fetch_20260209` | ✅ Yes |
| `code_execution_20260120` | ❌ No |
| Files API (for code execution) | ❌ No |
| Client tools (custom) | ✅ Yes (depends on API access) |

### Direct Tool Access (ZDR Mode for Programmatic Calling)

For programmatic tool calling in ZDR mode, use:

```python
"allowed_callers": ["direct"]
```

This allows programmatic tool calling without code execution, maintaining ZDR eligibility. However, the tools are called sequentially (no code execution sandbox), losing the performance benefit.

---

## 9. Cost

### Standard Pricing

Server tools are billed at the same token rates as regular messages. Additional usage fees apply:

| Tool | Additional Cost |
|------|----------------|
| `web_search_20260209` | Per-search fee (check current pricing) |
| `web_fetch_20260209` | Per-fetch fee (check current pricing) |
| `code_execution_20260120` | See below |

### Code Execution Pricing

**FREE** when bundled with `web_search_20260209` **or** `web_fetch_20260209` in the same request.

**Paid** when used standalone. Check the [Anthropic pricing page](https://www.anthropic.com/pricing) for current code execution session costs.

```python
# This combination: code_execution is FREE
tools = [
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "code_execution_20260120", "name": "code_execution"},
]

# This: code_execution may have a fee
tools = [
    {"type": "code_execution_20260120", "name": "code_execution"},
]
```

---

## 10. Platform Availability

Not all server tools are available on every platform:

| Tool | Anthropic API | Azure AI | Bedrock | Vertex AI |
|------|:---:|:---:|:---:|:---:|
| `web_search_20260209` | ✅ | ✅ | ❌ | ❌ |
| `web_fetch_20260209` | ✅ | ✅ | ❌ | ❌ |
| `code_execution_20260120` | ✅ | ✅ | ❌ | ❌ |
| `code_execution_20250825` (older) | ✅ | ⚠️ | ❌ | ❌ |
| Files API | ✅ | ✅ | ❌ | ❌ |
| Extended thinking | ✅ | ✅ | ✅ | ✅ |
| Prompt caching | ✅ | ✅ | ✅ | ✅ |

For cross-platform compatibility (Bedrock, Vertex): use client tools only.

---

## 11. Error Handling

### Server Tool Use Errors

Server tool errors are embedded in the response content, not raised as exceptions:

```python
for block in response.content:
    if block.type == "server_tool_use":
        if hasattr(block, "error"):
            print(f"Server tool error: {block.error}")
```

### Handling `pause_turn`

```python
import anthropic

client = anthropic.Anthropic()

def run_to_completion(
    initial_messages: list,
    tools: list,
    system: str = "",
    max_resumes: int = 10
) -> str:
    """Run a server-tool-enabled conversation to completion."""
    messages = list(initial_messages)

    for resume_count in range(max_resumes):
        kwargs = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 8192,
            "tools": tools,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        # Always append the assistant response
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract text from final response
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks)

        elif response.stop_reason == "pause_turn":
            # Server hit iteration limit — continue
            messages.append({
                "role": "user",
                "content": "Please continue with the analysis."
            })
            print(f"[pause_turn] Resuming (attempt {resume_count + 1}/{max_resumes})")
            continue

        elif response.stop_reason == "tool_use":
            # Client tools need execution
            tool_results = execute_client_tools(response, client_tool_registry)
            messages.append({"role": "user", "content": tool_results})
            continue

        else:
            raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"Did not complete after {max_resumes} resumes")


def execute_client_tools(response, registry: dict) -> list[dict]:
    results = []
    for block in response.content:
        if block.type == "tool_use":
            fn = registry.get(block.name)
            if fn:
                try:
                    content = fn(**block.input)
                except Exception as e:
                    content = f"Error: {str(e)}"
            else:
                content = f"Error: Unknown tool '{block.name}'"
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
    return results


# Example: research + code analysis pipeline
if __name__ == "__main__":
    tools = [
        {"type": "web_search_20260209", "name": "web_search"},
        {"type": "code_execution_20260120", "name": "code_execution"},
    ]

    result = run_to_completion(
        initial_messages=[{
            "role": "user",
            "content": (
                "Search for the latest benchmarks on scikit-learn RandomForest performance, "
                "then write and execute Python code to demonstrate a RandomForest classifier "
                "on the iris dataset, reporting accuracy."
            )
        }],
        tools=tools,
        system="You are a machine learning research assistant. Always run code to verify claims."
    )
    print(result)
```
