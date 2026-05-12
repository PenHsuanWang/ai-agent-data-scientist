# Anthropic Messages API — Complete Reference

## 1. Overview

The Messages API is the primary interface for interacting with Claude models. Each call represents a conversation turn (or a full multi-turn conversation) and returns Claude's response as a structured `Message` object.

**REST vs Python SDK**

Under the hood the Python SDK is a thin wrapper around the REST endpoint:

```
POST https://api.anthropic.com/v1/messages
```

The SDK handles authentication headers, serialisation, retries, and response parsing. You should always use the SDK in Python projects — it catches breaking changes and provides type safety.

**Two execution modes:**

| Mode | Client class | Use case |
|------|-------------|----------|
| Synchronous | `anthropic.Anthropic` | Scripts, notebooks, simple services |
| Asynchronous | `anthropic.AsyncAnthropic` | FastAPI, concurrent pipelines, high-throughput agents |

---

## 2. Client Initialization

### Synchronous Client

```python
import anthropic

# Minimal — reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Explicit key (avoid hardcoding in production)
client = anthropic.Anthropic(api_key="sk-ant-...")

# Full configuration
client = anthropic.Anthropic(
    api_key="sk-ant-...",              # SecretStr internally — never logged
    base_url="https://your-proxy/",    # For corporate proxies or testing
    max_retries=3,                     # Default: 2. Set 0 to disable
    timeout=60.0,                      # Seconds. Default: 600
)
```

### Asynchronous Client

```python
import anthropic

async_client = anthropic.AsyncAnthropic(
    api_key="sk-ant-...",
    max_retries=3,
    timeout=120.0,
)
```

### API Key Handling

The SDK wraps the key in a `SecretStr`-like type that redacts the value in logs and `repr()` output. Best practice in production:

```python
import os
from anthropic import Anthropic

# Read from environment — never commit keys to source
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
```

### Singleton Pattern for Services

Create the client once per process, not once per request:

```python
# config.py
import anthropic
import os

_client: anthropic.Anthropic | None = None

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_retries=2,
            timeout=120.0,
        )
    return _client
```

---

## 3. Basic Request

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",          # Required: model ID string
    max_tokens=1024,                     # Required: hard cap on output tokens
    system="You are a data scientist.", # Optional: system prompt
    messages=[
        {
            "role": "user",
            "content": "What is the mean of [1, 2, 3, 4, 5]?"
        }
    ],
    temperature=0.2,                    # 0.0–1.0; lower = more deterministic
    stop_sequences=["DONE"],            # Stop generation at these strings
    metadata={"user_id": "analyst-42"} # For abuse tracking; not sent to model
)

# Access the text
print(response.content[0].text)
# → "The mean of [1, 2, 3, 4, 5] is 3.0."

# Check why Claude stopped
print(response.stop_reason)
# → "end_turn"

# Token usage
print(response.usage.input_tokens, response.usage.output_tokens)
```

### All Parameters of `messages.create()`

| Parameter | Type | Required | Default | Notes |
|-----------|------|----------|---------|-------|
| `model` | `str` | ✅ | — | Model ID, e.g. `"claude-sonnet-4-6"` |
| `max_tokens` | `int` | ✅ | — | Hard cap; 128k for Opus, 64k for Sonnet/Haiku |
| `messages` | `list[dict]` | ✅ | — | Conversation turns |
| `system` | `str \| list` | ❌ | — | System prompt |
| `tools` | `list[dict]` | ❌ | — | Tool definitions |
| `tool_choice` | `dict` | ❌ | `auto` | Tool selection mode |
| `temperature` | `float` | ❌ | `1.0` | 0.0–1.0 |
| `top_p` | `float` | ❌ | — | Nucleus sampling |
| `top_k` | `int` | ❌ | — | Top-k sampling |
| `stop_sequences` | `list[str]` | ❌ | — | Stop generation tokens |
| `stream` | `bool` | ❌ | `False` | Use `.stream()` context manager instead |
| `metadata` | `dict` | ❌ | — | `{"user_id": "..."}` for tracking |
| `thinking` | `dict` | ❌ | — | `{"type": "enabled", "budget_tokens": N}` |
| `betas` | `list[str]` | ❌ | — | Beta feature opt-ins |

---

## 4. The `messages` Parameter

### Role Alternation Rules

The messages list must alternate between `"user"` and `"assistant"` roles. The first message must always be `"user"`.

```python
messages = [
    {"role": "user",      "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user",      "content": "Analyse this dataset."},
]
```

**Rules:**
- Must start with `"user"`
- Must alternate strictly (no two consecutive same roles)
- Do **not** include the system prompt in the messages list

### Content Types

**Simple string (most common):**
```python
{"role": "user", "content": "What is 2+2?"}
```

**List of content blocks (for multi-modal or tool results):**
```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this chart:"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "<base64_encoded_png>"
            }
        }
    ]
}
```

### CRITICAL: Tool Results Must Be in User Role

When Claude uses a tool, you must send the result back as a **user** message:

```python
# Claude returns a tool_use block
tool_use_block = response.content[0]  # type: ToolUseBlock

# You execute the tool, then send result in user role
messages.append({"role": "assistant", "content": response.content})
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": "The query returned 42 rows."  # or list of blocks
        }
    ]
})
```

### Multi-Turn Format for Agents

```python
conversation_history = []

def send_message(user_text: str) -> str:
    conversation_history.append({
        "role": "user",
        "content": user_text
    })
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=conversation_history
    )
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    return assistant_message
```

---

## 5. The `system` Parameter

### Single String

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a senior data scientist specialising in time-series analysis. "
           "Always explain your reasoning step by step. "
           "When you write code, use pandas and numpy.",
    messages=[{"role": "user", "content": "Detect anomalies in this series."}]
)
```

### List of Blocks (for prompt caching)

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a data scientist...",
            "cache_control": {"type": "ephemeral"}  # Cache this block
        }
    ],
    messages=[...],
    betas=["prompt-caching-2024-07-31"]
)
```

### Best Practices for Agent Systems

- **Be specific about role and expertise**: "You are a senior data scientist" outperforms "You are an AI assistant"
- **Define output format expectations**: "Always return code in fenced code blocks. Always explain results in plain English after code."
- **Set tool-use expectations**: "When the user asks a question about data, always write and execute code to verify your answer."
- **Avoid contradictions**: Don't say "be concise" and then "explain everything in detail"
- **Keep it under 4000 tokens**: Beyond that, detail rarely improves performance

---

## 6. Response Object Deep Dive

```python
response = client.messages.create(...)

# Full type: anthropic.types.Message
print(type(response))  # <class 'anthropic.types.message.Message'>
```

### Every Field

```python
response.id            # str  — "msg_01XFDUDYJgAACzvnptvVoYEL"
response.type          # str  — always "message"
response.role          # str  — always "assistant"
response.content       # list[ContentBlock] — the actual response
response.model         # str  — model that generated it, e.g. "claude-sonnet-4-6-20251015"
response.stop_reason   # str  — why generation stopped (see below)
response.stop_sequence # str | None — which stop_sequence triggered (if any)
response.usage         # Usage object
response.usage.input_tokens   # int
response.usage.output_tokens  # int
```

### The `stop_reason` Decision Tree

```
stop_reason
├── "end_turn"       → Claude finished naturally. Extract content[0].text.
├── "tool_use"       → Claude wants to call a tool. Find ToolUseBlock in content.
├── "max_tokens"     → Hit max_tokens limit. Response may be truncated.
│                      Consider increasing max_tokens or summarising context.
├── "stop_sequence"  → Hit one of your stop_sequences. Check stop_sequence field.
├── "refusal"        → Content policy refusal. Log and handle gracefully.
└── "pause_turn"     → Server-side loop hit iteration limit (server tools only).
                       Re-send conversation to continue.
```

```python
def handle_response(response):
    if response.stop_reason == "end_turn":
        return response.content[0].text
    elif response.stop_reason == "tool_use":
        return handle_tool_calls(response)
    elif response.stop_reason == "max_tokens":
        raise ValueError("Response truncated — increase max_tokens")
    elif response.stop_reason == "pause_turn":
        return continue_paused_turn(response)
    else:
        raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")
```

---

## 7. Content Blocks Catalog

The `response.content` field is always a list of content blocks. Iterate and type-narrow:

```python
import anthropic

for block in response.content:
    if block.type == "text":
        # block is TextBlock
        print("Text:", block.text)

    elif block.type == "tool_use":
        # block is ToolUseBlock
        print("Tool:", block.name)
        print("ID:", block.id)
        print("Input:", block.input)  # dict

    elif block.type == "thinking":
        # block is ThinkingBlock (extended thinking only)
        print("Thinking:", block.thinking)

    elif block.type == "server_tool_use":
        # block is ServerToolUseBlock (server tools only)
        print("Server tool:", block.name)
        print("Input:", block.input)
```

### TextBlock

```python
block.type    # "text"
block.text    # str — Claude's response text
```

### ToolUseBlock

```python
block.type    # "tool_use"
block.id      # str — "toolu_01A09q90qw90lq917835..." (unique per tool call)
block.name    # str — tool name you defined
block.input   # dict — arguments Claude passed, matching your input_schema
```

### ThinkingBlock (Extended Thinking)

```python
block.type      # "thinking"
block.thinking  # str — Claude's internal reasoning (may be long)
```

Thinking blocks appear **before** text blocks in the response. You must pass them back in subsequent messages unmodified:

```python
# Preserve thinking blocks in history
messages.append({
    "role": "assistant",
    "content": response.content  # Pass full list including ThinkingBlock
})
```

### ServerToolUseBlock

```python
block.type    # "server_tool_use"
block.id      # str
block.name    # str — e.g. "web_search_20260209"
block.input   # dict — the tool's input parameters
```

---

## 8. Usage & Pricing

```python
usage = response.usage

print(f"Input tokens:  {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")

# With prompt caching enabled, you also get:
# usage.cache_creation_input_tokens  — tokens written to cache
# usage.cache_read_input_tokens      — tokens read from cache
```

### Calculating Cost

```python
# Pricing per million tokens (MTok), May 2026
PRICING = {
    "claude-opus-4-7": {
        "input":  5.00 / 1_000_000,
        "output": 25.00 / 1_000_000,
        "cache_write": 6.25 / 1_000_000,   # 25% more than input
        "cache_read":  0.50 / 1_000_000,   # 90% discount
    },
    "claude-sonnet-4-6": {
        "input":  3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
        "cache_write": 3.75 / 1_000_000,
        "cache_read":  0.30 / 1_000_000,
    },
    "claude-haiku-4-5-20251001": {
        "input":  1.00 / 1_000_000,
        "output": 5.00 / 1_000_000,
        "cache_write": 1.25 / 1_000_000,
        "cache_read":  0.10 / 1_000_000,
    },
}

def calculate_cost(response, model: str) -> float:
    p = PRICING[model]
    usage = response.usage
    cost = (
        usage.input_tokens * p["input"] +
        usage.output_tokens * p["output"]
    )
    # Add cache costs if available
    if hasattr(usage, "cache_creation_input_tokens"):
        cost += usage.cache_creation_input_tokens * p["cache_write"]
        cost += usage.cache_read_input_tokens * p["cache_read"]
    return cost

cost = calculate_cost(response, "claude-sonnet-4-6")
print(f"Cost: ${cost:.6f}")
```

---

## 9. Error Handling

### Error Hierarchy

```
anthropic.APIError (base)
├── anthropic.APIConnectionError  — Network/DNS/TLS issues
├── anthropic.APIStatusError      — HTTP 4xx/5xx responses
│   ├── anthropic.BadRequestError       (400)
│   ├── anthropic.AuthenticationError   (401)
│   ├── anthropic.PermissionDeniedError (403)
│   ├── anthropic.NotFoundError         (404)
│   ├── anthropic.RateLimitError        (429)
│   └── anthropic.InternalServerError   (500)
└── anthropic.APITimeoutError     — Request exceeded timeout
```

### Basic Error Handling

```python
import anthropic
import time

client = anthropic.Anthropic()

def safe_create(messages: list, **kwargs) -> anthropic.types.Message:
    try:
        return client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=messages,
            **kwargs
        )
    except anthropic.RateLimitError as e:
        print(f"Rate limited: {e}")
        raise
    except anthropic.APIConnectionError as e:
        print(f"Connection error: {e}")
        raise
    except anthropic.BadRequestError as e:
        print(f"Bad request (check messages format): {e}")
        raise
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}")
        raise
```

### Exponential Backoff

```python
import anthropic
import time
import random

def create_with_backoff(
    client: anthropic.Anthropic,
    max_retries: int = 5,
    base_delay: float = 1.0,
    **kwargs
) -> anthropic.types.Message:
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)

        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited. Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(delay)

        except anthropic.APIConnectionError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Connection error. Retrying in {delay:.1f}s")
            time.sleep(delay)

        except anthropic.InternalServerError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Server error. Retrying in {delay:.1f}s")
            time.sleep(delay)

    raise RuntimeError("Max retries exceeded")

# Usage
response = create_with_backoff(
    client,
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

> **Note**: The SDK's built-in `max_retries` parameter handles this automatically for most cases. Manual backoff is only needed for custom retry logic.

---

## 10. Async Usage

### Basic Async

```python
import asyncio
import anthropic

async_client = anthropic.AsyncAnthropic()

async def analyse_dataset(question: str) -> str:
    response = await async_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are a data scientist.",
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

# Run
result = asyncio.run(analyse_dataset("What is regression analysis?"))
```

### Concurrent Requests with `asyncio.gather()`

```python
import asyncio
import anthropic

async_client = anthropic.AsyncAnthropic()

async def analyse_single(dataset_name: str, question: str) -> dict:
    response = await async_client.messages.create(
        model="claude-haiku-4-5-20251001",  # Use fast model for batch
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"For dataset '{dataset_name}': {question}"
        }]
    )
    return {
        "dataset": dataset_name,
        "analysis": response.content[0].text,
        "tokens": response.usage.input_tokens + response.usage.output_tokens
    }

async def analyse_all_datasets(datasets: list[str], question: str) -> list[dict]:
    tasks = [analyse_single(ds, question) for ds in datasets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    if failed:
        print(f"Warning: {len(failed)} requests failed")
    return successful

# Usage
datasets = ["sales_q1", "sales_q2", "sales_q3", "sales_q4"]
results = asyncio.run(analyse_all_datasets(datasets, "What is the trend?"))
```

### Async in FastAPI

```python
from fastapi import FastAPI
import anthropic

app = FastAPI()
client = anthropic.AsyncAnthropic()

@app.post("/analyse")
async def analyse(request: dict) -> dict:
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": request["question"]}]
    )
    return {
        "answer": response.content[0].text,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens
    }
```

---

## 11. Streaming

Streaming returns tokens as they are generated, improving perceived latency for long responses.

### Text Streaming

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Explain gradient descent in detail."}]
) as stream:
    # Stream tokens as they arrive
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print()  # Newline after streaming completes

    # Get the complete final message (available after stream ends)
    final_message = stream.get_final_message()
    print(f"\nTotal tokens: {final_message.usage.input_tokens + final_message.usage.output_tokens}")
```

### Async Streaming

```python
import asyncio
import anthropic

async_client = anthropic.AsyncAnthropic()

async def stream_response(question: str) -> str:
    full_text = ""
    async with async_client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": question}]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
    print()
    return full_text
```

### Handling Tool Calls in Streams

When using tools with streaming, you must handle tool_use events specially:

```python
import anthropic
import json

client = anthropic.Anthropic()

def stream_with_tools(messages: list, tools: list) -> dict:
    """Stream a response that may include tool calls."""
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=messages,
        tools=tools,
    ) as stream:
        # Collect all events
        for event in stream:
            pass  # Events are processed internally

        # After stream ends, get the complete message
        final = stream.get_final_message()

    return {
        "stop_reason": final.stop_reason,
        "content": final.content,
        "usage": final.usage,
    }
```

### Stream Event Types

The stream emits these events (accessible via `stream` iteration):

| Event type | When |
|-----------|------|
| `message_start` | Beginning of response; has initial message object |
| `content_block_start` | Start of a new content block |
| `content_block_delta` | Incremental text delta |
| `content_block_stop` | End of a content block |
| `message_delta` | Updates to message (stop_reason, usage) |
| `message_stop` | Stream finished |

---

## 12. Production Patterns

### Connection Pooling

The SDK automatically maintains an HTTP connection pool. Use a single client instance per process:

```python
# ✅ Good — one client, reused across requests
client = anthropic.Anthropic(timeout=120.0)

# ❌ Bad — new client per request (defeats connection pooling)
def handle_request(data):
    client = anthropic.Anthropic()  # Don't do this
    return client.messages.create(...)
```

### Timeout Configuration

```python
import httpx
import anthropic

# Fine-grained timeout control
client = anthropic.Anthropic(
    timeout=httpx.Timeout(
        connect=5.0,   # Seconds to establish connection
        read=120.0,    # Seconds to wait for data
        write=10.0,    # Seconds to write request
        pool=5.0,      # Seconds to wait for connection from pool
    )
)
```

### Request-Level Timeout Override

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[...],
    timeout=30.0  # Override client-level timeout for this request
)
```

### Structured Logging

```python
import anthropic
import logging
import time

logger = logging.getLogger(__name__)

def traced_create(client: anthropic.Anthropic, **kwargs) -> anthropic.types.Message:
    start = time.monotonic()
    try:
        response = client.messages.create(**kwargs)
        elapsed = time.monotonic() - start
        logger.info(
            "API call succeeded",
            extra={
                "model": response.model,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": round(elapsed * 1000),
            }
        )
        return response
    except anthropic.APIError as e:
        elapsed = time.monotonic() - start
        logger.error(
            "API call failed",
            extra={
                "error_type": type(e).__name__,
                "latency_ms": round(elapsed * 1000),
            }
        )
        raise
```
