# Anthropic Python SDK — Quick Reference Cheatsheet

---

## Client Initialization

```python
import anthropic

# Sync client
client = anthropic.Anthropic(api_key="sk-ant-...")  # or reads ANTHROPIC_API_KEY env var

# Async client (use in FastAPI / asyncio)
client = anthropic.AsyncAnthropic(api_key="sk-ant-...")

# With custom settings
client = anthropic.Anthropic(
    api_key="sk-ant-...",
    max_retries=3,           # default: 2
    timeout=60.0,            # seconds; default: 600
    base_url="https://...",  # for proxies
)
```

---

## messages.create() — All Parameters

```python
response = client.messages.create(
    # Required
    model="claude-sonnet-4-6",          # model ID (see table below)
    max_tokens=4096,                    # max tokens in response (required)
    messages=[                          # conversation history
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Tell me about Python."},
    ],

    # Optional
    system="You are a helpful assistant.",  # system prompt (string or list)
    temperature=1.0,           # 0-1; default: 1.0; lower = more deterministic
    top_p=0.999,               # nucleus sampling; mutually exclusive with top_k
    top_k=40,                  # token sampling cutoff
    stop_sequences=["END"],    # stop before these strings
    stream=False,              # True to stream (returns Stream object)
    metadata={"user_id": "u1"}, # arbitrary metadata (not sent to model)

    # Tools (for tool_use stop_reason)
    tools=[
        {
            "name": "get_weather",
            "description": "Get weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
            },
        }
    ],
    tool_choice={"type": "auto"},  # auto | any | tool (force specific tool)

    # Prompt caching
    # (add cache_control to content blocks — see Prompt Caching section)

    # Extended thinking (claude-3-7-sonnet and above)
    thinking={"type": "enabled", "budget_tokens": 5000},
)
```

---

## Response Fields

```python
response.id               # str: "msg_01XFDUDYJgAACzvnptvVoYEL"
response.type             # str: "message"
response.role             # str: "assistant"
response.model            # str: "claude-sonnet-4-6"
response.stop_reason      # str: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
response.stop_sequence    # str | None: which stop sequence triggered
response.content          # list[ContentBlock]
response.usage.input_tokens   # int
response.usage.output_tokens  # int
```

---

## Content Block Types

```python
# Text block
response.content[0].type  # "text"
response.content[0].text  # str

# Tool use block
response.content[0].type  # "tool_use"
response.content[0].id    # str: "toolu_01A09q90qw90lq917835lq9"
response.content[0].name  # str: "get_weather"
response.content[0].input # dict: {"city": "New York"}

# Thinking block (extended thinking)
response.content[0].type     # "thinking"
response.content[0].thinking # str (raw reasoning text)

# Type narrowing (recommended pattern)
from anthropic.types import TextBlock, ToolUseBlock
for block in response.content:
    if isinstance(block, TextBlock):
        print(block.text)
    elif isinstance(block, ToolUseBlock):
        print(block.name, block.input)
```

---

## Tool Use Patterns

### Server-side (Claude calls tools, you execute)

```python
# 1. Initial request with tools
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=[weather_tool_schema],
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
)

# 2. Check if tool was called
if response.stop_reason == "tool_use":
    tool_block = next(b for b in response.content if b.type == "tool_use")
    result = my_weather_function(**tool_block.input)

    # 3. Send tool result back
    followup = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[weather_tool_schema],
        messages=[
            {"role": "user", "content": "Weather in Tokyo?"},
            {"role": "assistant", "content": response.content},  # pass full content list
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": str(result),   # or list of content blocks
                    }
                ],
            },
        ],
    )
```

### Force a specific tool

```python
tool_choice={"type": "tool", "name": "get_weather"}
```

### Force any tool (not text)

```python
tool_choice={"type": "any"}
```

---

## stop_reason Decision Tree

```python
match response.stop_reason:
    case "end_turn":
        # Normal completion — extract text
        text = "".join(b.text for b in response.content if b.type == "text")

    case "tool_use":
        # Claude wants to call a tool — extract tool_use blocks
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        for call in tool_calls:
            result = dispatch(call.name, call.input)
            # ... build tool_result messages and continue

    case "max_tokens":
        # Response was cut off — consider increasing max_tokens
        # or splitting the task
        partial = "".join(b.text for b in response.content if b.type == "text")

    case "stop_sequence":
        # One of stop_sequences was hit
        # response.stop_sequence tells you which one

    case _:
        # Unknown — treat as error
        raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")
```

---

## Error Types

```python
# All are subclasses of anthropic.APIError
try:
    response = client.messages.create(...)
except anthropic.AuthenticationError:   # 401 — invalid API key
    ...
except anthropic.PermissionDeniedError: # 403 — no access to model
    ...
except anthropic.NotFoundError:         # 404 — resource not found
    ...
except anthropic.RateLimitError:        # 429 — slow down; has Retry-After header
    ...
except anthropic.BadRequestError:       # 400 — invalid request (check message)
    ...
except anthropic.InternalServerError:   # 500 — Anthropic server error
    ...
except anthropic.APITimeoutError:       # request timed out
    ...
except anthropic.APIConnectionError:    # network error
    ...
except anthropic.APIStatusError as e:   # base for HTTP errors
    print(e.status_code, e.response, e.message)
```

---

## Streaming

```python
# Sync streaming
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Count to 10."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Get final message after stream
final_message = stream.get_final_message()

# Async streaming
async with client.messages.stream(...) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)

# Raw event streaming
with client.messages.stream(...) as stream:
    for event in stream:
        # event.type: "message_start", "content_block_start",
        #             "content_block_delta", "content_block_stop", "message_stop"
        if event.type == "content_block_delta":
            print(event.delta.text, end="")
```

---

## Prompt Caching

Cache static content (system prompt, large documents) to reduce cost/latency:

```python
# Mark content blocks for caching with cache_control
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant.",
        },
        {
            "type": "text",
            "text": very_long_document_text,   # min ~1024 tokens to be worth caching
            "cache_control": {"type": "ephemeral"},  # cached for 5 minutes
        },
    ],
    messages=[{"role": "user", "content": "Summarise the document."}],
)

# Check cache usage
print(response.usage.cache_creation_input_tokens)   # tokens written to cache
print(response.usage.cache_read_input_tokens)       # tokens read from cache (free!)
```

Rules:
- Minimum ~1024 tokens for cache to activate
- `ephemeral` cache lasts 5 minutes (extended to 1 hour with beta flag)
- Max 4 cache control breakpoints per request

---

## Batch API

For large-scale offline processing:

```python
# Create a batch
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": "req-001",
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        },
        # ... up to 10,000 requests
    ]
)

print(batch.id)           # "msgbatch_..."
print(batch.processing_status)  # "in_progress" | "ended"

# Poll until done
import time
while True:
    batch = client.messages.batches.retrieve(batch.id)
    if batch.processing_status == "ended":
        break
    time.sleep(30)

# Stream results
for result in client.messages.batches.results(batch.id):
    print(result.custom_id, result.result.message.content[0].text)
```

---

## Model IDs (2025-2026)

| Model | ID | Context | Notes |
|---|---|---|---|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 200K | **Recommended for agents** |
| Claude Sonnet 4.5 | `claude-sonnet-4-5` | 200K | Previous Sonnet |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | Fastest, cheapest |
| Claude Opus 4 | `claude-opus-4-0` | 200K | Most capable |
| Claude Sonnet 3.7 | `claude-3-7-sonnet-20250219` | 200K | Extended thinking |
| Claude Haiku 3.5 | `claude-3-5-haiku-20241022` | 200K | — |

---

## Token Counting

```python
# Count tokens before sending (no charge)
token_count = client.messages.count_tokens(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello world!"}],
)
print(token_count.input_tokens)  # e.g. 10

# With system and tools
token_count = client.messages.count_tokens(
    model="claude-sonnet-4-6",
    system="You are an assistant.",
    tools=[my_tool_schema],
    messages=messages,
)
```

---

## Rate Limits and Retry

```python
# Built-in retry (default: 2 retries with exponential backoff)
client = anthropic.Anthropic(max_retries=3)

# Manual exponential backoff
import time
for attempt in range(5):
    try:
        response = client.messages.create(...)
        break
    except anthropic.RateLimitError as e:
        wait = 2 ** attempt
        print(f"Rate limited. Waiting {wait}s...")
        time.sleep(wait)

# Rate limit headers (on RateLimitError)
err.response.headers["retry-after"]           # seconds to wait
err.response.headers["anthropic-ratelimit-requests-remaining"]
err.response.headers["anthropic-ratelimit-tokens-remaining"]
```

---

## System Prompt as List (for multiple cache points)

```python
# String form (simple)
system="You are a helpful assistant."

# List form (for caching or image content)
system=[
    {"type": "text", "text": "You are a helpful assistant."},
    {"type": "text", "text": large_document, "cache_control": {"type": "ephemeral"}},
]
```

---

## Async Patterns (FastAPI)

```python
# Single shared client (recommended)
_client = anthropic.AsyncAnthropic()

async def call_claude(messages: list[dict]) -> str:
    response = await _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=messages,
    )
    return response.content[0].text

# FastAPI lifespan pattern
from contextlib import asynccontextmanager
from fastapi import FastAPI

_client: anthropic.AsyncAnthropic | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = anthropic.AsyncAnthropic()
    yield
    await _client.close()

app = FastAPI(lifespan=lifespan)
```
