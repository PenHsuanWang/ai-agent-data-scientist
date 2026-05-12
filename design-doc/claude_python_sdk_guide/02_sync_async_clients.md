# Chapter 2 — Sync & Async Clients

*← [Chapter 1: Core Concepts](01_core_concepts.md) | [Chapter 3: Streaming](03_streaming.md) →*

---

## Two Clients, One Interface

The SDK ships with two client classes that share an **identical API surface**. The only difference is whether calls are synchronous or asynchronous.

| Client | Import | When to use |
|--------|--------|-------------|
| `Anthropic` | `from anthropic import Anthropic` | Scripts, CLI tools, simple web frameworks (Flask, Django) |
| `AsyncAnthropic` | `from anthropic import AsyncAnthropic` | `asyncio`-based frameworks (FastAPI, Starlette, aiohttp servers) |

---

## Synchronous Client

```python
import anthropic

client = anthropic.Anthropic()   # ANTHROPIC_API_KEY from environment

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is Python?"}],
)
print(response.content[0].text)
```

The call **blocks** until Claude returns a complete response. This is simple and predictable, but means your program can't do anything else while waiting.

---

## Asynchronous Client

```python
import asyncio
import anthropic

async def main():
    client = anthropic.AsyncAnthropic()

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is Python?"}],
    )
    print(response.content[0].text)
    await client.close()

asyncio.run(main())
```

The `await` keyword suspends the coroutine while waiting for Claude, allowing the event loop to run other tasks in the meantime.

---

## Concurrent Fan-Out Pattern

The biggest advantage of async is the ability to send **multiple requests simultaneously** with `asyncio.gather`.

```python
import asyncio
import anthropic

async def ask(client: anthropic.AsyncAnthropic, question: str) -> str:
    resp = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text.strip()

async def main():
    client = anthropic.AsyncAnthropic()

    questions = [
        "What is the capital of Japan?",
        "Who wrote Pride and Prejudice?",
        "What is the boiling point of water in Celsius?",
    ]

    # All three requests run concurrently — total time ≈ slowest request
    answers = await asyncio.gather(*[ask(client, q) for q in questions])

    for q, a in zip(questions, answers):
        print(f"Q: {q}\nA: {a}\n")

    await client.close()

asyncio.run(main())
```

**Sequential:** ~3 × latency  
**Concurrent:** ~1 × latency (all requests inflight simultaneously)

---

## aiohttp Backend

By default the SDK uses `httpx` as the HTTP transport. For very high concurrency workloads (hundreds of simultaneous requests), the `aiohttp` backend can offer better performance.

```bash
pip install aiohttp
```

Usage depends on your SDK version. Check the changelog if the import path differs:

```python
import asyncio
import aiohttp
import anthropic

async def main():
    # Create a shared aiohttp session for connection pooling
    async with aiohttp.ClientSession() as session:
        # Pass a custom httpx client that wraps aiohttp if SDK supports it,
        # or use the default AsyncAnthropic for most use cases
        client = anthropic.AsyncAnthropic()
        resp = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(resp.content[0].text)

asyncio.run(main())
```

---

## Client Configuration Options

Both clients accept the same configuration parameters:

```python
import httpx
import anthropic

client = anthropic.Anthropic(
    api_key="sk-ant-...",           # explicit key (prefer env var)
    max_retries=3,                  # auto-retry on transient errors (default: 2)
    timeout=60.0,                   # seconds (default: 600)
    default_headers={               # headers added to every request
        "X-App-Name": "my-app",
    },
    http_client=anthropic.DefaultHttpxClient(
        proxies="http://my-proxy:8080",
        limits=httpx.Limits(max_connections=50),
    ),
)
```

---

## Per-Request Overrides with `with_options()`

You can override client-level settings for a single request without creating a new client:

```python
# Disable retries for this one call
resp = client.messages.with_options(max_retries=0).create(
    model="claude-haiku-4-5",
    max_tokens=32,
    messages=[{"role": "user", "content": "Quick test"}],
)

# Use a longer timeout for a complex request
resp = client.messages.with_options(timeout=120.0).create(
    model="claude-opus-4-7",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Write a detailed research report on…"}],
)
```

---

## Connection Management

### Context manager (recommended)

```python
# Automatically closes HTTP connections when the block exits
with anthropic.Anthropic() as client:
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi!"}],
    )
    print(resp.content[0].text)
# Connections closed here
```

Async version:

```python
async with anthropic.AsyncAnthropic() as client:
    resp = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi!"}],
    )
    print(resp.content[0].text)
```

### Manual close

```python
client = anthropic.Anthropic()
try:
    # ... make requests ...
    pass
finally:
    client.close()    # or await client.aclose() for async
```

> **Tip for long-running services:** Create one client at startup and reuse it for the lifetime of the process. Creating a new client per request is wasteful and exhausts connections.

---

## FastAPI Integration Example

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import anthropic

# Shared client — created once at startup
claude_client: anthropic.AsyncAnthropic | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global claude_client
    claude_client = anthropic.AsyncAnthropic()
    yield
    await claude_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(message: str):
    resp = await claude_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": message}],
    )
    return {"reply": resp.content[0].text}
```

---

## Choosing Between Sync and Async

```
Your app uses asyncio / FastAPI / Starlette?
  └─ YES → AsyncAnthropic
     ├─ Single request at a time → await client.messages.create(...)
     └─ Multiple requests concurrently → asyncio.gather(...)

Your app is a script / CLI / Flask / Django?
  └─ NO → Anthropic (sync)
     ├─ Simple usage → client.messages.create(...)
     └─ Want concurrency without async → use Message Batches API (Chapter 5)
```

---

*← [Chapter 1: Core Concepts](01_core_concepts.md) | [Chapter 3: Streaming](03_streaming.md) →*
