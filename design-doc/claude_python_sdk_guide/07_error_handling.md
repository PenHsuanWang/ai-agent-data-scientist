# Chapter 7 — Error Handling & Resilience

*← [Chapter 6: Files API](06_files_api.md) | [Chapter 8: Advanced Features](08_advanced_features.md) →*

---

## Error Hierarchy

All SDK errors inherit from `anthropic.APIError`. Understanding the hierarchy helps you write precise error handlers.

```
anthropic.APIError                    ← base class
├── anthropic.APIConnectionError      ← network failure, no HTTP status
├── anthropic.APITimeoutError         ← request timed out
└── anthropic.APIStatusError          ← HTTP 4xx / 5xx received
    ├── anthropic.BadRequestError              (400)
    ├── anthropic.AuthenticationError          (401)
    ├── anthropic.PermissionDeniedError        (403)
    ├── anthropic.NotFoundError                (404)
    ├── anthropic.UnprocessableEntityError     (422)
    ├── anthropic.RateLimitError               (429)
    └── anthropic.InternalServerError          (>=500)
```

### HTTP status code reference

| Status | Exception | Typical cause |
|--------|-----------|---------------|
| 400 | `BadRequestError` | Invalid parameters, malformed request |
| 401 | `AuthenticationError` | Wrong or missing API key |
| 403 | `PermissionDeniedError` | API key doesn't have access to this resource |
| 404 | `NotFoundError` | Model or resource doesn't exist |
| 422 | `UnprocessableEntityError` | Request is valid JSON but semantically wrong |
| 429 | `RateLimitError` | Too many requests; slow down |
| ≥500 | `InternalServerError` | Anthropic server error; retry |
| N/A | `APIConnectionError` | DNS failure, connection refused, etc. |
| N/A | `APITimeoutError` | Request exceeded timeout |

---

## Basic Error Handling

```python
import anthropic

client = anthropic.Anthropic()

try:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.content[0].text)

except anthropic.AuthenticationError:
    print("❌ Invalid API key. Check ANTHROPIC_API_KEY.")

except anthropic.RateLimitError as e:
    print(f"⏳ Rate limited. Request ID: {e.request_id}")
    # Implement back-off (see below)

except anthropic.InternalServerError as e:
    print(f"🔴 Anthropic server error ({e.status_code}). Request ID: {e.request_id}")
    # Retry after a delay

except anthropic.APIConnectionError:
    print("🌐 Network error. Check your internet connection.")

except anthropic.APITimeoutError:
    print("⌛ Request timed out. Consider using streaming for long responses.")

except anthropic.APIStatusError as e:
    # Catch-all for any other 4xx / 5xx
    print(f"API error {e.status_code}: {e.message}")
```

---

## The `request_id` — Your Debugging Lifeline

Every response and every error carries a `request_id` from the response headers. **Always log this value.** When contacting Anthropic support, the `request_id` lets them find the exact request in their logs.

```python
# From a successful response
response = client.messages.create(...)
print(f"request_id: {response._request_id}")

# From an exception
try:
    client.messages.create(...)
except anthropic.APIStatusError as e:
    print(f"Failed! request_id: {e.request_id}")
    print(f"Status: {e.status_code}")
    print(f"Message: {e.message}")
```

> **Note:** `_request_id` uses an underscore prefix on response objects, but it is **public** — the underscore is a convention, not a signal that it's private.

---

## Automatic Retries

The SDK automatically retries requests that encounter transient errors:

| Retried automatically | Not retried |
|-----------------------|-------------|
| `APIConnectionError` | `AuthenticationError` (401) |
| 408 Request Timeout | `BadRequestError` (400) |
| 409 Conflict | `PermissionDeniedError` (403) |
| `RateLimitError` (429) | `UnprocessableEntityError` (422) |
| `InternalServerError` (≥500) | — |

**Default behaviour:** 2 retries with exponential backoff.

### Configuring retries

```python
# Client-level: all requests get 5 attempts (1 initial + 4 retries)
resilient_client = anthropic.Anthropic(max_retries=4)

# Client-level: disable retries entirely
no_retry_client = anthropic.Anthropic(max_retries=0)

# Per-request override
response = client.messages.with_options(max_retries=0).create(
    model="claude-haiku-4-5",
    max_tokens=32,
    messages=[{"role": "user", "content": "Quick test."}],
)
```

---

## Timeouts

**Default timeout: 600 seconds (10 minutes)**

Timeouts raise `anthropic.APITimeoutError` and are automatically retried.

### Setting timeouts

```python
import httpx
import anthropic

# Simple float — total time for the entire request
client = anthropic.Anthropic(timeout=30.0)

# Granular control with httpx.Timeout
client = anthropic.Anthropic(
    timeout=httpx.Timeout(
        connect=5.0,    # TCP handshake
        read=60.0,      # waiting for response data
        write=10.0,     # sending request body
        pool=5.0,       # waiting for a connection from the pool
    )
)

# Per-request
response = client.messages.with_options(timeout=120.0).create(
    model="claude-opus-4-7",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Write a detailed technical document…"}],
)
```

> **Rule of thumb:** For requests with large `max_tokens`, always use **streaming** instead of increasing the timeout. Streaming keeps the connection active and avoids proxy timeouts.

---

## Manual Exponential Backoff

While the SDK's automatic retries cover most cases, you may need custom logic — for example, different retry counts per error type, alerting, or circuit-breaking.

```python
import time
import random
import anthropic

client = anthropic.Anthropic(max_retries=0)  # disable auto-retry for manual control

def create_with_backoff(max_attempts: int = 5, **kwargs):
    """
    Retry with exponential backoff + jitter.
    Only retries on rate limits and server errors.
    """
    base_delay = 1.0

    for attempt in range(max_attempts):
        try:
            return client.messages.create(**kwargs)

        except anthropic.RateLimitError as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited (attempt {attempt + 1}). "
                  f"Retrying in {delay:.1f}s… [request_id={e.request_id}]")
            time.sleep(delay)

        except anthropic.InternalServerError as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Server error {e.status_code} (attempt {attempt + 1}). "
                  f"Retrying in {delay:.1f}s…")
            time.sleep(delay)

        except (anthropic.AuthenticationError, anthropic.BadRequestError):
            raise   # never retry these


response = create_with_backoff(
    max_attempts=4,
    model="claude-haiku-4-5",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)
```

---

## Rate Limits

Anthropic enforces rate limits measured in:
- **RPM** — requests per minute
- **TPM** — tokens per minute (input + output)

Your limits increase automatically as you progress through usage tiers. View your current limits at **platform.claude.com → Settings → Limits**.

### Handling 429s gracefully

```python
import anthropic
import time

client = anthropic.Anthropic()

def safe_create(**kwargs):
    """Wrapper that respects rate limit headers."""
    try:
        return client.messages.create(**kwargs)
    except anthropic.RateLimitError:
        # Anthropic's retry-after header gives the optimal wait time
        # The SDK may expose this in future; for now use a fixed delay
        print("Rate limited. Waiting 60 seconds…")
        time.sleep(60)
        return client.messages.create(**kwargs)
```

---

## Handling Long Requests

For responses expected to take a long time (large `max_tokens`), **always use streaming**:

```python
import anthropic

client = anthropic.Anthropic()

# ❌ This raises ValueError if expected latency > ~10 minutes
# response = client.messages.create(model="claude-opus-4-7", max_tokens=100000, ...)

# ✅ Use streaming for long generations
with client.messages.stream(
    model="claude-opus-4-7",
    max_tokens=100_000,
    messages=[{"role": "user", "content": "Write a comprehensive guide…"}],
) as stream:
    for chunk in stream.text_stream:
        print(chunk, end="", flush=True)
```

---

## Structured Error Logging

A production-ready error logging pattern:

```python
import logging
import anthropic

logger = logging.getLogger(__name__)

def call_claude(messages: list, model: str = "claude-haiku-4-5", **kwargs):
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=kwargs.pop("max_tokens", 512),
            messages=messages,
            **kwargs,
        )
        logger.info(
            "Claude API success",
            extra={
                "request_id": response._request_id,
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )
        return response

    except anthropic.APIStatusError as e:
        logger.error(
            "Claude API error",
            extra={
                "request_id": e.request_id,
                "status_code": e.status_code,
                "error_type": type(e).__name__,
                "message": e.message,
            }
        )
        raise

    except anthropic.APIConnectionError as e:
        logger.error("Claude network error", extra={"error": str(e)})
        raise
```

---

## Summary Checklist

- [ ] Catch `AuthenticationError` separately — it means your key is wrong, not a transient issue
- [ ] Always log `request_id` / `e.request_id` for every error
- [ ] Use SDK auto-retry (`max_retries=2+`) for production workloads
- [ ] Use streaming for any request with `max_tokens > 4096` to avoid proxy timeouts
- [ ] Never retry `400` or `401` errors — they indicate bugs in your code or config
- [ ] For `429` errors, add jitter to your backoff so parallel workers don't all retry simultaneously

---

*← [Chapter 6: Files API](06_files_api.md) | [Chapter 8: Advanced Features](08_advanced_features.md) →*
