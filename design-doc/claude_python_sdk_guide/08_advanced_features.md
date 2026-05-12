# Chapter 8 — Advanced Features

*← [Chapter 7: Error Handling](07_error_handling.md) | [Chapter 9: Best Practices](09_best_practices.md) →*

---

## Token Counting

### Why count tokens?

- Predict cost before sending expensive requests
- Avoid hitting context window limits
- Decide whether to summarise history or truncate

### Pre-flight token count

```python
import anthropic

client = anthropic.Anthropic()

messages = [
    {"role": "user", "content": "Explain the theory of relativity in detail."}
]

count = client.messages.count_tokens(
    model="claude-haiku-4-5",
    messages=messages,
)
print(f"Estimated input tokens: {count.input_tokens}")
```

You can also count tokens for tool definitions and system prompts:

```python
count = client.messages.count_tokens(
    model="claude-haiku-4-5",
    system="You are an expert physicist.",
    tools=[{"name": "search_papers", "description": "...", "input_schema": {}}],
    messages=messages,
)
print(f"Total input tokens with tools: {count.input_tokens}")
```

### Post-request usage

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    messages=messages,
)
print(f"Input  tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Total tokens:  {response.usage.input_tokens + response.usage.output_tokens}")
```

---

## Paginated List Endpoints

Some APIs return pages of results. The SDK automatically handles pagination using a standard `for` loop:

### Models list (sync)

```python
import anthropic

client = anthropic.Anthropic()

# Automatically fetches subsequent pages
for model in client.models.list():
    print(f"  {model.id}  context={model.context_window}")
```

### Models list (async)

```python
import asyncio
import anthropic

async def list_models():
    client = anthropic.AsyncAnthropic()
    async for model in await client.models.list():
        print(f"  {model.id}")
    await client.aclose()

asyncio.run(list_models())
```

### Manual pagination control

```python
page = client.models.list()

while True:
    for model in page.data:
        print(model.id)
    if not page.has_next_page():
        break
    page = page.get_next_page()
```

---

## Type System

### Request parameters — TypedDicts

All request parameters are typed with TypedDicts, giving you autocompletion and type checking in VS Code / PyCharm.

Enable VS Code type checking:
```json
// .vscode/settings.json
{
    "python.analysis.typeCheckingMode": "basic"
}
```

```python
from anthropic.types import MessageParam, TextBlockParam

# The SDK accepts plain dicts — TypedDicts are optional but help editors
message: MessageParam = {
    "role": "user",
    "content": [TextBlockParam(type="text", text="Hello!")]
}
```

### Responses — Pydantic models

Claude responses are Pydantic v2 models:

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hi!"}],
)

# Convert to dict
as_dict = response.model_dump()

# Convert to JSON string (with indentation)
as_json = response.model_dump_json(indent=2)
print(as_json)
```

### Distinguishing null vs. absent fields

```python
import anthropic

response = client.messages.create(...)

# A field that wasn't included in the response
if response.stop_sequence is anthropic.NOT_GIVEN:
    print("stop_sequence: field absent")
elif response.stop_sequence is None:
    print("stop_sequence: explicitly null")
else:
    print(f"stop_sequence: {response.stop_sequence}")
```

---

## Raw HTTP Response Access

For cases where you need response headers, status codes, or binary body data:

### `.with_raw_response` — eagerly reads the full body

```python
import anthropic

client = anthropic.Anthropic()

raw = client.messages.with_raw_response.create(
    model="claude-haiku-4-5",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hello!"}],
)

print("HTTP status :", raw.http_response.status_code)        # 200
print("request-id  :", raw.http_response.headers.get("request-id"))
print("content-type:", raw.http_response.headers.get("content-type"))

# Parse the body into the typed Message model
message = raw.parse()
print("Text:", message.content[0].text)
```

### `.with_streaming_response` — lazy body reading

```python
with client.messages.with_streaming_response.create(
    model="claude-haiku-4-5",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hello!"}],
) as resp:
    # Body is NOT yet read — the connection is still open
    body_text = resp.text()    # read now as string
    # or: resp.json()          # parse as JSON dict
    # or: resp.read()          # raw bytes
    # or: resp.iter_lines()    # line-by-line iterator
```

> **Important:** Always use a context manager with `.with_streaming_response` to guarantee the connection is closed.

---

## Logging

The SDK uses Python's standard `logging` module under the logger name `"anthropic"`.

### Via environment variable

```bash
export ANTHROPIC_LOG=debug   # shows full request/response
export ANTHROPIC_LOG=info    # basic request info
export ANTHROPIC_LOG=warn    # warnings only
export ANTHROPIC_LOG=off     # silence the SDK logger
```

### Via Python code

```python
import logging

# Enable debug logging for the SDK
logging.getLogger("anthropic").setLevel(logging.DEBUG)

# Format log output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
```

Debug logs show:
- Full request URL, headers (API key masked), body
- Full response headers and body
- Retry attempts with reasons

---

## Custom & Undocumented Endpoints

### Calling undocumented endpoints

```python
# Raw HTTP verbs — retries and auth headers are still applied
response = client.get("/v1/models")
response = client.post("/v1/some/future/endpoint", body={"key": "value"})
```

### Extra parameters on documented calls

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hello!"}],
    # Append extra query params to the URL
    extra_query={"debug": "1"},
    # Merge extra fields into the JSON body
    extra_body={"experimental_feature": True},
    # Append extra HTTP headers
    extra_headers={"X-App-Name": "my-production-app"},
)
```

> **Security note:** Only pass `extra_body` / `extra_query` data from trusted sources. These bypass the SDK's type validation.

### Accessing undocumented response fields

```python
response = client.messages.create(...)

# Access an undocumented field directly
value = response.unknown_prop

# Get all extra fields as a dict
extra_fields = response.model_extra
```

---

## HTTP Client Customisation

### Custom proxy

```python
import anthropic

client = anthropic.Anthropic(
    http_client=anthropic.DefaultHttpxClient(
        proxies="http://my-proxy.company.com:8080",
    )
)
```

### Custom connection limits

```python
import httpx
import anthropic

client = anthropic.Anthropic(
    http_client=anthropic.DefaultHttpxClient(
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
    )
)
```

### Mocking for tests

```python
import httpx
from unittest.mock import MagicMock
import anthropic

# Use httpx's built-in mock transport for unit tests
mock_response = httpx.Response(
    200,
    json={
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
)

transport = httpx.MockTransport(lambda request: mock_response)
client = anthropic.Anthropic(
    http_client=anthropic.DefaultHttpxClient(transport=transport)
)
```

---

## Custom Headers and API Versions

The SDK automatically sends `anthropic-version: 2023-06-01`. You can override headers per client or per request:

```python
client = anthropic.Anthropic(
    default_headers={
        "anthropic-version": "2023-06-01",
        "X-Custom-Trace-ID": "abc123",
    }
)
```

> **Warning:** Overriding `anthropic-version` may break type safety and response parsing.

---

## Beta Features

Access beta features via the `beta` property and the `betas` parameter:

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}],
    betas=["output-300k-2026-03-24"],   # opt in to a beta feature
)
```

For the Files API, pass the header at client level:

```python
client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "files-api-2025-04-14"}
)
```

---

## Platform Integrations

All four platform clients share the same interface as the base `Anthropic` client:

```python
# AWS Bedrock (new projects)
from anthropic import AnthropicBedrockMantle

client = AnthropicBedrockMantle()   # uses AWS credentials automatically
response = client.messages.create(
    model="anthropic.claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello from Bedrock!"}],
)

# AWS Bedrock (legacy InvokeModel path)
from anthropic import AnthropicBedrock
client = AnthropicBedrock()

# Google Vertex AI
from anthropic import AnthropicVertex

client = AnthropicVertex(region="us-east5", project_id="my-project")
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello from Vertex!"}],
)

# Microsoft Foundry / Azure
from anthropic import AnthropicFoundry
client = AnthropicFoundry()
```

### Install extras

```bash
pip install "anthropic[bedrock]"    # AWS
pip install "anthropic[vertex]"     # Google
```

---

## Checking the SDK Version at Runtime

```python
import anthropic

version = anthropic.__version__
print(f"anthropic SDK version: {version}")
```

Use this to confirm which version is active in your environment if you're troubleshooting missing features.

---

*← [Chapter 7: Error Handling](07_error_handling.md) | [Chapter 9: Best Practices](09_best_practices.md) →*
