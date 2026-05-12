# Chapter 3 — Streaming Responses

*← [Chapter 2: Sync & Async Clients](02_sync_async_clients.md) | [Chapter 4: Tool Use](04_tool_use.md) →*

---

## Why Stream?

Without streaming, your app waits for Claude to finish generating the **entire** response before receiving any bytes. For short outputs this is fine. For longer outputs:

- Network proxies may drop idle TCP connections (default timeout is often 60–90 s)
- Users see a blank screen until the full response arrives
- Risk of hitting SDK timeout (default 10 minutes)

With streaming, tokens are sent to your app as Claude generates them, enabling:
- **Real-time UI updates** (chat interfaces)
- **Long responses** without timeout risk
- **Lower perceived latency** — users see the first word immediately

---

## How SSE Streaming Works

Claude uses [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events). The SDK abstracts the raw HTTP layer and exposes two streaming APIs.

### Event flow for every stream

```
message_start
  └─ content_block_start  (block index 0)
       content_block_delta  (text chunk)
       content_block_delta  (text chunk)
       ...
       content_block_stop
  └─ content_block_start  (block index 1, if multiple blocks)
       ...
       content_block_stop
message_delta  (stop_reason, usage stats)
message_stop
```

Occasionally `ping` events are interspersed to keep the connection alive.

---

## Method 1 — High-Level Helper (Recommended)

`client.messages.stream()` is a **context manager** that gives you:

- `.text_stream` — async/sync iterator yielding text chunks
- `.get_final_message()` — the complete `Message` object after the stream ends
- `.get_final_text()` — the concatenated text string
- Automatic connection cleanup

### Sync

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=512,
    messages=[{"role": "user", "content": "Tell me a short story about a robot."}],
) as stream:
    # Print tokens as they arrive
    for chunk in stream.text_stream:
        print(chunk, end="", flush=True)

# After the context manager exits the full message is available
final_message = stream.get_final_message()
print(f"\n\nTokens used: {final_message.usage.output_tokens} output")
```

### Async

```python
import asyncio
import anthropic

async def main():
    client = anthropic.AsyncAnthropic()

    async with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": "Explain how photosynthesis works."}],
    ) as stream:
        async for chunk in stream.text_stream:
            print(chunk, end="", flush=True)

    final = stream.get_final_message()
    print(f"\n[{final.usage.output_tokens} output tokens]")

asyncio.run(main())
```

---

## Method 2 — Low-Level Iterator

`client.messages.create(stream=True)` returns an iterator over raw SSE events. This is **lower memory** — no message accumulation — and gives you full access to every event type.

### Sync

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "List three facts about the Moon."}],
    stream=True,
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)
        elif event.type == "message_stop":
            print("\n[stream ended]")
```

### Event Types Reference

| Event type | When it fires | Useful data |
|-----------|--------------|-------------|
| `message_start` | Once at start | Partial `Message` with empty content |
| `content_block_start` | Start of each content block | `content_block.type`, `index` |
| `content_block_delta` | Each token chunk | `delta.text` (text), `delta.partial_json` (tool input) |
| `content_block_stop` | End of each content block | `index` |
| `message_delta` | Near end of stream | `delta.stop_reason`, cumulative `usage` |
| `message_stop` | Stream complete | — |
| `ping` | Periodically | Keep-alive, ignore |

---

## Collecting the Full Text Without Printing

Use `.get_final_text()` when you need the complete string but not incremental display:

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Summarise the water cycle in 3 sentences."}],
) as stream:
    full_text = stream.get_final_text()   # blocks until stream ends

print(full_text)
```

> **Note:** The SDK requires `stream=True` (internally via `.stream()`) for requests expected to take longer than ~10 minutes. Setting a large `max_tokens` on a non-streaming request triggers a `ValueError`.

---

## Streaming with a System Prompt

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a Shakespearean poet. Always write in iambic pentameter.",
    messages=[{"role": "user", "content": "Describe a software bug."}],
) as stream:
    for chunk in stream.text_stream:
        print(chunk, end="", flush=True)
print()
```

---

## Streaming with Tool Use

When Claude calls a tool during a stream, you receive `content_block_delta` events with `delta.type == "input_json_delta"` containing **partial JSON** for the tool's input. Accumulate these and parse the JSON once you see `content_block_stop`.

```python
import json
import anthropic

client = anthropic.Anthropic()

TOOLS = [{
    "name": "get_weather",
    "description": "Get weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"],
    },
}]

tool_input_buffer = ""

with client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    tools=TOOLS,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    stream=True,
) as stream:
    for event in stream:
        if event.type == "content_block_start" and event.content_block.type == "tool_use":
            print(f"Tool call starting: {event.content_block.name}")
            tool_input_buffer = ""
        elif event.type == "content_block_delta":
            if hasattr(event.delta, "partial_json"):
                tool_input_buffer += event.delta.partial_json
            elif hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)
        elif event.type == "content_block_stop" and tool_input_buffer:
            tool_input = json.loads(tool_input_buffer)
            print(f"\nTool input parsed: {tool_input}")
            tool_input_buffer = ""
```

---

## Error Recovery

If a stream is interrupted by a network error, you can reconstruct the request with the partial response and continue.

### Claude 4.5 and earlier — prefix resumption

```python
import anthropic

client = anthropic.Anthropic()

def stream_with_recovery(prompt: str) -> str:
    """Stream a response, recovering from interruptions."""
    partial_text = ""
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(3):
        try:
            with client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=messages,
            ) as stream:
                for chunk in stream.text_stream:
                    partial_text += chunk
                    print(chunk, end="", flush=True)
            return partial_text  # success

        except Exception as e:
            print(f"\n[Error on attempt {attempt + 1}: {e}]")
            if partial_text and attempt < 2:
                # Reconstruct with the partial response as prefix
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": partial_text},
                ]
                print("[Resuming from partial response…]")
            else:
                raise

    return partial_text

result = stream_with_recovery("Tell me a long story about a brave knight.")
print(f"\n\nFull text length: {len(result)} characters")
```

### Claude 4.6+ — continuation message

For Claude 4.6 models, send a user message that instructs Claude to continue:

```python
messages = [
    {"role": "user", "content": original_prompt},
    {"role": "assistant", "content": partial_text},
    {"role": "user", "content": "Please continue from where you left off."},
]
```

---

## Raw Response Body Streaming

For cases where you need to process the raw HTTP response body (e.g., proxying to a browser), use `.with_streaming_response`:

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.with_streaming_response.create(
    model="claude-haiku-4-5",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
) as response:
    # Iterate raw bytes without parsing
    for line in response.iter_lines():
        print(line)
```

---

## Streaming in Production

### Checklist

- [ ] Always use a **context manager** (`with`/`async with`) — ensures the connection is closed even on exceptions
- [ ] Handle `content_block_delta` events **gracefully** — unknown delta types may appear in future models
- [ ] Set a reasonable **timeout** for the first token: `client.messages.with_options(timeout=30.0).stream(...)`
- [ ] **Accumulate tool input JSON** before parsing — `partial_json` deltas are fragments, not valid JSON
- [ ] For chat UIs: **flush** your output buffer after each chunk for real-time display

---

*← [Chapter 2: Sync & Async Clients](02_sync_async_clients.md) | [Chapter 4: Tool Use](04_tool_use.md) →*
