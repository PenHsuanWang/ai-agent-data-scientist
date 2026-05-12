# Chapter 1 — Core Concepts: Messages API

*← [Chapter 0: Introduction](00_introduction.md) | [Chapter 2: Sync & Async Clients](02_sync_async_clients.md) →*

---

## The Messages API

The **Messages API** is the primary interface for interacting with Claude. Every request is structured as a conversation composed of **messages** — alternating between `user` and `assistant` roles.

```
POST https://api.anthropic.com/v1/messages
```

The Python SDK wraps this endpoint in `client.messages.create(...)`.

---

## Roles

| Role | Who sends it | When |
|------|-------------|------|
| `user` | Your application | Questions, instructions, tool results |
| `assistant` | Claude | Responses, reasoning, tool calls |

> **Rule:** Messages must strictly alternate roles. The first message must be `user`. The SDK will raise an error if you violate this order.

---

## Content Blocks

Both user and assistant messages use **content blocks** — typed objects that represent different kinds of content.

### Text block

```python
{"type": "text", "text": "Hello, Claude!"}
```

### Image block (vision)

```python
{
    "type": "image",
    "source": {
        "type": "base64",          # or "url" or "file" (Files API)
        "media_type": "image/jpeg",
        "data": "<base64-encoded-bytes>",
    }
}
```

### Document block (PDF / plain text)

```python
{
    "type": "document",
    "source": {
        "type": "base64",          # or "url" or "file" (Files API)
        "media_type": "application/pdf",
        "data": "<base64-encoded-bytes>",
    }
}
```

### Tool-use block (assistant → your code)

Claude emits this when it wants to call a function:

```python
{
    "type": "tool_use",
    "id": "toolu_01A09q90qw90lq917835lq9",
    "name": "get_weather",
    "input": {"location": "Paris", "unit": "celsius"}
}
```

### Tool-result block (your code → Claude)

Your code sends this back with the function's output:

```python
{
    "type": "tool_result",
    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
    "content": '{"temp": 18, "condition": "cloudy"}'
}
```

---

## Shorthand vs. Structured Content

For simple text-only messages you can pass a plain string instead of a list:

```python
# Shorthand (text only)
messages=[{"role": "user", "content": "What is 2 + 2?"}]

# Structured (supports multiple blocks, images, documents, …)
messages=[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image", "source": {"type": "url", "url": "https://…"}},
        ]
    }
]
```

---

## System Prompt

The **system prompt** sets Claude's persona, constraints, and instructions. It is separate from the `messages` list.

```python
message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    system="You are a helpful Python tutor. Always explain concepts with code examples.",
    messages=[{"role": "user", "content": "What is a list comprehension?"}],
)
```

The system prompt can also be a list of content blocks (useful for caching long system prompts — see Chapter 9):

```python
system=[
    {
        "type": "text",
        "text": "You are a helpful Python tutor…",
        "cache_control": {"type": "ephemeral"},
    }
]
```

---

## Multi-Turn Conversations

Claude has no built-in memory across API calls. To maintain a conversation, you must accumulate the full message history yourself and send it with each request.

```python
import anthropic

client = anthropic.Anthropic()

def run_conversation():
    history = []

    def chat(user_input: str) -> str:
        # 1. Append the new user message
        history.append({"role": "user", "content": user_input})

        # 2. Send the full history to Claude
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=history,
        )

        # 3. Extract the assistant reply
        assistant_reply = response.content[0].text

        # 4. Append Claude's reply to history for next turn
        history.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    print(chat("My name is Alice."))
    print(chat("What is my name?"))          # Claude remembers "Alice"
    print(chat("What did I tell you first?"))

run_conversation()
```

---

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model ID (e.g. `"claude-haiku-4-5"`) |
| `max_tokens` | `int` | Maximum tokens to generate (**required**) |
| `messages` | `list` | Conversation history (user/assistant turns) |
| `system` | `str \| list` | System prompt (optional) |
| `temperature` | `float` | Randomness 0.0–1.0 (default varies by model) |
| `top_p` | `float` | Nucleus sampling threshold |
| `top_k` | `int` | Top-k sampling |
| `stop_sequences` | `list[str]` | Stop generation when any string is encountered |
| `tools` | `list` | Tool definitions for function calling |
| `tool_choice` | `dict` | Controls whether/which tool Claude must use |
| `stream` | `bool` | Enable SSE streaming |
| `metadata` | `dict` | `{"user_id": "…"}` for abuse monitoring |

---

## Stop Reasons

| `stop_reason` | Meaning |
|--------------|---------|
| `"end_turn"` | Claude finished naturally |
| `"max_tokens"` | Hit the `max_tokens` limit — response may be truncated |
| `"tool_use"` | Claude wants to call one or more tools |
| `"stop_sequence"` | One of your custom `stop_sequences` was generated |

Always check `stop_reason` when your code depends on the response being complete:

```python
if response.stop_reason == "max_tokens":
    print("Warning: response was truncated. Increase max_tokens.")
```

---

## Token Budgeting

Tokens are the unit of billing and rate-limiting for Claude models.

```
1 token ≈ 0.75 English words ≈ 4 characters
```

**What consumes tokens:**
- Your system prompt (input)
- All messages in the conversation history (input)
- Tool definitions (input)
- Claude's response (output)

**Practical tips:**
- Use `client.messages.count_tokens(...)` before sending expensive requests
- Shorter history = fewer tokens = lower cost. Summarise old turns when appropriate
- `max_tokens` sets the *ceiling*, not a guaranteed length

```python
# Count tokens before sending
count = client.messages.count_tokens(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "Explain quantum physics."}],
)
print(f"Estimated input tokens: {count.input_tokens}")
```

---

## Complete Example

```python
import anthropic

client = anthropic.Anthropic()

# Single-turn with all common parameters
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    temperature=0.7,
    system="You are a creative writer. Respond with vivid, engaging prose.",
    messages=[
        {
            "role": "user",
            "content": "Write a two-sentence description of a sunset over the ocean."
        }
    ],
    stop_sequences=["###"],   # stop if Claude generates "###"
)

print(response.content[0].text)
print(f"\nTokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
print(f"Stop reason: {response.stop_reason}")
```

---

*← [Chapter 0: Introduction](00_introduction.md) | [Chapter 2: Sync & Async Clients](02_sync_async_clients.md) →*
