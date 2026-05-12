# Chapter 4 — Tool Use (Function Calling)

*← [Chapter 3: Streaming](03_streaming.md) | [Chapter 5: Message Batches](05_batch_processing.md) →*

---

## What Is Tool Use?

Tool use (also called **function calling**) lets Claude invoke functions that you define. Claude doesn't execute the code — it generates a structured call and your application runs the function. The result flows back to Claude so it can continue reasoning.

This pattern bridges language understanding with real-world data:

- Fetch live weather or stock prices
- Query a database
- Call a REST API
- Execute code
- Search a knowledge base

---

## The Agentic Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                        │
│                                                             │
│  1. Build messages + tool definitions                       │
│  2. POST /v1/messages  ──────────────────────────────────►  │
│                                          Claude API         │
│  3. Response: stop_reason = "tool_use"  ◄────────────────── │
│     └─ ToolUseBlock { name, input }                         │
│                                                             │
│  4. Execute the tool in YOUR code                           │
│  5. Append tool_result to messages                          │
│  6. POST /v1/messages again  ────────────────────────────►  │
│                                          Claude API         │
│  7. Response: stop_reason = "end_turn"  ◄────────────────── │
│     └─ Final text answer                                    │
└─────────────────────────────────────────────────────────────┘
```

The loop repeats until `stop_reason == "end_turn"` (Claude finished) or you decide to stop.

---

## Defining a Tool

Each tool definition contains three required fields:

```python
tool = {
    "name": "get_weather",                # must match ^[a-zA-Z0-9_-]{1,64}$
    "description": (
        "Get the current weather for a given city. "
        "Use this when the user asks about weather, temperature, or climate. "
        "Returns temperature and a short condition description."
    ),
    "input_schema": {                      # JSON Schema object
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'Paris' or 'New York'."
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit. Defaults to celsius."
            }
        },
        "required": ["location"],          # 'unit' is optional
    },
}
```

### Description best practices

| Do | Don't |
|----|-------|
| Explain **when** to use the tool | Leave the description vague |
| Describe what each parameter means | Skip explaining optional parameters |
| Note what the tool **does not** return | Assume Claude will guess |
| Write 3–4 sentences minimum | Use one line |

---

## Minimal Working Example

```python
import json
import anthropic

client = anthropic.Anthropic()

# --- Tool definition ---
TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Get current weather for a city. "
            "Returns temperature in celsius and a condition string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name."},
            },
            "required": ["location"],
        },
    }
]

# --- Mock implementation ---
def get_weather(location: str) -> dict:
    data = {
        "paris": {"temp": 18, "condition": "Partly cloudy"},
        "tokyo": {"temp": 28, "condition": "Sunny"},
    }
    return data.get(location.lower(), {"temp": 20, "condition": "Unknown"})

# --- Agentic loop ---
def run(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Always append Claude's full response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract and return final text
            return next(
                block.text for block in response.content
                if hasattr(block, "text")
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → Tool call: {block.name}({block.input})")
                    if block.name == "get_weather":
                        result = get_weather(**block.input)
                    else:
                        result = {"error": f"Unknown tool: {block.name}"}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "user", "content": tool_results})

answer = run("What's the weather like in Paris right now?")
print(f"\nAnswer: {answer}")
```

---

## Multiple Tools

Pass multiple tool definitions in the `tools` list. Claude chooses the right one(s) based on descriptions:

```python
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression. "
            "Use this for any arithmetic, percentages, or unit conversions. "
            "Input must be a valid Python expression."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]
```

Claude may call **multiple tools in a single response** (in parallel). Always iterate `response.content` and handle every `tool_use` block before sending results back.

---

## Parallel Tool Calls

When Claude needs data from multiple tools simultaneously, it returns them all in one response:

```python
# Example: "What's the weather in Paris and Tokyo, and what is 100 * 1.2?"
# Claude may return:
response.content = [
    TextBlock(text="I'll look those up for you…"),
    ToolUseBlock(id="tu_1", name="get_weather", input={"location": "Paris"}),
    ToolUseBlock(id="tu_2", name="get_weather", input={"location": "Tokyo"}),
    ToolUseBlock(id="tu_3", name="calculator", input={"expression": "100 * 1.2"}),
]
```

Collect **all** results before sending back:

```python
tool_results = []
for block in response.content:
    if block.type == "tool_use":
        result = dispatch(block)   # your dispatcher
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result),
        })
messages.append({"role": "user", "content": tool_results})
```

---

## Controlling Tool Use with `tool_choice`

| `tool_choice` | Behaviour |
|--------------|-----------|
| `{"type": "auto"}` | Claude decides whether to call a tool (default) |
| `{"type": "any"}` | Claude must call at least one tool |
| `{"type": "tool", "name": "calculator"}` | Claude must call the named tool |
| `{"type": "none"}` | No tools; ignore all definitions |

```python
# Force Claude to use the calculator
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    tools=TOOLS,
    tool_choice={"type": "tool", "name": "calculator"},
    messages=[{"role": "user", "content": "What is the square root of 169?"}],
)
```

> **Note:** `tool_choice: any` and `tool_choice: tool` are **not** compatible with extended thinking. Use `auto` or `none` when extended thinking is enabled.

---

## Strict Tool Use

Add `"strict": true` to a tool definition to guarantee that Claude's tool calls **always conform to the schema**. Useful when your dispatcher does no validation:

```python
{
    "name": "create_user",
    "description": "Create a new user account.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["username", "email"],
        "additionalProperties": False,
    },
}
```

---

## Returning Tool Errors

If a tool fails, return an error message in the `tool_result` content. Claude will read the error and decide how to proceed:

```python
try:
    result = my_tool(**block.input)
    content = json.dumps(result)
    is_error = False
except Exception as e:
    content = f"Error: {str(e)}"
    is_error = True

tool_results.append({
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": content,
    "is_error": is_error,   # tells Claude the tool failed
})
```

---

## Tool Use with Streaming

Tool input arrives as `partial_json` deltas. Accumulate them and parse after `content_block_stop`:

```python
import json
import anthropic

client = anthropic.Anthropic()
current_tool_id = None
current_tool_name = None
json_buffer = ""

with client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    tools=TOOLS,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    stream=True,
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                current_tool_id = block.id
                current_tool_name = block.name
                json_buffer = ""
        elif event.type == "content_block_delta":
            if hasattr(event.delta, "partial_json"):
                json_buffer += event.delta.partial_json
            elif hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)
        elif event.type == "content_block_stop" and json_buffer:
            tool_input = json.loads(json_buffer)
            print(f"\nTool {current_tool_name} called with: {tool_input}")
            json_buffer = ""
```

---

## Best Practices Summary

| Practice | Why |
|----------|-----|
| Write rich, detailed descriptions | Descriptions are Claude's primary guide for tool selection |
| Group related operations into one tool with an `action` param | Fewer tools = less ambiguity |
| Return only the data Claude needs | Bloated responses waste context and tokens |
| Use `strict: true` for production tools | Eliminates validation logic in your dispatcher |
| Always handle all `tool_use` blocks in a response | Claude may call multiple tools in one turn |
| Set a loop limit (e.g. `max_iterations = 10`) | Prevent infinite loops in agentic pipelines |

---

*← [Chapter 3: Streaming](03_streaming.md) | [Chapter 5: Message Batches](05_batch_processing.md) →*
