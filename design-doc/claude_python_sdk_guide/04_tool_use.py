"""
Lesson 04 — Tool Use (Function Calling)
========================================
Topics covered:
  • Defining tools with JSON Schema
  • The agentic loop — how Claude signals tool calls
  • Executing tools and returning results
  • Forcing tool use with tool_choice
  • Multi-tool orchestration
  • Tool helpers / @beta_tool decorator (SDK helper)
"""

import json
import anthropic

client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Mock "external" functions that tools will call
# ---------------------------------------------------------------------------
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather lookup."""
    data = {
        "paris": {"temp": 18, "condition": "partly cloudy"},
        "tokyo": {"temp": 25, "condition": "sunny"},
        "london": {"temp": 12, "condition": "rainy"},
    }
    weather = data.get(location.lower(), {"temp": 20, "condition": "unknown"})
    return {"location": location, "unit": unit, **weather}


def calculator(expression: str) -> dict:
    """Safely evaluate a simple math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Retrieve current weather conditions for a given city. "
            "Use this whenever the user asks about weather, temperature, "
            "or climate in a specific location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris' or 'Tokyo'."
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Defaults to celsius.",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Use standard Python operators: +, -, *, /, **, //, %. "
            "Do NOT use for non-math questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '(3 + 4) * 2'."
                }
            },
            "required": ["expression"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher — maps tool names to Python functions
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculator": calculator,
}


def dispatch_tool(tool_use_block) -> str:
    """Execute the requested tool and return the result as a JSON string."""
    fn = TOOL_REGISTRY.get(tool_use_block.name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_use_block.name}"})
    result = fn(**tool_use_block.input)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# 1. Single-step tool use (manual agentic loop)
# ---------------------------------------------------------------------------
def single_tool_call(user_message: str):
    """
    Agentic loop:
      1. Send user message + tool definitions to Claude.
      2. If Claude responds with stop_reason='tool_use', execute the tools.
      3. Send tool results back to Claude.
      4. Repeat until stop_reason='end_turn'.
    """
    messages = [{"role": "user", "content": user_message}]

    print(f"=== Tool Use ===\nUser: {user_message}")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Append Claude's response to the conversation
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # No more tool calls — extract and print final text
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"Claude: {block.text}")
            break

        if response.stop_reason == "tool_use":
            # Collect all tool results into one user message
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [Tool call: {block.name}({block.input})]")
                    result = dispatch_tool(block)
                    print(f"  [Tool result: {result}]")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    print()


# ---------------------------------------------------------------------------
# 2. Forcing a specific tool with tool_choice
# ---------------------------------------------------------------------------
def forced_tool_use():
    """
    tool_choice options:
      {"type": "auto"}   — Claude decides (default)
      {"type": "any"}    — Claude must use at least one tool
      {"type": "tool", "name": "calculator"}  — force a specific tool
      {"type": "none"}   — no tools
    """
    print("=== Forced Tool Use (calculator) ===")

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        tools=TOOLS,
        tool_choice={"type": "tool", "name": "calculator"},
        messages=[{"role": "user", "content": "What is the square root of 144?"}],
    )

    for block in response.content:
        if block.type == "tool_use":
            print(f"Tool: {block.name}, Input: {block.input}")
            print("Result:", dispatch_tool(block))
    print()


# ---------------------------------------------------------------------------
# 3. Multi-turn with multiple tool calls in one response
# ---------------------------------------------------------------------------
def multi_tool_demo():
    """Claude can request several tools in a single response."""
    single_tool_call(
        "What is the weather in Tokyo and in London? Also, what is 42 * 13?"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    single_tool_call("What's the weather like in Paris?")
    single_tool_call("Calculate (100 / 4) + 7 ** 2")
    forced_tool_use()
    multi_tool_demo()
