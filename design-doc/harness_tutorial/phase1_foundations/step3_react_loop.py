"""
PHASE 1 — STEP 3: The ReAct Loop (Reason + Act)
=================================================
Harness Pillar: Standard Workflows & Refinement Loops
SDK Docs: https://docs.anthropic.com/en/agents-and-tools/tool-use/how-tool-use-works

Goal: Build the ReAct (Reason+Act) agentic loop from scratch — the core
      pattern of every AI agent. No frameworks. Just the raw SDK + a while loop.

The ReAct Loop:
  1. Claude THINKS (reasoning — visible via Extended Thinking)
  2. Claude ACTS (tool_use block → your code executes)
  3. Claude OBSERVES (tool_result → Claude reads the output)
  4. Repeat until stop_reason == "end_turn"

This is Ring 1 → Ring 4 from the SDK tutorial, explained in Harness terms.
"""

import math
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# Tool definitions — the ACI (Agent-Computer Interface)
# These are the "hands" the agent can use.
# ─────────────────────────────────────────────
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Perform mathematical calculations. "
            "Accepts any valid Python math expression (e.g. '2**10', 'math.sqrt(144)'). "
            "Use this instead of computing in your head. "
            "Returns the numeric result as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python math expression to evaluate, e.g. '2 + 2' or 'math.sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "unit_converter",
        "description": (
            "Convert between units of measurement. "
            "Supported: temperature (celsius/fahrenheit/kelvin), "
            "length (km/miles/feet/meters), weight (kg/lbs/oz). "
            "Returns the converted value with units."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g. 'celsius', 'km', 'kg')"},
                "to_unit": {"type": "string", "description": "Target unit (e.g. 'fahrenheit', 'miles', 'lbs')"}
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    },
    {
        "name": "get_fact",
        "description": (
            "Look up a factual piece of information from the knowledge base. "
            "Use this for constants, country data, or well-known facts you're not sure about. "
            "Do NOT use for real-time data like stock prices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic to look up (e.g. 'speed of light', 'population of Japan')"}
            },
            "required": ["topic"]
        }
    },
]


# ─────────────────────────────────────────────
# Tool implementations — YOUR code runs here
# (In a real harness: DB calls, API calls, file ops)
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "speed of light": "299,792,458 meters per second",
    "population of japan": "approximately 124 million (2024)",
    "boiling point of water": "100°C (212°F) at sea level",
    "pi": "3.14159265358979323846",
    "avogadro's number": "6.022 × 10²³ mol⁻¹",
}

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"{result}"
    except Exception as e:
        return f"ERROR: {e}"

def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
        ("celsius", "kelvin"):     lambda v: v + 273.15,
        ("km", "miles"):           lambda v: v * 0.621371,
        ("miles", "km"):           lambda v: v * 1.60934,
        ("kg", "lbs"):             lambda v: v * 2.20462,
        ("lbs", "kg"):             lambda v: v / 2.20462,
        ("meters", "feet"):        lambda v: v * 3.28084,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"ERROR: Conversion from {from_unit} to {to_unit} not supported"

def get_fact(topic: str) -> str:
    return KNOWLEDGE_BASE.get(topic.lower(), f"No data found for '{topic}'")

def dispatch_tool(name: str, inputs: dict) -> str:
    """Route tool calls to their implementations."""
    if name == "calculator":
        return calculator(inputs["expression"])
    elif name == "unit_converter":
        return unit_converter(inputs["value"], inputs["from_unit"], inputs["to_unit"])
    elif name == "get_fact":
        return get_fact(inputs["topic"])
    return f"ERROR: Unknown tool '{name}'"


# ─────────────────────────────────────────────
# THE REACT LOOP — The heart of every AI agent
# ─────────────────────────────────────────────
def run_agent(user_query: str, verbose: bool = True) -> str:
    """
    Runs the full ReAct agentic loop.
    
    Loop anatomy:
      THINK  → Claude reasons (extended thinking, if enabled)
      ACT    → Claude emits tool_use block (stop_reason="tool_use")
      RUN    → We execute the tool
      OBSERVE → We send tool_result back
      REPEAT until stop_reason == "end_turn"
    """
    messages = [{"role": "user", "content": user_query}]
    iteration = 0
    MAX_ITERATIONS = 10  # safety guard — prevent infinite loops

    while iteration < MAX_ITERATIONS:
        iteration += 1
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            tools=TOOLS,
            system=(
                "You are a precise calculation agent. "
                "Always use the provided tools — never compute in your head. "
                "Show your reasoning step by step."
            ),
            messages=messages,
        )

        if verbose:
            print(f"Stop reason: {response.stop_reason}")
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"Claude text: {block.text[:200]}")
                elif block.type == "tool_use":
                    print(f"Tool call : {block.name}({json.dumps(block.input)})")

        # Append Claude's full response to conversation history
        messages.append({"role": "assistant", "content": response.content})

        # ── OBSERVE: Claude is done ─────────────────
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return final_text

        # ── ACT: Claude wants tools ─────────────────
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                # RUN the tool
                result = dispatch_tool(block.name, block.input)
                if verbose:
                    print(f"Tool result: {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            # Feed all results back in one user message
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason — exit safely
            break

    return "Agent reached iteration limit without finishing."


# ─────────────────────────────────────────────
# Demo: Multi-step reasoning
# ─────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "If a car travels 150 km at 60 km/h, how long does the trip take in hours and minutes?",
        "Convert the boiling point of water from Celsius to Fahrenheit, then tell me how many degrees above room temperature (22°C) that is.",
        "What is the square root of Avogadro's number? Show your steps.",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        answer = run_agent(query, verbose=True)
        print(f"\nFINAL ANSWER:\n{answer}")


# ─────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. The ReAct loop is a while loop keyed on stop_reason:
     "tool_use"  → execute tools, append results, loop
     "end_turn"  → Claude is done, extract the text answer

2. ALWAYS append the assistant's full content to messages[]:
     messages.append({"role": "assistant", "content": response.content})
   This is what gives Claude its "memory" within a session.

3. Tool results go back as a "user" message with tool_result blocks.
   The tool_use_id must match the id from the tool_use block.

4. Always cap iterations (MAX_ITERATIONS) to prevent runaway loops.

5. dispatch_tool() is the ACI dispatch layer — it routes tool names
   to your actual implementations. Keep this clean and testable.

6. This loop IS the agent. Everything else (memory, multi-agent,
   evaluation) is built around this core pattern.
""")
