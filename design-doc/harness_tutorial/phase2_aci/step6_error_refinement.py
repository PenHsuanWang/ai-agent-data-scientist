"""
PHASE 2 — STEP 6: Error-Handling Refinement Loop
==================================================
Harness Pillar: Standard Workflows & Refinement Loops
SDK Docs: https://docs.anthropic.com/en/agents-and-tools/tool-use/handle-tool-calls

Goal: Implement the self-correction refinement loop. When a tool fails,
      feed the stack trace or error back into the model so it can fix
      its own mistakes autonomously.

This is the difference between a fragile script and a resilient agent.
  - is_error=True  → signals to Claude that the tool failed
  - Stack trace    → Claude reads it and corrects its next call
  - Max retries    → prevent infinite correction loops
"""

import json
import subprocess
import sys
import traceback
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# Simulated tools that can fail realistically
# ─────────────────────────────────────────────
def run_python_code(code: str) -> tuple[str, bool]:
    """
    Execute Python code in a subprocess and return (output, is_error).
    Returns the full stderr on failure — this is fed back to Claude.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return result.stderr.strip(), True
        return result.stdout.strip() or "(no output)", False
    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out after 10 seconds.", True
    except Exception as e:
        return f"ERROR: {traceback.format_exc()}", True


def call_fake_api(endpoint: str, params: dict) -> tuple[str, bool]:
    """
    Simulate an external API call that can fail with various errors.
    This demonstrates the self-correction loop for API integration.
    """
    VALID_ENDPOINTS = {
        "/users": {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]},
        "/products": {"products": [{"id": "P1", "name": "Widget", "price": 9.99}]},
        "/orders": {"orders": []},
    }

    # Simulate common API errors
    if not endpoint.startswith("/"):
        return json.dumps({
            "error": "InvalidEndpoint",
            "message": f"Endpoint must start with '/'. Got: '{endpoint}'",
            "valid_endpoints": list(VALID_ENDPOINTS.keys()),
        }), True

    if endpoint not in VALID_ENDPOINTS:
        return json.dumps({
            "error": "NotFound",
            "message": f"Endpoint '{endpoint}' does not exist.",
            "valid_endpoints": list(VALID_ENDPOINTS.keys()),
        }), True

    if "limit" in params and not isinstance(params["limit"], int):
        return json.dumps({
            "error": "ValidationError",
            "message": f"'limit' must be an integer, got '{type(params['limit']).__name__}'",
        }), True

    return json.dumps(VALID_ENDPOINTS[endpoint]), False


# ─────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────
TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code and return the output. "
            "Use for calculations, data processing, and analysis. "
            "If the code fails, you will receive the full error traceback — "
            "read it, identify the bug, and fix your code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Must print() results to stdout."
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "call_api",
        "description": (
            "Call the internal REST API. "
            "Valid endpoints: /users, /products, /orders. "
            "Returns JSON. If you get an error, read the 'valid_endpoints' list in the response "
            "and correct your endpoint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint, must start with '/' (e.g. '/users')"
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters as a JSON object"
                }
            },
            "required": ["endpoint"],
        },
    },
]


# ─────────────────────────────────────────────
# THE REFINEMENT LOOP
# Key difference from Step 3: is_error=True
# Claude reads errors and self-corrects
# ─────────────────────────────────────────────
def run_self_correcting_agent(task: str, verbose: bool = True) -> str:
    """
    Agentic loop with error feedback — the refinement loop.

    When a tool fails:
      1. Set is_error=True in tool_result
      2. Include the full error message
      3. Claude reads it and retries with corrections
    """
    messages = [{"role": "user", "content": task}]
    iteration = 0
    MAX_ITERATIONS = 15  # higher than usual to allow for self-correction attempts

    tool_call_counts: dict[str, int] = {}  # track retry attempts per tool call

    while iteration < MAX_ITERATIONS:
        iteration += 1

        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2048,
            tools=TOOLS,
            system=(
                "You are a resilient coding and data agent. "
                "When a tool returns an error, carefully read the error message "
                "and fix your approach — do not give up on the first failure. "
                "You have up to 3 correction attempts per sub-task. "
                "Always explain what went wrong and how you're fixing it."
            ),
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if verbose:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    print(f"\n[Claude]: {block.text[:300]}")
                elif block.type == "tool_use":
                    print(f"\n[Tool Call]: {block.name}({json.dumps(block.input)[:200]})")

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                # Dispatch tool and capture success/failure
                is_error = False
                if block.name == "run_python":
                    result, is_error = run_python_code(block.input["code"])
                elif block.name == "call_api":
                    result, is_error = call_fake_api(
                        block.input["endpoint"],
                        block.input.get("params", {})
                    )
                else:
                    result, is_error = f"Unknown tool: {block.name}", True

                if verbose:
                    status = "❌ ERROR" if is_error else "✅ OK"
                    print(f"[Result {status}]: {result[:300]}")

                # The critical difference: is_error=True tells Claude it failed
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,   # ← THIS is what enables self-correction
                })

            messages.append({"role": "user", "content": tool_results})

    return "Agent reached maximum iterations."


# ─────────────────────────────────────────────
# Demo: Tasks that require self-correction
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tasks = [
        # This will fail first (NameError) then self-correct:
        "Write Python code to calculate the first 10 Fibonacci numbers using the 'fib' function, then print them.",

        # This will hit the wrong endpoint first, then self-correct:
        "Fetch the list of products from the API and count how many there are.",

        # This requires code + API together:
        "Get the user list from the API, then write Python code to find the user with the longest name.",
    ]

    for task in tasks:
        print("\n" + "=" * 60)
        print(f"TASK: {task}")
        print("=" * 60)
        answer = run_self_correcting_agent(task, verbose=True)
        print(f"\n✅ FINAL ANSWER:\n{answer}")


print("""
KEY TAKEAWAYS
=============
1. is_error=True in tool_result is the SIGNAL that activates self-correction.
   Without it, Claude assumes the tool succeeded even if the output is an error string.

2. Include the FULL error/stack trace in the tool_result content.
   Claude is trained to read Python tracebacks and correct its code.

3. Set MAX_ITERATIONS higher for self-correcting agents (10-15 vs 5-8 for simple agents).

4. The refinement loop is what separates a "demo" agent from a production agent.
   Real environments have errors — the harness must handle them gracefully.

5. Track retries per sub-task to avoid infinite correction loops on unsolvable problems.
""")
