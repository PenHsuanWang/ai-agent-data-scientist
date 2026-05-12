"""
PHASE 1 — STEP 1: API Mastery
==============================
Harness Pillar: Cognitive Framework
SDK Docs: https://docs.anthropic.com/en/api/sdks/python

Goal: Understand how to deterministically control Claude's output using
      API parameters. A Harness Engineer must own the model's behaviour,
      not just prompt it and hope.

Key parameters:
  - model         : which version of Claude to use
  - max_tokens    : hard cap on output length
  - temperature   : 0=deterministic, 1=creative (default ~1)
  - stop_sequences: force the model to stop at a token
  - system        : the Cognitive Framework / agent worldview
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# 1a. Basic call — observe the raw structure
# ─────────────────────────────────────────────
print("=" * 60)
print("1a. Basic call")
print("=" * 60)

response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is 17 × 24?"}],
)
print(f"Stop reason : {response.stop_reason}")
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens:{response.usage.output_tokens}")
print(f"Answer      : {response.content[0].text}")


# ─────────────────────────────────────────────
# 1b. Temperature = 0 (deterministic output)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("1b. Temperature=0 (deterministic)")
print("=" * 60)

for _ in range(3):
    r = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=64,
        temperature=0,                     # ← fully deterministic
        messages=[{"role": "user", "content": "Give me one word for 'happy'."}],
    )
    print(r.content[0].text.strip())


# ─────────────────────────────────────────────
# 1c. Stop sequences — force early termination
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("1c. Stop sequence")
print("=" * 60)

r = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=256,
    stop_sequences=["STOP"],              # model must stop here
    messages=[{
        "role": "user",
        "content": "Count from 1 to 10. After each number write 'STOP' on a new line."
    }],
)
print(r.content[0].text)
print(f"\n→ Stop reason: {r.stop_reason}")  # should be "stop_sequence"


# ─────────────────────────────────────────────
# 1d. System prompt — define the Cognitive Framework
#     (the agent's worldview and behavioural boundary)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("1d. System prompt (Cognitive Framework)")
print("=" * 60)

SYSTEM = """You are a financial data analyst assistant.
Your rules:
1. ONLY answer questions about financial data, markets, and analysis.
2. ALWAYS cite your reasoning step-by-step before giving a number.
3. When you don't have live data, say so explicitly — never invent figures.
4. Format all monetary values as: $X,XXX.XX
"""

r = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=256,
    system=SYSTEM,
    messages=[{"role": "user", "content": "What's the PE ratio of Apple?"}],
)
print(r.content[0].text)


# ─────────────────────────────────────────────
# 1e. Streaming — essential for long agentic turns
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("1e. Streaming response")
print("=" * 60)

with client.messages.stream(
    model="claude-opus-4-5-20251101",
    max_tokens=256,
    messages=[{"role": "user", "content": "List 5 Python best practices, one sentence each."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()
    print(f"\n\n→ Total tokens: {final.usage.input_tokens + final.usage.output_tokens}")


# ─────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Always set temperature=0 for deterministic agent actions.
2. Use stop_sequences to force the model to produce parseable output.
3. The system prompt IS your agent's Cognitive Framework —
   define its identity, boundaries, and output style here.
4. Check stop_reason after every call:
   - "end_turn"      → normal completion
   - "max_tokens"    → output was cut off — increase max_tokens!
   - "stop_sequence" → hit your stop token
   - "tool_use"      → model wants to call a tool (Phase 2)
5. Always log usage tokens — they directly control your costs.
""")
