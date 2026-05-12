"""
PHASE 1 — STEP 2: Structured Outputs
======================================
Harness Pillar: ACI (Agent-Computer Interface) — Structured JSON I/O
SDK Docs: https://docs.anthropic.com/en/build-with-claude/structured-outputs

Goal: Force Claude to always reply in strictly validated JSON schemas using
      Pydantic. A Harness Engineer never parses free text — they contract
      the model to emit machine-readable structured data.

SDK Features Used:
  - client.messages.parse()      ← SDK helper that wraps structured outputs
  - Pydantic BaseModel            ← schema definition + validation
  - output_config.format          ← JSON Schema constrained decoding
  - strict=True on tools          ← guaranteed tool input conformance
"""

import json
from typing import Literal
import anthropic
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# 2a. Basic Pydantic structured output
#     Use case: extract structured data from prose
# ─────────────────────────────────────────────
print("=" * 60)
print("2a. Basic structured extraction")
print("=" * 60)

class Company(BaseModel):
    name: str
    founded_year: int
    headquarters: str
    main_product: str

response = client.messages.parse(
    model="claude-opus-4-5-20251101",
    max_tokens=256,
    output_format=Company,                  # ← SDK converts Pydantic → JSON Schema
    messages=[{
        "role": "user",
        "content": "Tell me about Anthropic, the AI safety company."
    }],
)

company: Company = response.content[0].parsed
print(f"Name      : {company.name}")
print(f"Founded   : {company.founded_year}")
print(f"HQ        : {company.headquarters}")
print(f"Product   : {company.main_product}")
assert isinstance(company.founded_year, int), "Year must be int — schema enforced!"


# ─────────────────────────────────────────────
# 2b. Nested schema — agent tool call result
#     Use case: parse multi-field analysis report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2b. Nested structured schema")
print("=" * 60)

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)

class TextAnalysis(BaseModel):
    summary: str = Field(description="One sentence summary")
    sentiment: Sentiment
    key_topics: list[str] = Field(max_length=5)
    word_count: int

TEXT = """
Claude is an AI assistant made by Anthropic. It is designed to be helpful,
harmless, and honest. Many developers use it to build production AI agents,
and it has a rich Python SDK with support for streaming, tool use, and
structured outputs. The community finds it very capable for complex tasks.
"""

response = client.messages.parse(
    model="claude-opus-4-5-20251101",
    max_tokens=512,
    output_format=TextAnalysis,
    messages=[{"role": "user", "content": f"Analyze this text:\n{TEXT}"}],
)

analysis: TextAnalysis = response.content[0].parsed
print(f"Summary    : {analysis.summary}")
print(f"Sentiment  : {analysis.sentiment.label} ({analysis.sentiment.confidence:.0%})")
print(f"Topics     : {analysis.key_topics}")
print(f"Word count : {analysis.word_count}")


# ─────────────────────────────────────────────
# 2c. Strict tool use — guaranteed schema input
#     Use case: agent calls a tool and inputs MUST match schema exactly
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2c. Strict tool use (schema-guaranteed inputs)")
print("=" * 60)

class WeatherQuery(BaseModel):
    city: str
    country_code: str = Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 country code")
    units: Literal["celsius", "fahrenheit"] = "celsius"

tools = [
    {
        "name": "get_weather",
        "description": "Fetch current weather for a location. Use when the user asks about weather.",
        "input_schema": WeatherQuery.model_json_schema(),
        "strict": True,                     # ← Claude MUST match this schema exactly
    }
]

response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=256,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather like in London?"}],
)

if response.stop_reason == "tool_use":
    tool_call = next(b for b in response.content if b.type == "tool_use")
    # Validate the input against our Pydantic model
    weather_input = WeatherQuery(**tool_call.input)
    print(f"Tool called : {tool_call.name}")
    print(f"City        : {weather_input.city}")
    print(f"Country     : {weather_input.country_code}")
    print(f"Units       : {weather_input.units}")
    print(f"Raw JSON    : {json.dumps(tool_call.input, indent=2)}")
    print("\n✅ Schema validation passed! No free-text parsing needed.")


# ─────────────────────────────────────────────
# 2d. Decision schema — agent decision-making
#     Use case: instead of parsing "yes" or "no" from prose,
#               contract the agent to emit a typed Decision object
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2d. Decision schema (no regex parsing!)")
print("=" * 60)

class AgentDecision(BaseModel):
    action: Literal["approve", "reject", "escalate", "need_more_info"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    next_step: str

SCENARIO = """
A user wants to withdraw $50,000 from their savings account.
Their account balance is $52,000. Their account is 2 years old.
They have no history of fraud. This is their first large withdrawal.
"""

response = client.messages.parse(
    model="claude-opus-4-5-20251101",
    max_tokens=256,
    system="You are a bank fraud detection agent. Evaluate transactions and output decisions.",
    output_format=AgentDecision,
    messages=[{"role": "user", "content": f"Evaluate this transaction:\n{SCENARIO}"}],
)

decision: AgentDecision = response.content[0].parsed
print(f"Action      : {decision.action}")
print(f"Confidence  : {decision.confidence:.0%}")
print(f"Reason      : {decision.reason}")
print(f"Next step   : {decision.next_step}")
print("\n✅ Zero ambiguity — decision is a typed enum, not a string to parse!")


# ─────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. NEVER parse free text with regex — define a Pydantic schema instead.
2. client.messages.parse() + output_format=MyModel is the cleanest pattern.
3. strict=True on tools guarantees tool inputs match your schema exactly.
4. Use Literal["a", "b"] for constrained choices (like enums).
5. The SDK automatically removes unsupported constraints from Pydantic 
   (like ge/le) and adds them to field descriptions — Claude still sees 
   the intent, and the SDK validates the response for you.
6. Structured outputs = the foundation of reliable agentic pipelines.
""")
