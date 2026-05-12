# Model Reference — Choosing the Right Model for Data Science

## 1. Current Model Lineup (as of May 2026)

### Full Specification Table

| Spec | claude-opus-4-7 | claude-sonnet-4-6 | claude-haiku-4-5-20251001 |
|------|:---:|:---:|:---:|
| **Tier** | Most capable | Best balance | Fastest |
| **Input price** | $5.00 / MTok | $3.00 / MTok | $1.00 / MTok |
| **Output price** | $25.00 / MTok | $15.00 / MTok | $5.00 / MTok |
| **Cache write** | $6.25 / MTok | $3.75 / MTok | $1.25 / MTok |
| **Cache read** | $0.50 / MTok | $0.30 / MTok | $0.10 / MTok |
| **Context window** | 1M tokens | 1M tokens | 200K tokens |
| **Max output tokens** | 128K | 64K | 64K |
| **Knowledge cutoff** | Jan 2026 | (earlier) | (earlier) |
| **Input modalities** | Text, Image | Text, Image | Text, Image |
| **Vision** | ✅ | ✅ | ✅ |
| **Multilingual** | ✅ | ✅ | ✅ |
| **Extended thinking** | ✅ Adaptive | ✅ Extended + Adaptive | ❌ |
| **Code execution** | ✅ | ✅ | ✅ (basic) |
| **Streaming** | ✅ | ✅ | ✅ |
| **Batch API** | ✅ | ✅ | ✅ |
| **Prompt caching** | ✅ (60 min TTL) | ✅ (60 min TTL) | ✅ (5 min TTL) |
| **Programmatic calling** | ✅ | ✅ | ❌ |

### Model ID Strings

```python
# Use these exact strings in the `model` parameter:
"claude-opus-4-7"              # Latest Opus 4.7 snapshot
"claude-sonnet-4-6"            # Latest Sonnet 4.6 snapshot
"claude-haiku-4-5-20251001"    # Haiku 4.5 (pinned October 2025 snapshot)
```

> **Note on model IDs**: Opus and Sonnet IDs without a date suffix resolve to the latest snapshot. Haiku 4.5 uses a pinned snapshot ID. Use pinned snapshots in production for reproducibility (see Section 9).

---

## 2. Model Selection for Data Science

### Decision Guide

Use this flowchart to pick the right model:

```
Is this a production workload where cost matters?
├── YES: Is the analysis complex (multi-step reasoning, novel methods)?
│   ├── YES: Is extreme accuracy worth 5x the cost?
│   │   ├── YES → claude-opus-4-7
│   │   └── NO  → claude-sonnet-4-6
│   └── NO (routine aggregations, standard EDA):
│       ├── Latency-critical? → claude-haiku-4-5-20251001
│       └── Otherwise        → claude-sonnet-4-6
└── NO (prototyping, one-off analysis):
    └── Use claude-sonnet-4-6 (best all-round)
```

### By Task Type

| Task | Recommended Model | Reason |
|------|:-----------------:|--------|
| Exploratory data analysis | `claude-sonnet-4-6` | Good reasoning + fast enough |
| Simple statistics & SQL | `claude-haiku-4-5-20251001` | Fast, cheap, sufficient |
| Complex ML model design | `claude-opus-4-7` | Best reasoning for novel approaches |
| Real-time dashboards | `claude-haiku-4-5-20251001` | Lowest latency |
| Batch report generation | `claude-haiku-4-5-20251001` | 50% batch discount + cheapest |
| Research & literature review | `claude-opus-4-7` | Latest knowledge cutoff (Jan 2026) |
| Causal inference | `claude-opus-4-7` + extended thinking | Complex reasoning needed |
| Unit validation (many calls) | `claude-haiku-4-5-20251001` | High volume, simple task |
| Anomaly explanation | `claude-sonnet-4-6` | Balanced for explanations |
| Long-context code analysis | `claude-sonnet-4-6` | 1M context, good code understanding |

### Configuration Presets

```python
# Fast & cheap — for high-volume, simple tasks
FAST_CONFIG = {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 1024,
    "temperature": 0.0,
}

# Balanced — for most data analysis work
BALANCED_CONFIG = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 8192,
    "temperature": 0.2,
}

# Powerful — for complex, one-off analysis
POWERFUL_CONFIG = {
    "model": "claude-opus-4-7",
    "max_tokens": 16384,
    "temperature": 0.5,
    "thinking": {"type": "enabled", "budget_tokens": 8000},
}
```

---

## 3. Context Window Strategy

### The 1-Million-Token Context

Claude Opus 4.7 and Sonnet 4.6 have a **1-million-token context window** — approximately 750,000 words or ~3,000 pages of text. Haiku 4.5 has 200K tokens.

**What fits in 1M tokens:**
- An entire codebase (medium-sized project)
- Thousands of rows of data as CSV text
- Years of log files
- Complete research papers (dozens)
- Full conversation history of a long analysis session

### Practical Limits vs. Theoretical Limits

Despite the 1M token context, there are practical reasons to keep context shorter:

| Issue | Threshold | Mitigation |
|-------|-----------|------------|
| Latency increases | ~100K tokens | Use streaming |
| Cost increases linearly | — | Use prompt caching |
| Attention dilution | >200K tokens | Summarise older turns |
| Rate limits per minute | Varies | Use batch API |

### Context Management for Long Sessions

```python
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def summarise_if_needed(
    client,
    messages: list[dict],
    max_tokens: int = 100_000,
    summary_model: str = "claude-haiku-4-5-20251001"
) -> list[dict]:
    """
    If conversation is getting long, summarise older turns.
    Keeps the most recent 10 messages intact.
    """
    total_chars = sum(
        len(str(m.get("content", ""))) for m in messages
    )
    estimated_tokens = total_chars // 4

    if estimated_tokens <= max_tokens:
        return messages

    # Keep system-adjacent messages + recent turns
    keep_recent = 10
    to_summarise = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    summary_prompt = (
        "Summarise the following conversation history in a concise but complete way. "
        "Preserve all key findings, data statistics, decisions made, and tool results. "
        "Format as bullet points grouped by topic.\n\n"
        + "\n".join(f"{m['role']}: {str(m.get('content',''))[:500]}" for m in to_summarise)
    )

    summary_response = client.messages.create(
        model=summary_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    summary_text = summary_response.content[0].text

    # Replace old turns with summary
    summary_message = {
        "role": "user",
        "content": f"[Summary of earlier conversation]\n{summary_text}"
    }
    summary_ack = {
        "role": "assistant",
        "content": "Understood. I have reviewed the conversation summary and will continue from here."
    }

    return [summary_message, summary_ack] + recent
```

---

## 4. Extended Thinking

Extended thinking gives Claude extra "compute time" to reason through complex problems before answering. The model generates a thinking block (visible to you) before the final response.

### When to Enable It

Enable extended thinking for:
- Multi-step statistical derivations
- Causal inference and confounding analysis
- Novel ML architecture design
- Complex data pipeline debugging
- Research synthesis across many sources

Do **not** enable it for:
- Simple aggregations or statistics
- Retrieval tasks (what is in this CSV?)
- Real-time / latency-sensitive endpoints
- Tasks with `tool_choice: "any"` or `tool_choice: "tool"` (incompatible)

### Usage

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",    # Or claude-opus-4-7
    max_tokens=16000,              # Must be > budget_tokens
    messages=[{
        "role": "user",
        "content": (
            "I have a dataset of 10,000 patients with 50 features. "
            "I want to predict 30-day readmission risk. "
            "Design a full ML pipeline including feature engineering, "
            "model selection rationale, validation strategy to avoid data leakage, "
            "and how to handle the class imbalance (5% positive rate)."
        )
    }],
    thinking={
        "type": "enabled",
        "budget_tokens": 8000    # How much thinking to allow; more = better but slower
    }
)

# Response content:
for block in response.content:
    if block.type == "thinking":
        print("=== Claude's reasoning ===")
        print(block.thinking[:500], "...")  # May be very long
    elif block.type == "text":
        print("=== Final answer ===")
        print(block.text)
```

### Adaptive Thinking

Claude Opus 4.7 and Sonnet 4.6 also support **adaptive thinking**, where Claude decides autonomously how much thinking to apply:

```python
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=8192,
    messages=[...],
    thinking={
        "type": "enabled",
        "budget_tokens": 5000,
        "adaptive": True    # Claude allocates budget based on task complexity
    }
)
```

### `budget_tokens` Guidance

| Task complexity | Recommended `budget_tokens` |
|----------------|:--------------------------:|
| Moderate (standard analysis) | 2,000–5,000 |
| High (novel approaches) | 5,000–15,000 |
| Very high (research-level) | 15,000–30,000 |
| Maximum quality | 32,000+ |

> `max_tokens` must always be greater than `budget_tokens` + expected response length.

---

## 5. Cost Optimisation

### Prompt Caching

For repeated analysis sessions with the same system prompt, prompt caching provides a 90% discount on cached tokens.

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a senior data scientist specialising in retail analytics.
You have access to the company's data warehouse through the provided tools.
Always compute exact values using tools rather than estimating.
Present findings with:
1. A one-sentence headline insight
2. Supporting statistics with exact numbers
3. The code that produced the results
4. Business implications in plain language

Dataset schema:
- sales_transactions(date, product_id, region, units, revenue, cost)
- products(product_id, category, brand, launch_date)
- customers(customer_id, segment, lifetime_value, acquisition_channel)
- campaigns(campaign_id, start_date, end_date, budget, channel)
"""  # This is ~1200 tokens — meets the 1024-token minimum for caching

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}  # Cache this block
        }
    ],
    messages=[{"role": "user", "content": "What were the top 5 products last quarter?"}],
    betas=["prompt-caching-2024-07-31"]
)

# First call: cache_creation_input_tokens = ~300, input_tokens = ~300
# Subsequent calls (within 60 min): cache_read_input_tokens = ~300
# Savings: 90% on the system prompt tokens for every subsequent call
print(f"Cache created: {getattr(response.usage, 'cache_creation_input_tokens', 0)}")
print(f"Cache read:    {getattr(response.usage, 'cache_read_input_tokens', 0)}")
```

**Minimum cacheable sizes:**
- Sonnet and Opus: **1,024 tokens** minimum
- Haiku: **2,048 tokens** minimum

**Cache TTL:**
- Haiku: 5 minutes
- Sonnet, Opus: 60 minutes

### Batch API for Offline Analysis

When latency is not critical (nightly reports, bulk analysis):

```python
import anthropic
import json

client = anthropic.Anthropic()

# Prepare 100 analysis requests
requests = []
for product_id in product_ids[:100]:
    requests.append({
        "custom_id": f"analysis-{product_id}",
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": f"Summarise the sales performance of product {product_id} last month."
            }]
        }
    })

# Submit batch
batch = client.messages.batches.create(requests=requests)
print(f"Batch ID: {batch.id}")
print(f"Status: {batch.processing_status}")

# Poll for completion (results within 24 hours, often much faster)
import time

while True:
    batch = client.messages.batches.retrieve(batch.id)
    if batch.processing_status == "ended":
        break
    print(f"Processing... {batch.request_counts.processing} remaining")
    time.sleep(60)

# Collect results
results = {}
for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        results[result.custom_id] = result.result.message.content[0].text
    else:
        print(f"Failed: {result.custom_id}: {result.result.error}")

# Savings: 50% vs real-time API + haiku pricing = ~70% cheaper than opus real-time
```

---

## 6. Model Capabilities Matrix

Full feature support by model:

| Feature | claude-opus-4-7 | claude-sonnet-4-6 | claude-haiku-4-5-20251001 |
|---------|:---:|:---:|:---:|
| Text generation | ✅ | ✅ | ✅ |
| Vision / image input | ✅ | ✅ | ✅ |
| Tool use (client) | ✅ | ✅ | ✅ |
| Parallel tool calls | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ |
| Async | ✅ | ✅ | ✅ |
| Extended thinking | ✅ | ✅ | ❌ |
| Adaptive thinking | ✅ | ✅ | ❌ |
| Code execution (basic) | ✅ | ✅ | ✅ |
| Programmatic tool calling | ✅ | ✅ | ❌ |
| web_search server tool | ✅ | ✅ | ✅ |
| web_fetch server tool | ✅ | ✅ | ✅ |
| Files API | ✅ | ✅ | ✅ |
| Prompt caching | ✅ | ✅ | ✅ |
| Batch API | ✅ | ✅ | ✅ |
| 1M context window | ✅ | ✅ | ❌ (200K) |
| 128K max output | ✅ | ❌ (64K) | ❌ (64K) |
| Jan 2026 knowledge | ✅ | ❌ | ❌ |
| Bedrock | ✅ | ✅ | ✅ |
| Vertex AI | ✅ | ✅ | ✅ |
| Azure AI | ✅ | ✅ | ✅ |

---

## 7. Migrating Between Models

### How to Change the Model

```python
# In your agent settings or config file:
AGENT_MODEL = "claude-sonnet-4-6"  # Change this one line

# Or use an environment variable:
import os
AGENT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Then use in every request:
response = client.messages.create(
    model=AGENT_MODEL,
    ...
)
```

### Testing Impact on Analysis Quality

When migrating models, run a regression test on your benchmark queries:

```python
import anthropic
import json

client = anthropic.Anthropic()

BENCHMARK_QUERIES = [
    {
        "id": "statistics-basic",
        "query": "Calculate mean, median, and std dev of [23, 45, 12, 67, 34, 89, 56, 78, 11, 45]",
        "expected_mean": 46.0,
    },
    {
        "id": "trend-detection",
        "query": "Is this time series showing an upward trend? Values by month: [100, 105, 108, 115, 120, 130]",
        "expected_contains": "upward",
    },
    {
        "id": "outlier-detection",
        "query": "Identify outliers in: [10, 11, 10.5, 10.8, 11.2, 150, 10.3, 11.1]",
        "expected_contains": "150",
    }
]

def run_benchmark(model: str) -> dict:
    results = {"model": model, "passed": 0, "failed": 0, "costs": []}

    for test in BENCHMARK_QUERIES:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": test["query"]}]
        )
        answer = response.content[0].text.lower()
        cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
        results["costs"].append(cost)

        if "expected_mean" in test:
            passed = str(test["expected_mean"]) in answer or "46" in answer
        elif "expected_contains" in test:
            passed = test["expected_contains"].lower() in answer

        results["passed" if passed else "failed"] += 1

    results["avg_cost"] = sum(results["costs"]) / len(results["costs"])
    return results

# Compare models
for model in ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"]:
    result = run_benchmark(model)
    print(f"\n{result['model']}:")
    print(f"  Passed: {result['passed']}/{len(BENCHMARK_QUERIES)}")
    print(f"  Avg cost: ${result['avg_cost']:.6f}")
```

---

## 8. Recommended Configuration

Best settings for the Data Scientist Agent:

```python
# settings.py — recommended for the data-scientist-agent

from dataclasses import dataclass

@dataclass
class AgentSettings:
    # Model selection
    default_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"
    powerful_model: str = "claude-opus-4-7"

    # Token limits
    default_max_tokens: int = 8192
    max_output_tokens: int = 64000      # Sonnet/Haiku max
    thinking_budget: int = 5000         # For complex analyses

    # Client settings
    max_retries: int = 3
    timeout_seconds: float = 120.0

    # Caching
    use_prompt_caching: bool = True
    cache_system_prompt: bool = True    # Always cache long system prompts

    # Tools
    enable_code_execution: bool = True
    enable_web_search: bool = False     # Enable only when needed
    enable_files_api: bool = True

    # Behaviour
    temperature: float = 0.2           # Low for reproducible analysis
    max_agentic_iterations: int = 15   # Safety cap on tool loops

    # Cost controls
    max_cost_per_request: float = 1.0  # USD — alert if exceeded
    use_batch_for_bulk: bool = True    # Use batch API for >10 requests


SETTINGS = AgentSettings()

# System prompt for data scientist agent
DATA_SCIENTIST_SYSTEM_PROMPT = """You are a senior data scientist with deep expertise in:
- Statistical analysis and hypothesis testing
- Machine learning (sklearn, interpretability, validation)
- Data pipeline design and optimisation
- Data quality assessment and cleaning
- Time-series analysis and forecasting
- Experimental design and A/B testing

Working principles:
1. Always use available tools to compute exact values — never guess statistics
2. Show your reasoning and the code that produced results
3. Flag data quality issues when you encounter them
4. Explain findings in business terms after technical analysis
5. When uncertain, say so and quantify the uncertainty

Output format for analyses:
- Headline: one-sentence key finding
- Statistics: exact numbers with appropriate precision
- Code: the pandas/numpy/sklearn code used
- Interpretation: business implications
- Caveats: data quality issues or statistical limitations
"""
```

---

## 9. API Versioning

### How Model IDs Work

Anthropic uses two types of model IDs:

**1. Pinned snapshot IDs** (recommended for production):
```python
"claude-haiku-4-5-20251001"   # Pinned to October 2025 release — never changes
```

**2. Alias IDs** (resolve to latest snapshot):
```python
"claude-opus-4-7"             # May resolve to different snapshot over time
"claude-sonnet-4-6"           # May resolve to different snapshot over time
```

### Response Model Field

The `response.model` field always returns the **actual pinned snapshot** that was used:

```python
response = client.messages.create(model="claude-sonnet-4-6", ...)
print(response.model)  # e.g., "claude-sonnet-4-6-20251015"
```

Log this value in production to know exactly which snapshot ran.

### Deprecation Policy

1. Anthropic announces deprecation in advance (typically 6–12 months)
2. Deprecated models continue to work until the end-of-life date
3. Alias IDs (`claude-sonnet-4-6`) are updated to newer snapshots — if you're using aliases and notice behaviour changes, pin to a specific snapshot

### Pinning in Production

```python
# Production: pin to specific snapshot for reproducibility
PRODUCTION_MODEL = "claude-sonnet-4-6-20251015"   # From response.model

# Development: use alias to get latest
DEV_MODEL = "claude-sonnet-4-6"

import os
MODEL = PRODUCTION_MODEL if os.environ.get("ENV") == "production" else DEV_MODEL
```

### API Version Header

The Anthropic API also has an API version (separate from model version):

```python
# The SDK sets this automatically — current: "2023-06-01"
# You don't need to set this manually
```
