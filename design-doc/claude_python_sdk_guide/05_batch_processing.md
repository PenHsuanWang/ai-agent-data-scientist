# Chapter 5 — Message Batches API

*← [Chapter 4: Tool Use](04_tool_use.md) | [Chapter 6: Files API](06_files_api.md) →*

---

## What Is Batch Processing?

Instead of sending requests one at a time and waiting for each response, the **Message Batches API** lets you submit up to **100,000 requests in a single batch** for asynchronous processing.

### When to use batches vs. real-time

| Scenario | Use |
|----------|-----|
| User-facing chat, interactive app | Real-time Messages API |
| Bulk data analysis, nightly jobs | **Message Batches API** |
| Large-scale evaluations (evals) | **Message Batches API** |
| Content moderation pipelines | **Message Batches API** |
| Generating product descriptions at scale | **Message Batches API** |

### Benefits

- **50 % cost reduction** — batch pricing is half the standard per-token price
- **Higher throughput** — many requests processed in parallel by Anthropic
- **All Messages API features** — vision, tool use, system prompts, multi-turn all work

---

## Pricing

| Model | Batch Input | Batch Output |
|-------|------------|--------------|
| Claude Opus 4.7 | $2.50 / MTok | $12.50 / MTok |
| Claude Sonnet 4.6 | $1.50 / MTok | $7.50 / MTok |
| Claude Haiku 4.5 | $0.50 / MTok | $2.50 / MTok |

*(Standard prices are 2× — batches save 50 %)*

---

## Batch Limits

| Limit | Value |
|-------|-------|
| Requests per batch | 100,000 |
| Batch size | 256 MB |
| `custom_id` format | `^[a-zA-Z0-9_-]{1,64}$` |
| Processing time limit | 24 hours |
| Result retention | 29 days after creation |
| `min max_tokens` per request | 1 (0 is not supported) |

---

## Step-by-Step Walkthrough

### Step 1 — Create a batch

```python
import anthropic

client = anthropic.Anthropic()

# Build the list of requests
batch_requests = [
    {
        "custom_id": f"classify-{i}",      # unique ID for this request
        "params": {
            "model": "claude-haiku-4-5",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Classify the sentiment of this review as "
                        f"POSITIVE, NEGATIVE, or NEUTRAL:\n\n{review}"
                    ),
                }
            ],
        },
    }
    for i, review in enumerate([
        "Absolutely love this product! Five stars.",
        "Terrible experience. Never buying again.",
        "It's okay. Does what it says.",
    ])
]

# Submit the batch
batch = client.messages.batches.create(requests=batch_requests)
print(f"Batch created: {batch.id}")
print(f"Status: {batch.processing_status}")   # "in_progress"
```

### Step 2 — Poll for completion

```python
import time

def wait_for_batch(client, batch_id: str, poll_interval: int = 10):
    """Poll until the batch reaches 'ended' status."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{batch.processing_status}] "
            f"processing={counts.processing} "
            f"succeeded={counts.succeeded} "
            f"errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_interval)

batch = wait_for_batch(client, batch.id)
```

> **Tip:** Most batches complete within 1 hour. For production pipelines, use a scheduled job (cron / Celery beat) to poll rather than a busy loop.

### Step 3 — Retrieve results

Results are streamed as **JSONL** (one JSON object per line). Order is **not guaranteed** — always use `custom_id` to match results to requests.

```python
for result in client.messages.batches.results(batch_id):
    cid = result.custom_id

    if result.result.type == "succeeded":
        text = result.result.message.content[0].text
        print(f"[{cid}] ✅ {text.strip()}")

    elif result.result.type == "errored":
        error = result.result.error
        print(f"[{cid}] ❌ {error.type}: {error.message}")

    elif result.result.type == "expired":
        print(f"[{cid}] ⏰ Request expired (batch took > 24h)")

    elif result.result.type == "canceled":
        print(f"[{cid}] 🚫 Canceled before processing")
```

### Result types

| Type | Meaning | Billed? |
|------|---------|---------|
| `succeeded` | Request completed successfully | Yes |
| `errored` | Invalid request or server error | **No** |
| `expired` | Batch hit 24-hour limit | **No** |
| `canceled` | User canceled the batch | **No** |

---

## Complete Production Example

```python
import time
import json
import anthropic
from pathlib import Path

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5"


def batch_summarize(texts: list[str]) -> dict[str, str]:
    """
    Submit a list of texts for summarisation via the Batches API.
    Returns {custom_id: summary_text}.
    """
    # 1. Build requests
    requests = [
        {
            "custom_id": f"doc-{i}",
            "params": {
                "model": MODEL,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Summarise the following in 2 sentences:\n\n{text}"
                    }
                ],
            },
        }
        for i, text in enumerate(texts)
    ]

    # 2. Submit
    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {batch.id} with {len(requests)} requests")

    # 3. Poll
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  Waiting… ({batch.request_counts.processing} remaining)")
        time.sleep(15)

    # 4. Collect results
    summaries = {}
    errors = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            summaries[result.custom_id] = result.result.message.content[0].text
        else:
            errors.append(result.custom_id)

    if errors:
        print(f"Warning: {len(errors)} requests failed: {errors}")

    return summaries


# Usage
texts = [
    "The Python programming language was created by Guido van Rossum…",
    "Machine learning is a subset of artificial intelligence…",
]

results = batch_summarize(texts)
for cid, summary in results.items():
    print(f"\n{cid}:\n{summary}")
```

---

## Listing All Batches

```python
# The SDK handles pagination automatically
for batch in client.messages.batches.list(limit=10):
    print(f"{batch.id}  {batch.processing_status}  created={batch.created_at}")
```

---

## Canceling a Batch

```python
# Cancel a batch that is still processing
result = client.messages.batches.cancel(batch_id)
print(f"Status after cancel request: {result.processing_status}")
# → "canceling" (async), then "ended" once complete
```

Cancellation is asynchronous. Poll until `processing_status == "ended"` to confirm. Already-processed requests within the batch are returned.

---

## Prompt Caching with Batches

Prompt caching and batch discounts **stack** — you can save up to 90 %+ on repeated shared context.

```python
SHARED_SYSTEM = (
    "You are an expert literary analyst. "
    "Analyse the provided passage with nuance and depth."
)

PASSAGES = [
    "It was the best of times, it was the worst of times…",
    "Call me Ishmael. Some years ago—never mind how long precisely…",
    "Happy families are all alike; every unhappy family is unhappy in its own way.",
]

batch_requests = [
    {
        "custom_id": f"passage-{i}",
        "params": {
            "model": "claude-haiku-4-5",
            "max_tokens": 300,
            "system": [
                {
                    "type": "text",
                    "text": SHARED_SYSTEM,
                    "cache_control": {"type": "ephemeral"},  # cache the system prompt
                }
            ],
            "messages": [
                {"role": "user", "content": f"Analyse this opening line:\n\n{p}"}
            ],
        },
    }
    for i, p in enumerate(PASSAGES)
]

batch = client.messages.batches.create(requests=batch_requests)
print(f"Batch with prompt caching: {batch.id}")
```

> **Tip:** Use the 1-hour cache duration (`"cache_control": {"type": "ephemeral", "ttl": 3600}`) for batch workloads with shared context — the default 5-minute TTL may expire before all batch requests are processed.

---

## Extended Output (Beta)

For Claude Opus 4.7, Opus 4.6, and Sonnet 4.6, you can generate up to **300,000 output tokens per request** in a batch by adding a beta header:

```python
client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "output-300k-2026-03-24"}
)
```

Use this for book-length drafts, exhaustive data extraction, or large code scaffolding. Note that a single 300k-token generation can take over an hour.

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Matching results to requests by index | **Always use `custom_id`** — results may be reordered |
| Using `max_tokens: 0` in batch requests | Use at least `max_tokens: 1` |
| Assuming batch completes in < 1 hour | Design your polling loop for up to 24 hours |
| Downloading all results at once for huge batches | Use `client.messages.batches.results()` — it streams JSONL |
| Forgetting to delete sensitive batches | Use `DELETE /v1/messages/batches/{id}` when done |

---

*← [Chapter 4: Tool Use](04_tool_use.md) | [Chapter 6: Files API](06_files_api.md) →*
