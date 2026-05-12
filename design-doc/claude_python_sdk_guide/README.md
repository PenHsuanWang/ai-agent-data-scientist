# Anthropic Python SDK — Crash Course

A comprehensive crash course covering every major feature of the
[Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python),
with both a **textbook guide** (Markdown chapters) and **runnable code examples** (Python files).

## Prerequisites

- Python 3.9+
- An Anthropic API key — get one at <https://platform.claude.com/settings/keys>

## Setup

```bash
pip install anthropic python-dotenv
# Optional extras
pip install "anthropic[bedrock]"   # AWS Bedrock
pip install "anthropic[vertex]"    # Google Vertex AI
pip install aiohttp                # Better async performance
```

Create a `.env` file in this directory:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 📖 Textbook (Markdown Chapters)

Read these in order for a comprehensive understanding:

| Chapter | File | Topic |
|---------|------|-------|
| 0 | [`00_introduction.md`](00_introduction.md) | Introduction, setup, first message, model reference |
| 1 | [`01_core_concepts.md`](01_core_concepts.md) | Messages API, roles, content blocks, system prompts, multi-turn |
| 2 | [`02_sync_async_clients.md`](02_sync_async_clients.md) | Sync & async clients, concurrent fan-out, FastAPI integration |
| 3 | [`03_streaming.md`](03_streaming.md) | SSE streaming, helpers, raw events, tool streaming, error recovery |
| 4 | [`04_tool_use.md`](04_tool_use.md) | Tool definitions, agentic loop, parallel tools, strict tool use |
| 5 | [`05_batch_processing.md`](05_batch_processing.md) | Batches API, 50 % cost saving, polling, prompt caching in batches |
| 6 | [`06_files_api.md`](06_files_api.md) | Files API — upload, reference, manage, document Q&A pattern |
| 7 | [`07_error_handling.md`](07_error_handling.md) | Error hierarchy, retries, timeouts, exponential backoff |
| 8 | [`08_advanced_features.md`](08_advanced_features.md) | Token counting, pagination, type system, logging, platform integrations |
| 9 | [`09_best_practices.md`](09_best_practices.md) | Cost optimisation, security, observability, architecture patterns |

---

## 💻 Runnable Code Examples (Python Files)

| File | Topic |
|------|-------|
| [`01_basic_usage.py`](01_basic_usage.py) | Client setup, first message, sync API, multi-turn |
| [`02_async_usage.py`](02_async_usage.py) | Async client, concurrent fan-out, aiohttp backend |
| [`03_streaming.py`](03_streaming.py) | Streaming helpers, raw SSE events, error recovery |
| [`04_tool_use.py`](04_tool_use.py) | Tool definitions, agentic loop, forced tool use |
| [`05_batch_processing.py`](05_batch_processing.py) | Batch create/poll/results, prompt caching in batches |
| [`06_files_api.py`](06_files_api.py) | Upload, reference by file_id, list, delete files |
| [`07_error_handling.py`](07_error_handling.py) | Error types, request IDs, retries, backoff |
| [`08_advanced_features.py`](08_advanced_features.py) | Token counting, pagination, type system, raw HTTP, logging |

---

## Quick Model Reference (May 2026)

| Model | API ID | Input / Output (per MTok) | Context |
|-------|--------|--------------------------|---------|
| Claude Opus 4.7 | `claude-opus-4-7` | $5 / $25 | 1 M tokens |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | $3 / $15 | 1 M tokens |
| Claude Haiku 4.5 | `claude-haiku-4-5` | $1 / $5 | 200 k tokens |

## Running the examples

```bash
# Run any Python file directly:
python 01_basic_usage.py
python 03_streaming.py
python 04_tool_use.py
```
