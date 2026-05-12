# Chapter 0 — Introduction & Setup

## What is Anthropic?

Anthropic is an AI safety company that builds the **Claude** family of large language models. Claude models are designed to be helpful, harmless, and honest, making them suitable for production applications that require both capability and reliability.

## The Anthropic Python SDK

The [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) provides a convenient, type-safe Python interface to the Anthropic REST API. Key capabilities out of the box:

| Feature | Description |
|---------|-------------|
| Sync & Async clients | Both `Anthropic` and `AsyncAnthropic` with identical interfaces |
| Streaming | Server-Sent Events (SSE) with high-level helpers |
| Tool use | Function calling with full agentic loop support |
| Message Batches | Asynchronous bulk processing at 50 % cost |
| Files API | Upload once, reference many times across requests |
| Type safety | TypedDicts for requests, Pydantic models for responses |
| Auto-retries | Configurable exponential backoff on transient errors |
| Platform support | AWS Bedrock, Google Vertex AI, Microsoft Foundry |

---

## Installation

### Minimum install

```bash
pip install anthropic
```

### With optional extras

```bash
# AWS Bedrock integration
pip install "anthropic[bedrock]"

# Google Vertex AI integration
pip install "anthropic[vertex]"

# Better async performance with aiohttp
pip install anthropic aiohttp

# Recommended: load API keys from .env files
pip install python-dotenv
```

### Requirements

- **Python 3.9 or later**

---

## Authentication

The SDK reads your API key from the environment variable `ANTHROPIC_API_KEY`.

### Option 1 — `.env` file (recommended for local development)

Create a file named `.env` in your project root:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Then load it at the top of your script:

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ

import anthropic
client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY automatically
```

> **Security rule:** Never commit your API key to source control. Add `.env` to your `.gitignore`.

### Option 2 — Environment variable (CI/CD, production)

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
python my_script.py
```

### Option 3 — Explicit parameter (avoid where possible)

```python
client = anthropic.Anthropic(api_key="sk-ant-api03-...")
```

---

## Getting an API Key

1. Sign in to [platform.claude.com](https://platform.claude.com)
2. Go to **Settings → API Keys**
3. Click **Create Key**
4. Copy the key — it is shown only once

---

## Model Reference (as of May 2026)

| Model | API ID | Context | Max Output | Input $/MTok | Output $/MTok |
|-------|--------|---------|-----------|-------------|--------------|
| Claude Opus 4.7 | `claude-opus-4-7` | 1 M tokens | 128 k | $5 | $25 |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 1 M tokens | 64 k | $3 | $15 |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200 k tokens | 64 k | $1 | $5 |

**Choosing a model:**
- **Opus 4.7** — complex reasoning, agentic coding, tasks requiring deep understanding
- **Sonnet 4.6** — balanced speed and intelligence, the workhorse for most apps
- **Haiku 4.5** — fastest and cheapest; great for high-volume, simpler tasks

---

## Your First Claude Message

```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment

message = client.messages.create(
    model="claude-haiku-4-5",    # model ID
    max_tokens=256,              # maximum tokens to generate
    messages=[
        {
            "role": "user",
            "content": "Hello Claude! What are you?"
        }
    ]
)

print(message.content[0].text)
```

**Expected output:**
```
Hello! I'm Claude, an AI assistant made by Anthropic. I'm designed to be helpful,
harmless, and honest. How can I assist you today?
```

---

## Anatomy of a Response

```python
message.id           # "msg_01XFDUDYJgAACzvnptvVoYEL"
message.type         # "message"
message.role         # "assistant"
message.model        # "claude-haiku-4-5-20251001"
message.stop_reason  # "end_turn" | "max_tokens" | "tool_use" | "stop_sequence"
message.content      # list of content blocks (TextBlock, ToolUseBlock, …)
message.usage        # Usage(input_tokens=25, output_tokens=110)
message._request_id  # "req_01..." — quote this when contacting Anthropic support
```

---

## SDK Version Check

```python
import anthropic
print(anthropic.__version__)   # e.g. "0.50.0"
```

---

## Course Outline

| Chapter | File | Topic |
|---------|------|-------|
| 0 | `00_introduction.md` | Introduction & Setup ← *you are here* |
| 1 | `01_core_concepts.md` | Core Concepts: Messages API |
| 2 | `02_sync_async_clients.md` | Sync & Async Clients |
| 3 | `03_streaming.md` | Streaming Responses |
| 4 | `04_tool_use.md` | Tool Use (Function Calling) |
| 5 | `05_batch_processing.md` | Message Batches API |
| 6 | `06_files_api.md` | Files API |
| 7 | `07_error_handling.md` | Error Handling & Resilience |
| 8 | `08_advanced_features.md` | Advanced Features |
| 9 | `09_best_practices.md` | Best Practices & Production Guide |

---

*Next → [Chapter 1: Core Concepts: Messages API](01_core_concepts.md)*
