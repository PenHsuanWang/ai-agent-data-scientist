"""
PHASE 3 — STEP 7 & 8: Memory — RAG + Context Compaction
=========================================================
Harness Pillar: Context Management (The Brain's Workspace)
SDK Docs:
  Memory Tool: https://docs.anthropic.com/en/agents-and-tools/tool-use/memory-tool
  Prompt Cache: https://docs.anthropic.com/en/build-with-claude/prompt-caching
  Compaction:   https://docs.anthropic.com/en/build-with-claude/compaction

Goal:
  Step 7: Basic RAG — retrieve docs on demand instead of stuffing
          everything into the context window upfront.
  Step 8: Context compaction — summarize long history down to
          essential facts when approaching the token limit.

The MemGPT mental model:
  Context window  = RAM   (fast, limited, expensive)
  External store  = Hard drive  (slow, unlimited, cheap)
  → Load into RAM only what you need RIGHT NOW
"""

import json
import os
from pathlib import Path
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ═══════════════════════════════════════════════════════════
# STEP 7: Basic RAG (Retrieval-Augmented Generation)
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# Our "knowledge base" — in production this is
# a vector database (Pinecone, Weaviate, pgvector)
# For this tutorial: a simple keyword index
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "python_async": {
        "title": "Python Asyncio Guide",
        "keywords": ["async", "await", "asyncio", "coroutine", "event loop", "concurrent"],
        "content": """
Python's asyncio library enables concurrent code using async/await syntax.
Key concepts:
- async def: defines a coroutine function
- await: suspends execution until the awaitable completes
- asyncio.run(): runs the top-level coroutine
- asyncio.gather(): runs multiple coroutines concurrently
- Event loop: schedules and runs coroutines

Example:
    import asyncio
    async def fetch_data(url):
        await asyncio.sleep(1)  # simulates I/O
        return f"data from {url}"
    
    async def main():
        results = await asyncio.gather(
            fetch_data("api.com/1"),
            fetch_data("api.com/2"),
        )
        print(results)
    
    asyncio.run(main())
"""
    },
    "anthropic_sdk": {
        "title": "Anthropic Python SDK Guide",
        "keywords": ["anthropic", "claude", "sdk", "api", "messages", "tools", "streaming"],
        "content": """
The Anthropic Python SDK provides access to Claude's API.
Installation: pip install anthropic

Key classes:
- Anthropic(): sync client
- AsyncAnthropic(): async client

Core method: client.messages.create()
  Parameters:
    model: "claude-opus-4-5-20251101"
    max_tokens: integer (required)
    messages: list of {"role": "user"|"assistant", "content": str}
    system: system prompt (optional)
    tools: tool definitions (optional)
    temperature: 0-1 (optional, 0=deterministic)

Tool use:
  1. Define tools with JSON Schema
  2. Check stop_reason == "tool_use"
  3. Execute tool, send back tool_result
  4. Loop until stop_reason == "end_turn"
"""
    },
    "docker_basics": {
        "title": "Docker Containerization Guide",
        "keywords": ["docker", "container", "image", "dockerfile", "compose", "kubernetes"],
        "content": """
Docker packages applications into containers.
Key commands:
- docker build -t myapp .      # build image from Dockerfile
- docker run -p 8080:80 myapp  # run container
- docker compose up            # start multi-container app
- docker ps                    # list running containers
- docker logs <container_id>   # view logs

Dockerfile basics:
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["python", "app.py"]

Sandboxing: use --network none to disable network access,
            --read-only for read-only filesystem,
            --memory 512m to limit memory.
"""
    },
}


def simple_retrieval(query: str, top_k: int = 2) -> list[dict]:
    """
    Keyword-based retrieval. In production: embed query → vector search.
    Returns the top_k most relevant documents.
    """
    query_words = set(query.lower().split())
    scored = []
    for doc_id, doc in KNOWLEDGE_BASE.items():
        keywords = set(doc["keywords"])
        score = len(query_words & keywords)
        if score > 0:
            scored.append((score, doc_id, doc))

    scored.sort(reverse=True)
    return [{"id": doc_id, "title": doc["title"], "content": doc["content"]}
            for _, doc_id, doc in scored[:top_k]]


# ─────────────────────────────────────────────
# RAG Tool — the "Map" tool in ACI hierarchy
# ─────────────────────────────────────────────
RAG_TOOL = {
    "name": "search_docs",
    "description": (
        "Search the knowledge base for relevant documentation. "
        "Use this BEFORE answering any technical question — do not rely on your "
        "training data alone. Retrieve the relevant docs first, then answer. "
        "Returns the most relevant document excerpts for your query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query (e.g. 'how to run async functions in Python')"
            }
        },
        "required": ["query"],
    },
}

def run_rag_agent(question: str, verbose: bool = True) -> str:
    """Agent that retrieves docs on demand (just-in-time context)."""
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            tools=[RAG_TOOL],
            system=(
                "You are a technical documentation assistant. "
                "ALWAYS search the docs with search_docs before answering. "
                "Ground your answer in the retrieved documents, not training data."
            ),
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                docs = simple_retrieval(block.input["query"])
                result = json.dumps(docs, indent=2) if docs else '{"message": "No relevant docs found."}'
                if verbose:
                    print(f"\n[RAG] Retrieved {len(docs)} docs for: '{block.input['query']}'")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})


# ═══════════════════════════════════════════════════════════
# STEP 8: Context Compaction
# ═══════════════════════════════════════════════════════════

TOKEN_LIMIT = 4000  # threshold to trigger compaction (in practice: ~80% of model max)

def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~1 token per 4 characters."""
    return sum(len(json.dumps(m)) // 4 for m in messages)

def compact_history(messages: list) -> list:
    """
    Manual compaction: summarize the conversation history
    down to bullet points when approaching the token limit.
    
    Keeps:
      - The system prompt (injected separately)
      - The original first user message
      - The summary of all intermediate turns
      - The last 2 messages (recent context)
    """
    if len(messages) <= 4:
        return messages

    # Summarize the middle of the conversation
    to_summarize = messages[1:-2]
    history_text = "\n".join(
        f"{m['role'].upper()}: {json.dumps(m['content'])[:300]}"
        for m in to_summarize
    )

    print(f"\n[Compaction] Summarizing {len(to_summarize)} messages...")

    summary_response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "Summarize the following conversation into a compact bulleted list "
                "of the key facts, decisions, and results. Be very concise.\n\n"
                f"{history_text}"
            )
        }],
    )
    summary = summary_response.content[0].text

    # Replace middle messages with a single summary message
    compacted = [
        messages[0],  # first user message preserved
        {
            "role": "user",
            "content": f"[CONVERSATION SUMMARY]\n{summary}"
        },
        {"role": "assistant", "content": "Understood. I'll continue with this context."},
        *messages[-2:],  # last 2 messages preserved
    ]

    print(f"[Compaction] {len(messages)} → {len(compacted)} messages")
    print(f"[Compaction] Tokens: ~{estimate_tokens(messages)} → ~{estimate_tokens(compacted)}")
    return compacted


def run_long_session_agent(questions: list[str], verbose: bool = True) -> list[str]:
    """
    Agent that handles a long multi-turn session with context compaction.
    Automatically compacts when approaching the token limit.
    """
    messages = []
    answers = []

    for q in questions:
        messages.append({"role": "user", "content": q})

        # Check token budget BEFORE calling API — compact if needed
        token_estimate = estimate_tokens(messages)
        if verbose:
            print(f"\n[Context] ~{token_estimate} tokens in history")

        if token_estimate > TOKEN_LIMIT:
            messages = compact_history(messages)

        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=512,
            system="You are a helpful assistant with memory of our conversation.",
            messages=messages,
        )

        answer = response.content[0].text
        messages.append({"role": "assistant", "content": answer})
        answers.append(answer)

        if verbose:
            print(f"Q: {q}")
            print(f"A: {answer[:200]}")

    return answers


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7: RAG — Just-in-Time Context Retrieval")
    print("=" * 60)

    rag_questions = [
        "How do I run multiple async functions at the same time in Python?",
        "How do I implement tool use with the Anthropic Python SDK?",
    ]
    for q in rag_questions:
        print(f"\nQ: {q}")
        answer = run_rag_agent(q, verbose=True)
        print(f"A: {answer[:500]}")

    print("\n" + "=" * 60)
    print("STEP 8: Context Compaction")
    print("=" * 60)

    # Simulate a long multi-turn session
    long_conversation = [
        "My name is Alex and I'm building a Python web scraper.",
        "I want to use asyncio for concurrent requests. How does that work?",
        "What's the difference between asyncio.gather and asyncio.wait?",
        "How do I handle timeouts in async code?",
        "Can I use the Anthropic SDK in async mode?",
        "How do I containerize my async scraper with Docker?",
        "Given everything we've discussed, what's the recommended architecture for my project?",
    ]

    answers = run_long_session_agent(long_conversation, verbose=True)
    print(f"\nFinal answer: {answers[-1][:400]}")


print("""
KEY TAKEAWAYS
=============
Step 7 — RAG:
1. NEVER stuff all docs into the context window upfront (wastes tokens).
2. The search_docs tool retrieves on demand — only what's needed RIGHT NOW.
3. In production: replace keyword search with vector embeddings (OpenAI/Voyage).
4. This is the "Map" layer of the ACI — it gives the agent its navigational system.

Step 8 — Compaction:
1. Estimate tokens BEFORE every API call — compact proactively.
2. Manual compaction: summarize → replace middle messages with summary.
3. SDK auto-compaction: just add cache_control={"type":"ephemeral"} at 
   the top level and the SDK handles it server-side (on supported models).
4. Always preserve: the first user message + the last N messages.
5. The golden rule: treat the context window like RAM — keep only what's active.
""")
