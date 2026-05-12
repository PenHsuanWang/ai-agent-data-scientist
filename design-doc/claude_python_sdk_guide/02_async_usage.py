"""
Lesson 02 — Async Client
=========================
Topics covered:
  • AsyncAnthropic — identical interface, async/await instead
  • Running multiple requests concurrently with asyncio.gather
  • aiohttp backend for higher throughput
  • Async multi-turn conversation
"""

import asyncio
import anthropic


# ---------------------------------------------------------------------------
# 1. Basic async message
# ---------------------------------------------------------------------------
async def basic_async():
    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

    message = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "Name three colours of the rainbow."}],
    )

    print("=== Basic Async ===")
    print(message.content[0].text)
    print()
    await client.close()


# ---------------------------------------------------------------------------
# 2. Concurrent requests (fan-out pattern)
# ---------------------------------------------------------------------------
async def concurrent_requests():
    """Send multiple independent prompts at the same time."""
    client = anthropic.AsyncAnthropic()

    prompts = [
        "What is 2 + 2?",
        "Name the largest planet in our solar system.",
        "In one sentence, what is machine learning?",
    ]

    tasks = [
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": p}],
        )
        for p in prompts
    ]

    responses = await asyncio.gather(*tasks)

    print("=== Concurrent Requests ===")
    for prompt, resp in zip(prompts, responses):
        print(f"Q: {prompt}")
        print(f"A: {resp.content[0].text.strip()}")
        print()

    await client.close()


# ---------------------------------------------------------------------------
# 3. aiohttp backend — better concurrency under high load
# ---------------------------------------------------------------------------
async def aiohttp_backend():
    """
    Install: pip install anthropic aiohttp
    The aiohttp backend replaces the default httpx transport.
    """
    try:
        from anthropic.adapters import AiohttpTransport  # type: ignore

        transport = AiohttpTransport()
        client = anthropic.AsyncAnthropic(http_client=transport)

        message = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello!"}],
        )
        print("=== aiohttp Backend ===")
        print(message.content[0].text)
        await client.close()
    except (ImportError, AttributeError):
        # aiohttp adapter API may differ across SDK versions
        print("=== aiohttp Backend ===")
        print("(aiohttp adapter not available in this SDK version; "
              "using default httpx transport.)")
        client = anthropic.AsyncAnthropic()
        message = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello!"}],
        )
        print(message.content[0].text)
        await client.close()
    print()


# ---------------------------------------------------------------------------
# 4. Async multi-turn conversation
# ---------------------------------------------------------------------------
async def async_multi_turn():
    client = anthropic.AsyncAnthropic()
    history = []

    async def chat(text: str) -> str:
        history.append({"role": "user", "content": text})
        resp = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=history,
        )
        reply = resp.content[0].text
        history.append({"role": "assistant", "content": reply})
        return reply

    print("=== Async Multi-turn ===")
    print("User: My favourite colour is blue.")
    print("Claude:", await chat("My favourite colour is blue."))
    print("User: What is my favourite colour?")
    print("Claude:", await chat("What is my favourite colour?"))
    print()
    await client.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    await basic_async()
    await concurrent_requests()
    await aiohttp_backend()
    await async_multi_turn()


if __name__ == "__main__":
    asyncio.run(main())
