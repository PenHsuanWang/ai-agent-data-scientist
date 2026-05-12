"""
Lesson 03 — Streaming Responses
================================
Topics covered:
  • Stream with client.messages.stream() — high-level helper
  • Stream with client.messages.create(stream=True) — low-level iterator
  • Async streaming
  • Accessing accumulated final message
  • Raw SSE event inspection
  • Error recovery for interrupted streams
"""

import asyncio
import anthropic


client = anthropic.Anthropic()
async_client = anthropic.AsyncAnthropic()


# ---------------------------------------------------------------------------
# 1. High-level streaming helper — recommended for most use cases
# ---------------------------------------------------------------------------
def stream_with_helper():
    """
    client.messages.stream() returns a context-manager that gives you:
      • on_text callback / iterator
      • .get_final_message()  — full Message after stream ends
      • .get_final_text()     — plain string
    """
    print("=== Stream Helper (sync) ===")

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Count from 1 to 10, one per line."}],
    ) as stream:
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)

    # After the context manager exits the full message is ready:
    final = stream.get_final_message()
    print(f"\n[done — {final.usage.output_tokens} output tokens]")
    print()


# ---------------------------------------------------------------------------
# 2. Low-level streaming — iterate raw events
# ---------------------------------------------------------------------------
def stream_low_level():
    """
    client.messages.create(stream=True) returns an iterator of SSE events.
    Lower memory usage — no automatic message accumulation.
    """
    print("=== Stream Low-level (sync) ===")

    with client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "Write a haiku about Python."}],
        stream=True,
    ) as stream:
        for event in stream:
            # Event types: message_start, content_block_start,
            #   content_block_delta, content_block_stop, message_delta,
            #   message_stop, ping
            if event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# 3. Async streaming helper
# ---------------------------------------------------------------------------
async def async_stream_with_helper():
    print("=== Async Stream Helper ===")

    async with async_client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "List five programming languages."}],
    ) as stream:
        async for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)

    final = stream.get_final_message()
    print(f"\n[done — {final.usage.output_tokens} output tokens]")
    print()


# ---------------------------------------------------------------------------
# 4. Async low-level streaming
# ---------------------------------------------------------------------------
async def async_stream_low_level():
    print("=== Async Stream Low-level ===")

    async with async_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "Say 'hello world' in 5 languages."}],
        stream=True,
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# 5. Collecting the full text without printing incrementally
# ---------------------------------------------------------------------------
def stream_collect_text():
    """Use .get_final_text() when you only need the finished string."""
    print("=== Collect Full Text ===")

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "What is 7 × 8?"}],
    ) as stream:
        full_text = stream.get_final_text()

    print("Answer:", full_text)
    print()


# ---------------------------------------------------------------------------
# 6. Error recovery — resume an interrupted stream (Claude 4.5 and earlier)
# ---------------------------------------------------------------------------
def stream_error_recovery():
    """
    If a stream is interrupted, you can reconstruct the conversation with
    the partial assistant response and send a continuation request.
    
    NOTE: For Claude 4.6+, send a user message asking it to continue instead.
    """
    print("=== Error Recovery Pattern ===")

    partial_text = ""

    try:
        with client.messages.stream(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": "Tell me a short story."}],
        ) as stream:
            for chunk in stream.text_stream:
                partial_text += chunk
                print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\n[Stream interrupted: {e}]")
        print("[Resuming…]")

        # Build a continuation request with whatever we received so far
        if partial_text:
            continuation = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                messages=[
                    {"role": "user", "content": "Tell me a short story."},
                    # Partial assistant response acts as a "prefix"
                    {"role": "assistant", "content": partial_text},
                ],
            )
            print(continuation.content[0].text)
    else:
        print()

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def run_async():
    await async_stream_with_helper()
    await async_stream_low_level()


if __name__ == "__main__":
    stream_with_helper()
    stream_low_level()
    stream_collect_text()
    stream_error_recovery()
    asyncio.run(run_async())
