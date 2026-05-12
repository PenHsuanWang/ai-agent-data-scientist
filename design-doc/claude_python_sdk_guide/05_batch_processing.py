"""
Lesson 05 — Message Batches API
=================================
Topics covered:
  • Why use batch processing (50 % cost saving, async processing)
  • Creating a batch with custom_id per request
  • Polling for completion
  • Retrieving and processing results
  • Cancelling a batch
  • Prompt caching inside batches
  • Limitations (100 k requests / 256 MB, 24-hour TTL)
"""

import time
import anthropic

client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 1. Create a batch
# ---------------------------------------------------------------------------
def create_batch_example():
    """
    Each item needs:
      • custom_id  — unique string (used to match results to requests)
      • params     — same params as the Messages API
    """
    requests = [
        anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": f"What is {a} + {b}?"}],
        )
        for a, b in [(1, 2), (10, 20), (100, 200)]
    ]

    # Build batch request items
    batch_requests = [
        {
            "custom_id": f"addition-{i}",
            "params": req,
        }
        for i, req in enumerate(requests)
    ]

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Batch created: {batch.id}  status={batch.processing_status}")
    return batch.id


# ---------------------------------------------------------------------------
# 2. Poll until the batch finishes
# ---------------------------------------------------------------------------
def wait_for_batch(batch_id: str, poll_interval: int = 5) -> None:
    """Poll every `poll_interval` seconds until processing_status == 'ended'."""
    print(f"Polling batch {batch_id}…")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(f"  status={status}  "
              f"processing={counts.processing}  "
              f"succeeded={counts.succeeded}  "
              f"errored={counts.errored}")
        if status == "ended":
            break
        time.sleep(poll_interval)
    print("Batch finished.\n")


# ---------------------------------------------------------------------------
# 3. Retrieve and process results
# ---------------------------------------------------------------------------
def get_batch_results(batch_id: str) -> None:
    """
    Results are streamed as JSONL. Order is NOT guaranteed — always use
    custom_id to match results to the original requests.
    """
    print("=== Batch Results ===")
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            print(f"[{cid}] ✅ {text.strip()}")
        elif result.result.type == "errored":
            print(f"[{cid}] ❌ Error: {result.result.error}")
        elif result.result.type == "expired":
            print(f"[{cid}] ⏰ Expired (batch took > 24 h)")
        elif result.result.type == "canceled":
            print(f"[{cid}] 🚫 Canceled")
    print()


# ---------------------------------------------------------------------------
# 4. List all batches in the workspace
# ---------------------------------------------------------------------------
def list_batches_example() -> None:
    print("=== All Batches ===")
    # Paginate with a for loop — SDK handles paging automatically
    for batch in client.messages.batches.list(limit=5):
        print(f"  {batch.id}  status={batch.processing_status}  "
              f"created={batch.created_at}")
    print()


# ---------------------------------------------------------------------------
# 5. Cancel a batch
# ---------------------------------------------------------------------------
def cancel_batch_example(batch_id: str) -> None:
    """
    Cancellation is asynchronous. The batch transitions to 'canceling' then
    'ended'. Already-completed requests within the batch are returned.
    """
    try:
        result = client.messages.batches.cancel(batch_id)
        print(f"Cancellation requested. status={result.processing_status}")
    except anthropic.NotFoundError:
        print(f"Batch {batch_id} not found (may already be ended).")
    print()


# ---------------------------------------------------------------------------
# 6. Prompt caching inside a batch
# ---------------------------------------------------------------------------
def batch_with_prompt_caching():
    """
    Mark large shared content with cache_control to improve cache hit rates.
    Prompt caching and batch discounts stack — double savings!
    """
    SHARED_CONTEXT = (
        "You are an expert in Python. "
        "Always give concise, correct answers."
    )

    questions = [
        "What is a list comprehension?",
        "What is the difference between a list and a tuple?",
        "How does a Python decorator work?",
    ]

    batch_requests = [
        {
            "custom_id": f"q-{i}",
            "params": {
                "model": MODEL,
                "max_tokens": 200,
                "system": [
                    {
                        "type": "text",
                        "text": SHARED_CONTEXT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": q}],
            },
        }
        for i, q in enumerate(questions)
    ]

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"=== Batch with Prompt Caching ===")
    print(f"Batch created: {batch.id}")
    print("(Polling skipped in this demo — check the Console for results.)\n")
    return batch.id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Create and wait for a small batch ---
    batch_id = create_batch_example()

    # NOTE: Real batches can take minutes. For this demo we poll briefly.
    # Replace the sleep with wait_for_batch(batch_id) in production.
    print("Waiting 10 seconds before checking status…")
    time.sleep(10)

    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status == "ended":
        get_batch_results(batch_id)
    else:
        print(f"Batch still processing (status={batch.processing_status}).")
        print("Run get_batch_results(batch_id) later when it completes.")

    list_batches_example()
    batch_with_prompt_caching()
