"""
Lesson 07 — Error Handling, Retries & Timeouts
================================================
Topics covered:
  • Error hierarchy (APIError and its subclasses)
  • Catching specific error types
  • Request IDs for debugging
  • Configuring retries (max_retries)
  • Configuring timeouts (timeout)
  • Per-request overrides with with_options()
  • Connection management (context manager / .close())
"""

import anthropic

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------
#   anthropic.APIError                 (base)
#   ├── anthropic.APIConnectionError   (network problem — no HTTP status)
#   ├── anthropic.APIStatusError       (base for HTTP error codes)
#   │   ├── anthropic.BadRequestError          (400)
#   │   ├── anthropic.AuthenticationError      (401)
#   │   ├── anthropic.PermissionDeniedError    (403)
#   │   ├── anthropic.NotFoundError            (404)
#   │   ├── anthropic.UnprocessableEntityError (422)
#   │   ├── anthropic.RateLimitError           (429)
#   │   └── anthropic.InternalServerError      (>=500)
#   └── anthropic.APITimeoutError      (request timed out)


# ---------------------------------------------------------------------------
# 1. Catching specific errors
# ---------------------------------------------------------------------------
def error_handling_demo():
    print("=== Error Handling ===")

    try:
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}],
        )
    except anthropic.AuthenticationError as e:
        print(f"Auth error (401): {e.message}  request_id={e.request_id}")
    except anthropic.RateLimitError as e:
        print(f"Rate limited (429): retry after a moment.  request_id={e.request_id}")
    except anthropic.InternalServerError as e:
        print(f"Server error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError as e:
        print(f"Network error: {e}")
    except anthropic.APITimeoutError:
        print("Request timed out — consider using streaming for long requests.")
    except anthropic.APIStatusError as e:
        # Catch-all for any other 4xx/5xx
        print(f"API error {e.status_code}: {e.message}")
    else:
        print("Request succeeded (no error).")

    print()


# ---------------------------------------------------------------------------
# 2. Accessing request_id for debugging / support
# ---------------------------------------------------------------------------
def request_id_demo():
    """
    Every response carries _request_id from the response header.
    Log this to quickly report issues to Anthropic support.
    """
    print("=== Request ID ===")
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=32,
            messages=[{"role": "user", "content": "Say hi!"}],
        )
        print(f"request_id: {resp._request_id}")
    except anthropic.APIStatusError as e:
        print(f"Error request_id: {e.request_id}")
    print()


# ---------------------------------------------------------------------------
# 3. Configuring retries
# ---------------------------------------------------------------------------
def retry_demo():
    """
    By default the SDK retries 2 times on:
      - Connection errors
      - 408 Request Timeout
      - 409 Conflict
      - 429 Rate Limit
      - >=500 Internal errors

    Set max_retries=0 to disable, or increase the count.
    """
    print("=== Retries ===")

    # Client-level: all requests retry up to 5 times
    resilient_client = anthropic.Anthropic(max_retries=5)

    # Per-request override: disable retries for this one call
    resp = client.messages.with_options(max_retries=0).create(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "What year is it?"}],
    )
    print("No-retry response:", resp.content[0].text.strip())
    print()


# ---------------------------------------------------------------------------
# 4. Configuring timeouts
# ---------------------------------------------------------------------------
def timeout_demo():
    """
    Default timeout: 10 minutes (600 seconds).
    Timeouts raise anthropic.APITimeoutError and are retried by default.

    TIP: For long-running requests, use streaming instead of increasing
    the timeout — some proxies drop idle TCP connections.
    """
    import httpx

    print("=== Timeouts ===")

    # Client-level: 30-second global timeout
    fast_client = anthropic.Anthropic(timeout=30.0)

    # Fine-grained timeout control using httpx.Timeout:
    #   connect   — TCP handshake
    #   read      — waiting for response data
    #   write     — sending request body
    #   pool      — waiting for a connection from the pool
    detailed_timeout = httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0,
    )
    custom_client = anthropic.Anthropic(timeout=detailed_timeout)

    # Per-request override
    resp = client.messages.with_options(timeout=60.0).create(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "What is 1 + 1?"}],
    )
    print("Response:", resp.content[0].text.strip())
    print()


# ---------------------------------------------------------------------------
# 5. Context-manager for proper resource cleanup
# ---------------------------------------------------------------------------
def context_manager_demo():
    """Use a with-block to ensure HTTP connections are closed after use."""
    print("=== Context Manager ===")
    with anthropic.Anthropic() as c:
        resp = c.messages.create(
            model="claude-haiku-4-5",
            max_tokens=32,
            messages=[{"role": "user", "content": "Goodbye!"}],
        )
        print(resp.content[0].text.strip())
    # HTTP connections are closed automatically here
    print()


# ---------------------------------------------------------------------------
# 6. Implementing exponential back-off for rate limits
# ---------------------------------------------------------------------------
def manual_backoff_demo():
    """
    The SDK handles retries automatically, but here is a manual pattern
    useful when you need custom logic (e.g., jitter, alerting).
    """
    import time
    import random

    print("=== Manual Backoff Pattern ===")
    max_attempts = 4
    base_delay = 1.0

    for attempt in range(max_attempts):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=32,
                messages=[{"role": "user", "content": "Hello!"}],
            )
            print(f"Success on attempt {attempt + 1}: {resp.content[0].text.strip()}")
            break
        except anthropic.RateLimitError:
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {delay:.1f}s…")
                time.sleep(delay)
            else:
                print("Max retries reached.")
                raise
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    error_handling_demo()
    request_id_demo()
    retry_demo()
    timeout_demo()
    context_manager_demo()
    manual_backoff_demo()
