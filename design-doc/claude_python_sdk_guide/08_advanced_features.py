"""
Lesson 08 — Advanced Features
===============================
Topics covered:
  • Token counting (before & after a request)
  • Paginated list endpoints
  • Type system — TypedDicts for requests, Pydantic for responses
  • Handling null vs missing response fields
  • Raw HTTP response access (.with_raw_response)
  • Streaming response body (.with_streaming_response)
  • Logging / debug output
  • Custom / undocumented endpoints & params
  • HTTP client customisation (proxies, custom headers)
  • Checking the installed SDK version
  • Platform integrations (Bedrock, Vertex AI)
"""

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 1. Token counting
# ---------------------------------------------------------------------------
def token_counting_demo():
    """
    Count tokens before sending to avoid surprises / control cost.
    Token counts are returned in every response via message.usage.
    """
    print("=== Token Counting ===")

    messages = [
        {"role": "user", "content": "Explain quantum entanglement in simple terms."}
    ]

    # --- Pre-flight count ---
    count = client.messages.count_tokens(
        model=MODEL,
        messages=messages,
    )
    print(f"Estimated input tokens: {count.input_tokens}")

    # --- Actual request ---
    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=messages,
    )
    print(f"Actual input tokens : {resp.usage.input_tokens}")
    print(f"Actual output tokens: {resp.usage.output_tokens}")
    print(f"Total tokens        : {resp.usage.input_tokens + resp.usage.output_tokens}")
    print()


# ---------------------------------------------------------------------------
# 2. Models API — list available models (paginated)
# ---------------------------------------------------------------------------
def list_models_demo():
    """The SDK automatically handles pagination with the for-loop syntax."""
    print("=== Available Models ===")
    for model in client.models.list():
        print(f"  {model.id}")
    print()


# ---------------------------------------------------------------------------
# 3. Type system
# ---------------------------------------------------------------------------
def type_system_demo():
    """
    Request parameters: TypedDicts → great for editor autocomplete.
    Responses: Pydantic models → serialisable, inspectable.
    """
    print("=== Type System ===")

    resp = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "Name the colours in a traffic light."}],
    )

    # Pydantic model → dict
    as_dict = resp.model_dump()
    print("Response as dict (keys):", list(as_dict.keys()))

    # Pydantic model → JSON string
    as_json = resp.model_dump_json(indent=2)
    print("Response as JSON (first 200 chars):", as_json[:200])
    print()


# ---------------------------------------------------------------------------
# 4. Null vs missing fields
# ---------------------------------------------------------------------------
def null_vs_missing_demo():
    """
    Pydantic distinguishes between a field explicitly set to None
    and a field that was absent from the response.
    """
    print("=== Null vs Missing Fields ===")

    resp = client.messages.create(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi!"}],
    )

    # system is optional; check whether it was included
    if resp.system is anthropic.NOT_GIVEN:        # field was absent
        print("system field: absent from response")
    elif resp.system is None:                      # field explicitly null
        print("system field: explicitly null")
    else:
        print(f"system field: {resp.system}")
    print()


# ---------------------------------------------------------------------------
# 5. Raw HTTP response access
# ---------------------------------------------------------------------------
def raw_response_demo():
    """
    .with_raw_response eagerly reads the full body and exposes headers.
    Useful for accessing x-request-id, rate-limit headers, etc.
    """
    print("=== Raw HTTP Response ===")

    raw = client.messages.with_raw_response.create(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Say hello!"}],
    )

    print("HTTP status :", raw.http_response.status_code)
    print("request-id  :", raw.http_response.headers.get("request-id", "n/a"))
    message = raw.parse()          # parse into the typed Message model
    print("Content     :", message.content[0].text.strip())
    print()


# ---------------------------------------------------------------------------
# 6. Streaming response body
# ---------------------------------------------------------------------------
def streaming_body_demo():
    """
    .with_streaming_response delays reading the body until you call
    .read(), .text(), .json(), or iterate .iter_lines().
    The context manager ensures the connection is always closed.
    """
    print("=== Streaming Response Body ===")

    with client.messages.with_streaming_response.create(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Count to three."}],
    ) as resp:
        text = resp.text()   # reads and decodes the full body
    print("Body:", text[:200])
    print()


# ---------------------------------------------------------------------------
# 7. Enabling SDK logging
# ---------------------------------------------------------------------------
def logging_demo():
    """
    Set ANTHROPIC_LOG=debug (or info/warn/off) to enable structured logging.
    This shows the full request/response cycle including headers.
    """
    import os
    import logging

    print("=== Logging ===")
    print("Set env var ANTHROPIC_LOG=debug before running to see full logs.")
    print(f"Current ANTHROPIC_LOG={os.environ.get('ANTHROPIC_LOG', 'not set')}")

    # You can also configure the standard library logger directly:
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    print()


# ---------------------------------------------------------------------------
# 8. Custom / undocumented endpoints
# ---------------------------------------------------------------------------
def custom_endpoints_demo():
    """
    Use client.get / client.post to call undocumented endpoints.
    The extra_query / extra_body / extra_headers options add params
    to any documented call without changing the typed interface.
    """
    print("=== Custom Endpoints ===")

    # Extra query params on a documented call:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi!"}],
        extra_headers={"X-Custom-Header": "crash-course"},
    )
    print(f"Response with custom header, request_id={resp._request_id}")
    print()


# ---------------------------------------------------------------------------
# 9. HTTP client customisation
# ---------------------------------------------------------------------------
def custom_http_client_demo():
    """
    Pass a custom httpx client to control proxies, connection limits,
    SSL settings, etc.  Use DefaultHttpxClient to preserve SDK defaults.
    """
    import httpx

    print("=== Custom HTTP Client ===")

    http_client = anthropic.DefaultHttpxClient(
        proxies=None,                     # set to your proxy URL if needed
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=5,
        ),
    )

    custom = anthropic.Anthropic(http_client=http_client)
    resp = custom.messages.create(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Quick test."}],
    )
    print("Custom HTTP client response:", resp.content[0].text.strip())
    custom.close()
    print()


# ---------------------------------------------------------------------------
# 10. SDK version
# ---------------------------------------------------------------------------
def sdk_version_demo():
    print("=== SDK Version ===")
    print(f"anthropic version: {anthropic.__version__}")
    print()


# ---------------------------------------------------------------------------
# 11. Platform integrations (Bedrock / Vertex)
# ---------------------------------------------------------------------------
def platform_integrations_note():
    """
    All four client classes are importable from the base package.
    They share the same interface — just swap the client constructor.
    """
    print("=== Platform Integration Clients ===")
    print(
        "  from anthropic import AnthropicBedrock\n"
        "  from anthropic import AnthropicBedrockMantle  # preferred for new projects\n"
        "  from anthropic import AnthropicVertex\n"
        "  from anthropic import AnthropicFoundry\n"
        "\n"
        "Install extras:\n"
        "  pip install anthropic[bedrock]\n"
        "  pip install anthropic[vertex]\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    token_counting_demo()
    list_models_demo()
    type_system_demo()
    raw_response_demo()
    sdk_version_demo()
    platform_integrations_note()
    custom_endpoints_demo()
    logging_demo()
    # Uncomment to test (require network):
    # streaming_body_demo()
    # custom_http_client_demo()
