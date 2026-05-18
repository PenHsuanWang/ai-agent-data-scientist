"""Context window management — sliding window + Anthropic prompt caching.

This module provides the public ``optimize_context_window`` function that
the agent loop (or any caller) should use when preparing messages for an
Anthropic API call.  It combines two concerns:

1. **Sliding window** — trim the oldest messages so the history never
   exceeds ``max_history`` entries.
2. **Ephemeral cache injection** — add ``cache_control: {type: ephemeral}``
   to the two most-recent *user* messages so Anthropic can cache the
   conversation prefix and reduce per-call token cost.

Usage::

    from app.services.context_manager import optimize_context_window
    api_messages = optimize_context_window(session_state.messages, max_history=20)
    response = await client.messages.create(..., messages=api_messages)
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from app.domain.state_models import AgentMessage

logger = logging.getLogger(__name__)

# Maximum number of user messages to inject ephemeral cache markers into.
_MAX_CACHE_BREAKPOINTS = 2


def optimize_context_window(
    messages: list[AgentMessage],
    max_history: int = 20,
) -> list[dict[str, Any]]:
    """Apply sliding window and inject Anthropic prompt-caching markers.

    Steps
    -----
    1. Retain only the *most recent* ``max_history`` messages.
    2. Ensure the retained window starts with a ``"user"`` turn (Anthropic
       rejects histories that begin with an ``"assistant"`` message).
    3. Convert each ``AgentMessage`` to a plain ``dict`` compatible with the
       Anthropic SDK ``messages=`` parameter.
    4. Inject ``cache_control: {"type": "ephemeral"}`` into the two most
       recent user messages to enable server-side KV-cache reuse.

    Args:
        messages: Full ordered conversation history.
        max_history: Maximum number of messages to retain.  Defaults to 20.

    Returns:
        A list of ``dict`` objects ready to pass to
        ``AsyncAnthropic().messages.create(messages=...)``.

    Notes:
        - If a user message's ``content`` is already a ``list`` (i.e. it
          contains structured blocks), ``cache_control`` is injected into the
          last ``"text"`` block found in that list.  If no text block exists,
          the message is left unchanged.
        - This function never mutates the input ``messages`` list or any of
          its elements; it always works on deep-copies for the cache injection
          step.
    """
    # ── 1. Sliding window ─────────────────────────────────────────────── #
    trimmed: list[AgentMessage] = (
        messages[-max_history:] if len(messages) > max_history else list(messages)
    )
    if len(messages) > max_history:
        logger.info(
            "Context window: trimmed %d → %d messages (max_history=%d)",
            len(messages),
            len(trimmed),
            max_history,
        )

    # ── 2. Ensure window starts with a user turn ──────────────────────── #
    while trimmed and trimmed[0].role != "user":
        trimmed = trimmed[1:]
    if not trimmed and messages:
        trimmed = [messages[-1]]

    # ── 3. Convert to API dicts (shallow copy — content is not mutated yet) #
    api_messages: list[dict[str, Any]] = [
        {"role": msg.role, "content": msg.content} for msg in trimmed
    ]

    # ── 4. Inject ephemeral cache markers on up to 2 recent user msgs ─── #
    breakpoints_applied = 0
    for msg_dict in reversed(api_messages):
        if msg_dict["role"] != "user":
            continue

        content = msg_dict["content"]

        if isinstance(content, str):
            # Wrap string in a single text block with cache_control.
            msg_dict["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            breakpoints_applied += 1

        elif isinstance(content, list):
            # Find the last text block and inject cache_control there.
            # Work on a deep copy so we never mutate AgentMessage objects.
            blocks: list[dict[str, Any]] = copy.deepcopy(content)
            injected = False
            for block in reversed(blocks):
                if block.get("type") == "text":
                    block["cache_control"] = {"type": "ephemeral"}
                    injected = True
                    break
            if injected:
                msg_dict["content"] = blocks
                breakpoints_applied += 1

        if breakpoints_applied >= _MAX_CACHE_BREAKPOINTS:
            break

    return api_messages
