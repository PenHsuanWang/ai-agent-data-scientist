"""
Lesson 01 — Basic Usage (Synchronous Client)
=============================================
Topics covered:
  • Installing and importing the SDK
  • Creating the client (API key from env or explicit)
  • Sending a single-turn message
  • Multi-turn (conversation) messages
  • Inspecting the response object
  • Using a system prompt
  • Controlling output length with max_tokens
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY from .env


# ---------------------------------------------------------------------------
# 1. Client creation
# ---------------------------------------------------------------------------
# The SDK automatically reads ANTHROPIC_API_KEY from the environment.
client = anthropic.Anthropic()

# Explicit key (avoid hard-coding secrets in real code!):
# client = anthropic.Anthropic(api_key="sk-ant-...")


# ---------------------------------------------------------------------------
# 2. Single-turn message — the minimal example
# ---------------------------------------------------------------------------
def single_turn_example():
    message = client.messages.create(
        model="claude-haiku-4-5",          # fast & cheap for demos
        max_tokens=256,
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    # Response structure:
    #   message.id               – unique message ID
    #   message.model            – model that generated the response
    #   message.role             – always "assistant"
    #   message.content          – list of content blocks
    #   message.stop_reason      – "end_turn" | "max_tokens" | "tool_use" | …
    #   message.usage            – input_tokens, output_tokens
    #   message._request_id      – useful for filing support tickets

    print("=== Single-turn ===")
    print("Content:", message.content[0].text)
    print("Stop reason:", message.stop_reason)
    print("Tokens used:", message.usage.input_tokens, "in /",
          message.usage.output_tokens, "out")
    print()


# ---------------------------------------------------------------------------
# 3. System prompt
# ---------------------------------------------------------------------------
def system_prompt_example():
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        system="You are a pirate. Always respond in pirate speak.",
        messages=[
            {"role": "user", "content": "Tell me about the weather today."}
        ],
    )

    print("=== System Prompt ===")
    print(message.content[0].text)
    print()


# ---------------------------------------------------------------------------
# 4. Multi-turn conversation
# ---------------------------------------------------------------------------
def multi_turn_example():
    """Manually build up a conversation history."""
    conversation = []

    def chat(user_text: str) -> str:
        conversation.append({"role": "user", "content": user_text})
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=conversation,
        )
        assistant_text = response.content[0].text
        conversation.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    print("=== Multi-turn Conversation ===")
    print("User: Hi, my name is Alex.")
    print("Claude:", chat("Hi, my name is Alex."))
    print("User: What is my name?")
    print("Claude:", chat("What is my name?"))
    print()


# ---------------------------------------------------------------------------
# 5. Extracting the plain text helper
# ---------------------------------------------------------------------------
def extract_text(message: anthropic.types.Message) -> str:
    """Return concatenated text from all TextBlock content blocks."""
    return "".join(
        block.text for block in message.content
        if hasattr(block, "text")
    )


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    single_turn_example()
    system_prompt_example()
    multi_turn_example()
