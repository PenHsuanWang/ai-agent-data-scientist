"""DataScienceAgentService — Anthropic native Tool Calling loop.

Implements the tool-use loop driven by the Anthropic SDK:
1. Call Claude API with cached system prompt and TOOL_DEFINITIONS via tools= param.
2. If stop_reason == "tool_use": dispatch each tool_use block, collect tool_results.
3. Append tool_result user message and loop.
4. If stop_reason == "end_turn": return the text answer.
5. Apply a sliding window before each call to prevent context overflow.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable

from anthropic import AsyncAnthropic
from anthropic.types import Message

from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, PhysicalUnit
from app.domain.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMContextOverflowError,
    ReActLoopError,
)
from app.infrastructure.code_runner import CodeRunner, CodeRunnerFactory
from app.infrastructure.unit_registry import (
    check_magnitude,
    convert_units,
    validate_physical_units,
)
from app.services.data_tools import (
    execute_python_code,
    export_notebook_tool,
    get_execution_variables,
    get_figure,
    list_figures,
    save_figure_tool,
)
from app.services.knowledge_tools import (
    describe_columns,
    get_coding_standards,
    inspect_dataset,
    list_datasets,
    list_domain_documents,
    read_domain_document,
    search_domain_knowledge,
)
from app.services.tool_definitions import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────── #
# System prompt (cached — static across all calls)                       #
# ──────────────────────────────────────────────────────────────────── #

_SYSTEM_PROMPT_TEXT = """You are an expert Data Scientist AI Agent specialising in physical process analysis.

## Workflow Rules

1. ALWAYS call get_coding_standards at the start of any analysis task.
2. ALWAYS read domain documents to understand physical context before analysing data.
3. ALWAYS validate physical quantities (efficiency, temperature, pressure, power) using validate_physical_units.
4. If a computed result is outside expected ranges, investigate before reporting it.
5. If a tool returns an error, read the message carefully and try to correct your approach.

## Code Generation Rules

6. Use print() in execute_python_code — never rely on expression evaluation.
7. Load datasets: pd.read_csv('data/datasets/<file>') or pd.read_parquet(...)
8. Use the pre-configured helpers from the style preamble: COLORS, PALETTE, C_GOOD, C_WARN, C_BAD,
   label_bars(), add_reference_line(), format_axis_units(), engineering_plot().
9. Every plot MUST have: xlabel with unit, ylabel with unit, title, plt.show().
10. Use descriptive variable names — no single letters except loop indices.
11. Add a section header print() before each analysis step.

## Physical Validation Reminder

A thermal efficiency > 100% violates the First Law of Thermodynamics.
A negative absolute temperature violates the Third Law.
Always check your results make physical sense before presenting them.

## Reporting Standard

End every response with a structured summary:
- Dataset, shape, key metric with units
- Physical validation status (✅ or ⚠)
- Any anomalies detected
- Recommendation if applicable
"""

# Prompt caching: the static system block is marked ephemeral so Anthropic
# can reuse the KV cache across requests, reducing input-token cost and latency.
_CACHED_SYSTEM: list[dict[str, Any]] = [
    {
        "type": "text",
        "text": _SYSTEM_PROMPT_TEXT,
        "cache_control": {"type": "ephemeral"},
    }
]


def _build_cached_tools() -> list[dict[str, Any]]:
    """Return TOOL_DEFINITIONS with cache_control on the last entry.

    Placing cache_control on the final tool definition tells Anthropic to
    cache the entire tools prefix up to that point, eliminating repeated
    token processing for the static schema block.
    """
    tools: list[dict[str, Any]] = list(TOOL_DEFINITIONS)
    last = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools[:-1] + [last]


_CACHED_TOOLS: list[dict[str, Any]] = _build_cached_tools()

# ──────────────────────────────────────────────────────────────────── #
# Helpers                                                                #
# ──────────────────────────────────────────────────────────────────── #


def _content_to_dict(content_blocks: list) -> list[dict[str, Any]]:
    """Convert Anthropic SDK content blocks to serialisable dicts."""
    result: list[dict[str, Any]] = []
    for block in content_blocks:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input),
                }
            )
    return result


def _apply_sliding_window(
    messages: list[dict[str, Any]],
    max_total: int,
) -> list[dict[str, Any]]:
    """Trim message history to at most *max_total* entries.

    Oldest messages are dropped first.  After trimming the result is
    guaranteed to start with a user-role message so the Anthropic API
    never rejects the history due to a leading assistant turn.
    """
    if len(messages) <= max_total:
        return messages

    kept = messages[-max_total:]
    # Advance past any leading assistant turns.
    while kept and kept[0].get("role") != "user":
        kept = kept[1:]
    if not kept:
        kept = messages[-1:]

    logger.info(
        "Sliding window: %d → %d messages (max=%d)",
        len(messages),
        len(kept),
        max_total,
    )
    return kept


# ──────────────────────────────────────────────────────────────────── #
# Anthropic client                                                       #
# ──────────────────────────────────────────────────────────────────── #

_client = AsyncAnthropic(
    api_key=settings.anthropic_api_key.get_secret_value(),
    base_url=settings.anthropic_base_url,
    max_retries=settings.max_retries,
)


# ──────────────────────────────────────────────────────────────────── #
# DataScienceAgentService                                               #
# ──────────────────────────────────────────────────────────────────── #


class DataScienceAgentService:
    """Orchestrates the native tool-calling loop for one user turn.

    Maintains a per-session CodeRunner in a dict keyed by session_id.
    """

    def __init__(self) -> None:
        self._runners: dict[str, CodeRunner] = {}

    def _get_runner(self, session: AnalysisSession) -> CodeRunner:
        if session.session_id not in self._runners:
            self._runners[session.session_id] = CodeRunnerFactory.create(
                session_id=session.session_id
            )
        return self._runners[session.session_id]

    async def _call_claude_with_retry(
        self,
        session: AnalysisSession,
    ) -> Message:
        """Call the Anthropic API with cached system + tools and structured error classification.

        Applies the sliding window *before* the call to prevent context overflow.
        The SDK-level ``max_retries`` handles transient connection errors and 429s.
        Application-level exception mapping:

        - ``LLMAuthenticationError`` — fast-fail; no retry makes sense.
        - ``LLMContextOverflowError`` — 400 "prompt too long"; caller should
          inform the user to start a new session.
        - ``LLMAPIError`` — any other Anthropic API failure.
        """
        import anthropic as _ant

        # Proactive sliding window — trim before we hit the API limit.
        session.messages = _apply_sliding_window(
            session.messages, max_total=settings.max_context_messages
        )

        try:
            response = await _client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_tokens,
                system=_CACHED_SYSTEM,
                tools=_CACHED_TOOLS,
                messages=session.messages,
            )
        except _ant.AuthenticationError as exc:
            raise LLMAuthenticationError(
                f"Anthropic API key is invalid or revoked: {exc}"
            ) from exc
        except _ant.BadRequestError as exc:
            lower = str(exc).lower()
            if any(
                kw in lower
                for kw in (
                    "prompt is too long",
                    "context_length_exceeded",
                    "too many tokens",
                    "context window",
                    "max_tokens",
                )
            ):
                raise LLMContextOverflowError(
                    "Message history exceeds the model's context window. "
                    "Please start a new session to continue. "
                    f"Detail: {exc}"
                ) from exc
            raise LLMAPIError(f"Anthropic API bad request: {exc}", status_code=400) from exc
        except _ant.APIError as exc:
            status = getattr(exc, "status_code", None)
            raise LLMAPIError(
                f"Anthropic API error ({type(exc).__name__}): {exc}",
                status_code=status,
            ) from exc
        except Exception as exc:
            raise LLMAPIError(
                f"Unexpected error calling Claude API: {type(exc).__name__}: {exc}"
            ) from exc

        return response

    def _build_tool_registry(
        self, session: AnalysisSession, runner: CodeRunner
    ) -> dict[str, Callable[..., str]]:
        """Build the tool dispatch table for this session."""

        def _wrap_validate(inp: dict[str, Any]) -> str:
            res = validate_physical_units(inp["quantity"], inp["value"], inp["unit"])
            try:
                data = json.loads(res)
                session.log_unit_validation(
                    PhysicalUnit(
                        quantity=data.get("quantity", inp["quantity"]),
                        value=data.get("value", inp["value"]),
                        unit=data.get("unit", inp["unit"]),
                        is_valid=data.get("is_valid", False),
                        message=data.get("message", ""),
                        canonical_value=data.get("canonical_value"),
                        canonical_unit=data.get("canonical_unit"),
                    )
                )
            except Exception as exc:
                logger.error("Failed to log unit validation: %s", exc)
            return res

        def _wrap_check_magnitude(inp: dict[str, Any]) -> str:
            res = check_magnitude(inp["quantity"], inp["value"], inp["unit"])
            try:
                data = json.loads(res)
                session.log_unit_validation(
                    PhysicalUnit(
                        quantity=data.get("quantity", inp["quantity"]),
                        value=data.get("value", inp["value"]),
                        unit=data.get("unit", inp["unit"]),
                        is_valid=data.get("is_plausible", False),
                        message=data.get("message", ""),
                    )
                )
            except Exception as exc:
                logger.error("Failed to log magnitude check: %s", exc)
            return res

        return {
            # Knowledge
            "list_domain_documents": lambda _: list_domain_documents(),
            "read_domain_document": lambda inp: read_domain_document(inp["file_name"]),
            "search_domain_knowledge": lambda inp: search_domain_knowledge(
                inp["query"], inp.get("top_k", 3)
            ),
            "get_coding_standards": lambda _: get_coding_standards(),
            "list_datasets": lambda _: list_datasets(),
            "inspect_dataset": lambda inp: inspect_dataset(inp["file_name"]),
            "describe_columns": lambda inp: describe_columns(inp["file_name"], inp["columns"]),
            # Execution
            "execute_python_code": lambda inp: execute_python_code(
                inp["code"], session, runner
            ),
            "get_execution_variables": lambda _: get_execution_variables(runner),
            "get_figure": lambda inp: get_figure(inp["figure_id"], session),
            "list_figures": lambda _: list_figures(session),
            "export_notebook": lambda inp: export_notebook_tool(session, inp["title"]),
            "save_figure": lambda inp: save_figure_tool(
                session, inp["figure_id"], inp["filename"]
            ),
            # Validation
            "validate_physical_units": _wrap_validate,
            "convert_units": lambda inp: convert_units(
                inp["value"], inp["from_unit"], inp["to_unit"]
            ),
            "check_magnitude": _wrap_check_magnitude,
        }

    async def run(self, session: AnalysisSession, user_message: str) -> str:
        """Execute the tool-calling loop for one user turn.

        Args:
            session: The AnalysisSession (mutated in place).
            user_message: The user's natural language request.

        Returns:
            The final answer text from Claude.

        Raises:
            ReActLoopError: If the loop exceeds MAX_REACT_ITERATIONS or hits
                            an unexpected stop_reason.
            LLMContextOverflowError: If context cannot be recovered by sliding window.
            LLMAuthenticationError: If the API key is invalid.
            LLMAPIError: For any other Anthropic API failure.

        Note:
            On LLM API errors the session message history is rolled back to
            its state before this call, so the session remains consistent.
        """
        runner = self._get_runner(session)
        tool_registry = self._build_tool_registry(session, runner)

        # Snapshot message history BEFORE appending anything.
        # Rolled back if an LLM API error is raised, keeping the session clean.
        checkpoint = len(session.messages)

        try:
            return await self._run_loop(
                session=session,
                user_message=user_message,
                tool_registry=tool_registry,
            )
        except (LLMAPIError, LLMContextOverflowError, LLMAuthenticationError):
            # Roll back any partial messages added during this run
            if len(session.messages) > checkpoint:
                logger.warning(
                    "LLM API error — rolling back session %s messages from %d → %d",
                    session.session_id,
                    len(session.messages),
                    checkpoint,
                )
                session.messages = session.messages[:checkpoint]
            raise

    async def _run_loop(
        self,
        session: AnalysisSession,
        user_message: str,
        tool_registry: dict,
    ) -> str:
        """Inner native tool-calling loop (separated from run() for clean rollback)."""
        session.add_user_message(user_message)

        for iteration in range(settings.max_react_iterations):
            logger.debug(
                "Tool-use iteration %d/%d (session=%s)",
                iteration + 1,
                settings.max_react_iterations,
                session.session_id,
            )

            response = await self._call_claude_with_retry(session)

            # Serialise content blocks and persist as the assistant turn.
            assistant_content = _content_to_dict(response.content)
            session.add_assistant_message(assistant_content)

            # ── Terminal: Claude is done ───────────────────────────── #
            if response.stop_reason == "end_turn":
                answer = next(
                    (b["text"] for b in assistant_content if b.get("type") == "text"),
                    "",
                )
                session.append_react_step(
                    thought="", action="Final Answer", observation=answer
                )
                logger.info(
                    "Tool-use loop completed in %d iterations (session=%s)",
                    iteration + 1,
                    session.session_id,
                )
                return answer

            # ── Tool dispatch ──────────────────────────────────────── #
            if response.stop_reason == "tool_use":
                # Extract optional preceding thought text.
                thought = next(
                    (b["text"] for b in assistant_content if b.get("type") == "text"),
                    "",
                )
                tool_results: list[dict[str, Any]] = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    handler = tool_registry.get(block.name)
                    if handler is None:
                        observation = (
                            f"Error: Unknown tool '{block.name}'. "
                            f"Available tools: {list(tool_registry.keys())}"
                        )
                    else:
                        try:
                            observation = handler(dict(block.input))
                        except Exception as exc:
                            logger.error(
                                "Tool '%s' raised unexpectedly: %s",
                                block.name,
                                exc,
                                exc_info=True,
                            )
                            observation = f"Error: Tool '{block.name}' failed — {exc}"

                    MAX_OBS = 8000
                    if len(observation) > MAX_OBS:
                        observation = (
                            observation[:MAX_OBS] + f"\n[...truncated at {MAX_OBS} chars]"
                        )

                    logger.info(
                        "Dispatched tool '%s' with input: %s (session=%s)",
                        block.name,
                        str(dict(block.input))[:200],
                        session.session_id,
                    )
                    session.append_react_step(
                        thought=thought,
                        action=f"{block.name}({json.dumps(dict(block.input))})",
                        observation=observation,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": observation,
                        }
                    )

                session.add_tool_results(tool_results)
                continue

            # ── Unexpected stop_reason ─────────────────────────────── #
            logger.warning(
                "Unexpected stop_reason '%s' (session=%s, iteration=%d)",
                response.stop_reason,
                session.session_id,
                iteration + 1,
            )
            session.append_react_step(
                thought="",
                action="__unexpected_stop__",
                observation=f"stop_reason={response.stop_reason}",
            )
            raise ReActLoopError(
                f"Unexpected stop_reason: '{response.stop_reason}'",
                iterations=iteration + 1,
            )

        raise ReActLoopError(
            f"Reached maximum iterations ({settings.max_react_iterations}) without a Final Answer.",
            iterations=settings.max_react_iterations,
        )

    def shutdown_session(self, session_id: str) -> None:
        """Release the CodeRunner for a given session."""
        runner = self._runners.pop(session_id, None)
        if runner:
            runner.shutdown()


# Module-level singleton
data_science_agent = DataScienceAgentService()
