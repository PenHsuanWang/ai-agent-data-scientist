"""DataScienceAgentService — ReAct reasoning loop.

Implements the Thought → Action → Observation protocol:
1. Build system prompt with available tools and domain context.
2. Call Claude API with the full message history.
3. Parse Claude's text response for Thought/Action/Action Input OR Final Answer.
4. Dispatch Action to the tool registry.
5. Append Observation to the session and loop.
6. Terminate on Final Answer or MAX_REACT_ITERATIONS.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Any, Callable

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, PhysicalUnit
from app.domain.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMContextOverflowError,
    ReActLoopError,
    ReActParseError,
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
# ReAct regex patterns                                                   #
# ──────────────────────────────────────────────────────────────────── #

_THOUGHT_RE = re.compile(
    r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_RE = re.compile(r"Action:\s*([a-z_][a-z0-9_]*)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})\s*$", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)


def _parse_react(text: str) -> dict[str, Any]:
    """Parse a Claude response into a structured ReAct dict.

    Returns one of:
      {"type": "final_answer", "thought": str, "answer": str}
      {"type": "action", "thought": str, "action": str, "action_input": dict}
      {"type": "parse_error", "raw": str, "reason": str}
    """
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    # Final answer path
    final_m = _FINAL_ANSWER_RE.search(text)
    if final_m:
        return {
            "type": "final_answer",
            "thought": thought,
            "answer": final_m.group(1).strip(),
        }

    # Action path
    action_m = _ACTION_RE.search(text)
    if not action_m:
        # No action found — treat entire text as thought, signal parse error
        return {
            "type": "parse_error",
            "raw": text,
            "reason": "No 'Action:' or 'Final Answer:' found in Claude's response.",
        }

    action_name = action_m.group(1).strip()

    # Extract JSON input
    input_m = _ACTION_INPUT_RE.search(text)
    if input_m:
        raw_input = input_m.group(1).strip()
        try:
            action_input = json.loads(raw_input)
        except json.JSONDecodeError:
            # Fallback: try ast.literal_eval for Python-dict-style strings
            try:
                action_input = ast.literal_eval(raw_input)
            except Exception:
                return {
                    "type": "parse_error",
                    "raw": text,
                    "reason": f"Action Input is not valid JSON: {raw_input[:200]}",
                }
    else:
        action_input = {}

    return {
        "type": "action",
        "thought": thought,
        "action": action_name,
        "action_input": action_input,
    }


# ──────────────────────────────────────────────────────────────────── #
# System prompt builder                                                  #
# ──────────────────────────────────────────────────────────────────── #

_TOOL_SUMMARY = "\n".join(
    f"- **{t['name']}**: {t['description'].split('.')[0]}"
    for t in TOOL_DEFINITIONS
)

_SYSTEM_PROMPT_TEMPLATE = """You are an expert Data Scientist AI Agent. Your task is to answer questions about datasets by reasoning step-by-step and using the available tools.

## Available Tools

{tool_summary}

## Instructions

Always follow this exact format for EVERY response:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: {{"param": "value"}}

When you have enough information to answer the user's question:

Thought: <final reasoning>
Final Answer: <your complete, well-reasoned answer to the user>

## Workflow Rules

1. ALWAYS call get_coding_standards at the start of any analysis task.
2. ALWAYS read domain documents to understand physical context before analysing data.
3. ALWAYS validate physical quantities (efficiency, temperature, pressure, power) using validate_physical_units.
4. If a computed result is outside expected ranges, investigate before reporting it.
5. If a tool returns an error, read the message carefully and try to correct your approach.
6. Keep Action Input as valid JSON with double quotes.

## Code Generation Rules

7. Use print() in execute_python_code — never rely on expression evaluation.
8. Load datasets: pd.read_csv('data/datasets/<file>') or pd.read_parquet(...)
9. Use the pre-configured helpers from the style preamble: COLORS, PALETTE, C_GOOD, C_WARN, C_BAD,
   label_bars(), add_reference_line(), format_axis_units(), engineering_plot().
10. Every plot MUST have: xlabel with unit, ylabel with unit, title, plt.show().
11. Use descriptive variable names — no single letters except loop indices.
12. Add a section header print() before each analysis step.

## Physical Validation Reminder

A thermal efficiency > 100% violates the First Law of Thermodynamics.
A negative absolute temperature violates the Third Law.
Always check your results make physical sense before presenting them.

## Reporting Standard

End every Final Answer with a structured summary:
- Dataset, shape, key metric with units
- Physical validation status (✅ or ⚠)
- Any anomalies detected
- Recommendation if applicable
"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(tool_summary=_TOOL_SUMMARY)


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
    """Orchestrates the ReAct loop for one user turn.

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
        system_prompt: str,
    ) -> str:
        """Call the Anthropic API with structured error classification.

        The SDK-level ``max_retries`` setting already handles transient
        connection errors and 429 rate-limit responses.  This method adds
        application-level exception classification on top:

        - ``LLMAuthenticationError`` — fast-fail; no retry makes sense.
        - ``LLMContextOverflowError`` — 400 "prompt too long"; caller must
          inform the user to start a new session.
        - ``LLMAPIError`` — any other Anthropic API failure.
        """
        import anthropic as _ant

        try:
            response = await _client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_tokens,
                system=system_prompt,
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

        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text = block.text
                break
        return raw_text

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
        """Execute the ReAct loop for one user turn.

        Args:
            session: The AnalysisSession (mutated in place).
            user_message: The user's natural language request.

        Returns:
            The Final Answer text.

        Raises:
            ReActLoopError: If the loop exceeds MAX_REACT_ITERATIONS or encounters
                            unrecoverable parsing failures.
            LLMContextOverflowError: If the message history is too long.
            LLMAuthenticationError: If the API key is invalid.
            LLMAPIError: For any other Anthropic API failure.

        Note:
            On LLM API errors the session message history is rolled back to
            its state before this call, so the session remains consistent.
        """
        runner = self._get_runner(session)
        tool_registry = self._build_tool_registry(session, runner)
        system_prompt = _build_system_prompt()

        # Snapshot message history BEFORE appending anything (Gap 2).
        # Rolled back if an LLM API error is raised, keeping the session clean.
        checkpoint = len(session.messages)

        try:
            return await self._run_loop(
                session=session,
                user_message=user_message,
                tool_registry=tool_registry,
                system_prompt=system_prompt,
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
        system_prompt: str,
    ) -> str:
        """Inner ReAct loop (separated from run() to keep rollback logic clean)."""
        session.add_user_message(user_message)

        last_thought = ""
        parse_error_count = 0

        for iteration in range(settings.max_react_iterations):
            logger.debug(
                "ReAct iteration %d/%d (session=%s)",
                iteration + 1,
                settings.max_react_iterations,
                session.session_id,
            )

            # ── Call Claude (classified, retried by SDK) ───────────── #
            raw_text = await self._call_claude_with_retry(session, system_prompt)

            logger.debug("Claude raw response (iter %d): %.300s", iteration, raw_text)

            # ── Parse ReAct format ─────────────────────────────────── #
            parsed = _parse_react(raw_text)

            if parsed["type"] == "final_answer":
                final_answer = parsed["answer"]
                last_thought = parsed.get("thought", "")
                session.add_assistant_message(raw_text)
                session.append_react_step(
                    thought=last_thought,
                    action="Final Answer",
                    observation=final_answer,
                )
                logger.info(
                    "ReAct completed in %d iterations (session=%s)",
                    iteration + 1,
                    session.session_id,
                )
                return final_answer

            if parsed["type"] == "parse_error":
                parse_error_count += 1
                logger.warning(
                    "ReAct parse error (session=%s, attempt %d/3): %s",
                    session.session_id,
                    parse_error_count,
                    parsed["reason"],
                )
                # Record parse errors in the trace with a sentinel action (Gap 16)
                session.append_react_step(
                    thought=last_thought,
                    action="__parse_error__",
                    observation=(
                        f"Parse error {parse_error_count}/3: {parsed['reason']} "
                        f"| raw={parsed['raw'][:200]}"
                    ),
                )
                if parse_error_count >= 3:
                    raise ReActLoopError(
                        f"Repeated parse failures: {parsed['reason']}",
                        iterations=iteration + 1,
                        last_thought=last_thought,
                    )
                # Inject correction message
                correction = (
                    f"Your response did not follow the required format. "
                    f"Reason: {parsed['reason']}\n\n"
                    "Please respond using EXACTLY this format:\n"
                    "Thought: <your reasoning>\n"
                    "Action: <tool_name>\n"
                    'Action Input: {"param": "value"}\n\n'
                    "OR if you have the final answer:\n"
                    "Thought: <final reasoning>\n"
                    "Final Answer: <your answer>"
                )
                session.add_assistant_message(raw_text)
                session.add_user_message(correction)
                continue

            # ── Dispatch tool ──────────────────────────────────────── #
            thought = parsed.get("thought", "")
            action_name = parsed["action"]
            action_input = parsed.get("action_input", {})
            last_thought = thought

            logger.info(
                "Dispatching tool '%s' with input: %s (session=%s)",
                action_name,
                str(action_input)[:200],
                session.session_id,
            )

            handler = tool_registry.get(action_name)
            if handler is None:
                observation = (
                    f"Error: Unknown tool '{action_name}'. "
                    f"Available tools: {list(tool_registry.keys())}"
                )
            else:
                try:
                    observation = handler(action_input)
                except Exception as exc:
                    logger.error(
                        "Tool '%s' raised unexpectedly: %s",
                        action_name, exc, exc_info=True
                    )
                    observation = f"Error: Tool '{action_name}' failed — {exc}"

            # Truncate very long observations
            MAX_OBS = 8000
            if len(observation) > MAX_OBS:
                observation = observation[:MAX_OBS] + f"\n[...truncated at {MAX_OBS} chars]"

            session.append_react_step(
                thought=thought,
                action=f"{action_name}({json.dumps(action_input)})",
                observation=observation,
            )

            # Build the next assistant + user turn
            assistant_content = (
                f"Thought: {thought}\n"
                f"Action: {action_name}\n"
                f"Action Input: {json.dumps(action_input)}"
            )
            observation_content = f"Observation: {observation}"

            session.add_assistant_message(assistant_content)
            session.add_user_message(observation_content)

        raise ReActLoopError(
            f"Reached maximum iterations ({settings.max_react_iterations}) without a Final Answer.",
            iterations=settings.max_react_iterations,
            last_thought=last_thought,
        )

    def shutdown_session(self, session_id: str) -> None:
        """Release the CodeRunner for a given session."""
        runner = self._runners.pop(session_id, None)
        if runner:
            runner.shutdown()


# Module-level singleton
data_science_agent = DataScienceAgentService()
