# Phase 3 — The ReAct Loop Engine

## Overview

The ReAct (Reasoning + Acting) loop is the heart of the agent. Claude receives a system prompt that instructs it to produce **only** `Thought:` / `Action:` / `Action Input:` / `Observation:` blocks. The loop parser reads each response, dispatches the requested tool, injects the result as `Observation:`, and continues until `Final Answer:` is reached.

---

## 1. ReActParser — Complete Implementation

```python
# app/services/react_parser.py
from __future__ import annotations
import json
import re
from typing import Any
from app.domain.analysis_models import ReActStep


class ReActParser:
    THOUGHT_PATTERN = re.compile(
        r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    ACTION_PATTERN = re.compile(r"\nAction:\s*([a-zA-Z0-9_]+)", re.IGNORECASE)
    ACTION_INPUT_PATTERN = re.compile(r"\nAction Input:\s*(\{.+?\})", re.DOTALL)
    FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)

    def parse(self, text: str) -> ReActStep:
        thought = ""
        thought_match = self.THOUGHT_PATTERN.search(text)
        if thought_match:
            thought = thought_match.group(1).strip()

        final_match = self.FINAL_ANSWER_PATTERN.search(text)
        if final_match:
            return ReActStep(
                thought=thought,
                final_answer=final_match.group(1).strip(),
                raw_text=text,
            )

        action_match = self.ACTION_PATTERN.search(text)
        if not action_match:
            return ReActStep(thought=thought, raw_text=text)

        action = action_match.group(1).strip()
        action_input: dict[str, Any] = {}
        input_match = self.ACTION_INPUT_PATTERN.search(text)
        if input_match:
            raw_json = input_match.group(1).strip()
            try:
                action_input = json.loads(raw_json)
            except json.JSONDecodeError:
                action_input = {"_raw": raw_json, "_parse_error": "invalid JSON"}

        return ReActStep(
            thought=thought,
            action=action,
            action_input=action_input,
            raw_text=text,
        )

    def is_final(self, text: str) -> bool:
        return bool(self.FINAL_ANSWER_PATTERN.search(text))

    def extract_final_answer(self, text: str) -> str:
        match = self.FINAL_ANSWER_PATTERN.search(text)
        return match.group(1).strip() if match else ""

    def has_action(self, text: str) -> bool:
        return bool(self.ACTION_PATTERN.search(text))
```

---

## 2. PhysicalContextInjector — Complete Implementation

```python
# app/services/context_injector.py
from __future__ import annotations
import json
from app.domain.analysis_models import AnalysisSession
from app.services.knowledge_tools import list_datasets, list_domain_documents

TOOL_LIST_TEXT = """\
Knowledge tools:
  - list_domain_documents  -> no input required
  - read_domain_document   -> {"file_name": "..."}
  - search_domain_knowledge -> {"query": "..."}
  - list_datasets          -> no input required
  - inspect_dataset        -> {"file_name": "..."}
  - describe_columns       -> {"file_name": "...", "columns": [...]}

Execution tools:
  - execute_python_code    -> {"code": "..."}
  - get_execution_variables -> no input required
  - list_figures           -> no input required
  - get_figure             -> {"figure_id": "fig_000"}
  - export_notebook        -> {"title": "..."}
  - save_figure            -> {"figure_id": "fig_000", "filename": "..."}

Validation tools:
  - validate_physical_units -> {"quantity_name": "...", "value": 0.0, "unit": "...", "domain_key": "..."}
  - convert_units          -> {"value": 0.0, "from_unit": "...", "to_unit": "..."}
  - check_magnitude        -> {"value": 0.0, "unit": "...", "domain_key": "..."}"""

REACT_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Data Scientist AI Agent with deep knowledge of {domain_context}.

You analyse datasets using a structured Thought/Action/Observation loop.
You MUST follow this EXACT format:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param": "value"}}

Or for the final answer:

Final Answer: [Complete answer with units and physical interpretation]

## Rules
1. Always start with Thought:.
2. Action must be a single tool name from the list below.
3. Action Input must be valid JSON (double-quoted keys, no trailing commas).
4. After every Observation, continue with Thought: or write Final Answer:.
5. Never skip Action Input, even for tools with no arguments (use {{}}).
6. Validate key numerical results with validate_physical_units before Final Answer.
7. Include units in every numerical answer.
8. Flag physically implausible values explicitly.

## Available Tools
{tool_list}

## Domain Context
{domain_summary}

## Available Datasets
{dataset_summary}

## Physical Constraints
- Thermal efficiency: 25-50% for real power plants
- Temperatures must include units (degC or K)
- Efficiency > 100% or < 0% indicates a calculation error

Begin your analysis now."""


class PhysicalContextInjector:
    def __init__(self, domain_context: str = "power plant thermodynamics"):
        self.domain_context = domain_context

    async def build_system_prompt(self, session: AnalysisSession) -> str:
        data_dir = session.data_dir

        try:
            docs = json.loads(list_domain_documents(data_dir=data_dir))
            if docs:
                domain_summary = (
                    "Available domain knowledge documents:\n"
                    + "\n".join(f"  - {d}" for d in docs)
                    + "\n\nRead these first to understand physical constraints."
                )
            else:
                domain_summary = "No domain documents available."
        except Exception:
            domain_summary = "Domain documents could not be listed."

        try:
            datasets = json.loads(list_datasets(data_dir=data_dir))
            if datasets:
                dataset_summary = (
                    "Available datasets (use inspect_dataset for full schema):\n"
                    + "\n".join(
                        f"  - {d['file_name']} ({d['format'].upper()}, {d['size_bytes']//1024} KB)"
                        for d in datasets
                    )
                )
            else:
                dataset_summary = "No datasets found in data/datasets/."
        except Exception:
            dataset_summary = "Datasets could not be listed."

        return REACT_SYSTEM_PROMPT_TEMPLATE.format(
            domain_context=self.domain_context,
            tool_list=TOOL_LIST_TEXT,
            domain_summary=domain_summary,
            dataset_summary=dataset_summary,
        )
```

---

## 3. DataScienceAgentService — Complete Implementation

```python
# app/services/data_agent.py
from __future__ import annotations
import json
import logging
from typing import Any
import anthropic
from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, ReActStep
from app.domain.exceptions import CodeExecutionError, ReActLoopError
from app.services.context_injector import PhysicalContextInjector
from app.services.react_parser import ReActParser
from app.services.knowledge_tools import (
    list_domain_documents, list_datasets, read_domain_document,
    search_domain_knowledge, inspect_dataset, describe_columns,
)
from app.services.data_tools import (
    execute_python_code, get_execution_variables, get_figure,
    list_figures, validate_physical_units_tool, convert_units_tool,
    check_magnitude_tool, export_notebook, save_figure,
)

logger = logging.getLogger(__name__)
_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


class AnalysisResponse:
    def __init__(self, response: str, session: AnalysisSession):
        self.response = response
        self.session_id = session.session_id
        self.status = session.status
        self.react_trace = session.react_trace
        self.figures = session.figure_ids
        self.notebook_available = session.notebook_available
        self.validated_units = [u.to_dict() for u in session.validated_units]

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "session_id": self.session_id,
            "status": self.status,
            "react_trace": self.react_trace,
            "figures": self.figures,
            "notebook_available": self.notebook_available,
            "validated_units": self.validated_units,
        }


def _extract_text(response: anthropic.types.Message) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


class DataScienceAgentService:
    """
    Orchestrates the text-based ReAct loop.

    The session object carries all state: messages, figures, react_trace,
    validated_units, notebook_path. The service itself is stateless.
    """

    def __init__(self, domain_context: str = "power plant thermodynamics"):
        self.parser = ReActParser()
        self.context_injector = PhysicalContextInjector(domain_context=domain_context)
        self.max_iterations = settings.max_react_iterations

    async def run(self, session: AnalysisSession) -> AnalysisResponse:
        """
        Execute the ReAct loop.

        Mutates session in-place (adds messages, react_trace entries, figures).
        Returns AnalysisResponse when Final Answer is reached.
        Raises ReActLoopError if MAX_REACT_ITERATIONS is exceeded.
        """
        system_prompt = await self.context_injector.build_system_prompt(session)
        logger.info("Starting ReAct loop session=%s max=%d", session.session_id, self.max_iterations)

        for iteration in range(self.max_iterations):
            logger.debug("Iteration %d/%d", iteration + 1, self.max_iterations)

            # ── Call Claude (NO tools= parameter — text-based ReAct) ───────
            try:
                api_response = await _client.messages.create(
                    model=settings.claude_model,
                    system=system_prompt,
                    max_tokens=settings.max_tokens,
                    messages=session.messages,
                )
            except anthropic.APIStatusError as e:
                session.mark_error(str(e))
                raise

            text = _extract_text(api_response)
            logger.debug("Response[%d]: %s...", iteration + 1, text[:150])

            # ── Parse ──────────────────────────────────────────────────────
            step = self.parser.parse(text)

            # ── Final Answer ───────────────────────────────────────────────
            if step.is_final:
                logger.info("Final Answer at iteration %d", iteration + 1)
                await self._validate_final_answer(step.final_answer, session)
                session.add_assistant_message(text)
                session.mark_complete(step.final_answer)
                return AnalysisResponse(response=step.final_answer, session=session)

            # ── Action ─────────────────────────────────────────────────────
            if step.has_action:
                observation = await self._dispatch(step.action, step.action_input, session)
                logger.debug("Tool %s -> %s...", step.action, observation[:80])

                session.add_react_step(step, observation=observation)
                session.add_assistant_message(text)
                session.add_user_message(f"Observation: {observation}")
                continue

            # ── Malformed — nudge Claude ───────────────────────────────────
            logger.warning("Iteration %d: malformed response, nudging", iteration + 1)
            session.add_assistant_message(text)
            session.add_user_message(
                "Your response did not contain a valid Action: or Final Answer: block. "
                "Please continue:\n\nThought: [reasoning]\nAction: [tool_name]\n"
                "Action Input: {}\n\nor:\n\nFinal Answer: [answer]"
            )

        # ── Limit exceeded ─────────────────────────────────────────────────
        last_thought = session.react_trace[-1].get("thought", "") if session.react_trace else ""
        session.mark_error("Exceeded MAX_REACT_ITERATIONS")
        raise ReActLoopError(
            "Exceeded MAX_REACT_ITERATIONS without Final Answer",
            iterations=self.max_iterations,
            last_thought=last_thought,
        )

    async def _dispatch(
        self,
        action: str,
        action_input: dict[str, Any],
        session: AnalysisSession,
    ) -> str:
        """Route action name to tool function. Returns observation string."""
        data_dir = session.data_dir
        try:
            match action:
                case "list_domain_documents":
                    return list_domain_documents(data_dir=data_dir)
                case "read_domain_document":
                    fn = action_input.get("file_name", "")
                    return read_domain_document(fn, data_dir=data_dir) if fn else "ERROR: file_name required"
                case "search_domain_knowledge":
                    q = action_input.get("query", "")
                    top_k = int(action_input.get("top_k", 3))
                    return search_domain_knowledge(q, data_dir=data_dir, top_k=top_k)
                case "list_datasets":
                    return list_datasets(data_dir=data_dir)
                case "inspect_dataset":
                    fn = action_input.get("file_name", "")
                    return inspect_dataset(fn, data_dir=data_dir) if fn else "ERROR: file_name required"
                case "describe_columns":
                    fn = action_input.get("file_name", "")
                    cols = action_input.get("columns", [])
                    return describe_columns(fn, cols, data_dir=data_dir)
                case "execute_python_code":
                    code = action_input.get("code", "")
                    return execute_python_code(code, session) if code else "ERROR: code required"
                case "get_execution_variables":
                    return get_execution_variables(session)
                case "list_figures":
                    return list_figures(session)
                case "get_figure":
                    fid = action_input.get("figure_id", "")
                    return get_figure(fid, session)
                case "export_notebook":
                    title = action_input.get("title", "Analysis")
                    return export_notebook(title, session)
                case "save_figure":
                    fid = action_input.get("figure_id", "")
                    fname = action_input.get("filename", "figure")
                    return save_figure(fid, fname, session)
                case "validate_physical_units":
                    return validate_physical_units_tool(
                        quantity_name=action_input.get("quantity_name", "unknown"),
                        value=float(action_input.get("value", 0)),
                        unit=action_input.get("unit", ""),
                        domain_key=action_input.get("domain_key"),
                        session=session,
                    )
                case "convert_units":
                    return convert_units_tool(
                        float(action_input.get("value", 0)),
                        action_input.get("from_unit", ""),
                        action_input.get("to_unit", ""),
                    )
                case "check_magnitude":
                    return check_magnitude_tool(
                        float(action_input.get("value", 0)),
                        action_input.get("unit", ""),
                        action_input.get("domain_key", ""),
                    )
                case _:
                    tools = self._get_tool_names()
                    return f"ERROR: Unknown action '{action}'. Available: {', '.join(tools)}"
        except CodeExecutionError as e:
            return f"ERROR (code execution {e.backend}): {e}"
        except Exception as e:
            logger.exception("Tool %s raised unexpected error", action)
            return f"ERROR: {type(e).__name__}: {e}"

    async def _validate_final_answer(self, final_answer: str, session: AnalysisSession) -> None:
        """Log warnings for any physically invalid quantities in session."""
        invalid = [u for u in session.validated_units if not u.is_valid]
        if invalid:
            logger.warning(
                "Session %s: %d physically invalid quantities in Final Answer",
                session.session_id, len(invalid),
            )

    @staticmethod
    def _get_tool_names() -> list[str]:
        return [
            "list_domain_documents", "read_domain_document", "search_domain_knowledge",
            "list_datasets", "inspect_dataset", "describe_columns",
            "execute_python_code", "get_execution_variables", "list_figures",
            "get_figure", "export_notebook", "save_figure",
            "validate_physical_units", "convert_units", "check_magnitude",
        ]
```

---

## 4. Example Trace

A complete trace for "What is the mean thermal efficiency?":

```
[Iteration 1]
Claude: Thought: I should inspect the available datasets first.
        Action: list_datasets
        Action Input: {}

Observation: [{"file_name": "power_plant_data.csv", "format": "csv", "size_bytes": 40000}]

[Iteration 2]
Claude: Thought: I'll inspect the CSV to understand its columns.
        Action: inspect_dataset
        Action Input: {"file_name": "power_plant_data.csv"}

Observation: {"rows": 500, "columns": 7, "column_names": ["timestamp", "efficiency_pct", ...]}

[Iteration 3]
Claude: Thought: The column efficiency_pct contains the thermal efficiency. Let me compute the mean.
        Action: execute_python_code
        Action Input: {"code": "import pandas as pd\ndf = pd.read_csv('data/datasets/power_plant_data.csv')\nmean_eff = df['efficiency_pct'].mean()\nprint(f'Mean efficiency: {mean_eff:.2f}%')"}

Observation: Mean efficiency: 36.73%

[Iteration 4]
Claude: Thought: I should validate this value is physically plausible.
        Action: validate_physical_units
        Action Input: {"quantity_name": "thermal efficiency", "value": 36.73, "unit": "percent", "domain_key": "thermal_efficiency"}

Observation: {"is_valid": true, "warning": "", "expected_range": [25.0, 50.0]}

[Iteration 5]
Claude: Final Answer: The mean thermal efficiency of the power plant dataset is 36.73 percent.
        This value is physically plausible for a modern coal-fired power plant
        (expected range: 25-50%). The dataset contains 500 hourly measurements.
```

---

## 5. Testing Phase 3

```python
# tests/test_phase3_react.py
"""Tests for the ReAct parser and agent logic (mocked API calls)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.domain.analysis_models import AnalysisSession, ReActStep
from app.domain.exceptions import ReActLoopError
from app.services.react_parser import ReActParser
from app.services.data_agent import DataScienceAgentService, AnalysisResponse


class TestReActParser:
    def setup_method(self):
        self.parser = ReActParser()

    def test_parse_action_step(self):
        text = (
            "Thought: I need to list the datasets first.\n"
            "Action: list_datasets\n"
            "Action Input: {}"
        )
        step = self.parser.parse(text)
        assert step.thought == "I need to list the datasets first."
        assert step.action == "list_datasets"
        assert step.action_input == {}
        assert not step.is_final

    def test_parse_final_answer(self):
        text = (
            "Thought: I have enough information.\n"
            "Final Answer: The mean efficiency is 36.7 percent."
        )
        step = self.parser.parse(text)
        assert step.is_final
        assert "36.7 percent" in step.final_answer

    def test_parse_action_with_params(self):
        text = (
            "Thought: Let me inspect the dataset.\n"
            "Action: inspect_dataset\n"
            'Action Input: {"file_name": "power_plant_data.csv"}'
        )
        step = self.parser.parse(text)
        assert step.action == "inspect_dataset"
        assert step.action_input == {"file_name": "power_plant_data.csv"}

    def test_parse_invalid_json_input(self):
        text = (
            "Thought: Something.\n"
            "Action: inspect_dataset\n"
            "Action Input: {file_name: missing_quotes}"
        )
        step = self.parser.parse(text)
        assert "_parse_error" in step.action_input

    def test_parse_malformed_no_action_no_final(self):
        text = "Thought: I'm confused.\nI'll just think about it."
        step = self.parser.parse(text)
        assert not step.is_final
        assert not step.has_action

    def test_is_final_quick_check(self):
        assert self.parser.is_final("Final Answer: done")
        assert not self.parser.is_final("Thought: still thinking")


class TestDataScienceAgentService:
    """Integration tests with mocked Anthropic API."""

    def _make_mock_response(self, text: str):
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.content = [block]
        response.stop_reason = "end_turn"
        return response

    @pytest.mark.asyncio
    async def test_single_step_final_answer(self):
        """Agent that produces Final Answer on first iteration."""
        session = AnalysisSession.new("Simple question.", data_dir="data")
        service = DataScienceAgentService()

        final_text = "Thought: I know this.\nFinal Answer: The answer is 42."
        mock_response = self._make_mock_response(final_text)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with patch.object(service.context_injector, "build_system_prompt",
                              new=AsyncMock(return_value="System prompt")):
                result = await service.run(session)

        assert "42" in result.response
        assert result.status == "completed"
        assert result.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_two_step_trace(self):
        """Agent that calls one tool then gives Final Answer."""
        session = AnalysisSession.new("List datasets and summarise.", data_dir="data")
        service = DataScienceAgentService()

        step1_text = (
            "Thought: Let me list datasets.\n"
            "Action: list_datasets\n"
            "Action Input: {}"
        )
        step2_text = "Thought: OK.\nFinal Answer: There is 1 dataset available."

        responses = [
            self._make_mock_response(step1_text),
            self._make_mock_response(step2_text),
        ]

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(side_effect=responses)
            with patch.object(service.context_injector, "build_system_prompt",
                              new=AsyncMock(return_value="System prompt")):
                with patch("app.services.data_agent.list_datasets",
                           return_value='[{"file_name": "data.csv", "format": "csv", "size_bytes": 1000}]'):
                    result = await service.run(session)

        assert result.status == "completed"
        assert len(session.react_trace) == 1
        assert session.react_trace[0]["action"] == "list_datasets"

    @pytest.mark.asyncio
    async def test_exceeds_max_iterations(self):
        """Agent never produces Final Answer -> ReActLoopError."""
        session = AnalysisSession.new("Impossible task.", data_dir="data")
        service = DataScienceAgentService()
        service.max_iterations = 3

        malformed = self._make_mock_response("I'm just thinking endlessly...")

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=malformed)
            with patch.object(service.context_injector, "build_system_prompt",
                              new=AsyncMock(return_value="System prompt")):
                with pytest.raises(ReActLoopError) as exc_info:
                    await service.run(session)

        assert exc_info.value.iterations == 3
        assert session.status == "error"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_observation(self):
        """Unknown action name returns an error observation, loop continues."""
        session = AnalysisSession.new("Test.", data_dir="data")
        service = DataScienceAgentService()
        service.max_iterations = 2

        step1_text = (
            "Thought: Let me use a nonexistent tool.\n"
            "Action: nonexistent_tool\n"
            "Action Input: {}"
        )
        step2_text = "Thought: Understood.\nFinal Answer: Done."

        responses = [
            self._make_mock_response(step1_text),
            self._make_mock_response(step2_text),
        ]

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(side_effect=responses)
            with patch.object(service.context_injector, "build_system_prompt",
                              new=AsyncMock(return_value="System prompt")):
                result = await service.run(session)

        # The error observation should contain "Unknown action"
        assert any(
            "Unknown action" in step.get("observation", "")
            for step in session.react_trace
        )
        assert result.status == "completed"
```

```bash
pytest tests/test_phase3_react.py -v
```

---

## 6. Trace Storage

Each `add_react_step()` call appends to `session.react_trace`:

```python
[
    {
        "thought": "I need to list the datasets first.",
        "action": "list_datasets",
        "action_input": {},
        "observation": '[{"file_name": "power_plant_data.csv", ...}]',
        "final_answer": "",
    },
    {
        "thought": "Let me compute the mean efficiency.",
        "action": "execute_python_code",
        "action_input": {"code": "import pandas as pd\n..."},
        "observation": "Mean efficiency: 36.73%",
        "final_answer": "",
    },
]
```

This trace is returned in `AnalysisResponse.react_trace` and can be rendered in the UI for transparency.

---

## Checkpoint

After Phase 3:

```
app/services/react_parser.py       -- ReActParser (THOUGHT/ACTION/FINAL patterns)
app/services/context_injector.py   -- PhysicalContextInjector + REACT_SYSTEM_PROMPT_TEMPLATE
app/services/data_agent.py         -- DataScienceAgentService.run() + _dispatch() + _validate
tests/test_phase3_react.py         -- Parser + agent integration tests (mocked API)
```

-> Next: 05_phase4_validation.md -- Physical unit validation deep dive
