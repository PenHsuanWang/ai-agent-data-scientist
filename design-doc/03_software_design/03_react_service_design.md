# ReAct Service Design — DataScienceAgentService

**Document Version:** 1.0  
**Status:** Approved  
**Bounded Context:** Agent Reasoning — ReAct Protocol Implementation  

---

## 1. Responsibility

`DataScienceAgentService` is the brain of the Data Scientist Agent. It owns:

- **System prompt construction** via `PhysicalContextInjector` (reading domain docs, extracting units)
- **ReAct protocol execution**: the Thought → Action → Observation loop
- **Tool dispatch**: translating parsed action names to `ToolRegistry` calls
- **Observation formatting**: wrapping tool results in the `Observation:` block format
- **Final answer detection**: recognizing `Final Answer:` and applying the physical validation gate
- **ReAct trace recording**: storing each iteration on `AnalysisSession.react_trace`

`DataScienceAgentService` does **NOT** own:
- Code execution (delegated to `CodeRunner` via `data_tools.py`)
- Unit validation logic (delegated to `UnitRegistry`)
- HTTP handling (delegated to `api/v1/analysis.py`)
- Jupyter kernel lifecycle (delegated to `JupyterKernelManager`)

---

## 2. ReAct Protocol Specification

### 2.1 Text Format

The ReAct protocol uses a strict plaintext format. Claude is instructed via the system prompt to always produce output in this format:

```
Thought: <free-form reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON object matching tool's parameter schema>
```

Or, when the analysis is complete:

```
Thought: <final reasoning>
Final Answer: <human-readable conclusion for the user>
```

### 2.2 Regex Patterns

```python
import re

# Matches "Thought: ..." up to the next "Action:" or "Final Answer:"
THOUGHT_PATTERN = re.compile(
    r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)",
    re.DOTALL | re.IGNORECASE,
)

# Matches "Action: tool_name" — tool name is a single identifier
ACTION_PATTERN = re.compile(
    r"Action:\s*([a-z_][a-z0-9_]*)",
    re.IGNORECASE,
)

# Matches "Action Input: {...}" — greedy to handle nested JSON
ACTION_INPUT_PATTERN = re.compile(
    r"Action Input:\s*(\{.*?\})\s*$",
    re.DOTALL,
)

# Matches "Final Answer: ..." to end of string
FINAL_ANSWER_PATTERN = re.compile(
    r"Final Answer:\s*(.+)",
    re.DOTALL | re.IGNORECASE,
)
```

### 2.3 Edge Cases

| Situation | Handling |
|---|---|
| Claude outputs text before "Thought:" | Strip preamble, parse from first "Thought:" |
| Action Input is not valid JSON | Attempt `json.loads()`, on failure raise `ReActParseError` |
| Action name not in ToolRegistry | Return `"Error: Unknown tool '{name}'. Available: ..."` as Observation |
| Claude skips "Action:" and goes straight to "Final Answer:" | Accepted; skip tool dispatch |
| Multi-line Action Input JSON | `re.DOTALL` handles newlines inside `{}` |
| Claude uses single quotes in JSON | Attempt `ast.literal_eval()` as fallback before failing |
| "Thought:" missing entirely | Treat entire response as thought, continue with empty thought |

---

## 3. State Machine

The full finite state machine for one `run()` call:

```
                        ┌─────────────┐
                        │    START     │
                        │ (run called) │
                        └──────┬───────┘
                               │ build system prompt
                               │ add user message to session
                               ▼
                    ┌──────────────────────┐
           ┌───────►│  CALL_LLM            │◄──────────────┐
           │        │  Claude API request   │               │
           │        └──────────┬───────────┘               │
           │                   │                            │
           │         ┌─────────▼──────────┐                │
           │         │  PARSE_RESPONSE     │                │
           │         │  ReActParser.parse()│                │
           │         └────┬──────────┬─────┘                │
           │              │ OK        │ ParseError           │
           │              │           │                      │
           │    ┌─────────▼──┐    ┌───▼─────────────────┐  │
           │    │ HAS_FINAL   │    │ HANDLE_PARSE_ERROR  │  │
           │    │ ANSWER?     │    │ append error msg     │  │
           │    └──┬──────┬───┘    │ increment iteration  │  │
           │  YES  │      │ NO     └───────────────────────┘  │
           │       │      │                   │ continue      │
           │  ┌────▼───┐  │           ┌───────▼──────┐       │
           │  │VALIDAT-│  │           │ DISPATCH_TOOL │       │
           │  │ION_GATE│  │           │ ToolRegistry  │       │
           │  └────┬───┘  │           │ .dispatch()   │       │
           │  PASS │FAIL   │           └───────┬───────┘       │
           │       │   ┌───▼──────┐            │ result        │
           │  ┌────▼─┐ │APPEND_RE-│    ┌───────▼──────────┐   │
           │  │RETURN│ │ACT_STEP  │    │ FORMAT_OBSERVATION│   │
           │  │result│ │+ continue│    │ "Observation: ..." │   │
           │  └──────┘ └──────────┘    └───────┬──────────┘   │
           │                                    │               │
           │                          ┌─────────▼─────────┐    │
           │                          │ append to messages  │────┘
           │                          │ check iterations   │
           │                          └─────────┬──────────┘
           │                                    │ < MAX
           │                          ┌─────────▼──────────┐
           └──────────────────────────│ MAX_ITERS EXCEEDED? │
                                      │ raise              │
                                      │ReActMaxItersError  │
                                      └────────────────────┘
```

---

## 4. PhysicalContextInjector

`PhysicalContextInjector` runs once at service startup (or lazily on first request) to build the system prompt.

### 4.1 Class Design

```python
class PhysicalContextInjector:
    """
    Reads all domain knowledge documents and extracts:
    - Unit definitions (regex-based)
    - Domain-specific quantity ranges
    - Terminology for the system prompt

    The resulting system prompt is cached in memory.
    """
    
    def __init__(self, docs_dir: Path) -> None:
        self._docs_dir = docs_dir
        self._cached_prompt: str | None = None

    def build_system_prompt(self) -> str:
        """
        Returns the cached system prompt, building it on first call.
        Thread-safe via Python GIL (single-process assumption for MVP).
        """
        if self._cached_prompt is None:
            self._cached_prompt = self._build()
        return self._cached_prompt

    def _build(self) -> str:
        doc_summaries = self._load_and_summarize_docs()
        unit_definitions = self._extract_unit_definitions(doc_summaries)
        return _SYSTEM_PROMPT_TEMPLATE.format(
            domain_context=doc_summaries,
            unit_definitions=unit_definitions,
        )

    def _load_and_summarize_docs(self) -> str:
        """
        Reads up to MAX_DOCS (10) domain documents.
        Truncates each to MAX_DOC_CHARS (4000) characters.
        Returns concatenated summaries.
        """
        MAX_DOCS = 10
        MAX_DOC_CHARS = 4000
        parts = []
        docs = sorted(self._docs_dir.glob("*"))[:MAX_DOCS]
        for doc in docs:
            if doc.suffix.lower() in {".md", ".txt", ".rst"}:
                try:
                    text = doc.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"--- {doc.name} ---\n{text[:MAX_DOC_CHARS]}")
                except OSError:
                    pass
        return "\n\n".join(parts)

    def _extract_unit_definitions(self, text: str) -> str:
        """
        Extracts lines matching unit definition patterns, e.g.:
        - "MW: megawatt, unit of power"
        - "η (eta): thermal efficiency, dimensionless 0–1"
        - "°C: degrees Celsius, temperature"
        """
        import re
        patterns = [
            re.compile(r"^[-*]\s*\w+\s*(?:\([^)]+\))?\s*:\s*.+", re.MULTILINE),
            re.compile(r"\b(?:unit|measured in|expressed in|in units of)\b.{5,60}", re.IGNORECASE),
        ]
        definitions = set()
        for p in patterns:
            for m in p.finditer(text):
                definitions.add(m.group(0).strip())
        return "\n".join(sorted(definitions)[:50])  # cap at 50 definitions
```

### 4.2 System Prompt Template

```python
_SYSTEM_PROMPT_TEMPLATE = """You are an expert Data Scientist AI assistant with deep knowledge \
of physical systems, engineering, and data analysis.

## Your Capabilities
You can analyze datasets, write and execute Python code, generate visualizations, \
and validate results against physical laws.

## ReAct Protocol
You MUST always follow this exact format:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"key": "value"}}

When analysis is complete, use:
Thought: [Final reasoning]
Final Answer: [Your conclusion for the user]

## Available Tools
{tool_list}

## Domain Knowledge Context
{domain_context}

## Known Unit Definitions
{unit_definitions}

## Physical Validation Rules
- Efficiency values must be between 0% and 100%
- Temperature in Kelvin must be > 0
- Always validate computed values against known physical ranges before reporting
- If a computed result seems physically unreasonable, investigate before concluding
"""
```

---

## 5. ReActParser

### 5.1 Class Design

```python
import ast
import json
import re
from dataclasses import dataclass

from app.domain.exceptions import ReActParseError


@dataclass
class ParsedReActResponse:
    """Result of parsing one LLM response in the ReAct loop."""
    thought: str
    is_final_answer: bool
    final_answer: str              # non-empty if is_final_answer=True
    action_name: str               # non-empty if is_final_answer=False
    action_input: dict             # parsed JSON input for action
    raw_response: str


class ReActParser:
    """
    Parses Claude's text output into structured ReAct components.

    Stateless — all methods are pure functions operating on the input string.
    """

    def parse_response(self, response: str, iteration: int) -> ParsedReActResponse:
        """
        Main entry point. Raises ReActParseError if response is malformed
        beyond recovery.
        """
        thought = self.extract_thought(response)
        
        # Check for Final Answer first
        final_match = FINAL_ANSWER_PATTERN.search(response)
        if final_match:
            return ParsedReActResponse(
                thought=thought,
                is_final_answer=True,
                final_answer=final_match.group(1).strip(),
                action_name="",
                action_input={},
                raw_response=response,
            )

        # Must have Action + Action Input
        action_name = self.extract_action(response, iteration)
        action_input = self.extract_action_input(response, iteration)
        
        return ParsedReActResponse(
            thought=thought,
            is_final_answer=False,
            final_answer="",
            action_name=action_name,
            action_input=action_input,
            raw_response=response,
        )

    def extract_thought(self, response: str) -> str:
        """Extracts thought text. Returns empty string if not found."""
        m = THOUGHT_PATTERN.search(response)
        if m:
            return m.group(1).strip()
        return ""

    def extract_action(self, response: str, iteration: int) -> str:
        """
        Extracts action tool name. Raises ReActParseError if not found.
        """
        m = ACTION_PATTERN.search(response)
        if not m:
            raise ReActParseError(response, iteration)
        return m.group(1).strip().lower()

    def extract_action_input(self, response: str, iteration: int) -> dict:
        """
        Extracts and parses Action Input JSON.
        Tries json.loads() first, then ast.literal_eval() as fallback.
        Raises ReActParseError if neither succeeds.
        """
        m = ACTION_INPUT_PATTERN.search(response)
        if not m:
            # No braces found: return empty dict (some tools take no args)
            if "Action Input:" in response and "{" not in response:
                return {}
            raise ReActParseError(response, iteration)
        
        raw = m.group(1).strip()
        
        # Try standard JSON
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        # Fallback: ast.literal_eval (handles single-quoted strings)
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
        
        raise ReActParseError(response, iteration)

    def handle_malformed(self, response: str, iteration: int) -> str:
        """
        Called when parse_response raises ReActParseError.
        Returns an Observation string that nudges Claude back to format.
        """
        return (
            "Observation: Your previous response was not in the correct ReAct format. "
            "You MUST use exactly:\n"
            "Thought: [reasoning]\n"
            "Action: [tool_name]\n"
            'Action Input: {"key": "value"}\n'
            "OR:\n"
            "Thought: [reasoning]\n"
            "Final Answer: [answer]\n"
            "Please try again."
        )
```

---

## 6. The run() Method — Annotated Walkthrough

```python
async def run(self, session: AnalysisSession) -> str:
    """
    Executes the ReAct loop for the given session.
    
    Precondition: session.messages already has the latest user message appended.
    Postcondition: Returns the final answer string, session.react_trace is updated.
    
    Raises:
        ReActMaxIterationsError: If MAX_REACT_ITERATIONS exceeded.
        ReActLoopError:          For unrecoverable loop failures.
    """
    # Step 1: Build system prompt (cached after first call)
    system_prompt = self._context_injector.build_system_prompt()
    
    # Step 2: Inject tool list into prompt
    tool_names = "\n".join(f"- {name}" for name in TOOL_REGISTRY.keys())
    full_system = system_prompt.format(tool_list=tool_names, ...)

    for iteration in range(self._max_iterations):
        # Step 3: Call Claude API in text mode (NOT tool-use mode)
        response_text = await self._call_claude(session, full_system)
        
        # Step 4: Parse the response
        try:
            parsed = self._parser.parse_response(response_text, iteration)
        except ReActParseError:
            # Recovery: append format error as observation, continue loop
            error_obs = self._parser.handle_malformed(response_text, iteration)
            session.add_user_message(error_obs)
            session.append_react_step(
                thought="[parse error]",
                action="[malformed]",
                observation=error_obs,
            )
            continue
        
        # Step 5: Handle Final Answer
        if parsed.is_final_answer:
            # Physical validation gate
            validated_answer = await self._validate_final_answer(
                parsed.final_answer, session
            )
            session.append_react_step(
                thought=parsed.thought,
                action="Final Answer",
                observation=validated_answer,
            )
            return validated_answer
        
        # Step 6: Dispatch tool
        observation = self._dispatch_tool(
            parsed.action_name, parsed.action_input, session
        )
        
        # Step 7: Record trace and append observation to message history
        session.append_react_step(
            thought=parsed.thought,
            action=f"{parsed.action_name}({json.dumps(parsed.action_input)})",
            observation=observation,
        )
        observation_message = f"Observation: {observation}"
        session.add_user_message(observation_message)
    
    # Step 8: Max iterations exceeded
    last_thought = session.react_trace[-1]["thought"] if session.react_trace else ""
    raise ReActMaxIterationsError(self._max_iterations, last_thought)
```

---

## 7. Tool Dispatch Integration

```python
def _dispatch_tool(
    self,
    action_name: str,
    action_input: dict,
    session: AnalysisSession,
) -> str:
    """
    Looks up action_name in TOOL_REGISTRY and calls the function.
    Tools that need session context receive it via a thread-local or closure.

    Returns the tool's string result (tools never raise; they return "Error: ...").
    """
    if action_name not in TOOL_REGISTRY:
        available = ", ".join(TOOL_REGISTRY.keys())
        return f"Error: Unknown tool '{action_name}'. Available tools: {available}"
    
    tool_fn = TOOL_REGISTRY[action_name]
    
    # Inject session for tools that need it (data_tools, output_tools)
    if _needs_session(tool_fn):
        return tool_fn(session=session, **action_input)
    else:
        return tool_fn(**action_input)
```

**Tool Registry Structure:**

```python
# All tool functions are registered here
TOOL_REGISTRY: dict[str, Callable] = {
    # Knowledge group
    "list_domain_documents":   list_domain_documents,
    "read_domain_document":    read_domain_document,
    "search_domain_knowledge": search_domain_knowledge,
    "list_datasets":           list_datasets,
    "inspect_dataset":         inspect_dataset,
    "describe_columns":        describe_columns,
    # Data/code group
    "execute_python_code":     execute_python_code,     # needs session
    "get_execution_variables": get_execution_variables, # needs session
    "get_figure":              get_figure,               # needs session
    "list_figures":            list_figures,             # needs session
    # Physical validation group
    "validate_physical_units": validate_physical_units,
    "convert_units":           convert_units,
    "check_magnitude":         check_magnitude,
    # Output group
    "export_notebook":         export_notebook,          # needs session
    "save_figure":             save_figure,              # needs session
}

_SESSION_TOOLS = {
    "execute_python_code", "get_execution_variables", "get_figure",
    "list_figures", "export_notebook", "save_figure",
}

def _needs_session(fn: Callable) -> bool:
    import inspect
    return "session" in inspect.signature(fn).parameters
```

---

## 8. Observation Formatting

Tool results are always strings (by design). The observation is wrapped:

```python
def _format_observation(self, tool_name: str, result: str) -> str:
    """
    Wraps a tool result in an Observation block.
    Truncates very long results to avoid overflowing context window.
    """
    MAX_OBS_CHARS = 8000
    if len(result) > MAX_OBS_CHARS:
        result = result[:MAX_OBS_CHARS] + f"\n... [truncated, {len(result)} total chars]"
    return f"Observation: {result}"
```

**Example observations in the message history:**

```
Observation: {"file_name": "power_plant.csv", "shape": [1000, 8], "dtypes": {...}}

Observation: {"stdout": "count    1000\nmean     0.423\n...", "success": true, "figures": []}

Observation: {"quantity": "thermal_efficiency", "value": 0.423, "is_valid": true, "message": "OK"}

Observation: Error: Dataset 'unknown.csv' is not loaded in this session.
```

---

## 9. Final Answer Detection and Validation Gate

```python
async def _validate_final_answer(
    self, final_answer: str, session: AnalysisSession
) -> str:
    """
    Scans the final answer for numeric values with units and validates them.
    If any value is physically implausible, appends a warning to the answer.
    
    This gate is non-optional: every final answer passes through here.
    """
    import re
    
    # Simple heuristic: find patterns like "42.3%", "450 MW", "1500 °C"
    # and validate each one.
    value_unit_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(°C|°F|K|MW|GW|kW|bar|kPa|MPa|%|m/s|kg/s|mol|pH)",
        re.IGNORECASE,
    )
    
    warnings = []
    for match in value_unit_pattern.finditer(final_answer):
        value_str, unit = match.group(1), match.group(2)
        value = float(value_str)
        
        # Quick hard-constraint checks (no domain range needed)
        if unit in {"%"} and value > 100:
            warnings.append(
                f"⚠ Physical warning: {value}{unit} exceeds 100% — "
                f"check if this is a percentage or a fraction."
            )
        elif unit == "K" and value <= 0:
            warnings.append(
                f"⚠ Physical warning: {value}{unit} is at or below absolute zero."
            )
        elif unit in {"°C"} and value < -273.15:
            warnings.append(
                f"⚠ Physical warning: {value}{unit} is below absolute zero."
            )
    
    if warnings:
        warning_block = "\n\n**Physical Validation Warnings:**\n" + "\n".join(warnings)
        return final_answer + warning_block
    
    return final_answer
```

---

## 10. Error Recovery

| Error Condition | Recovery Action |
|---|---|
| `ReActParseError` | Append format correction as Observation, continue loop, decrement effective iterations |
| Unknown tool name | Return `"Error: Unknown tool..."` as Observation, Claude can self-correct |
| Tool returns `"Error: ..."` | Observation carries error text; Claude can try a different approach |
| `CodeTimeoutError` | Tool catches it and returns `"Error: Code execution timed out after 30s"` |
| `CodeSecurityError` | Tool returns `"Error: Code security check failed: ..."` |
| `ReActMaxIterationsError` | Propagates to API handler; 500 response with `react_trace` for debugging |

---

## 11. Iteration Guard

```python
MAX_REACT_ITERATIONS = 20  # configurable via settings.max_react_iterations

# In the run() loop:
for iteration in range(self._max_iterations):
    ...

# After the loop:
raise ReActMaxIterationsError(
    iterations=self._max_iterations,
    last_thought=session.react_trace[-1]["thought"] if session.react_trace else "",
)
```

The 20-iteration limit is chosen to balance:
- Complex multi-step analyses (inspect → execute → validate → plot → summarize = ~8 steps)
- Infinite loop protection (bad prompts or confused models)
- Claude API cost control (20 × average_tokens ≈ reasonable per-request cost)

---

## 12. Complete data_agent.py

```python
# services/data_agent.py
"""
DataScienceAgentService: ReAct loop for data science tasks.

Architecture:
  - Stateless service: all state lives in AnalysisSession
  - PhysicalContextInjector: builds system prompt from domain docs (cached)
  - ReActParser: parses Claude's text output into structured ReAct components
  - ToolRegistry: maps action names to tool functions
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anthropic

from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, PhysicalUnit
from app.domain.exceptions import (
    ReActLoopError,
    ReActMaxIterationsError,
    ReActParseError,
)
from app.services.knowledge_tools import (
    list_domain_documents,
    read_domain_document,
    search_domain_knowledge,
    list_datasets,
    inspect_dataset,
    describe_columns,
)
from app.services.data_tools import (
    execute_python_code,
    get_execution_variables,
    get_figure,
    list_figures,
)
from app.infrastructure.unit_registry import (
    validate_physical_units,
    convert_units,
    check_magnitude,
    export_notebook,
    save_figure,
)

# ── Regex patterns ──────────────────────────────────────────────────────────

THOUGHT_PATTERN = re.compile(
    r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)",
    re.DOTALL | re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"Action:\s*([a-z_][a-z0-9_]*)", re.IGNORECASE)
ACTION_INPUT_PATTERN = re.compile(r"Action Input:\s*(\{.*?\})\s*$", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)
VALUE_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(°C|°F|K|MW|GW|kW|bar|kPa|MPa|%|m/s|kg/s|mol|pH)",
    re.IGNORECASE,
)

# ── Tool Registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable] = {
    "list_domain_documents":   list_domain_documents,
    "read_domain_document":    read_domain_document,
    "search_domain_knowledge": search_domain_knowledge,
    "list_datasets":           list_datasets,
    "inspect_dataset":         inspect_dataset,
    "describe_columns":        describe_columns,
    "execute_python_code":     execute_python_code,
    "get_execution_variables": get_execution_variables,
    "get_figure":              get_figure,
    "list_figures":            list_figures,
    "validate_physical_units": validate_physical_units,
    "convert_units":           convert_units,
    "check_magnitude":         check_magnitude,
    "export_notebook":         export_notebook,
    "save_figure":             save_figure,
}

_SESSION_TOOLS = frozenset({
    "execute_python_code", "get_execution_variables", "get_figure",
    "list_figures", "export_notebook", "save_figure",
})

# ── System Prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Data Scientist AI assistant.

## ReAct Protocol (MANDATORY FORMAT)
Every response MUST use one of these formats:

Format A — When taking an action:
Thought: [your reasoning]
Action: [tool_name]
Action Input: {{"key": "value"}}

Format B — When analysis is complete:
Thought: [final reasoning]
Final Answer: [conclusion for the user]

## Available Tools
{tool_list}

## Domain Knowledge
{domain_context}

## Unit Definitions
{unit_definitions}

## Physical Constraints
- Efficiency: 0% to 100%
- Temperature (K): strictly > 0
- Validate all computed quantities before reporting
"""

# ── ParsedReActResponse ──────────────────────────────────────────────────────

@dataclass
class ParsedReActResponse:
    thought: str
    is_final_answer: bool
    final_answer: str
    action_name: str
    action_input: dict
    raw_response: str


# ── ReActParser ──────────────────────────────────────────────────────────────

class ReActParser:
    """Stateless parser for Claude ReAct responses."""

    def parse_response(self, response: str, iteration: int) -> ParsedReActResponse:
        thought = self.extract_thought(response)

        final_match = FINAL_ANSWER_PATTERN.search(response)
        if final_match:
            return ParsedReActResponse(
                thought=thought,
                is_final_answer=True,
                final_answer=final_match.group(1).strip(),
                action_name="",
                action_input={},
                raw_response=response,
            )

        action_name = self.extract_action(response, iteration)
        action_input = self.extract_action_input(response, iteration)

        return ParsedReActResponse(
            thought=thought,
            is_final_answer=False,
            final_answer="",
            action_name=action_name,
            action_input=action_input,
            raw_response=response,
        )

    def extract_thought(self, response: str) -> str:
        m = THOUGHT_PATTERN.search(response)
        return m.group(1).strip() if m else ""

    def extract_action(self, response: str, iteration: int) -> str:
        m = ACTION_PATTERN.search(response)
        if not m:
            raise ReActParseError(response, iteration)
        return m.group(1).strip().lower()

    def extract_action_input(self, response: str, iteration: int) -> dict:
        m = ACTION_INPUT_PATTERN.search(response)
        if not m:
            if "Action Input:" in response and "{" not in response:
                return {}
            raise ReActParseError(response, iteration)

        raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
        raise ReActParseError(response, iteration)

    def handle_malformed(self, response: str, iteration: int) -> str:
        return (
            "Observation: Your response was not in the correct ReAct format. "
            "You MUST use:\nThought: ...\nAction: tool_name\n"
            'Action Input: {"key": "value"}\nOR:\n'
            "Thought: ...\nFinal Answer: [conclusion]\nPlease retry."
        )


# ── PhysicalContextInjector ──────────────────────────────────────────────────

class PhysicalContextInjector:
    """Builds the system prompt from domain knowledge documents."""

    _MAX_DOCS = 10
    _MAX_DOC_CHARS = 4000

    def __init__(self, docs_dir: Path) -> None:
        self._docs_dir = docs_dir
        self._cached_prompt: str | None = None

    def build_system_prompt(self) -> str:
        if self._cached_prompt is None:
            self._cached_prompt = self._build()
        return self._cached_prompt

    def _build(self) -> str:
        domain_context = self._load_docs()
        unit_definitions = self._extract_units(domain_context)
        tool_list = "\n".join(f"- {name}" for name in TOOL_REGISTRY)
        return _SYSTEM_PROMPT_TEMPLATE.format(
            tool_list=tool_list,
            domain_context=domain_context or "(no domain documents found)",
            unit_definitions=unit_definitions or "(none extracted)",
        )

    def _load_docs(self) -> str:
        parts = []
        docs = sorted(self._docs_dir.glob("*"))[: self._MAX_DOCS]
        for doc in docs:
            if doc.suffix.lower() in {".md", ".txt", ".rst"}:
                try:
                    text = doc.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"--- {doc.name} ---\n{text[: self._MAX_DOC_CHARS]}")
                except OSError:
                    pass
        return "\n\n".join(parts)

    def _extract_units(self, text: str) -> str:
        pattern = re.compile(
            r"^[-*]\s*\w+\s*(?:\([^)]+\))?\s*:\s*.+", re.MULTILINE
        )
        defs = {m.group(0).strip() for m in pattern.finditer(text)}
        return "\n".join(sorted(defs)[:50])


# ── DataScienceAgentService ──────────────────────────────────────────────────

class DataScienceAgentService:
    """
    Runs the ReAct agentic loop for data science tasks.
    One instance shared across all requests (stateless; state is in AnalysisSession).
    """

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._parser = ReActParser()
        self._context_injector = PhysicalContextInjector(
            docs_dir=settings.domain_docs_dir
        )
        self._max_iterations = settings.max_react_iterations

    async def run(self, session: AnalysisSession) -> str:
        system_prompt = self._context_injector.build_system_prompt()

        for iteration in range(self._max_iterations):
            # Call Claude
            response_text = await self._call_claude(session, system_prompt)

            # Append raw response as assistant message
            session.add_assistant_message(response_text)

            # Parse
            try:
                parsed = self._parser.parse_response(response_text, iteration)
            except ReActParseError:
                error_obs = self._parser.handle_malformed(response_text, iteration)
                session.add_user_message(error_obs)
                session.append_react_step("[parse error]", "[malformed]", error_obs)
                continue

            if parsed.is_final_answer:
                validated = self._validate_final_answer(parsed.final_answer, session)
                session.append_react_step(parsed.thought, "Final Answer", validated)
                return validated

            # Dispatch tool
            observation = self._dispatch_tool(
                parsed.action_name, parsed.action_input, session
            )
            formatted_obs = self._format_observation(observation)

            session.append_react_step(
                parsed.thought,
                f"{parsed.action_name}({json.dumps(parsed.action_input)})",
                observation,
            )
            session.add_user_message(formatted_obs)

        last_thought = (
            session.react_trace[-1]["thought"] if session.react_trace else ""
        )
        raise ReActMaxIterationsError(self._max_iterations, last_thought)

    async def _call_claude(self, session: AnalysisSession, system: str) -> str:
        response = await self._client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens,
            system=system,
            messages=session.messages,
        )
        blocks = response.content
        return "\n".join(
            b.text for b in blocks if hasattr(b, "text")
        )

    def _dispatch_tool(
        self, action_name: str, action_input: dict, session: AnalysisSession
    ) -> str:
        if action_name not in TOOL_REGISTRY:
            available = ", ".join(TOOL_REGISTRY.keys())
            return f"Error: Unknown tool '{action_name}'. Available: {available}"

        fn = TOOL_REGISTRY[action_name]
        try:
            if action_name in _SESSION_TOOLS:
                return fn(session=session, **action_input)
            return fn(**action_input)
        except TypeError as e:
            return f"Error: Invalid arguments for '{action_name}': {e}"
        except Exception as e:
            return f"Error: Tool '{action_name}' raised unexpected error: {e}"

    def _format_observation(self, result: str) -> str:
        MAX_OBS_CHARS = 8000
        if len(result) > MAX_OBS_CHARS:
            result = result[:MAX_OBS_CHARS] + f"\n...[truncated, {len(result)} total]"
        return f"Observation: {result}"

    def _validate_final_answer(self, answer: str, session: AnalysisSession) -> str:
        warnings: list[str] = []
        for m in VALUE_UNIT_PATTERN.finditer(answer):
            val, unit = float(m.group(1)), m.group(2)
            if unit == "%" and val > 100:
                warnings.append(f"⚠ {val}% exceeds 100% — verify this is not a fraction.")
            elif unit == "K" and val <= 0:
                warnings.append(f"⚠ {val} K is at or below absolute zero.")
            elif unit == "°C" and val < -273.15:
                warnings.append(f"⚠ {val}°C is below absolute zero.")

        if warnings:
            return answer + "\n\n**Physical Validation Warnings:**\n" + "\n".join(warnings)
        return answer
```
