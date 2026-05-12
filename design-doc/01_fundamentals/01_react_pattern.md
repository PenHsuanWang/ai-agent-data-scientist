# The ReAct Pattern

> *"ReAct synergizes reasoning and acting in an interleaved manner, allowing the model to generate reasoning traces while also performing actions that interact with the environment."*
> — Yao et al., 2022

---

## Table of Contents

1. [What is ReAct](#1-what-is-react)
2. [The Three Components](#2-the-three-components)
3. [ReAct vs Other Patterns](#3-react-vs-other-patterns)
4. [ReAct Protocol Format](#4-react-protocol-format)
5. [Text-Based vs Native Tool-Use](#5-text-based-vs-native-tool-use)
6. [ReAct for Data Science](#6-react-for-data-science)
7. [Implementation Patterns](#7-implementation-patterns)
8. [Common Pitfalls](#8-common-pitfalls)
9. [The Iteration Guard](#9-the-iteration-guard)

---

## 1. What is ReAct

### The Original Paper

ReAct was introduced in the paper **"ReAct: Synergizing Reasoning and Acting in Language Models"** by Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao (arXiv:2210.03629, 2022, accepted at ICLR 2023).

The paper tested LLMs on two classes of tasks:
- **Reasoning tasks** (HotpotQA, FEVER) — require multi-hop reasoning over facts
- **Decision-making tasks** (ALFWorld, WebShop) — require sequential actions in environments

For both classes, interleaving reasoning traces with actions outperformed:
- Chain-of-thought alone (reasoning without acting — no external info)
- Direct acting alone (tool-use without reasoning — no error recovery)

The improvement came from two directions: reasoning helped the model *plan before acting* and *recover from errors*; acting let the model *gather information* that pure reasoning cannot fabricate.

### The Core Insight

The key insight is that **language model generations can serve two purposes simultaneously**:

1. **Reasoning generation** — the model writes "let me think about this" text that helps it plan, self-correct, and stay on track
2. **Action generation** — the model writes a structured tool call that triggers an external operation

When these are interleaved in the same token stream — rather than happening in separate API calls — the reasoning *contextually informs* the action, and the action result *contextually informs* the next round of reasoning. This produces a coherent problem-solving loop.

### Why ReAct Outperforms CoT Alone

Chain-of-thought (Wei et al., 2022) showed that saying "let's think step by step" dramatically improves reasoning on multi-step problems. But CoT has a critical weakness: it can only reason about information the model already has. When the model needs to check a value, look up a formula, or run a calculation, pure CoT will hallucinate — it will generate what looks like a numeric result without actually computing it.

ReAct solves this by giving the model a way to say: *"I need to look something up before I can continue reasoning."* The action step interrupts the reasoning, fetches real information, and returns it as an observation. The model then continues reasoning with verified facts.

For data science, this is especially important: you cannot reason about what a dataset contains without actually loading it, and you cannot validate a calculation without actually running it.

---

## 2. The Three Components

### 2.1 Thought

A **Thought** is the model's explicit reasoning about its current state, what it knows, what it doesn't know, and what it should do next.

**Format**: `Thought: <free-form reasoning text>`

**What a good Thought contains**:
- Summary of what is known from previous observations
- Identification of what information is still needed
- Explicit consideration of alternatives
- Unit/scale considerations before computation
- A clear decision about the next action

**Example of a high-quality Thought**:
```
Thought: The dataset has been loaded and I can see 8,760 rows (hourly data for one year)
         with 23 columns. Looking at the column names, I can see:
         - fuel_flow_kg_min (note: per MINUTE, not per second)
         - fuel_lhv_mj_kg  
         - gross_power_mw
         - auxiliary_power_mw
         
         Before computing thermal efficiency, I need to convert fuel_flow to kg/s by
         dividing by 60. Power is in MW and LHV in MJ/kg, so heat_input = 
         (fuel_flow_kg_s) * LHV = MW, which is consistent with gross_power in MW.
         
         I'll compute efficiency and then immediately validate it against the Carnot 
         limit I found in the domain document (66% for this temperature range).
```

**What a bad Thought looks like**:
```
Thought: I will calculate the efficiency.
```
This adds no reasoning value. The model has not considered units, ranges, or validation. The subsequent action is just as likely to be wrong as if there were no Thought at all.

### 2.2 Action

An **Action** is a structured invocation of a specific tool with specific parameters.

**Format**: `Action: tool_name(param1="value1", param2="value2")`

**What a good Action looks like**:
```
Action: execute_python(code="""
import pandas as pd
df = pd.read_csv("plant_sensors.csv")
fuel_flow_kg_s = df['fuel_flow_kg_min'] / 60.0
heat_input_mw = fuel_flow_kg_s * df['fuel_lhv_mj_kg']
net_work_mw = df['gross_power_mw'] - df['auxiliary_power_mw']
efficiency_pct = (net_work_mw / heat_input_mw) * 100.0
result = efficiency_pct.describe().to_dict()
print(result)
""")
```

**Properties of a good Action**:
- Uses exactly one tool per Action step
- Parameters are fully specified (not implicit)
- Code includes comments explaining unit conversions
- Code prints a result dict, not raw DataFrames (which are too verbose for observations)
- References are to values established in prior Thoughts/Observations, not hallucinated

### 2.3 Observation

An **Observation** is the verbatim output of the tool call, returned by the execution engine and injected into the model's context as a new message.

**Format**: `Observation: <tool_output>`

**What an Observation contains**:
- The exact output from the tool (string, JSON, error message)
- No interpretation — interpretation happens in the next Thought

**Example**:
```
Observation: {
  "count": 8760.0,
  "mean": 36.2,
  "std": 1.8,
  "min": 33.1,
  "25%": 34.9,
  "50%": 36.1,
  "75%": 37.5,
  "max": 38.9
}
```

**Why Observations must be faithful**: The entire value of the ReAct loop depends on the Observation being truthful. If a tool returns a misleading success when it has actually failed (e.g., returning `0.0` instead of raising an error when a column is not found), the model will reason on false premises. This is why the design mandate is: **tools never raise, tools always return descriptive error strings on failure**.

---

## 3. ReAct vs Other Patterns

| Criterion | Standard Prompt | Chain-of-Thought | Tree-of-Thought | Standard Tool-Use | **ReAct** |
|-----------|----------------|-----------------|-----------------|-------------------|-----------|
| **External information** | ✗ None | ✗ None | ✗ None | ✅ Tool calls | ✅ Tool calls |
| **Visible reasoning** | ✗ None | ✅ Yes | ✅ Yes | ✗ None | ✅ Yes |
| **Reasoning before action** | N/A | N/A | N/A | ✗ Implicit | ✅ Explicit |
| **Error recovery** | ✗ None | ✗ Limited | ✅ Backtrack | ✗ None | ✅ Via Thought |
| **Audit trail** | ✗ None | ✅ Trace | ✅ Tree | ✗ None | ✅ Full trace |
| **Complexity** | Very low | Low | High | Low | **Medium** |
| **Token usage** | Minimal | Moderate | High | Moderate | **Moderate** |
| **Best for** | Simple Q&A | Multi-step reasoning | Exploration | Stateless APIs | **Tool-using agents** |

**When to use each**:
- **Standard prompt**: Simple factual retrieval, single-step generation
- **Chain-of-thought**: Math word problems, logical reasoning, no external data needed
- **Tree-of-thought**: Creative tasks, code optimization where multiple approaches should be explored
- **Standard tool-use**: Stateless operations where the reasoning is trivial (lookup, format conversion)
- **ReAct**: Any multi-step task involving external state (datasets, files, APIs, computation) where reasoning about intermediate results matters

---

## 4. ReAct Protocol Format

### The Text Format

In the original ReAct paper (and in this system's text-based implementation), the protocol is plain text embedded in the model's response:

```
Thought: <reasoning — free form, can be multiple sentences>
Action: <tool_name>(<param_name>="<value>", ...)
Observation: <tool_output — injected by the system, not generated by model>
```

Iterations continue until the model emits a **Final Answer**:

```
Thought: I have computed the efficiency (36.2%), validated it against the Carnot 
         limit, and confirmed it is within the expected design range. I have also
         identified the Q3→Q4 efficiency drop. I can now provide a final answer.

Final Answer: The plant's Q4 thermal efficiency averaged 36.2% (range: 33.1–38.9%),
              which is within the design specification of 35–42% and well below the 
              Carnot limit of 66%. This represents a 3.2% decrease from Q3 (37.4%),
              consistent with the reported 8% fuel increase at constant load. The 
              most likely cause is boiler fouling — recommend inspection of heat 
              transfer surfaces.
```

### Regex for Parsing

When implementing a text-based ReAct parser, you need to extract action names and parameters from the model's text output:

```python
import re
from dataclasses import dataclass
from typing import Optional

# Pattern matches: Action: tool_name(param1="val1", param2=123)
ACTION_PATTERN = re.compile(
    r"^Action:\s*(\w+)\((.*)?\)\s*$",
    re.MULTILINE | re.DOTALL,
)

# Pattern for Final Answer
FINAL_ANSWER_PATTERN = re.compile(
    r"^Final Answer:\s*(.+)$",
    re.MULTILINE | re.DOTALL,
)

# Pattern for each Thought block
THOUGHT_PATTERN = re.compile(
    r"^Thought:\s*(.+?)(?=^(?:Action:|Final Answer:|Thought:)|\Z)",
    re.MULTILINE | re.DOTALL,
)

@dataclass
class ParsedReActStep:
    thought: Optional[str] = None
    action_name: Optional[str] = None
    action_args: Optional[str] = None
    final_answer: Optional[str] = None

def parse_react_step(text: str) -> ParsedReActStep:
    """Parse a single ReAct step from model output."""
    step = ParsedReActStep()
    
    thought_match = THOUGHT_PATTERN.search(text)
    if thought_match:
        step.thought = thought_match.group(1).strip()
    
    final_match = FINAL_ANSWER_PATTERN.search(text)
    if final_match:
        step.final_answer = final_match.group(1).strip()
        return step  # Final answer means no action
    
    action_match = ACTION_PATTERN.search(text)
    if action_match:
        step.action_name = action_match.group(1).strip()
        step.action_args = action_match.group(2).strip()
    
    return step
```

### Parsing Action Parameters

Action parameters in text format require careful parsing because values can contain embedded strings, numbers, or multiline code:

```python
import ast
from typing import Any

def parse_action_args(args_str: str) -> dict[str, Any]:
    """
    Parse action arguments from text format.
    
    Handles:
      - String values: param="hello"
      - Numeric values: count=42
      - Multiline strings: code=\"\"\"...\"\"\"
    """
    if not args_str.strip():
        return {}
    
    # Wrap in a fake function call so ast.parse can handle it
    try:
        fake_call = f"f({args_str})"
        tree = ast.parse(fake_call, mode="eval")
        call_node = tree.body
        
        result = {}
        for kw in call_node.keywords:
            result[kw.arg] = ast.literal_eval(kw.value)
        return result
    except (ValueError, SyntaxError) as e:
        # Fallback: try simple key=value splitting for basic cases
        return _parse_simple_args(args_str)

def _parse_simple_args(args_str: str) -> dict[str, Any]:
    """Fallback parser for simple key=value pairs."""
    result = {}
    # This is intentionally simple — production code should handle edge cases
    for match in re.finditer(r'(\w+)=(".*?"|\'.*?\'|\d+(?:\.\d+)?)', args_str):
        key = match.group(1)
        val_str = match.group(2)
        try:
            result[key] = ast.literal_eval(val_str)
        except ValueError:
            result[key] = val_str.strip('"\'')
    return result
```

---

## 5. Text-Based vs Native Tool-Use

There are two fundamentally different ways to implement the ReAct pattern with the Anthropic SDK. Understanding both is essential for choosing the right approach.

### 5.1 Text-Based ReAct (Original Paper Approach)

The model generates the entire `Thought/Action/Observation` sequence as plain text. The system parses the model's output to extract tool calls, executes them, and injects the results as text.

```python
# Text-based ReAct: parse the model's text output to find tool calls
def run_react_text_based(client, messages: list[dict]) -> str:
    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=messages,
            system=REACT_SYSTEM_PROMPT,  # tells model to use Thought/Action/Final Answer
        )
        
        text = response.content[0].text
        step = parse_react_step(text)
        
        if step.final_answer:
            return step.final_answer
        
        if step.action_name:
            # Execute the tool
            observation = execute_tool(step.action_name, step.action_args)
            
            # Add the model's response + observation to messages
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        else:
            # No action found — model may be confused
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": "Please continue with a Thought and Action, or provide a Final Answer."
            })
    
    return "Max iterations reached without final answer."
```

**Pros**:
- Fully visible reasoning — every character the model generates is part of the trace
- Works with any model that can follow text instructions (not just Claude)
- Easy to debug — you see exactly what the model wrote before parsing
- Historical: matches the original ReAct paper implementation

**Cons**:
- Parsing is fragile — models sometimes deviate from the expected format
- Parameters with complex values (multiline code, JSON) are hard to parse reliably
- No schema validation on tool inputs — the model might hallucinate parameter names
- More tokens per turn than native tool-use (the format markers are overhead)

### 5.2 Native Tool-Use (Anthropic SDK Approach)

The model generates a `tool_use` content block (structured JSON), not text. The SDK handles the structure; you handle the execution.

```python
# Native tool-use: Anthropic SDK handles structure, you handle execution
def run_react_native_tools(client, messages: list[dict], tools: list[dict]) -> str:
    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
        )
        
        if response.stop_reason == "end_turn":
            # Extract text from response
            text_blocks = [b for b in response.content if b.type == "text"]
            return text_blocks[0].text if text_blocks else ""
        
        if response.stop_reason == "tool_use":
            # Build the assistant message with all content blocks
            messages.append({"role": "assistant", "content": response.content})
            
            # Execute each tool_use block and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached."
```

**Pros**:
- Structured — inputs are validated against the tool schema (especially with `strict: true`)
- More reliable — no regex parsing, no format deviations
- Less overhead — no format markers needed in the response
- Official SDK support — stop_reason, tool_use_id, content blocks are first-class

**Cons**:
- Reasoning is implicit — you don't see the model's Thought unless you add a text tool or scratchpad
- Tied to Claude's tool-use API format (not portable to other models)
- Tool_use blocks can be processed in parallel by the SDK, which changes the interaction pattern

### 5.3 Hybrid Approach (This System)

This system uses **native tool-use for tool calls** but adds a `think` tool or relies on text blocks between tool calls for the Thought component. This gives us:
- Schema-validated tool calls (reliability)
- Visible reasoning via text blocks (auditability)
- Native SDK support (correctness)

```python
# The model generates text blocks (Thought) interspersed with tool_use blocks (Action)
# This is the default behavior when you don't set tool_choice="any"

for block in response.content:
    if block.type == "text":
        # This is a Thought — store it in the trace
        session.thought_trace.append(block.text)
    elif block.type == "tool_use":
        # This is an Action — execute it
        result = execute_tool(block.name, block.input)
        session.observation_trace.append(result)
```

---

## 6. ReAct for Data Science

### Why Data Analysis Benefits Especially from ReAct

Standard data science with AI assistants typically looks like: "Write me code to analyze X." The model generates code, you run it, you send back the output, the model responds. This is one Action per conversation turn.

Data science *analysis* — as opposed to data science *code generation* — requires a much tighter loop:

1. **Load → inspect → decide how to proceed** (you can't plan a cleaning strategy without seeing the data)
2. **Clean → check → re-clean** (data quality is iterative)
3. **Compute → validate → re-compute** (physical plausibility requires intermediate checking)
4. **Interpret → cross-reference → conclude** (conclusions require context from domain docs)

Each of these phases requires the model to reason about what it just observed before deciding what to do next. Without explicit Thoughts, the model will either:
- Plan everything upfront (and fail when reality doesn't match the plan)
- React to each observation in isolation (and lose track of the overall goal)

The ReAct loop's explicit Thought step is the mechanism by which the model maintains analytical coherence across many sequential steps.

### Physical Reasoning Before Acting

For datasets with physical quantities, the Thought step serves an additional purpose: **pre-computation unit analysis**. Before writing code that computes a derived quantity, the model should reason about:

- What units will the result be in?
- What is the expected order of magnitude?
- What physical laws constrain the result?
- What would be a red flag value?

This reasoning happens in the Thought step, *before* the code is written. Then the code is written to be unit-explicit. Then the result is validated against the pre-stated constraints. This three-part pattern (anticipate → compute → validate) is a formal extension of the basic ReAct loop for physical domains.

### Multi-Step Validation

A single validation is rarely sufficient. A typical data analysis session might involve:

1. Validate raw column units against domain document
2. Validate intermediate computed quantities (e.g., heat input in correct units)
3. Validate final derived quantities (e.g., efficiency within physical bounds)
4. Validate time-series consistency (e.g., no discontinuous jumps that suggest sensor failure)
5. Validate against expected design parameters (from domain doc)

Each validation is a separate `Observation` that informs the next `Thought`. The ReAct loop naturally accommodates this multi-step validation pattern — it would be extremely awkward to implement in a single-shot prompt.

### Auditable Trace

For engineering and scientific applications, the ReAct trace *is* the deliverable. When the engineer submits the analysis to a peer reviewer, the reviewer needs to see:
- What domain knowledge was consulted
- How units were handled at each step
- What physical validation was performed
- What the intermediate values were

This is exactly what the ReAct trace provides. The notebook export packages the trace into a standard format that any scientist can review.

---

## 7. Implementation Patterns

### 7.1 A Simple ReAct Parser Class

```python
from dataclasses import dataclass, field
from typing import Optional
import re

@dataclass
class ReActTrace:
    """Accumulates the full Thought/Action/Observation trace for a session."""
    steps: list[dict] = field(default_factory=list)
    
    def add_thought(self, text: str) -> None:
        self.steps.append({"type": "thought", "content": text})
    
    def add_action(self, tool_name: str, tool_input: dict, tool_use_id: str) -> None:
        self.steps.append({
            "type": "action",
            "tool": tool_name,
            "input": tool_input,
            "tool_use_id": tool_use_id,
        })
    
    def add_observation(self, tool_use_id: str, result: str) -> None:
        self.steps.append({
            "type": "observation",
            "tool_use_id": tool_use_id,
            "result": result,
        })
    
    def to_markdown(self) -> str:
        """Format the trace as readable markdown."""
        lines = []
        step_num = 1
        for step in self.steps:
            if step["type"] == "thought":
                lines.append(f"### Thought {step_num}")
                lines.append(step["content"])
                lines.append("")
            elif step["type"] == "action":
                lines.append(f"### Action {step_num} — `{step['tool']}`")
                lines.append("```python")
                import json
                lines.append(json.dumps(step["input"], indent=2))
                lines.append("```")
                lines.append("")
                step_num += 1
            elif step["type"] == "observation":
                lines.append(f"### Observation {step_num - 1}")
                lines.append("```")
                lines.append(step["result"])
                lines.append("```")
                lines.append("")
        return "\n".join(lines)
    
    def to_notebook_cells(self) -> list[dict]:
        """Convert trace to nbformat cell dicts."""
        cells = []
        code_counter = 1
        for step in self.steps:
            if step["type"] == "thought":
                cells.append({
                    "cell_type": "markdown",
                    "source": f"### 💭 Thought\n\n{step['content']}",
                    "metadata": {},
                })
            elif step["type"] == "action" and step["tool"] == "execute_python":
                source = step["input"].get("code", "")
                cells.append({
                    "cell_type": "code",
                    "source": source,
                    "metadata": {"cell_id": f"action_{code_counter}"},
                    "outputs": [],
                    "execution_count": code_counter,
                })
                code_counter += 1
        return cells
```

### 7.2 The Agent Loop with ReAct Trace

```python
import anthropic
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AnalysisSession:
    session_id: str
    messages: list[dict] = field(default_factory=list)
    trace: ReActTrace = field(default_factory=ReActTrace)
    iteration_count: int = 0

MAX_REACT_ITERATIONS = 20

def run_analysis_session(
    client: anthropic.Anthropic,
    session: AnalysisSession,
    tools: list[dict],
    tool_executor,  # Callable[[str, dict], str]
    system_prompt: str,
) -> str:
    """
    Run the ReAct loop for a data science analysis session.
    
    Returns the final answer text, or an error/timeout message.
    """
    while session.iteration_count < MAX_REACT_ITERATIONS:
        session.iteration_count += 1
        
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=system_prompt,
            messages=session.messages,
            tools=tools,
        )
        
        # Process content blocks — could be text (Thought) or tool_use (Action)
        has_tool_use = False
        response_content = []
        
        for block in response.content:
            response_content.append(block)
            
            if block.type == "text" and block.text.strip():
                session.trace.add_thought(block.text.strip())
            
            elif block.type == "tool_use":
                has_tool_use = True
                session.trace.add_action(block.name, block.input, block.id)
        
        # Add the full assistant message to history
        session.messages.append({
            "role": "assistant",
            "content": response_content,
        })
        
        # If end_turn with no tool calls, we're done
        if response.stop_reason == "end_turn" and not has_tool_use:
            text_blocks = [b for b in response.content if b.type == "text"]
            return text_blocks[0].text if text_blocks else ""
        
        # Execute all tool calls and collect results
        if has_tool_use:
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    observation = tool_executor(block.name, block.input)
                    session.trace.add_observation(block.id, str(observation))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(observation),
                    })
            
            session.messages.append({
                "role": "user",
                "content": tool_results,
            })
    
    return (
        f"Analysis reached the maximum iteration limit ({MAX_REACT_ITERATIONS} steps). "
        f"Partial results are available in the session trace."
    )
```

---

## 8. Common Pitfalls

### Pitfall 1: Infinite Loops

**What happens**: The agent gets stuck in a cycle — it calls `inspect_dataset`, observes the schema, calls `inspect_dataset` again (perhaps with slightly different reasoning), observes the same schema, and repeats.

**Why it happens**: The model's context grows with each turn, but the most recent messages dominate attention. If the most recent Thought says "I should inspect the dataset to understand its structure," and the Observation says "here is the schema," but the next Thought *again* says "I need to understand the structure," the model has lost track of what it already knows.

**Solutions**:
1. Always enforce `MAX_REACT_ITERATIONS` (non-negotiable)
2. Add a "progress check" to the system prompt: *"If you have called the same tool with the same arguments twice, you must explain in your Thought why a different approach is needed."*
3. Track tool calls in the session and inject a reminder if the same tool+args combination is seen twice

```python
from collections import Counter

def check_for_loops(trace: ReActTrace, max_repeats: int = 2) -> Optional[str]:
    """Return a warning message if any tool+args pair is repeated."""
    action_counts = Counter()
    for step in trace.steps:
        if step["type"] == "action":
            key = (step["tool"], str(sorted(step["input"].items())))
            action_counts[key] += 1
            if action_counts[key] >= max_repeats:
                return (
                    f"Warning: You have called {step['tool']} with the same arguments "
                    f"{action_counts[key]} times. If you need different information, "
                    f"please use a different tool or different parameters."
                )
    return None
```

### Pitfall 2: Malformed Actions

**What happens**: In text-based ReAct, the model generates `Action: execute_python(code=df.describe())` — without quotes around the code argument. The parser fails.

**Why it happens**: The model is generating text and may not reliably wrap string arguments in the expected delimiters.

**Solutions**:
- Use native tool-use instead of text-based parsing (the model generates structured JSON, not free-form text)
- If using text-based: add format examples to the system prompt with explicit quoting
- If using text-based: implement a lenient fallback parser that handles common deviations
- Add validation that re-prompts the model if the action can't be parsed

### Pitfall 3: Partial Observations

**What happens**: A `describe()` on a large DataFrame produces thousands of characters. The observation is truncated, and the model reasons on incomplete information.

**Why it happens**: Tool results have no size limit by default. Large pandas outputs can easily exceed 10,000 characters.

**Solutions**:
- Every tool that can produce large output should truncate proactively
- Add a `max_observation_chars` parameter to the tool executor
- For DataFrames: always convert to `head(5).to_dict()` or `describe().to_dict()`, never to `to_string()`

```python
def truncate_observation(obs: str, max_chars: int = 3000) -> str:
    """Truncate a tool observation to prevent context overflow."""
    if len(obs) <= max_chars:
        return obs
    half = max_chars // 2
    return (
        obs[:half]
        + f"\n\n[... TRUNCATED: {len(obs) - max_chars} characters omitted ...]\n\n"
        + obs[-half:]
    )
```

### Pitfall 4: Token Limit Exhaustion

**What happens**: A long session accumulates thousands of tokens of Thought/Action/Observation history. Eventually the context window fills up and the API returns `stop_reason="max_tokens"`.

**Why it happens**: Each turn adds the full prior history to the prompt. ReAct is token-intensive because every Thought and Observation is preserved.

**Solutions**:
- Monitor token count and warn when approaching the limit
- Implement context compression: summarize old Thought/Observation pairs into a single "Session Summary" message
- For data science: 20 iterations × ~500 tokens each = ~10,000 tokens — well within Claude's 200k context for Haiku

```python
def estimate_session_tokens(session: AnalysisSession) -> int:
    """Rough token estimate: ~4 characters per token."""
    total_chars = sum(
        len(str(msg.get("content", "")))
        for msg in session.messages
    )
    return total_chars // 4

MAX_SESSION_TOKENS = 150_000  # Leave headroom for response

def should_compress_context(session: AnalysisSession) -> bool:
    return estimate_session_tokens(session) > MAX_SESSION_TOKENS
```

### Pitfall 5: Action Hallucination

**What happens**: The model generates `Action: fetch_real_time_data("turbine_12", "current")` — a tool that doesn't exist — because it has reasoned that it *should* exist.

**Why it happens**: The model knows what tools would be useful and sometimes generates tool calls for tools it wishes existed, especially in text-based ReAct where there's no schema validation.

**Solutions**:
- Native tool-use prevents this: the model can only generate tool_use blocks for registered tools
- In text-based ReAct: maintain a registry of valid tool names and return a clear error when an unknown tool is called
- Include the complete list of available tools in the system prompt with their exact names

```python
AVAILABLE_TOOLS = {
    "list_local_documents", "read_local_document", "search_documents",
    "list_datasets", "inspect_dataset", "load_dataset", "describe_dataset",
    "execute_python", "get_figure", "validate_units", "convert_units",
    "check_magnitude", "export_notebook", "save_figure",
}

def safe_dispatch_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in AVAILABLE_TOOLS:
        available = ", ".join(sorted(AVAILABLE_TOOLS))
        return (
            f"Error: Unknown tool '{tool_name}'. "
            f"Available tools: {available}"
        )
    return dispatch_tool(tool_name, tool_args)
```

---

## 9. The Iteration Guard

### Why `MAX_REACT_ITERATIONS = 20` for Data Science

The MVP agent uses `_MAX_LOOP_ITERATIONS = 10`. That's appropriate for a Q&A agent where each question should be answerable in a handful of tool calls (typically: search → read → respond).

Data science analysis is structurally different:

| Phase | Typical tool calls |
|-------|-------------------|
| Domain knowledge loading | 2–4 (read docs, search for relevant sections) |
| Dataset inspection | 2–3 (list, inspect, check specific columns) |
| Exploratory analysis | 4–6 (load, describe, compute statistics, plot) |
| Physical validation | 3–5 (validate each derived quantity) |
| Subsystem deep-dive | 4–8 (additional analysis on flagged issues) |
| Notebook export | 1–2 (save figures, export notebook) |
| **Total** | **16–28** |

Setting `MAX_REACT_ITERATIONS = 20` covers most analysis sessions while still providing a hard bound against runaway loops. For particularly complex analyses, you can increase it session-by-session:

```python
@dataclass
class AnalysisConfig:
    max_iterations: int = 20
    model: str = "claude-sonnet-4-6"
    max_tokens_per_turn: int = 8192
    observation_truncation_chars: int = 3000
    
    @classmethod
    def for_quick_analysis(cls) -> "AnalysisConfig":
        return cls(max_iterations=10, model="claude-haiku-4-5-20251001")
    
    @classmethod
    def for_deep_analysis(cls) -> "AnalysisConfig":
        return cls(max_iterations=30, model="claude-opus-4-7", max_tokens_per_turn=16384)
```

### Communicating the Limit to the Model

The system prompt should tell the model about the iteration limit so it can plan accordingly:

```python
REACT_SYSTEM_PROMPT = """
You are a domain-aware data science analyst. You follow the ReAct pattern:
Reason explicitly before every action, then act, then observe.

You have a budget of {max_iterations} reasoning steps for this session.
Use them wisely:
- Do not read the same document twice unless you need a specific section
- Do not compute the same quantity twice
- Combine related operations into single code blocks
- Prioritize validation of computed quantities before moving to new analysis

When you have gathered sufficient evidence for a conclusion, provide a 
Final Answer immediately rather than exhausting your step budget.
"""
```

---

*Next: [02_clean_architecture.md](02_clean_architecture.md) — How the four-layer architecture supports the ReAct agent.*
