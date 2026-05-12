"""
PHASE 3 — STEP 9: State Machine (Plan→Execute→Review)
======================================================
Harness Pillar: Standard Workflows & Refinement Loops
SDK Docs: https://docs.anthropic.com/en/agents-and-tools/tool-use/tool-runner

Goal: Implement a strict Plan→Execute→Review workflow using a state machine.
      Force the agent into programmatic, verifiable stages instead of letting
      it wander freely. This prevents the agent from skipping planning,
      rushing to execute, or forgetting to validate its own output.

State Machine:
  ┌──────────┐     ┌─────────┐     ┌────────┐     ┌──────────┐
  │  PLAN    │────>│ EXECUTE │────>│ REVIEW │────>│  DONE    │
  └──────────┘     └─────────┘     └────────┘     └──────────┘
       │                                │ (fail)
       │                                ▼
       │                          ┌──────────┐
       └──────────────────────────│  REVISE  │
                                  └──────────┘
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# State machine definition
# ─────────────────────────────────────────────
class State(str, Enum):
    PLAN    = "PLAN"
    EXECUTE = "EXECUTE"
    REVIEW  = "REVIEW"
    REVISE  = "REVISE"
    DONE    = "DONE"
    ERROR   = "ERROR"

@dataclass
class AgentContext:
    task: str
    state: State = State.PLAN
    plan: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    review_feedback: str = ""
    revision_count: int = 0
    final_output: str = ""
    MAX_REVISIONS: int = 3


# ─────────────────────────────────────────────
# Stage implementations — each stage is a separate LLM call
# with a SPECIFIC system prompt for that stage's role
# ─────────────────────────────────────────────
def stage_plan(ctx: AgentContext) -> AgentContext:
    """
    PLAN stage: The agent produces a structured execution plan.
    Output is a validated list of steps — never free prose.
    """
    print(f"\n{'='*50}\n[STATE: PLAN]\n{'='*50}")

    from pydantic import BaseModel

    class ExecutionPlan(BaseModel):
        steps: list[str]
        estimated_complexity: str  # "low" | "medium" | "high"
        potential_risks: list[str]

    response = client.messages.parse(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        output_format=ExecutionPlan,
        system=(
            "You are a task planner. Your job is ONLY to create plans, not execute them. "
            "Break the task into 3-6 concrete, actionable steps. "
            "Identify risks that could cause the plan to fail."
        ),
        messages=[{
            "role": "user",
            "content": f"Create an execution plan for this task:\n{ctx.task}"
        }],
    )

    plan_data = response.content[0].parsed
    ctx.plan = plan_data.steps
    ctx.state = State.EXECUTE

    print(f"Plan ({plan_data.estimated_complexity} complexity):")
    for i, step in enumerate(ctx.plan, 1):
        print(f"  {i}. {step}")
    print(f"Risks: {plan_data.potential_risks}")

    return ctx


def stage_execute(ctx: AgentContext) -> AgentContext:
    """
    EXECUTE stage: The agent works through each plan step
    using available tools. Results are captured per step.
    """
    print(f"\n{'='*50}\n[STATE: EXECUTE]\n{'='*50}")

    # Tools available during execution
    EXEC_TOOLS = [
        {
            "name": "complete_step",
            "description": "Mark a plan step as complete and record the result.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "step_description": {"type": "string"},
                    "result": {"type": "string", "description": "What was accomplished"},
                    "output": {"type": "string", "description": "Any data/code/output produced"},
                },
                "required": ["step_number", "step_description", "result"],
            },
        },
    ]

    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(ctx.plan))
    messages = [{
        "role": "user",
        "content": (
            f"Execute this plan step by step for task: {ctx.task}\n\n"
            f"PLAN:\n{plan_text}\n\n"
            f"Use complete_step() for EACH step as you finish it."
        )
    }]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            tools=EXEC_TOOLS,
            system=(
                "You are a task executor. Execute the given plan step by step. "
                "Call complete_step() for EACH step when done. "
                "Do not skip steps. Do not merge steps."
            ),
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                ctx.results.append(block.input)
                print(f"  ✅ Step {block.input['step_number']}: {block.input['result'][:100]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Step {block.input['step_number']} recorded.",
                })
            messages.append({"role": "user", "content": tool_results})

    ctx.state = State.REVIEW
    return ctx


def stage_review(ctx: AgentContext) -> AgentContext:
    """
    REVIEW stage: A separate LLM call (acting as QA judge)
    evaluates the execution results against the original task.
    This is the LLM-as-judge pattern.
    """
    print(f"\n{'='*50}\n[STATE: REVIEW]\n{'='*50}")

    from pydantic import BaseModel
    from typing import Literal

    class ReviewResult(BaseModel):
        verdict: Literal["pass", "fail", "partial"]
        score: float  # 0.0 - 1.0
        issues: list[str]
        feedback: str
        final_output: str

    results_text = json.dumps(ctx.results, indent=2)

    response = client.messages.parse(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        output_format=ReviewResult,
        system=(
            "You are a quality assurance judge. Review the task execution "
            "and determine if it was completed correctly and completely. "
            "Be strict — partial completion is a failure. "
            "Provide specific, actionable feedback if the work needs revision."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Original task: {ctx.task}\n\n"
                f"Plan: {json.dumps(ctx.plan)}\n\n"
                f"Execution results:\n{results_text}\n\n"
                "Did the agent complete the task correctly and completely?"
            )
        }],
    )

    review = response.content[0].parsed
    print(f"Verdict  : {review.verdict} (score: {review.score:.0%})")
    print(f"Issues   : {review.issues}")
    print(f"Feedback : {review.feedback[:200]}")

    if review.verdict == "pass":
        ctx.state = State.DONE
        ctx.final_output = review.final_output
    elif ctx.revision_count >= ctx.MAX_REVISIONS:
        print(f"⚠️  Max revisions ({ctx.MAX_REVISIONS}) reached.")
        ctx.state = State.ERROR
        ctx.final_output = f"Incomplete after {ctx.MAX_REVISIONS} revisions. Last output: {review.final_output}"
    else:
        ctx.state = State.REVISE
        ctx.review_feedback = review.feedback
        ctx.revision_count += 1

    return ctx


def stage_revise(ctx: AgentContext) -> AgentContext:
    """
    REVISE stage: Update the plan based on review feedback,
    then go back to EXECUTE.
    """
    print(f"\n{'='*50}\n[STATE: REVISE — attempt {ctx.revision_count}]\n{'='*50}")
    print(f"Feedback: {ctx.review_feedback}")

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        system="You are a plan revisor. Update the plan based on the QA feedback.",
        messages=[{
            "role": "user",
            "content": (
                f"Task: {ctx.task}\n"
                f"Original plan: {json.dumps(ctx.plan)}\n"
                f"QA Feedback: {ctx.review_feedback}\n\n"
                "Provide a revised plan as a JSON list of steps."
            )
        }],
    )

    text = response.content[0].text
    try:
        # Extract JSON list from response
        import re
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            ctx.plan = json.loads(match.group())
    except Exception:
        pass

    ctx.results = []  # clear previous results
    ctx.state = State.EXECUTE
    return ctx


# ─────────────────────────────────────────────
# The State Machine Runner
# ─────────────────────────────────────────────
def run_state_machine(task: str) -> str:
    """
    Run the full Plan→Execute→Review state machine.
    The agent cannot skip stages — the harness enforces the workflow.
    """
    ctx = AgentContext(task=task)

    # State transition table
    transitions = {
        State.PLAN:    stage_plan,
        State.EXECUTE: stage_execute,
        State.REVIEW:  stage_review,
        State.REVISE:  stage_revise,
    }

    print(f"\n🚀 Starting state machine for task:\n  {task}\n")

    while ctx.state not in (State.DONE, State.ERROR):
        handler = transitions.get(ctx.state)
        if not handler:
            ctx.state = State.ERROR
            break
        ctx = handler(ctx)

    print(f"\n{'='*50}")
    print(f"[FINAL STATE: {ctx.state}]")
    print(f"{'='*50}")
    return ctx.final_output


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tasks = [
        "Write a Python function to parse a CSV file and compute the average of a specified numeric column, with error handling for malformed data.",
        "Create a step-by-step guide for setting up a basic FastAPI web server with a health check endpoint.",
    ]

    for task in tasks:
        result = run_state_machine(task)
        print(f"\n📋 FINAL OUTPUT:\n{result}")
        print("\n" + "─" * 60)


print("""
KEY TAKEAWAYS
=============
1. A state machine FORCES the agent through verified stages.
   It cannot skip planning or rush to action.

2. Each stage has a DIFFERENT system prompt with a different role:
   - Planner: "do not execute, only plan"
   - Executor: "execute the plan, call complete_step() for each step"
   - Judge: "be strict, partial completion is failure"
   - Revisor: "update the plan based on feedback"

3. The REVIEW stage (LLM-as-judge) is the evaluation guardrail.
   It uses a SEPARATE LLM call — the judge should never be the same
   conversation as the executor to avoid self-serving bias.

4. Structured outputs (Pydantic) at every stage ensure the harness
   can reliably read the agent's decisions and route accordingly.

5. The state machine is the embodiment of the "Standard Workflows"
   pillar — it removes ambiguity from the agentic process.
""")
