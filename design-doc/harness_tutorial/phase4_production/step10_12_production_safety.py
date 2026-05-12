"""
PHASE 4 — STEPS 10–12: Production & Safety
============================================
Harness Pillars: Evaluation & Guardrails + ACI (Compute + Action layers)
SDK Docs:
  Code Execution: https://docs.anthropic.com/en/agents-and-tools/tool-use/code-execution-tool
  Structured Outputs: https://docs.anthropic.com/en/build-with-claude/structured-outputs

Goal:
  Step 10: Sandboxed code execution (Anthropic's secure container)
  Step 11: LLM-as-judge evaluation suite
  Step 12: Human-in-the-Loop (HITL) approval gate for high-stakes actions

These three components form the complete safety harness for production agents.
"""

import json
import time
from dataclasses import dataclass
from typing import Callable, Any
import anthropic
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ═══════════════════════════════════════════════════════════
# STEP 10: Sandboxed Code Execution
# Using Anthropic's server-side secure container
# SDK Feature: code_execution tool (runs in Anthropic's sandbox)
# ═══════════════════════════════════════════════════════════

def run_sandboxed_agent(task: str, verbose: bool = True) -> str:
    """
    Agent with access to Anthropic's sandboxed code execution tool.
    
    The sandbox:
      - Runs Python 3.11 in an isolated container
      - No network access (internet completely disabled)
      - 5 GiB RAM, 5 GiB disk
      - Pre-installed: pandas, numpy, matplotlib, scikit-learn, etc.
      - Container expires after 30 days
    
    This replaces the need to set up your own Docker-based sandbox.
    """
    print(f"\n{'='*50}\n[STEP 10: Sandboxed Code Execution]\n{'='*50}")

    response = client.beta.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        betas=["code-execution-2025-08-25"],    # ← enable the sandbox
        tools=[{"type": "code_execution_20250825"}],
        system=(
            "You are a data analyst. Write and execute Python code to complete tasks. "
            "Always use pandas for data manipulation and matplotlib for visualization. "
            "Print your results clearly."
        ),
        messages=[{"role": "user", "content": task}],
    )

    if verbose:
        for block in response.content:
            btype = getattr(block, "type", "unknown")
            if btype == "text":
                print(f"\n[Claude]: {block.text[:500]}")
            elif btype == "tool_use":
                print(f"\n[Code executed]:")
                if hasattr(block, "input") and "code" in block.input:
                    print(block.input["code"][:400])
            elif btype == "tool_result":
                print(f"\n[Output]: {str(block)[:400]}")

    return next(
        (b.text for b in response.content if getattr(b, "type", "") == "text"),
        "No text output"
    )


# ═══════════════════════════════════════════════════════════
# STEP 11: LLM-as-Judge Evaluation Suite
# Build a pipeline that tests your agent and grades outputs
# ═══════════════════════════════════════════════════════════

@dataclass
class TestCase:
    id: str
    task: str
    expected_elements: list[str]  # things that MUST appear in the output
    forbidden_elements: list[str] = None  # things that MUST NOT appear


class EvalResult(BaseModel):
    test_id: str
    verdict: Literal["pass", "fail"]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    missing_elements: list[str]
    issues: list[str]


def llm_judge(test: TestCase, agent_output: str) -> EvalResult:
    """
    Use Claude as a judge to evaluate another Claude's output.
    This is the LLM-as-judge pattern for automated evaluation.
    
    IMPORTANT: Use a SEPARATE client call for judging — never the same
               conversation context as the agent being evaluated.
    """
    response = client.messages.parse(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        output_format=EvalResult,
        system=(
            "You are a strict technical evaluator. "
            "Grade the agent's output objectively based on the criteria. "
            "A 'pass' requires ALL expected elements to be present and NO forbidden elements."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Task: {test.task}\n\n"
                f"Required elements (ALL must be present):\n"
                f"{json.dumps(test.expected_elements)}\n\n"
                f"Forbidden elements (NONE must be present):\n"
                f"{json.dumps(test.forbidden_elements or [])}\n\n"
                f"Agent output:\n{agent_output}\n\n"
                "Evaluate whether the output meets the criteria. "
                "Set test_id to: " + test.id
            )
        }],
    )
    return response.content[0].parsed


def run_evaluation_suite(
    agent_fn: Callable[[str], str],
    test_cases: list[TestCase],
) -> dict:
    """
    Run a full evaluation suite against an agent function.
    Returns a report with pass rates and detailed results.
    """
    print(f"\n{'='*50}\n[STEP 11: Evaluation Suite]\n{'='*50}")
    results = []
    passed = 0

    for test in test_cases:
        print(f"\n📋 Test: {test.id}")
        print(f"   Task: {test.task[:80]}...")

        try:
            output = agent_fn(test.task)
            result = llm_judge(test, output)
            results.append(result)

            status = "✅ PASS" if result.verdict == "pass" else "❌ FAIL"
            print(f"   {status} (score: {result.score:.0%})")
            if result.issues:
                print(f"   Issues: {result.issues[:2]}")
            if result.verdict == "pass":
                passed += 1

        except Exception as e:
            print(f"   💥 ERROR: {e}")
            results.append(EvalResult(
                test_id=test.id,
                verdict="fail",
                score=0.0,
                reasoning=f"Agent raised exception: {e}",
                missing_elements=[],
                issues=[str(e)],
            ))

    pass_rate = passed / len(test_cases) if test_cases else 0

    print(f"\n{'─'*50}")
    print(f"RESULTS: {passed}/{len(test_cases)} passed ({pass_rate:.0%})")

    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": len(test_cases),
        "results": [r.model_dump() for r in results],
    }


# ═══════════════════════════════════════════════════════════
# STEP 12: Human-in-the-Loop (HITL) Approval Gate
# Pause the agent on high-stakes actions and wait for human approval
# ═══════════════════════════════════════════════════════════

# Define which tools require human approval
HIGH_STAKES_TOOLS = {"send_email", "write_database", "deploy_code", "delete_files"}

class HITLDecision(BaseModel):
    action: Literal["approve", "reject", "modify"]
    modified_input: dict | None = None
    reason: str


def mock_approval_ui(tool_name: str, tool_input: dict) -> HITLDecision:
    """
    In production: send to Slack/email/web UI and wait for response.
    Here: simulate a human approving or modifying the action.
    
    Real implementations:
      - Slack: send message, wait for button click
      - Email: send approval link, poll for response
      - Web UI: show modal, block until user responds
    """
    print(f"\n{'🚨'*20}")
    print(f"HUMAN APPROVAL REQUIRED")
    print(f"  Tool   : {tool_name}")
    print(f"  Input  : {json.dumps(tool_input, indent=4)}")
    print(f"{'🚨'*20}")
    print("(Auto-approving for demo — in production, pause here for human input)")
    time.sleep(1)

    # Simulate human approval
    return HITLDecision(action="approve", reason="Demo auto-approval")


def run_hitl_agent(task: str, verbose: bool = True) -> str:
    """
    Agent with Human-in-the-Loop approval gates.
    High-stakes tools are paused until a human approves.
    """
    print(f"\n{'='*50}\n[STEP 12: HITL Agent]\n{'='*50}")

    # All tools available to the agent
    TOOLS = [
        {
            "name": "read_data",
            "description": "Read records from the database. Safe, no approval needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "send_email",
            "description": "Send an email. REQUIRES human approval before execution.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"]
            }
        },
        {
            "name": "write_database",
            "description": "Write a record to the database. REQUIRES human approval.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "record": {"type": "object"},
                },
                "required": ["table", "record"]
            }
        }
    ]

    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            tools=TOOLS,
            system=(
                "You are a business automation agent. "
                "Some actions require human approval (send_email, write_database). "
                "Always confirm what you're about to do before calling those tools."
            ),
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_input = block.input

                # ── HITL GATE ─────────────────────────────
                if block.name in HIGH_STAKES_TOOLS:
                    decision = mock_approval_ui(block.name, tool_input)

                    if decision.action == "reject":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"ACTION REJECTED by human. Reason: {decision.reason}",
                            "is_error": True,
                        })
                        continue

                    if decision.action == "modify" and decision.modified_input:
                        tool_input = decision.modified_input
                        print(f"[HITL] Human modified input to: {tool_input}")

                # ── Execute the (approved) tool ────────────
                if block.name == "read_data":
                    result = f"[DB Read] Found 3 records matching: {tool_input['query']}"
                elif block.name == "send_email":
                    result = f"[Email sent] To: {tool_input['to']} | Subject: {tool_input['subject']}"
                elif block.name == "write_database":
                    result = f"[DB Write] Inserted 1 record into {tool_input['table']}"
                else:
                    result = f"Unknown tool: {block.name}"

                if verbose:
                    print(f"\n[Executed] {block.name}: {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Step 11: Evaluation suite
    def simple_agent(task: str) -> str:
        r = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=512,
            messages=[{"role": "user", "content": task}]
        )
        return r.content[0].text

    test_cases = [
        TestCase(
            id="tc-001",
            task="Write a Python function to reverse a string",
            expected_elements=["def ", "return", "[::-1]"],
            forbidden_elements=["import os", "subprocess"],
        ),
        TestCase(
            id="tc-002",
            task="Explain what a Python decorator is in one paragraph",
            expected_elements=["function", "wrap", "@"],
            forbidden_elements=[],
        ),
        TestCase(
            id="tc-003",
            task="What is 2 + 2?",
            expected_elements=["4"],
            forbidden_elements=[],
        ),
    ]

    report = run_evaluation_suite(simple_agent, test_cases)
    print(f"\nEval Report: {json.dumps(report, indent=2)[:500]}")

    # Step 12: HITL
    print("\n" + "=" * 60)
    result = run_hitl_agent(
        "Check our database for new customer signups from today, "
        "then send a welcome email to each one, "
        "and log the sent emails back to the database."
    )
    print(f"\nFinal: {result[:300]}")


print("""
KEY TAKEAWAYS
=============
Step 10 — Sandboxed Code Execution:
1. Use Anthropic's code_execution tool for zero-setup sandboxing.
2. It runs in a fully isolated container — no network, no host access.
3. Pre-installed data science stack (pandas, numpy, matplotlib, sklearn).
4. For your own Docker sandbox: use --network none --read-only flags.

Step 11 — Evaluation Suite:
1. LLM-as-judge uses a SEPARATE Claude call — never the agent's context.
2. Use Pydantic structured outputs for the judge → deterministic verdicts.
3. Define expected_elements and forbidden_elements per test case.
4. Run this suite in CI/CD before deploying agent updates.
5. Pass rate < 80% = do not deploy. Fix the agent first.

Step 12 — HITL:
1. HIGH_STAKES_TOOLS is your "permission bridge" — explicit allowlist.
2. The approval gate PAUSES execution — not cancels, just waits.
3. The human can APPROVE, REJECT, or MODIFY the tool input.
4. Rejections send is_error=True back to Claude → it adapts gracefully.
5. In production: Slack button clicks, email approval links, or web modals.

PUTTING IT ALL TOGETHER — The Complete Harness:
  ┌─────────────────────────────────────────────────┐
  │              AGENT HARNESS                       │
  │                                                  │
  │  Cognitive Framework (system prompt + agents.md) │
  │  ┌─────────────────────────────────────────┐    │
  │  │  State Machine (Plan→Execute→Review)     │    │
  │  │  ┌───────────────────────────────────┐  │    │
  │  │  │  ReAct Loop (tool_use while loop) │  │    │
  │  │  │  ┌─────────────────────────────┐  │  │    │
  │  │  │  │  ACI Tools                  │  │  │    │
  │  │  │  │  • RAG (Map layer)          │  │  │    │
  │  │  │  │  • SQL (Observation)        │  │  │    │
  │  │  │  │  • Code Execution (Compute) │  │  │    │
  │  │  │  │  • Email/DB (Action)        │  │  │    │
  │  │  │  └──────────┬──────────────────┘  │  │    │
  │  │  │             │ is_error feedback    │  │    │
  │  │  └─────────────▼───────────────────  │  │    │
  │  │      Refinement Loop (self-correct)   │  │    │
  │  └───────────────────────────────────────┘  │    │
  │  Context Management (RAG + compaction)       │    │
  │  Guardrails: HITL + LLM-as-judge + limits    │    │
  └─────────────────────────────────────────────┘
""")
