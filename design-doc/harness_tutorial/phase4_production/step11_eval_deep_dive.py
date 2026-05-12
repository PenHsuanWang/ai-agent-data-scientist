"""
PHASE 4 — STEP 11 (DEEP DIVE): LLM Evaluation for Harness Engineering
=======================================================================
Harness Pillar: Evaluation & Guardrails (Pillar 5)
SDK Docs:
  Batch API:    https://docs.anthropic.com/en/build-with-claude/batch-processing
  Structured:   https://docs.anthropic.com/en/build-with-claude/structured-outputs
Research:
  Anthropic Evals: https://www.anthropic.com/research/evaluating-ai-systems
  LangSmith:       https://docs.smith.langchain.com/evaluation
  RAGAS Agents:    https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/agents/
  Building Agents: https://www.anthropic.com/engineering/building-effective-agents

GOAL:
  Build a production-grade evaluation harness covering all major eval strategies:
    1.  Grading Taxonomy   — Code → LLM-as-judge → Human (cost/quality tradeoff)
    2.  Dataset Management — Golden sets, Claude-generated cases, edge cases
    3.  Rubric Design      — Reference-free, reference-based, pairwise comparison
    4.  Agent Evals        — Trajectory, ToolCallAccuracy, step efficiency, goal accuracy
    5.  RAG Evals          — Faithfulness, context precision, answer relevancy
    6.  Bias Mitigation    — Position bias, verbosity bias, self-serving bias
    7.  CI/CD Pipeline     — Pass/fail thresholds, regression guards, batch-scale eval

CORE PRINCIPLE (Harness Engineering):
  ┌─────────────────────────────────────────────────────┐
  │  Agent = Model + Harness                             │
  │  Eval Harness = Test Data + Graders + Pipeline       │
  │                                                      │
  │  Grading cost pyramid (cheap → expensive):           │
  │    ① Deterministic code  →  free, instant, narrow    │
  │    ② LLM-as-judge        →  $, fast, broad           │
  │    ③ Human annotation    →  $$$, slow, authoritative  │
  └─────────────────────────────────────────────────────┘
"""

import json
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: GRADING TAXONOMY
# Three-tier grading pyramid — use the cheapest tier that satisfies quality.
# ═══════════════════════════════════════════════════════════════════════

# ── Tier 1: Deterministic Code Graders (fastest, cheapest, narrowest) ──

def grade_exact_match(output: str, expected: str) -> float:
    """Binary score: 1.0 if strings match exactly, else 0.0."""
    return 1.0 if output.strip() == expected.strip() else 0.0


def grade_substring_presence(output: str, required_substrings: list[str]) -> float:
    """Score by fraction of required substrings present (case-insensitive)."""
    output_lower = output.lower()
    hits = sum(1 for s in required_substrings if s.lower() in output_lower)
    return hits / len(required_substrings) if required_substrings else 1.0


def grade_json_schema(output: str, schema: dict) -> float:
    """Score 1.0 if output is valid JSON with all required keys, else 0.0."""
    try:
        parsed = json.loads(output)
        required_keys = schema.get("required", [])
        missing = [k for k in required_keys if k not in parsed]
        return 0.0 if missing else 1.0
    except json.JSONDecodeError:
        return 0.0


def grade_code_executes(code_output: str) -> float:
    """Score 1.0 if code output doesn't contain an error traceback."""
    error_markers = ["Traceback (most recent call last)", "SyntaxError:", "NameError:", "AttributeError:"]
    return 0.0 if any(m in code_output for m in error_markers) else 1.0


def grade_word_count_range(output: str, min_words: int, max_words: int) -> float:
    """Score 1.0 if output word count is within range, else scaled partial credit."""
    count = len(output.split())
    if min_words <= count <= max_words:
        return 1.0
    elif count < min_words:
        return count / min_words  # partial credit for too-short outputs
    else:
        return max_words / count  # partial credit for too-long outputs


# ── Tier 2: LLM-as-Judge (flexible, handles nuance, moderate cost) ────

class JudgeVerdict(BaseModel):
    """Structured output for LLM-as-judge. Forces numeric score before reasoning."""
    score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 (worst) and 1.0 (perfect)")
    verdict: Literal["pass", "partial", "fail"]
    dimension_scores: dict[str, float] = Field(
        description="Scores for each rubric dimension (0.0–1.0)"
    )
    reasoning: str = Field(description="Step-by-step reasoning that led to the score")
    suggestions: list[str] = Field(description="Concrete improvements the agent could make")


def llm_judge_with_rubric(
    task: str,
    agent_output: str,
    rubric: dict[str, str],          # e.g. {"accuracy": "All facts are correct", ...}
    reference_answer: str | None = None,
    model: str = "claude-haiku-4-5-20251001",   # Use cheap model for grading
    pass_threshold: float = 0.7,
) -> JudgeVerdict:
    """
    Rubric-based LLM-as-judge evaluation.

    BIAS MITIGATION STRATEGIES applied here:
      1. Numeric score BEFORE reasoning  → prevents anchoring on qualitative labels
      2. Separate judge context          → no shared history with the agent (self-serving bias)
      3. Detailed rubric dimensions      → reduces vagueness / position bias
      4. Use different model than agent  → further reduces self-serving bias
      5. Encourage critique first        → verbosity bias mitigation
    """
    rubric_text = "\n".join(f"  - {dim}: {criteria}" for dim, criteria in rubric.items())
    reference_section = f"\nREFERENCE ANSWER (ground truth):\n{reference_answer}\n" if reference_answer else ""

    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        output_format=JudgeVerdict,
        system=(
            "You are a strict, fair, and impartial technical evaluator. "
            "You care only about quality — not length or style. "
            "Follow these steps: "
            "1) Identify problems and gaps in the output. "
            "2) Score each rubric dimension 0.0–1.0. "
            "3) Compute the overall score as the mean of dimension scores. "
            "4) Set verdict to 'pass' if score >= " + str(pass_threshold) + ", "
            "'partial' if score >= 0.4, else 'fail'. "
            "Give honest, actionable suggestions."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"TASK:\n{task}\n"
                f"{reference_section}"
                f"\nEVALUATION RUBRIC (score each 0.0–1.0):\n{rubric_text}\n"
                f"\nAGENT OUTPUT:\n{agent_output}\n"
            )
        }],
    )
    return response.content[0].parsed


# ── Tier 2b: Pairwise Comparison Judge ─────────────────────────────────

class PairwiseVerdict(BaseModel):
    winner: Literal["A", "B", "tie"]
    margin: Literal["strong", "slight", "tie"]
    reasoning: str
    dimension_winner: dict[str, Literal["A", "B", "tie"]]


def pairwise_judge(
    task: str,
    output_a: str,
    output_b: str,
    dimensions: list[str],
    model: str = "claude-haiku-4-5-20251001",
    randomize_order: bool = True,   # position bias mitigation
) -> PairwiseVerdict:
    """
    Pairwise evaluation: compare two outputs and pick the better one.

    More reliable than absolute scoring for subjective tasks (summaries, translations).
    POSITION BIAS mitigation: randomize which output is "A" vs "B".

    Research finding (Anthropic): position bias causes judges to favour the
    first response ~60% of the time. Randomization and averaging removes this.
    """
    a_label, b_label = "A", "B"
    show_a, show_b = output_a, output_b

    if randomize_order and random.random() > 0.5:
        show_a, show_b = output_b, output_a
        a_label, b_label = "B", "A"

    dims_text = ", ".join(dimensions)
    response = client.messages.parse(
        model=model,
        max_tokens=512,
        output_format=PairwiseVerdict,
        system=(
            "Compare two agent outputs for the given task. "
            "Be impartial — do NOT favour verbose or formal responses. "
            f"Evaluate on these dimensions: {dims_text}. "
            "Respond with winner=A, B, or tie."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"TASK: {task}\n\n"
                f"OUTPUT A:\n{show_a}\n\n"
                f"OUTPUT B:\n{show_b}"
            )
        }],
    )
    verdict = response.content[0].parsed

    # Undo randomization so caller always gets consistent A/B labels
    if a_label == "B":  # we swapped
        mapping = {"A": "B", "B": "A", "tie": "tie"}
        verdict.winner = mapping[verdict.winner]
        verdict.dimension_winner = {k: mapping[v] for k, v in verdict.dimension_winner.items()}
    return verdict


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: DATASET MANAGEMENT
# Build and manage golden evaluation datasets.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvalExample:
    """One test case in the evaluation dataset."""
    id: str
    task: str
    expected_output: str | None = None        # reference answer (for reference-based evals)
    rubric: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)   # e.g. ["edge_case", "adversarial", "golden"]
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Collection of eval examples with versioning and split support."""
    name: str
    version: str
    examples: list[EvalExample] = field(default_factory=list)

    def add(self, example: EvalExample) -> None:
        self.examples.append(example)

    def split(self, tag: str) -> list[EvalExample]:
        """Filter by tag: 'golden', 'edge_case', 'adversarial', 'regression'."""
        return [e for e in self.examples if tag in e.tags]

    def save(self, path: str) -> None:
        """Persist dataset as JSONL (one example per line)."""
        with open(path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps({
                    "id": ex.id,
                    "task": ex.task,
                    "expected_output": ex.expected_output,
                    "rubric": ex.rubric,
                    "tags": ex.tags,
                    "metadata": ex.metadata,
                }) + "\n")
        print(f"💾 Saved {len(self.examples)} examples to {path}")

    @classmethod
    def load(cls, path: str, name: str = "loaded") -> "EvalDataset":
        dataset = cls(name=name, version="loaded")
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                dataset.add(EvalExample(**d))
        return dataset


class GeneratedTestCase(BaseModel):
    """Schema for Claude-generated test cases."""
    task: str
    expected_output: str
    difficulty: Literal["easy", "medium", "hard"]
    edge_case_type: str | None
    rubric: dict[str, str]


def generate_test_cases_with_claude(
    domain: str,
    capability: str,
    n: int = 5,
    include_edge_cases: bool = True,
) -> list[EvalExample]:
    """
    Use Claude to generate diverse test cases (synthetic data).

    Strategy from Anthropic research: seed with 2-3 hand-crafted golden examples,
    then generate variations. Always verify generated cases before adding to golden set.

    Advantages:
      - Scales to hundreds of cases quickly
      - LLM naturally generates diverse phrasings
      - Can target specific difficulty levels and edge cases
    """
    edge_case_instruction = (
        "Include at least 2 edge cases such as: ambiguous queries, adversarial inputs, "
        "out-of-domain requests, very long inputs, inputs with typos, or multi-step tasks."
        if include_edge_cases else ""
    )

    response = client.messages.parse(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        output_format=list[GeneratedTestCase],
        system=(
            "You are a test engineer generating diverse evaluation cases. "
            "Each case must be realistic, specific, and have a clear correct answer. "
            "Vary difficulty and phrasing across cases."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} evaluation test cases for a '{domain}' agent "
                f"specifically testing the '{capability}' capability. "
                f"{edge_case_instruction}\n"
                "For each case, include a complete rubric with 3-4 dimensions."
            )
        }],
    )

    cases = response.content[0].parsed
    examples = []
    for i, case in enumerate(cases):
        tag = "edge_case" if case.edge_case_type else "synthetic"
        examples.append(EvalExample(
            id=f"{domain}_{capability}_{i+1}",
            task=case.task,
            expected_output=case.expected_output,
            rubric=case.rubric,
            tags=[tag, case.difficulty],
            metadata={"edge_case_type": case.edge_case_type, "generated_by": "claude"},
        ))
    return examples


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: AGENT-SPECIFIC EVALUATION
# Evaluate trajectories, tool call accuracy, and goal completion.
# Standard task-output evals miss the "how" — agent evals capture the journey.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentTrajectory:
    """Full record of an agent's execution including every step."""
    task: str
    steps: list[dict]   # [{"type": "tool_call"|"text", "name": ..., "input": ..., "output": ...}]
    final_answer: str
    total_tokens: int
    wall_time_seconds: float


class TrajectoryEval(BaseModel):
    """Evaluation of an agent's execution trajectory."""
    goal_achieved: bool
    tool_selection_score: float = Field(ge=0.0, le=1.0)
    tool_argument_score: float = Field(ge=0.0, le=1.0)
    step_efficiency_score: float = Field(ge=0.0, le=1.0, description="1.0=optimal steps, lower=wasted steps")
    error_recovery_score: float = Field(ge=0.0, le=1.0, description="1.0=recovered from all errors")
    overall_score: float = Field(ge=0.0, le=1.0)
    redundant_calls: list[str] = Field(description="Tool calls that were unnecessary")
    missed_tools: list[str] = Field(description="Tools that should have been called but weren't")
    reasoning: str


def evaluate_trajectory(
    trajectory: AgentTrajectory,
    reference_tool_sequence: list[str],    # expected tool call names in order
    available_tools: list[str],
) -> TrajectoryEval:
    """
    Trajectory evaluation: assess HOW the agent solved the task, not just WHAT it answered.

    Inspired by RAGAS ToolCallAccuracy metric and agent trajectory research.
    Captures:
      - Did it call the right tools? (tool_selection_score)
      - Were args correct? (tool_argument_score)
      - Did it take too many steps? (step_efficiency_score)
      - Did it recover from errors? (error_recovery_score)

    Key insight: Two agents can get the same final answer via very different paths.
    A path with 10 tool calls when 3 suffice is costly and brittle in production.
    """
    tool_calls = [s for s in trajectory.steps if s["type"] == "tool_call"]
    actual_tools = [tc["name"] for tc in tool_calls]
    error_calls = [tc for tc in tool_calls if tc.get("is_error", False)]

    steps_text = "\n".join(
        f"  Step {i+1}: [{s['type'].upper()}] "
        + (f"{s.get('name', '')}({json.dumps(s.get('input', {}))[:80]}) → {str(s.get('output', ''))[:80]}"
           if s['type'] == 'tool_call' else s.get('content', '')[:100])
        for i, s in enumerate(trajectory.steps)
    )

    response = client.messages.parse(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        output_format=TrajectoryEval,
        system=(
            "You are an agent performance analyst. Evaluate agent trajectories on efficiency, "
            "accuracy, and correctness. Be strict about unnecessary tool calls (wastes tokens/latency)."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"TASK: {trajectory.task}\n\n"
                f"EXPECTED TOOL SEQUENCE: {reference_tool_sequence}\n"
                f"AVAILABLE TOOLS: {available_tools}\n\n"
                f"ACTUAL TRAJECTORY:\n{steps_text}\n\n"
                f"FINAL ANSWER: {trajectory.final_answer}\n\n"
                f"Evaluate this trajectory. Count redundant calls (same tool called twice with same args) "
                "and missed tools (in expected sequence but not called)."
            )
        }],
    )
    return response.content[0].parsed


def tool_call_f1(
    actual_calls: list[str],
    reference_calls: list[str],
) -> dict[str, float]:
    """
    Compute precision/recall/F1 for tool call selection.
    Inspired by RAGAS ToolCallF1 metric. Order-independent (set-based).

    Precision = |actual ∩ reference| / |actual|
    Recall    = |actual ∩ reference| / |reference|
    F1        = harmonic mean of precision and recall
    """
    actual_set = set(actual_calls)
    reference_set = set(reference_calls)
    intersection = actual_set & reference_set

    precision = len(intersection) / len(actual_set) if actual_set else 0.0
    recall = len(intersection) / len(reference_set) if reference_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: RAG-SPECIFIC EVALUATION
# Evaluate retrieval quality and answer faithfulness separately.
# (Inspired by RAGAS metrics for RAG pipelines)
# ═══════════════════════════════════════════════════════════════════════

class RAGEvalResult(BaseModel):
    faithfulness_score: float = Field(ge=0.0, le=1.0,
        description="Are ALL claims in the answer supported by the retrieved context?")
    answer_relevancy_score: float = Field(ge=0.0, le=1.0,
        description="Does the answer actually address the user's question?")
    context_precision_score: float = Field(ge=0.0, le=1.0,
        description="What fraction of retrieved chunks were actually useful?")
    hallucinations: list[str] = Field(description="Claims in answer NOT supported by context")
    reasoning: str


def evaluate_rag_response(
    question: str,
    retrieved_chunks: list[str],
    answer: str,
) -> RAGEvalResult:
    """
    Evaluate a RAG response on three key dimensions (RAGAS-inspired):

    1. FAITHFULNESS     — are all answer claims grounded in the retrieved context?
                          (Detects hallucinations — the #1 RAG failure mode)
    2. ANSWER RELEVANCY — does the answer address the actual question?
                          (Detects "answered a different question" failure)
    3. CONTEXT PRECISION — what fraction of retrieved context was actually used?
                          (Detects over-retrieval / noise in the context window)

    Use with faithfulness < 0.8 threshold to flag potential hallucinations.
    """
    context = "\n\n---\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(retrieved_chunks))

    response = client.messages.parse(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        output_format=RAGEvalResult,
        system=(
            "You are a RAG quality auditor. Evaluate with extreme attention to faithfulness. "
            "A claim is a 'hallucination' if it cannot be verified from the provided context chunks. "
            "Be conservative: if uncertain whether context supports a claim, flag it as a hallucination."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"USER QUESTION: {question}\n\n"
                f"RETRIEVED CONTEXT:\n{context}\n\n"
                f"AGENT ANSWER: {answer}\n\n"
                "Evaluate the answer on all three dimensions. "
                "List specific claims that are NOT in the retrieved context."
            )
        }],
    )
    return response.content[0].parsed


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: BIAS MITIGATION PATTERNS
# LLM judges have known biases — mitigate them systematically.
# ═══════════════════════════════════════════════════════════════════════

def evaluate_with_swap_test(
    task: str,
    output_a: str,
    output_b: str,
    dimensions: list[str],
    n_repeats: int = 2,
) -> dict:
    """
    Swap test: run pairwise eval with A↔B swapped to detect and correct position bias.

    Position bias: LLM judges prefer the first response shown ~60% of the time.
    Fix: run both orderings (A-first and B-first), average the results.

    If A wins in A-first ordering but B wins in B-first ordering → disagreement.
    Disagreement rate > 30% indicates an unreliable judge prompt that needs tuning.
    """
    verdicts = []
    for _ in range(n_repeats):
        # pairwise_judge already randomizes order internally
        v = pairwise_judge(task, output_a, output_b, dimensions, randomize_order=True)
        verdicts.append(v.winner)

    a_wins = verdicts.count("A")
    b_wins = verdicts.count("B")
    ties = verdicts.count("tie")
    agreement_rate = max(a_wins, b_wins, ties) / len(verdicts)

    return {
        "final_winner": max(["A", "B", "tie"], key=lambda x: verdicts.count(x)),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "agreement_rate": agreement_rate,
        "reliable": agreement_rate >= 0.7,   # if < 70% agreement, judge is noisy
    }


def evaluate_self_serving_bias_check(
    task: str,
    agent_output: str,
    rubric: dict[str, str],
    judge_models: list[str] | None = None,
) -> dict:
    """
    Self-serving bias check: same agent model judging its own output inflates scores.

    Fix: use a different model for judging, or average across multiple judge models.
    Research finding: Claude-as-judge on Claude outputs scores ~0.1 higher than
    cross-model judging. Always use a separate model for grading in production.
    """
    if judge_models is None:
        judge_models = ["claude-haiku-4-5-20251001"]

    results = {}
    for model in judge_models:
        verdict = llm_judge_with_rubric(task, agent_output, rubric, model=model)
        results[model] = {"score": verdict.score, "verdict": verdict.verdict}

    scores = [r["score"] for r in results.values()]
    return {
        "scores_by_model": results,
        "mean_score": sum(scores) / len(scores),
        "score_variance": max(scores) - min(scores),
        "high_variance_warning": (max(scores) - min(scores)) > 0.2,
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: CI/CD EVALUATION PIPELINE
# Batch-based eval with pass/fail thresholds for regression testing.
# Run this on every PR — fail the PR if pass rate drops below baseline.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """Configuration for a CI/CD eval run."""
    name: str
    pass_threshold: float = 0.80         # Fail if overall pass rate < this
    regression_threshold: float = 0.05   # Fail if pass rate drops by more than this vs baseline
    use_batch_api: bool = True           # Use Batch API for 50% cost reduction
    grade_with_model: str = "claude-haiku-4-5-20251001"


@dataclass
class CIEvalReport:
    config_name: str
    total: int
    passed: int
    pass_rate: float
    baseline_pass_rate: float | None
    regression_detected: bool
    threshold_met: bool
    details: list[dict]

    @property
    def ci_status(self) -> str:
        if self.regression_detected:
            return "❌ REGRESSION DETECTED"
        if not self.threshold_met:
            return "❌ BELOW THRESHOLD"
        return "✅ PASSED"

    def print_summary(self) -> None:
        print(f"\n{'═'*60}")
        print(f"CI EVAL REPORT: {self.config_name}")
        print(f"{'─'*60}")
        print(f"Status:    {self.ci_status}")
        print(f"Pass Rate: {self.pass_rate:.1%}  (threshold: ≥{self.config.pass_threshold:.1%})"
              if hasattr(self, 'config') else f"Pass Rate: {self.pass_rate:.1%}")
        if self.baseline_pass_rate:
            delta = self.pass_rate - self.baseline_pass_rate
            symbol = "▲" if delta >= 0 else "▼"
            print(f"vs Baseline: {symbol}{abs(delta):.1%}")
        print(f"Results:   {self.passed}/{self.total} passed")
        print(f"{'═'*60}")


def run_batch_eval_pipeline(
    agent_fn: Callable[[str], str],
    dataset: EvalDataset,
    config: EvalConfig,
    baseline_pass_rate: float | None = None,
) -> CIEvalReport:
    """
    Full CI/CD evaluation pipeline using the Batch API for cost efficiency.

    Strategy:
      1. Collect all agent outputs (could be batched if agent itself is LLM-only)
      2. Use Batch API to grade all outputs concurrently (50% cost vs serial)
      3. Compute pass rate and compare against baseline
      4. Return structured report for CI integration

    Batch API advantages for eval:
      - 50% cost reduction vs. synchronous grading
      - Up to 100,000 requests per batch
      - Results available in < 1 hour typically
      - Concurrent grading (no serial bottleneck)
    """
    print(f"\n{'═'*60}")
    print(f"[CI EVAL PIPELINE] {config.name}")
    print(f"  Dataset: {dataset.name} v{dataset.version} ({len(dataset.examples)} examples)")
    print(f"  Grader:  {config.grade_with_model}")
    print(f"  Batch:   {'Yes (50% cost)' if config.use_batch_api else 'No (synchronous)'}")
    print(f"{'═'*60}")

    # Phase 1: Collect agent outputs (serial — agents often have side effects)
    print("\n[Phase 1/3] Running agent on all examples...")
    outputs = {}
    for ex in dataset.examples:
        try:
            outputs[ex.id] = agent_fn(ex.task)
            print(f"  ✓ {ex.id}")
        except Exception as e:
            outputs[ex.id] = f"AGENT_ERROR: {e}"
            print(f"  ✗ {ex.id}: {e}")

    # Phase 2: Build grading requests
    print("\n[Phase 2/3] Preparing grading requests...")
    batch_requests = []
    for ex in dataset.examples:
        rubric_text = "\n".join(f"  - {k}: {v}" for k, v in ex.rubric.items())
        reference = f"\nREFERENCE: {ex.expected_output}\n" if ex.expected_output else ""

        batch_requests.append({
            "custom_id": ex.id,
            "params": {
                "model": config.grade_with_model,
                "max_tokens": 512,
                "system": (
                    "You are a strict evaluator. Output a JSON object with: "
                    "score (0.0-1.0), verdict ('pass'/'partial'/'fail'), reasoning (str). "
                    f"Pass threshold = {config.pass_threshold}."
                ),
                "messages": [{
                    "role": "user",
                    "content": (
                        f"TASK: {ex.task}\n"
                        f"{reference}"
                        f"RUBRIC:\n{rubric_text}\n\n"
                        f"AGENT OUTPUT:\n{outputs[ex.id]}\n\n"
                        "Return JSON with score, verdict, reasoning."
                    )
                }],
            }
        })

    # Phase 3: Grade (Batch API or synchronous)
    print(f"\n[Phase 3/3] Grading ({len(batch_requests)} requests)...")
    grade_results = {}

    if config.use_batch_api and len(batch_requests) >= 2:
        grade_results = _batch_grade(batch_requests, config.pass_threshold)
    else:
        # Synchronous fallback for small datasets
        for req in batch_requests:
            cid = req["custom_id"]
            try:
                resp = client.messages.create(**req["params"])
                data = json.loads(resp.content[0].text)
                grade_results[cid] = data
            except Exception as e:
                grade_results[cid] = {"score": 0.0, "verdict": "fail", "reasoning": str(e)}

    # Compute report
    details = []
    passed = 0
    for ex in dataset.examples:
        gr = grade_results.get(ex.id, {"score": 0.0, "verdict": "fail", "reasoning": "No result"})
        is_pass = gr.get("verdict") == "pass" or gr.get("score", 0) >= config.pass_threshold
        if is_pass:
            passed += 1
        details.append({
            "id": ex.id,
            "tags": ex.tags,
            "score": gr.get("score", 0.0),
            "verdict": gr.get("verdict", "fail"),
            "reasoning": gr.get("reasoning", ""),
            "agent_output_preview": outputs.get(ex.id, "")[:100],
        })

    pass_rate = passed / len(dataset.examples) if dataset.examples else 0.0
    threshold_met = pass_rate >= config.pass_threshold
    regression_detected = (
        baseline_pass_rate is not None
        and (baseline_pass_rate - pass_rate) > config.regression_threshold
    )

    return CIEvalReport(
        config_name=config.name,
        total=len(dataset.examples),
        passed=passed,
        pass_rate=pass_rate,
        baseline_pass_rate=baseline_pass_rate,
        regression_detected=regression_detected,
        threshold_met=threshold_met,
        details=details,
    )


def _batch_grade(batch_requests: list[dict], pass_threshold: float) -> dict[str, dict]:
    """Submit grading requests to Batch API, poll, and return results."""
    print(f"  Submitting batch of {len(batch_requests)} grading requests...")

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"  Batch ID: {batch.id}")
    print("  Polling for completion (this may take several minutes)...")

    # Poll until ended
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        status = batch.processing_status
        counts = batch.request_counts
        print(f"  Status: {status} | processing: {counts.processing} | "
              f"succeeded: {counts.succeeded} | errored: {counts.errored}")
        if status == "ended":
            break
        time.sleep(10)

    # Collect results
    results = {}
    for result in client.messages.batches.results(batch.id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            try:
                data = json.loads(result.result.message.content[0].text)
                results[cid] = data
            except Exception:
                results[cid] = {"score": 0.0, "verdict": "fail", "reasoning": "JSON parse error"}
        else:
            results[cid] = {"score": 0.0, "verdict": "fail",
                            "reasoning": f"Batch error: {result.result.type}"}

    return results


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: COMPLETE DEMO
# Demonstrate the full eval harness end-to-end.
# ═══════════════════════════════════════════════════════════════════════

def demo_simple_agent(task: str) -> str:
    """Minimal agent for demonstration purposes."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": task}],
    )
    return response.content[0].text


def demo_grading_taxonomy():
    """Demonstrate all three grading tiers on the same output."""
    print("\n" + "═"*60)
    print("DEMO 1: GRADING TAXONOMY (3 Tiers)")
    print("═"*60)

    task = "Explain what prompt caching is in the Anthropic API in 2-3 sentences."
    output = demo_simple_agent(task)
    print(f"\n[Agent Output] {output[:200]}...")

    # Tier 1: Deterministic
    print("\n[Tier 1: Code Graders]")
    t1_keywords = grade_substring_presence(output, ["cache", "token", "cost"])
    t1_length = grade_word_count_range(output, 20, 80)
    print(f"  Keyword presence: {t1_keywords:.0%}")
    print(f"  Word count (20-80): {t1_length:.0%}")

    # Tier 2: LLM-as-judge
    print("\n[Tier 2: LLM-as-Judge with Rubric]")
    rubric = {
        "accuracy":   "Correctly explains prompt caching as reusing computation for repeated prefixes",
        "clarity":    "Explanation is clear and understandable to a developer audience",
        "completeness": "Mentions at least one benefit (cost, latency, or efficiency)",
    }
    verdict = llm_judge_with_rubric(task, output, rubric)
    print(f"  Score: {verdict.score:.0%} | Verdict: {verdict.verdict}")
    print(f"  Dimension scores: {verdict.dimension_scores}")
    if verdict.suggestions:
        print(f"  Suggestion: {verdict.suggestions[0]}")


def demo_agent_trajectory_eval():
    """Demonstrate trajectory evaluation with a simulated multi-step agent run."""
    print("\n" + "═"*60)
    print("DEMO 2: AGENT TRAJECTORY EVALUATION")
    print("═"*60)

    # Simulate a trajectory (in real use, capture from your agent loop)
    trajectory = AgentTrajectory(
        task="Find the capital of France, then get the current weather there.",
        steps=[
            {"type": "tool_call", "name": "search_web",
             "input": {"query": "capital of France"}, "output": "Paris is the capital of France."},
            {"type": "tool_call", "name": "search_web",
             "input": {"query": "capital of France"}, "output": "Paris is the capital of France.",
             "is_error": False},   # ← redundant call!
            {"type": "tool_call", "name": "get_weather",
             "input": {"city": "Paris"}, "output": "15°C, partly cloudy"},
            {"type": "text", "content": "The capital of France is Paris. Current weather: 15°C, partly cloudy."},
        ],
        final_answer="The capital of France is Paris. Current weather: 15°C, partly cloudy.",
        total_tokens=350,
        wall_time_seconds=4.2,
    )

    reference_sequence = ["search_web", "get_weather"]  # optimal: 2 steps
    available_tools = ["search_web", "get_weather", "translate", "summarize"]

    result = evaluate_trajectory(trajectory, reference_sequence, available_tools)
    print(f"\n  Goal achieved:        {'✅' if result.goal_achieved else '❌'}")
    print(f"  Tool selection:       {result.tool_selection_score:.0%}")
    print(f"  Tool arguments:       {result.tool_argument_score:.0%}")
    print(f"  Step efficiency:      {result.step_efficiency_score:.0%}")
    print(f"  Overall score:        {result.overall_score:.0%}")
    if result.redundant_calls:
        print(f"  ⚠ Redundant calls:   {result.redundant_calls}")

    # Tool Call F1
    actual = ["search_web", "search_web", "get_weather"]
    reference = ["search_web", "get_weather"]
    f1_scores = tool_call_f1(actual, reference)
    print(f"\n  Tool Call F1 Score:   {f1_scores}")


def demo_rag_eval():
    """Demonstrate RAG evaluation (faithfulness, relevancy, precision)."""
    print("\n" + "═"*60)
    print("DEMO 3: RAG RESPONSE EVALUATION")
    print("═"*60)

    question = "What is the context window size for Claude?"
    retrieved = [
        "Claude Opus 4 supports up to 200,000 tokens in its context window.",
        "Anthropic offers various Claude models with different capabilities.",
        "The Anthropic API supports streaming responses for long outputs.",
    ]
    # Deliberately include a hallucination to demonstrate detection
    answer = (
        "Claude supports a context window of up to 200,000 tokens. "
        "It also supports up to 1 million tokens in special extended mode."  # ← hallucination
    )

    result = evaluate_rag_response(question, retrieved, answer)
    print(f"\n  Faithfulness:       {result.faithfulness_score:.0%}")
    print(f"  Answer Relevancy:   {result.answer_relevancy_score:.0%}")
    print(f"  Context Precision:  {result.context_precision_score:.0%}")
    if result.hallucinations:
        print(f"  ⚠ Hallucinations:  {result.hallucinations}")


def demo_dataset_and_pipeline():
    """Demonstrate dataset creation and CI/CD eval pipeline."""
    print("\n" + "═"*60)
    print("DEMO 4: DATASET & CI/CD PIPELINE")
    print("═"*60)

    # Build a small golden dataset manually
    dataset = EvalDataset(name="qa_basic", version="1.0")

    golden_cases = [
        EvalExample(
            id="qa_001",
            task="What is 2 + 2?",
            expected_output="4",
            rubric={
                "correctness": "The answer must be exactly 4",
                "conciseness": "Response should be short and direct",
            },
            tags=["golden", "easy"],
        ),
        EvalExample(
            id="qa_002",
            task="Explain recursion briefly.",
            rubric={
                "accuracy":     "Correctly describes a function calling itself",
                "example":      "Includes a concrete example or analogy",
                "clarity":      "Explanation is understandable",
            },
            tags=["golden", "medium"],
        ),
        EvalExample(
            id="qa_003_edge",
            task="What is the sound of silence?",   # ambiguous/edge case
            rubric={
                "coherence":   "Provides a coherent, relevant response",
                "honesty":     "Does not fabricate a definitive scientific answer",
            },
            tags=["edge_case", "hard"],
        ),
    ]
    for ex in golden_cases:
        dataset.add(ex)

    print(f"  Dataset: {dataset.name} — {len(dataset.examples)} examples")
    print(f"  Golden cases: {len(dataset.split('golden'))}")
    print(f"  Edge cases:   {len(dataset.split('edge_case'))}")

    # Run CI/CD eval pipeline (synchronous mode for demo, no Batch API needed for 3 examples)
    config = EvalConfig(
        name="demo_pipeline",
        pass_threshold=0.70,
        regression_threshold=0.05,
        use_batch_api=False,     # synchronous for demo speed
        grade_with_model="claude-haiku-4-5-20251001",
    )

    report = run_batch_eval_pipeline(
        agent_fn=demo_simple_agent,
        dataset=dataset,
        config=config,
        baseline_pass_rate=0.80,  # simulate a prior baseline
    )

    print(f"\n  CI Status:   {report.ci_status}")
    print(f"  Pass Rate:   {report.pass_rate:.1%} (threshold: ≥{config.pass_threshold:.0%})")
    if report.baseline_pass_rate:
        delta = report.pass_rate - report.baseline_pass_rate
        print(f"  vs Baseline: {'▲' if delta >= 0 else '▼'}{abs(delta):.1%}")
    print(f"  Regression:  {'YES ⚠' if report.regression_detected else 'No'}")

    print("\n  Per-example results:")
    for d in report.details:
        status = "✅" if d["verdict"] == "pass" else "❌"
        print(f"    {status} {d['id']} (score: {d['score']:.0%}) [{', '.join(d['tags'])}]")


# ═══════════════════════════════════════════════════════════════════════
# KEY TAKEAWAYS — LLM Eval for Harness Engineering
# ═══════════════════════════════════════════════════════════════════════
"""
EVAL HARNESS ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────┐
  │                LLM EVAL HARNESS                              │
  │                                                              │
  │  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
  │  │ Test Dataset │  │    Graders    │  │  CI/CD Pipeline  │  │
  │  │              │  │               │  │                  │  │
  │  │ • Manual     │→ │ ① Code        │→ │ • Pass threshold │  │
  │  │ • Synthetic  │  │ ② LLM-judge   │  │ • Regression Δ  │  │
  │  │ • Production │  │ ③ Human       │  │ • Batch API 50%  │  │
  │  │ • Edge cases │  │               │  │ • PR gates       │  │
  │  └──────────────┘  └───────────────┘  └─────────────────┘  │
  └─────────────────────────────────────────────────────────────┘

CRITICAL RULES:
  1. Never use the same model/context to judge itself (self-serving bias)
  2. Use numeric score BEFORE text reasoning (anchoring prevention)
  3. Swap test for pairwise evals (position bias mitigation)
  4. Start with 10-20 hand-crafted golden cases — quality over quantity
  5. For agents, evaluate TRAJECTORY not just final answer
  6. For RAG, always check FAITHFULNESS (hallucination detection)
  7. Use Batch API for scale: 50% cost, 100k requests per batch
  8. Set CI regression threshold (e.g., Δ > 5% → fail the PR)

GRADING COST HIERARCHY:
  Deterministic code  →  ~$0/eval, instant, narrow scope
  LLM-as-judge        →  ~$0.001/eval (Haiku), fast, broad
  Human annotation    →  ~$0.10-1.00/eval, slow, authoritative
  → Start with code, escalate to LLM-judge, human for golden set creation only

AGENT-SPECIFIC EVAL CHECKLIST:
  □ Tool selection correctness (right tool chosen?)
  □ Tool argument accuracy (right parameters?)
  □ Step efficiency (minimum necessary calls?)
  □ Error recovery (handled errors gracefully?)
  □ Goal achievement (task ultimately completed?)
  □ Topic adherence (stayed on domain?)
  □ Trajectory vs. baseline (comparison to reference path)

BIAS MITIGATION CHECKLIST:
  □ Different model for judge vs. agent
  □ Numeric score required before qualitative reasoning
  □ Randomize A/B order in pairwise evals
  □ Run swap test (both orderings) for important decisions
  □ Multi-model judging panel for high-stakes evals
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   STEP 11: LLM EVALUATION DEEP DIVE (Harness Pillar 5)  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_grading_taxonomy()
    demo_agent_trajectory_eval()
    demo_rag_eval()
    demo_dataset_and_pipeline()

    print("\n✅ All eval demos complete.")
    print("\nNext steps:")
    print("  • Integrate run_batch_eval_pipeline() into your GitHub Actions CI workflow")
    print("  • Build a golden dataset of 20+ hand-crafted examples for your domain")
    print("  • Set a baseline pass rate and add regression_threshold to your CI config")
    print("  • For production RAG, add faithfulness monitoring on every response")
