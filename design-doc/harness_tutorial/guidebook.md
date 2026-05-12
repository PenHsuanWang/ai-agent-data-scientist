# Harness Engineering Guidebook: AI Agents From Scratch

## Introduction: The Philosophy of Harness Engineering

> **Core Principle:** `Agent = Model + Harness`

In the rapidly evolving landscape of artificial intelligence, a common misconception is that the Large Language Model (LLM) itself is the agent. In reality, the LLM is merely the "brain"—a powerful next-token predictor. The true capability of an autonomous agent emerges from its *Harness*: the surrounding infrastructure, memory management, tool access, safety guardrails, and evaluation loops that allow the LLM to interact with the world reliably.

A naked LLM is simply a chatbot. A harnessed LLM is an autonomous, reliable, and production-ready agent. This guidebook delves into the fundamental theory and practical implementation of Harness Engineering, providing a structured curriculum to build this critical infrastructure layer by layer.

### The 5 Pillars of Harness Engineering

Harness Engineering rests on five foundational pillars. Mastering these is essential for moving AI applications from impressive demos to robust production systems.

1. **Context Management (The Brain's Workspace):** 
   The LLM's context window is its active memory (RAM). It is fast but strictly limited and expensive. Harness engineering involves dynamically assembling prompts, proactively compacting conversation history to avoid token overflow, and fetching information "just-in-time" rather than stuffing the prompt upfront.
2. **Cognitive Framework (The Agent's Worldview):** 
   An agent needs a clearly defined operational identity and behavioral boundaries. This is enforced through meticulously crafted system prompts and instruction files (like an `agents.md` file). The framework dictates how the agent should think, format its output, and what rules it must never break.
3. **ACI - Agent-Computer Interface (The Agent's Hands):** 
   Just as humans use a GUI (Graphical User Interface) and developers use an API, agents need an ACI. This goes beyond simple function calling. It involves designing a structured hierarchy of tools (Map → Observe → Compute → Action) with strict JSON-in/JSON-out schemas, ensuring the agent interacts with external systems predictably.
4. **Standard Workflows (The Operating Mechanics):** 
   Left to their own devices, LLMs can wander, hallucinate, or skip crucial steps. Harness engineering imposes programmatic structure using refinement loops. Paradigms like ReAct (Reason + Act) and strict State Machines (Plan → Execute → Review) force the agent to follow a verifiable process.
5. **Evaluation & Guardrails (The Safety Net):** 
   "If you can't measure it, you can't improve it." Production agents require a robust safety net. This includes deterministic checks, Human-in-the-Loop (HITL) approval gates for high-stakes actions, and comprehensive LLM-as-judge automated evaluation suites to run in CI/CD pipelines.

---

## Phase 1: Foundations (The Stateless Core)

Before orchestrating complex multi-agent systems or long-running memory loops, an engineer must master the stateless, single-turn interaction with the model. This phase is about absolute control.

### API Mastery & Deterministic Control
To build a reliable harness, you must dictate the model's behavior rather than leaving it to chance.
*   **Temperature (`temperature=0`):** While creativity (temperature ~1.0) is great for writing poetry, agents require determinism. Always set temperature to 0 when the agent is planning or using tools to ensure consistent, repeatable actions.
*   **Stop Sequences (`stop_sequences=["STOP"]`):** Force the model to terminate its generation exactly where you want it to. This is crucial for parsing custom output formats cleanly without trailing conversational filler.
*   **The System Prompt as Cognitive Framework:** This is not just a polite request; it is the absolute boundary of the agent. A good system prompt defines the persona ("You are a strict data analyst"), the rules ("Never invent figures"), and the output style ("Format all values as $X,XXX.XX").
*   **Streaming (`client.messages.stream`):** Essential for long agentic turns. Streaming prevents connection timeouts during complex generations and provides immediate feedback, keeping the system responsive.

### Structured Outputs (The Foundation of ACI)
A cardinal rule of Harness Engineering: **Never parse free text with regular expressions.** Instead, contract the model to emit machine-readable, strictly validated structured data.
*   **Pydantic & JSON Schema:** Define your expected output using strong typing (e.g., Pydantic models in Python). The SDK converts these into JSON Schemas that the LLM must follow.
*   **Strict Tool Inputs (`strict=True`):** When defining tools, enforcing strict schema matching guarantees that the tool input will exactly match your code's expectations. No missing fields, no hallucinated parameters.
*   **Decision Schemas:** Instead of asking the model "Should we approve this?" and parsing a "yes" or "no" from a paragraph of prose, force it to output a `Decision` object: `{"action": "approve", "confidence": 0.95, "reason": "..."}`. This eliminates ambiguity.

### The ReAct Loop (Reason + Act)
The ReAct loop is the heartbeat of every autonomous agent. Stripped of all frameworks, it is fundamentally a `while` loop keyed on the model's `stop_reason`:
1.  **THINK:** The model receives the user query and reasons about it (often visible via Extended Thinking blocks).
2.  **ACT:** The model determines it needs external data and emits a `tool_use` block, causing the API to return `stop_reason="tool_use"`.
3.  **OBSERVE:** The harness intercepts this, executes the corresponding Python function (the tool), and feeds the result back to the model as a `tool_result` block.
4.  **REPEAT:** The loop continues until the model has enough information to formulate a final answer, at which point it returns `stop_reason="end_turn"`.

---

## Phase 2: ACI & Tools (The Agent's Hands)

The Agent-Computer Interface (ACI) defines how the model touches the real world. A well-designed ACI protects the underlying systems while empowering the agent.

### Safe Tool Design and Guardrails
Tools should not be a flat list; they form an operational hierarchy:
*   **Context/Maps (Discovery):** Tools like `list_tables` or `get_schema`. The agent should be trained to always consult the map before navigating.
*   **Observation (READ):** These tools must be fiercely protected. For example, a `safe_sql_query` tool must implement programmatic guardrails *before* executing the agent's SQL:
    1.  Block all mutation keywords (`DROP`, `UPDATE`, `INSERT`).
    2.  Automatically inject `LIMIT` clauses to prevent catastrophic full-table scans.
*   **Compute (PROCESS):** Tools for data manipulation and calculation.
*   **Action (WRITE):** High-stakes tools that alter state (covered in Phase 4).

### The Error Refinement Loop (Self-Correction)
The difference between a fragile script and a resilient agent is how it handles failure.
*   When a standard script hits an exception, it crashes. 
*   When an agent's tool hits an exception (e.g., a SQL syntax error), the harness catches it, formats it as JSON, and sends the raw stack trace back to the LLM.
*   Crucially, the harness flags this with `is_error=True`. This signals to the model: *"Your last action failed. Read this error, figure out why, and try a different approach."* This allows the agent to autonomously debug and self-correct.

### Complex Workflows: The Data Analyst Pattern
ACI extends far beyond simple REST endpoints. By combining the Files API and secure Code Execution environments, agents can perform deep, domain-aware workflows.
*   **Spec-Driven Context:** Instead of just analyzing raw numbers, the harness uploads a physical specification document (PDF/TXT) via the Files API. The agent reads this first to understand the physical constraints (e.g., "Voltage over 4.9V is clipping") and domain-specific indices (e.g., SNR, THD) required.
*   **Sandboxed Code Execution:** The agent is given access to a secure, network-isolated Python sandbox pre-installed with `pandas`, `scipy`, and `matplotlib`. It writes and executes code to analyze the data against the spec.
*   **Container Reuse (State Persistence):** By passing the same `container_id` across multiple turns, the filesystem state is preserved. The agent can process data, generate a Jupyter Notebook, execute it to generate PNG plots, and save them—all of which the harness can later download for the user.

---

## Phase 3: Memory & Orchestration (The Brain)

For an agent to be truly useful over long sessions or across massive codebases, it requires an architecture for memory and strict operational orchestration.

### Context Management (RAM vs. Hard Drive)
Using the MemGPT mental model, the context window is the agent's RAM (fast, expensive, limited), while external storage is the hard drive. 
*   **Just-in-Time Retrieval (RAG):** Never stuff all available documentation into the system prompt upfront. Provide a `search_docs` tool so the agent retrieves only what is strictly necessary for the current turn.
*   **Context Compaction:** Monitor the token length of the `messages` array. As it approaches the model's limit (e.g., 80% capacity), the harness proactively triggers a "compaction" phase. It uses the LLM to summarize the middle of the conversation into dense bullet points, effectively freeing up "RAM" while maintaining the overarching narrative.

### Zero-Database File-Based RAG
Implementing memory does not strictly require deploying complex vector databases like Postgres/pgvector or Pinecone. For highly portable, auditable agent memory, flat Markdown files on disk are unparalleled.
*   **The Format:** Memories are stored as `.md` files with YAML frontmatter (Title, Tags, Date, Importance) categorized into `episodic/`, `semantic/`, and `procedural/` folders. This makes memory instantly human-readable, editable, and Git-versionable.
*   **Retrieval Strategies:**
    1.  **BM25:** Fast, keyword-based retrieval. Ideal for finding specific hardware spec parameters or exact error codes where semantic matching fails.
    2.  **Numpy Cosine:** Pure Python semantic similarity using `sentence-transformers`. Embeddings are cached as local `.npy` files.
    3.  **FAISS:** For scaling up to tens of thousands of memories using a local binary index.
    4.  **LanceDB:** An embedded columnar database that supports powerful hybrid searches (e.g., semantic search combined with a strict metadata filter like `importance='high'`).

### Strict Orchestration (State Machines)
ReAct loops are powerful but can lead to wandering if the task is complex. For mission-critical tasks, the harness must enforce a **Plan → Execute → Review** workflow using a strict State Machine.
*   **PLAN State:** The agent uses a specific system prompt restricted *only* to planning. It outputs a Pydantic-validated list of steps. It is physically blocked from taking action.
*   **EXECUTE State:** The agent is given access to tools and executes the locked-in plan step-by-step, recording the output of each.
*   **REVIEW State (LLM-as-Judge):** A separate, independent LLM call reviews the execution against the original prompt. If it detects failure or partial completion, it kicks the state machine into a **REVISE** state, forcing the agent to try again. This prevents the agent from "grading its own homework."

---

## Phase 4: Production & Safety (The Harness)

A prototype works when the developer watches it; a production agent works when no one is looking. This phase focuses on the absolute safety and continuous evaluation of the system.

### Production Guardrails
*   **Sandboxed Execution:** Any code generated by an agent must run in an isolated environment. Utilize server-side sandboxing tools or strictly configured local Docker containers (`--network none`, `--read-only`) to ensure rogue or hallucinated code cannot damage the host system.
*   **Human-in-the-Loop (HITL) Approval Gates:** Define an explicit allowlist of high-stakes tools (`HIGH_STAKES_TOOLS = {"send_email", "write_database", "deploy_code"}`). When the LLM requests these tools, the harness intercepts the call, pauses execution, and alerts a human. The human can Review, Approve, Reject, or even Modify the tool parameters before allowing the loop to continue.

### LLM Evaluation Deep Dive (Pillar 5)
Evaluation is the most critical component of production readiness. It is governed by a grading taxonomy based on cost and reliability:
1.  **Deterministic Code (Tier 1):** Python assertions, regex matches, and JSON schema validation. Free, instant, but rigid.
2.  **LLM-as-Judge (Tier 2):** Using an LLM to grade outputs against a strict rubric. Fast, inexpensive, and nuanced.
3.  **Human Annotation (Tier 3):** Slow and expensive. Reserved for creating the "Golden" dataset of reference answers.

#### Advanced Evaluation Methodologies:
*   **Agent Trajectory Evaluation:** Do not just grade the final answer; grade the journey. Did the agent pick the optimal tools? Did it make redundant calls? Did it recover from errors smoothly? (e.g., scoring Tool Selection Accuracy and Step Efficiency).
*   **RAG Evals:** Hallucinations are the #1 failure mode of RAG. Evaluate specifically for **Faithfulness** (are all claims firmly grounded in the retrieved chunks?) and **Context Precision** (was the retrieved data actually relevant to the prompt?).
*   **Bias Mitigation:** LLM judges suffer from biases. 
    *   *Position Bias:* (Favoring option A over B). Mitigate this by running "Swap Tests" and randomizing the order of answers.
    *   *Self-Serving Bias:* (Claude grading Claude higher than GPT-4 grading Claude). Mitigate this by always using a different model for the Judge than the one used for the Agent.
*   **CI/CD Batch Pipelines:** Evaluation datasets grow large. Use the API's Batch processing capabilities to grade hundreds of interactions concurrently at a 50% cost reduction. Run these suites automatically on every pull request, enforcing a strict regression threshold (e.g., "Fail the PR if the pass rate drops by > 5%").

By mastering these five pillars, engineers can transition from simply prompting models to building robust, self-healing, and deeply integrated AI systems.