# AI Agent From Scratch — Harness Engineering Tutorial
## Cross-referenced with Anthropic Claude Python SDK

> **Core Principle:** `Agent = Model + Harness`
>
> The LLM is the brain. The *Harness* is the surrounding infrastructure: memory, tools, context management, safety guardrails, and evaluation loops. This tutorial builds that harness, layer by layer.

---

## The 5 Pillars (from Harness Engineering)

| Pillar | What It Does | SDK Feature Used |
|--------|-------------|-----------------|
| **Context Management** | Assemble prompts, compact history, avoid overflow | Prompt Caching, Compaction, Memory Tool |
| **Cognitive Framework** | Agent's worldview, behavioral boundaries | System prompt, `agents.md`, Extended Thinking |
| **ACI (Agent-Computer Interface)** | JSON-in/JSON-out tool design hierarchy | Tool Use, Structured Outputs, Code Execution |
| **Standard Workflows** | Plan→Generate→Evaluate refinement loops | Agentic Loop, `is_error`, ReAct |
| **Evaluation & Guardrails** | Deterministic checks + LLM-as-judge | Strict Tool Use, HITL, Batch API |

---

## Tutorial Structure

```
harness_tutorial/
├── README.md                              ← You are here
├── requirements.txt
│
├── phase1_foundations/                    ── STATELESS CORE ──
│   ├── step1_api_mastery.py               ← API control: temperature, streaming, caching
│   ├── step2_structured_outputs.py        ← Pydantic + strict JSON schemas
│   └── step3_react_loop.py                ← ReAct loop from scratch (the agent core)
│
├── phase2_aci/                            ── ACI: THE AGENT'S HANDS ──
│   ├── step4_sql_tool.py                  ← Safe read-only SQL tool with guardrails
│   ├── step5_data_analyst_agent.py        ← ★ Data analyst: spec→analysis→notebook→figures
│   └── step6_error_refinement.py          ← Self-correction with is_error=True
│
├── phase3_memory/                         ── MEMORY & ORCHESTRATION ──
│   ├── step7_8_rag_compaction.py          ← RAG (keyword retrieval) + context compaction
│   ├── step7b_file_based_rag.py           ← ★ File-based RAG: BM25/Numpy/FAISS/LanceDB
│   └── step9_state_machine.py             ← Plan→Execute→Review state machine
│
└── phase4_production/                     ── PRODUCTION SAFETY HARNESS ──
    ├── step10_12_production_safety.py     ← Sandboxed execution + HITL approval gates
    └── step11_eval_deep_dive.py           ← LLM eval deep dive: rubrics, trajectory, CI/CD
```

---

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# Run each step in order:
python phase1_foundations/step1_api_mastery.py
python phase1_foundations/step2_structured_outputs.py
python phase1_foundations/step3_react_loop.py
python phase2_aci/step4_sql_tool.py
python phase2_aci/step5_data_analyst_agent.py   # ★ new
python phase2_aci/step6_error_refinement.py
python phase3_memory/step7_8_rag_compaction.py
python phase3_memory/step9_state_machine.py
python phase4_production/step10_12_production_safety.py
python phase4_production/step11_eval_deep_dive.py
```

---

## Learning Path Map

```
Phase 1: Foundations (Stateless)
  Step 1: Control the model — temperature, stop sequences, streaming, prompt caching
  Step 2: Force structured output — Pydantic + client.messages.parse()
  Step 3: ReAct loop from scratch — the while stop_reason=="tool_use" core pattern
       ↓
Phase 2: ACI & Tools (The Hands)
  Step 4: Safe SQL tool — SELECT-only guardrail, LIMIT injection, is_error feedback
  Step 5: ★ Data Analyst Agent — spec-driven analysis, Jupyter notebook, figures
            ├── Files API (upload spec PDF + CSV data, download generated figures)
            ├── Code Execution sandbox (pandas, numpy, scipy, matplotlib pre-installed)
            ├── Jupyter notebook (create .ipynb + execute via nbconvert in sandbox)
            ├── Physical constraint extraction (spec defines valid ranges + indices)
            └── Container reuse (multi-turn analysis on persistent filesystem)
  Step 6: Self-correction loop — is_error=True activates Claude's error recovery
       ↓
Phase 3: Memory & Orchestration (The Brain)
  Step 7+8: RAG (keyword retrieval) + context compaction (manual summarise & trim)
  Step 7b:  ★ File-based RAG — 4 backends, zero database:
              A) BM25 + Markdown      no embeddings, rank_bm25 only
              B) Numpy cosine + .npy  sentence-transformers, pure numpy
              C) FAISS + .index       faiss-cpu, ANN at scale
              D) LanceDB              Apache Lance files, hybrid vector+metadata filter
            All 4 share the same .md memory files as source of truth.
            Vector index = disposable derived cache. Memory = the .md files.
  Step 9:   Plan→Execute→Review state machine — per-stage system prompts, Pydantic routing
       ↓
Phase 4: Production & Safety (The Harness)
  Step 10-12: Sandboxed code execution, HITL approval gates, high-stakes tool control
  Step 11:    LLM eval deep dive — 3-tier grading, trajectory eval, RAG faithfulness,
              bias mitigation, CI/CD pipeline with Batch API (50% cost)
```

---

## Step 5 — Data Analyst Agent (ACI Deep Dive)

The centrepiece of Phase 2. Demonstrates how the **Agent-Computer Interface** extends
beyond simple JSON tools to a full data analysis workflow:

```
Spec Document (PDF/txt)    Raw Data (CSV)
         │                       │
         └──── Files API ────────┘
                    │  (upload once → file_id)
                    ▼
         ┌──────────────────────────┐
         │  SANDBOXED CONTAINER     │
         │  (Python 3.11, no net)   │
         │                          │
         │  ① Read spec            │    SDK: document content block
         │  ② Extract limits       │    Pattern: Cognitive Framework → ACI
         │  ③ Filter data by spec  │    Library: pandas (pre-installed)
         │  ④ Compute indices      │    Library: scipy, numpy (pre-installed)
         │  ⑤ Write .ipynb         │    Library: nbformat (pip install in sandbox)
         │  ⑥ Execute notebook     │    Command: jupyter nbconvert --execute
         │  ⑦ Save PNG figures     │    Library: matplotlib, seaborn
         └──────────────────────────┘
                    │
         Files API download
                    │
         ┌──────────▼──────────┐
         │  analyst_output/    │
         │  ├── analysis.ipynb │
         │  ├── executed.ipynb │
         │  ├── executed.html  │
         │  └── figures/       │
         │      ├── fig_timeseries.png
         │      ├── fig_histogram.png
         │      └── fig_fft_spectrum.png
         └─────────────────────┘
```

**SDK Beta headers required:**
```python
betas=["code-execution-2025-08-25", "files-api-2025-04-14"]
```

**Key patterns:**
| Pattern | SDK Feature | Why It Matters |
|---------|------------|----------------|
| `container_upload` | Files API | Loads CSV into sandbox filesystem |
| `document` block | Files API | Claude reads spec as in-context text |
| `pause_turn` handling | Code Execution | Long notebook execution without timeout |
| `container_id` reuse | Code Execution | Filesystem persists across follow-up turns |
| Files API download | `client.beta.files.download()` | Retrieve generated PNGs and .ipynb |

---

## SDK Feature → Tutorial Step Matrix

| SDK Feature | Steps |
|-------------|-------|
| `client.messages.create()` params | 1 |
| `client.messages.parse()` + Pydantic | 2, 9, 11 |
| Tool Use + ReAct while-loop | 3, 4, 5, 6 |
| `is_error=True` self-correction | 6 |
| Prompt Caching (`cache_control`) | 1, 7 |
| Context Compaction | 7+8 |
| RAG (retrieval-augmented context) | 7+8 |
| State Machine (Plan→Execute→Review) | 9 |
| **Code Execution sandbox** | **5, 10-12** |
| **Files API (upload + download)** | **5** |
| **PDF / spec document ingestion** | **5** |
| **Jupyter Notebook creation** | **5** |
| **Container reuse** | **5** |
| LLM-as-judge evaluation | 11 |
| Batch API (50% cost) | 11 |
| Human-in-the-Loop (HITL) | 10-12 |
