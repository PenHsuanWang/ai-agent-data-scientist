# Troubleshooting Guide — Data Scientist Agent

Problem → Cause → Solution format for all major failure modes.

---

## 1. ReAct Loop Issues

### ❌ Loop exits without "Final Answer:"

**Symptom:** The agent returns an empty or truncated response. No `Final Answer:` in the output.

**Cause A:** Claude output a `Thought:` without a `Action:` or `Final Answer:` — typically when it's "stuck" and unsure what to do.

**Solution:**
- Check the last `Thought:` in the react_trace. Does it reference a tool that doesn't exist?
- Ensure all tool names in the system prompt match actual registered tools.
- Verify the system prompt includes `"Final Answer:"` as the exit phrase, not `"Final Answer"` (with colon).

```python
# Debug: Print all tool names the agent thinks exist
print(system_prompt.split("AVAILABLE TOOLS")[1][:500])
```

**Cause B:** Claude outputted a tool call in a format the parser couldn't parse (e.g., used `:` instead of `\nAction:`).

**Solution:** Widen the regex pattern in `ReActParser`:
```python
# More lenient pattern
ACTION_RE = re.compile(
    r"Action\s*:?\s*(.+?)(?=Action Input|$)",
    re.IGNORECASE | re.DOTALL
)
```

---

### ❌ `ReActLoopError: Maximum iterations (20) exceeded`

**Symptom:** `ReActLoopError` raised, session ends without result.

**Cause A:** Agent is cycling — calling the same tool repeatedly with the same inputs.

**Solution:**
1. Log the react_trace and look for repeated (action, action_input) pairs.
2. Add a repetition guard:
```python
# In DataScienceAgentService._run_loop()
seen_actions = set()
action_key = f"{action}::{json.dumps(action_input, sort_keys=True)}"
if action_key in seen_actions:
    observation = "Error: You already tried this exact action. Try a different approach."
else:
    seen_actions.add(action_key)
    observation = await self._dispatch(action, action_input)
```

**Cause B:** Task is genuinely complex and needs more iterations.

**Solution:** Increase `MAX_REACT_ITERATIONS` in config:
```python
# app/core/config.py
class Settings(BaseSettings):
    max_react_iterations: int = 30  # Increase from 20
```

---

### ❌ `Action Input:` contains invalid JSON

**Symptom:** `json.JSONDecodeError` when parsing `Action Input:`, agent crashes.

**Cause:** Claude sometimes produces Python dict syntax (`{'key': 'value'}`) instead of valid JSON (`{"key": "value"}`).

**Solution:** Apply a tolerant JSON parser before strict parsing:
```python
import ast
import json

def parse_action_input(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            # Attempt to evaluate as Python literal (handles single quotes)
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # Return as string — tool will handle/reject
            return {"_raw": raw}
```

---

### ❌ Agent skips domain document loading

**Symptom:** Agent calls `inspect_dataset` or `execute_python_code` before calling `list_domain_documents`.

**Cause:** System prompt domain rule not strong enough.

**Solution:** Add explicit enforcement in the ReAct system prompt:
```
MANDATORY FIRST STEP
--------------------
Your FIRST action MUST be: list_domain_documents
Do not call any other tool until you have called list_domain_documents.
```

Also add a check in `_dispatch()`:
```python
if action != "list_domain_documents" and not session.domain_docs_loaded:
    return "Error: You must call list_domain_documents before any other tool."
```

---

## 2. Tool Errors

### ❌ `list_domain_documents` returns empty list

**Symptom:** Tool returns `"No documents found in docs directory."`.

**Cause A:** `DOCS_DIR` environment variable not set or points to wrong path.

**Solution:**
```bash
# Check the config
echo $DOCS_DIR

# In .env file
DOCS_DIR=/absolute/path/to/docs/directory
```

**Cause B:** Docs directory exists but contains no `.md` or `.txt` files.

**Solution:** Add supported file extension to the tool:
```python
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}  # extend as needed
```

---

### ❌ `inspect_dataset` returns "File not found"

**Symptom:** Tool returns error even though file exists.

**Cause A:** Claude is providing just the filename but `DATA_DIR` is not configured correctly.

**Solution:**
1. Verify `DATA_DIR` in settings: `print(settings.data_dir)`
2. Verify the file exists: `ls $DATA_DIR/*.csv`
3. Check that the tool constructs the full path: `path = settings.data_dir / file_name`

**Cause B:** Path traversal guard is rejecting a legitimate path.

**Solution:**
```python
# In the tool implementation — debug path resolution
data_dir = Path(settings.data_dir).resolve()
file_path = (data_dir / file_name).resolve()
print(f"data_dir: {data_dir}")
print(f"file_path: {file_path}")
print(f"is child: {data_dir in file_path.parents or file_path.parent == data_dir}")
```

---

### ❌ `read_domain_document` returns "Permission denied"

**Symptom:** File system permission error.

**Solution:**
```bash
ls -la $DOCS_DIR/
chmod 644 $DOCS_DIR/*.md $DOCS_DIR/*.txt
```

---

## 3. Code Execution Issues

### ❌ `execute_python_code` times out

**Symptom:** Tool returns `"Error: Code execution timed out after 30 seconds."`.

**Cause A:** Data file is very large — loading takes too long.

**Solution:** Increase timeout for large datasets:
```python
# app/core/config.py
code_execution_timeout: int = 120  # seconds
```

**Cause B:** Code has an infinite loop or blocking call.

**Solution:** The timeout is the correct protection. Tell the agent to use chunked reading:
```python
# Agent should use:
df = pd.read_csv('file.csv', nrows=10000)  # sample first
# Not: df = pd.read_csv('file.csv')  # full load
```

---

### ❌ Variables not persisted between code executions

**Symptom:** Second `execute_python_code` call can't access variables from the first.

**Cause A (SubprocessRunner):** Variable serialization failed — unsupported type.

**Solution:** Ensure code exports results to JSON or saves to a temp file:
```python
# Pattern: explicit variable export
import json
result = {"df_shape": df.shape, "columns": list(df.columns)}
print(f"__VARS__{json.dumps(result)}")
```

**Cause B (SubprocessRunner):** DataFrame too large to pickle.

**Solution:** Save to temp parquet instead:
```python
df.to_parquet('/tmp/session_{session_id}_df.parquet')
# Next call:
df = pd.read_parquet('/tmp/session_{session_id}_df.parquet')
```

**Cause C:** Using `SubprocessRunner` but should use `JupyterBridge` for stateful sessions.

**Solution:** Enable Jupyter bridge in settings:
```bash
ENABLE_JUPYTER=true
CODE_EXECUTION_BACKEND=jupyter
```

---

### ❌ matplotlib figures not captured

**Symptom:** `get_figure` returns "Figure not found" after code execution.

**Cause A:** `matplotlib` backend is not set to `Agg` — interactive backend tries to open a window.

**Solution:** Set backend at the top of executed code or in runner:
```python
# In SubprocessCodeRunner.execute()
preamble = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
full_code = preamble + code
```

**Cause B:** Figure was created but `plt.savefig()` was not called.

**Solution:** Add auto-save hook in the runner:
```python
# After code execution, auto-save all open figures
auto_save = """
import matplotlib.pyplot as plt
for i, fig_num in enumerate(plt.get_fignums()):
    fig = plt.figure(fig_num)
    fig.savefig(f'/tmp/figure_{session_id}_{i}.png', bbox_inches='tight', dpi=150)
plt.close('all')
"""
subprocess.run([sys.executable, "-c", full_code + "\n" + auto_save], ...)
```

---

### ❌ `AnthropicCodeExecRunner` returns "Tool not available"

**Symptom:** Using cloud code execution but getting availability error.

**Cause A:** Account does not have access to `code_execution_20260120`.

**Solution:** Check API key access tier. As of May 2026, code execution is available on Claude API but NOT on Amazon Bedrock or Google Vertex AI.

**Cause B:** Container idle timeout — session state was lost after 4.5 minutes of inactivity.

**Solution:** Send a keepalive execution every 4 minutes:
```python
# app/infrastructure/anthropic_code_exec.py
IDLE_TIMEOUT_SECONDS = 240  # 4 minutes (before 4.5 min container timeout)

async def keepalive(self, session_id: str) -> None:
    await self.execute(session_id, code="1+1")  # no-op execution
```

---

## 4. Physical Validation Issues

### ❌ `validate_physical_units` returns "Unknown unit"

**Symptom:** Tool returns `"Error: Unit 'degC' not recognized"` or similar.

**Cause:** `pint` uses specific unit string formats.

**Solution:** Use exact pint unit strings:
```python
# Common pint strings
"degC"        # Celsius (NOT "°C", "C", "celsius")
"kelvin"      # Kelvin (NOT "K" — ambiguous)
"degF"        # Fahrenheit
"Pa"          # Pascal (NOT "pa")
"kPa"         # kilopascal
"bar"         # bar
"psi"         # pounds per square inch
"W"           # Watt
"kW"          # kilowatt
"m^3/s"       # cubic metres per second (NOT "m3/s")
"percent"     # percentage (NOT "%")
```

Also check pint documentation:
```python
import pint
ureg = pint.UnitRegistry()
print(ureg.parse_expression("your_unit_string"))  # Test your unit string
```

---

### ❌ Physical validation passes for impossible values

**Symptom:** Efficiency of 150% passes validation without error.

**Cause:** DOMAIN_RANGES entry for efficiency is missing or min/max are wrong.

**Solution:** Check the DOMAIN_RANGES dict:
```python
# app/infrastructure/unit_registry.py
DOMAIN_RANGES[("efficiency", "percent")]  # Should be (0.0, 100.0, "...")
```

Also verify HARD_BOUNDS includes efficiency:
```python
HARD_BOUNDS = {
    ("efficiency", "percent"): (0.0, 100.0, "First Law of Thermodynamics"),
    # ...
}
```

---

### ❌ pint `DimensionalityError` during unit conversion

**Symptom:** `pint.DimensionalityError: Cannot convert from 'degC' to 'Pa'`.

**Cause:** Attempting to convert between physically incompatible units.

**Solution:** This is the correct behavior — pint is protecting you from a unit error in the analysis. Check the quantity/unit pairing in the tool call:
```
validate_physical_units(quantity="temperature", value=101325, unit="Pa", ...)
# ❌ WRONG: Pa is pressure, not temperature

validate_physical_units(quantity="pressure", value=101325, unit="Pa", ...)
# ✅ CORRECT
```

---

## 5. Jupyter Bridge Issues

### ❌ `JupyterKernelManager.start()` fails with "No module named 'jupyter_client'"

**Solution:**
```bash
pip install jupyter_client nbformat ipykernel
# Or with uv:
uv add jupyter_client nbformat ipykernel
```

---

### ❌ Kernel hangs — `execute_cell()` never returns

**Symptom:** `await kernel.execute_cell(code)` blocks indefinitely.

**Cause A:** The iopub drain loop is not detecting `execution_state == "idle"`.

**Solution:** Add a timeout to the iopub loop:
```python
async def _drain_iopub(self, client, timeout=60.0) -> list:
    messages = []
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            msg = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.get_iopub_msg(timeout=1.0)
                ),
                timeout=2.0
            )
            messages.append(msg)
            if msg["header"]["msg_type"] == "status":
                if msg["content"]["execution_state"] == "idle":
                    break
        except (queue.Empty, asyncio.TimeoutError):
            break
    return messages
```

**Cause B:** Kernel crashed (OOM, segfault).

**Solution:**
```python
# Check kernel status before executing
if not client.is_alive():
    await self.restart(session_id)
```

---

### ❌ `export_notebook` produces a file with empty cells

**Symptom:** `.ipynb` file downloads fine but all cells are empty.

**Cause:** `jupyter_cells` list on `AnalysisSession` was not populated during execution.

**Solution:** Ensure cells are appended after each execution:
```python
# In data_tools.py execute_python_code tool
session.jupyter_cells.append({
    "cell_type": "code",
    "source": code,
    "outputs": [{"output_type": "stream", "text": result.stdout}],
})
```

---

## 6. API Issues

### ❌ POST `/api/v1/analysis/chat` returns 422 Unprocessable Entity

**Cause:** Request body doesn't match `AnalysisRequest` schema.

**Solution:** Check the required fields:
```bash
curl -s http://localhost:8000/openapi.json | python -m json.tool | grep -A 20 "AnalysisRequest"
```

Minimum valid request:
```json
{"user_message": "analyze the data"}
```

---

### ❌ GET `/api/v1/analysis/{session_id}/notebook` returns 404

**Cause A:** Session ID is wrong.

**Solution:**
```bash
# Get session ID from the POST /analysis/chat response
curl -s -X POST http://localhost:8000/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{"user_message": "test"}' | python -m json.tool | grep session_id
```

**Cause B:** Notebook was not generated (no `export_notebook` tool was called).

**Solution:** Ask the agent explicitly: `"Please export the analysis as a Jupyter notebook."`

---

### ❌ CORS error when calling API from browser

**Solution:** Ensure CORS middleware is configured in `main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 7. Anthropic API Errors

### ❌ `anthropic.AuthenticationError`

**Cause:** `ANTHROPIC_API_KEY` not set or invalid.

**Solution:**
```bash
echo $ANTHROPIC_API_KEY  # Should not be empty
export ANTHROPIC_API_KEY=sk-ant-...  # Or set in .env file
```

---

### ❌ `anthropic.RateLimitError`

**Cause:** Token rate limit exceeded (common for large context analyses).

**Solution:**
1. Add exponential backoff:
```python
import asyncio
from anthropic import RateLimitError

async def call_with_retry(client, **kwargs, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.messages.create(**kwargs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
```
2. Reduce `max_tokens` per call.
3. Use `claude-haiku-4-5` for knowledge/exploration calls, `claude-sonnet-4-6` only for final analysis.

---

### ❌ `anthropic.BadRequestError: prompt is too long`

**Cause:** Context window exceeded (1M tokens for claude-sonnet-4-6, but still possible with large datasets in prompts).

**Solution:**
1. Never put raw dataset content into prompts — use `inspect_dataset` which provides only schema + sample.
2. Truncate domain documents in `PhysicalContextInjector`:
```python
summary = text[:500]  # Limit each doc to 500 chars in system prompt
```
3. Prune old messages from the session if conversation is very long:
```python
# Keep only last N messages
if len(session.messages) > 40:
    session.messages = session.messages[-40:]
```

---

### ❌ `stop_reason: "pause_turn"` — analysis is cut off

**Cause:** Anthropic's server-side iteration limit was hit (not the same as `max_tokens`). This happens with long agentic loops using server tools.

**Solution:** Re-send the conversation to continue:
```python
# In DataScienceAgentService._run_loop()
while True:
    response = await self.client.messages.create(**kwargs)
    if response.stop_reason == "pause_turn":
        # Append Claude's partial response and continue
        messages.append({"role": "assistant", "content": response.content})
        kwargs["messages"] = messages
        continue
    break
```

---

## 8. Environment & Configuration

### ❌ FastAPI server fails to start — "Address already in use"

```bash
# Find the process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
# Or use a different port
uvicorn app.main:app --port 8001
```

---

### ❌ Settings validation error on startup

**Cause:** `.env` file has wrong format or missing required fields.

**Solution:**
```bash
# Check .env file format (no spaces around =)
cat .env
# ANTHROPIC_API_KEY=sk-ant-...  ✅
# ANTHROPIC_API_KEY = sk-ant-...  ❌ (spaces cause issues with some dotenv parsers)
```

Required fields in `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
DOCS_DIR=/path/to/docs
DATA_DIR=/path/to/data
```

Optional fields with defaults:
```env
MODEL_NAME=claude-sonnet-4-6
MAX_REACT_ITERATIONS=20
CODE_EXECUTION_TIMEOUT=30
ENABLE_JUPYTER=false
CODE_EXECUTION_BACKEND=subprocess
MAX_FILE_SIZE_MB=100
```

---

## Debugging Checklist

Use this checklist when an analysis fails:

```
□ 1. Is ANTHROPIC_API_KEY set and valid?
      echo $ANTHROPIC_API_KEY | head -c 20

□ 2. Does DATA_DIR exist and contain the expected files?
      ls $DATA_DIR

□ 3. Does DOCS_DIR exist and contain domain documents?
      ls $DOCS_DIR

□ 4. Is the ReAct trace showing the expected tool calls?
      Check response.react_trace in the API response

□ 5. Did the agent call list_domain_documents first?
      react_trace[0].action should be "list_domain_documents"

□ 6. Did any tool call return an error?
      Look for "Error:" prefix in react_trace[*].observation

□ 7. Did the loop hit max iterations?
      Check for ReActLoopError in the logs

□ 8. Are physical units being validated?
      Look for "validate_physical_units" in react_trace

□ 9. Is the final answer physically plausible?
      Check unit validation results in the trace

□ 10. For code execution — did matplotlib figures save?
       Check that figure_ids appear in response.figures
```
