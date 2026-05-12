"""
PHASE 2 — STEP 5: Data Analyst Agent
=====================================
Harness Pillars: ACI (Agent-Computer Interface) + Standard Workflows
SDK Docs:
  Code Execution:       https://docs.anthropic.com/en/agents-and-tools/tool-use/code-execution-tool
  Files API:            https://docs.anthropic.com/en/build-with-claude/files
  PDF Support:          https://docs.anthropic.com/en/build-with-claude/pdf-support
  Programmatic Tools:   https://docs.anthropic.com/en/agents-and-tools/tool-use/programmatic-tool-calling

GOAL:
  Build a domain-aware data analyst agent that:
    1. Reads a specification document to extract physical constraints & valid ranges
    2. Uploads raw measurement data (CSV) into the sandboxed container
    3. Runs Python analysis — filtered and scoped to the spec limits
    4. Creates a Jupyter Notebook (.ipynb) with embedded visualizations
    5. Executes the notebook inside the sandbox and inspects figure outputs
    6. Downloads the notebook and figures via Files API

WHY SPEC-DRIVEN ANALYSIS MATTERS (Harness Engineering):
  A generic data analyst sees numbers; a domain-aware analyst sees MEANING.
  The spec document acts as a Cognitive Framework constraint — it tells the agent:
    ✓ What physical ranges are valid (filter out-of-spec noise)
    ✓ What indices to compute (SNR, linearity, THD — not arbitrary statistics)
    ✓ What units to display (physical units, not raw ADC counts)
    ✓ What constitutes a pass/fail result

ARCHITECTURE:
  ┌───────────────────────────────────────────────────────────────────┐
  │                   DATA ANALYST HARNESS                             │
  │                                                                    │
  │  ┌─────────────────────┐                                          │
  │  │   INPUT ARTIFACTS   │                                          │
  │  │                     │  Files API (upload once, reuse by id)    │
  │  │  ① spec.txt / PDF   │────────────────────────────────────┐    │
  │  │  ② data.csv         │────────────────────────────────────┤    │
  │  └─────────────────────┘                                    │    │
  │                                                             ▼    │
  │                              ┌────────────────────────────────┐  │
  │                              │    SANDBOXED CONTAINER         │  │
  │                              │    (Python 3.11, no network)   │  │
  │                              │                                │  │
  │  System prompt:              │  Pre-installed:                │  │
  │  "You are a domain-aware     │    pandas, numpy, scipy        │  │
  │   data analyst. Always read  │    matplotlib, seaborn         │  │
  │   the spec first."           │    scikit-learn, statsmodels   │  │
  │                              │    openpyxl (Excel)            │  │
  │                              │    pdfplumber (PDF text)       │  │
  │                              │                                │  │
  │                              │  Agent workflow:               │  │
  │                              │   ① Read spec → limits        │  │
  │                              │   ② Load CSV → DataFrame      │  │
  │                              │   ③ Filter by spec ranges     │  │
  │                              │   ④ Compute domain indices    │  │
  │                              │   ⑤ Write .ipynb (nbformat)   │  │
  │                              │   ⑥ Execute notebook          │  │
  │                              │   ⑦ Save PNG figures          │  │
  │                              └──────────────┬─────────────────┘  │
  │                                             │                    │
  │                              Files API (download generated files)│
  │                              ┌──────────────▼─────────────────┐  │
  │                              │  OUTPUT ARTIFACTS              │  │
  │                              │    analysis.ipynb              │  │
  │                              │    executed.ipynb              │  │
  │                              │    fig_timeseries.png          │  │
  │                              │    fig_histogram.png           │  │
  │                              │    fig_fft_spectrum.png        │  │
  │                              └────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────┘

SDK BETA HEADERS REQUIRED:
  "code-execution-2025-08-25"   — sandbox execution + file operations
  "files-api-2025-04-14"        — upload spec/data, download figures

CONTAINER REUSE PATTERN:
  Each API call returns a container_id. Pass it back in subsequent calls
  to maintain filesystem state (files written in turn 1 are visible in turn 2).
"""

import io
import json
import time
import textwrap
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ─── Output directory for downloaded artifacts ───────────────────────────────
OUTPUT_DIR = Path("./analyst_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SAMPLE DATA GENERATION
# In production, these would be your real spec PDF and measurement CSV.
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_SPEC = """\
SENSOR MODULE SPECIFICATION — SM-2040 ADC Board
================================================
Document: SM-2040-SPEC-v2.1

1. OPERATING RANGES (all measurements must fall within these limits)
   Input Voltage Range:   -5.0 V  to  +5.0 V   (signals outside are clipping artifacts)
   Valid Temperature:      0 °C   to  85 °C     (out-of-range = thermal runaway flag)
   Sampling Rate:         1000 Hz (each row = 1 ms timestep)

2. PERFORMANCE INDICES (compute these; not raw statistics)
   Signal-to-Noise Ratio (SNR):
     SNR_dB = 20 * log10(RMS_signal / RMS_noise)
     PASS criterion: SNR >= 60 dB
   Total Harmonic Distortion (THD):
     THD_% = (sqrt(V2^2 + V3^2 + V4^2) / V1) * 100
     PASS criterion: THD <= 0.5%
   Linearity Error:
     linearity_error = max(|measured - ideal|) / full_scale * 100
     PASS criterion: linearity_error <= 0.1%
   Dynamic Range:
     DR_dB = 20 * log10(V_max / V_noise_floor)
     PASS criterion: DR >= 80 dB

3. DATA QUALITY FLAGS
   CLIP flag: |voltage| > 4.9 V  → mark as clipped sample, exclude from analysis
   TEMP flag: temperature outside [0, 85] → thermal anomaly, exclude
   NOISE flag: sample-to-sample delta > 2.0 V in < 1 ms → spike artifact

4. VISUALIZATION REQUIREMENTS
   Plot 1: Time-series of voltage (valid samples only, highlight clipped/flagged)
   Plot 2: Amplitude histogram with normal fit overlay
   Plot 3: FFT power spectrum (log scale, mark fundamental and harmonics)
   All plots: label axes with physical units, include spec limits as horizontal lines.
"""


def generate_sample_csv(n_samples: int = 2000) -> str:
    """
    Generate synthetic ADC measurement data with realistic characteristics:
    - Sinusoidal signal at 50 Hz with harmonics (for THD calculation)
    - Gaussian noise floor
    - A few clipping artifacts (|V| > 4.9)
    - Temperature drift
    """
    import math
    import random

    random.seed(42)
    lines = ["timestamp_ms,voltage_V,temperature_C,channel"]
    for i in range(n_samples):
        t = i / 1000.0  # seconds (1 kHz sampling)

        # Fundamental at 50 Hz + 2nd, 3rd harmonics
        v_signal = (
            2.0 * math.sin(2 * math.pi * 50 * t)
            + 0.05 * math.sin(2 * math.pi * 100 * t)  # 2nd harmonic (-32 dB)
            + 0.02 * math.sin(2 * math.pi * 150 * t)  # 3rd harmonic (-40 dB)
        )
        v_noise = random.gauss(0, 0.002)  # ~60 dB SNR
        v_total = v_signal + v_noise

        # Inject clipping artifacts
        if 100 <= i <= 105 or 1500 <= i <= 1503:
            v_total = 5.1 if v_total > 0 else -5.1  # clips outside ±5V

        # Temperature: slow drift from 22 °C to 35 °C with noise
        temp = 22 + (13 * i / n_samples) + random.gauss(0, 0.3)

        channel = "CH_A"
        lines.append(f"{i},{v_total:.6f},{temp:.2f},{channel}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FILES API — UPLOAD ARTIFACTS
# Upload once, reference by file_id across multiple turns and sessions.
# ═══════════════════════════════════════════════════════════════════════════════

def upload_artifacts() -> tuple[str, str]:
    """
    Upload the spec document and data CSV to the Files API.

    Returns:
        (spec_file_id, data_file_id)

    File type → content block mapping:
      text/plain  → "document" block (Claude reads it as context)
      text/plain  → "container_upload" block (loaded into sandbox filesystem)
      application/pdf → "document" block (Claude reads visually + text)
    """
    print("[FILES API] Uploading artifacts...")

    # Upload spec as plain text (in production: use application/pdf for real datasheets)
    spec_bytes = SAMPLE_SPEC.encode("utf-8")
    spec_response = client.beta.files.upload(
        file=("sm2040_spec.txt", io.BytesIO(spec_bytes), "text/plain"),
    )
    spec_file_id = spec_response.id
    print(f"  ✓ Spec doc  → file_id: {spec_file_id}")

    # Upload CSV data — will be loaded into container filesystem as "sensor_data.csv"
    csv_bytes = generate_sample_csv(2000).encode("utf-8")
    data_response = client.beta.files.upload(
        file=("sensor_data.csv", io.BytesIO(csv_bytes), "text/plain"),
    )
    data_file_id = data_response.id
    print(f"  ✓ Data CSV  → file_id: {data_file_id}")

    return spec_file_id, data_file_id


def cleanup_files(file_ids: list[str]) -> None:
    """Delete uploaded files when done to free storage quota."""
    for fid in file_ids:
        try:
            client.beta.files.delete(fid)
            print(f"  🗑  Deleted file: {fid}")
        except Exception as e:
            print(f"  ⚠ Could not delete {fid}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: THE DATA ANALYST AGENT LOOP
# Core ReAct loop augmented with:
#   - Code Execution sandbox (bash + Python)
#   - Files API (spec document + data upload)
#   - Container reuse for persistent filesystem state
#   - pause_turn handling for long-running notebook execution
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a precision data analyst specialized in physical measurement systems.

YOUR WORKFLOW (follow this order every time):
  1. READ THE SPEC DOCUMENT first. Extract:
     - Valid operating ranges (voltage, temperature, etc.)
     - Required performance indices (SNR, THD, linearity, dynamic range)
     - Data quality flags and exclusion criteria
     - Visualization requirements

  2. LOAD THE DATA from sensor_data.csv. Apply spec constraints:
     - Filter and flag out-of-spec samples (do NOT silently drop them)
     - Report how many samples were excluded and why

  3. COMPUTE DOMAIN INDICES as defined in the spec (not generic statistics).
     Always report pass/fail against each spec limit.

  4. CREATE A JUPYTER NOTEBOOK:
     - Install nbformat + nbconvert if needed: pip install -q nbformat nbconvert jupyter
     - Use nbformat.v4 to programmatically build the notebook
     - Cell 1: imports and constants (voltage limits from spec)
     - Cell 2: data loading + quality flagging
     - Cell 3: compute all performance indices
     - Cell 4+: one matplotlib figure per visualization requirement
     - Save figures as PNG files in ./figures/ before embedding in notebook
     - Final cell: summary table of pass/fail results

  5. EXECUTE THE NOTEBOOK:
     jupyter nbconvert --to notebook --execute analysis.ipynb --output executed.ipynb
     jupyter nbconvert --to html executed.ipynb

  6. VERIFY FIGURES: list all PNG files in ./figures/, describe what each shows.

RULES:
  - Always use physical units in labels (V, °C, dB, %) — never raw counts
  - Mark spec limits as horizontal dashed lines on all relevant plots
  - Separate the analysis from the visualization in the notebook structure
  - If a performance index FAILS the spec, highlight it clearly in the summary
"""


def run_analyst_agent(
    spec_file_id: str,
    data_file_id: str,
    task: str = "Perform a full spec-driven analysis of the sensor data.",
    max_turns: int = 15,
    verbose: bool = True,
) -> dict:
    """
    Run the data analyst agent with Files API + Code Execution.

    Key SDK patterns demonstrated:
      1. container_upload — loads CSV into sandbox filesystem
      2. document block   — Claude reads spec as in-context text
      3. stop_reason == "pause_turn" — long-running notebook execution
      4. container_id reuse — maintain filesystem state across turns
      5. Files API download — retrieve generated artifacts
    """
    print(f"\n{'═'*65}")
    print("DATA ANALYST AGENT — Starting")
    print(f"{'═'*65}")

    messages = [
        {
            "role": "user",
            "content": [
                # Spec document: loaded as in-context text Claude can read
                {
                    "type": "text",
                    "text": "Here is the sensor specification document:",
                },
                {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": spec_file_id,
                    },
                },
                # Data CSV: loaded directly into sandbox filesystem as a file
                {
                    "type": "text",
                    "text": (
                        "Here is the raw measurement data (sensor_data.csv). "
                        "It will be available at /home/user/sensor_data.csv in the sandbox:"
                    ),
                },
                {
                    "type": "container_upload",
                    "file_id": data_file_id,
                },
                # The actual task
                {
                    "type": "text",
                    "text": task,
                },
            ],
        }
    ]

    container_id = None          # Will be set from first response, reused after
    generated_file_ids = []      # Track files created in the sandbox
    turn = 0

    while turn < max_turns:
        turn += 1
        if verbose:
            print(f"\n[Turn {turn}] Calling API...")

        # Build container config — reuse on turns 2+ to persist filesystem state
        container_config = {"type": "auto"}
        if container_id:
            container_config = {"type": "persistent", "id": container_id}

        try:
            response = client.beta.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=8192,
                betas=[
                    "code-execution-2025-08-25",   # sandbox + file ops
                    "files-api-2025-04-14",         # Files API integration
                ],
                system=SYSTEM_PROMPT,
                tools=[{"type": "code_execution_20250825"}],
                container=container_config,
                messages=messages,
            )
        except anthropic.APIError as e:
            print(f"  ✗ API error: {e}")
            break

        # Extract container_id from first response
        if container_id is None and hasattr(response, "container") and response.container:
            container_id = response.container.id
            print(f"  📦 Container: {container_id} (will be reused)")

        stop_reason = response.stop_reason
        if verbose:
            print(f"  stop_reason: {stop_reason}")

        # ── Process response content blocks ──────────────────────────────────
        for block in response.content:
            if block.type == "text":
                if verbose:
                    preview = block.text[:300].replace("\n", " ")
                    print(f"  💬 {preview}{'...' if len(block.text) > 300 else ''}")

            elif block.type == "tool_use":
                # Code execution: show what's being run
                if verbose and hasattr(block, "input"):
                    inp = block.input
                    if isinstance(inp, dict):
                        code = inp.get("command", inp.get("code", str(inp)))
                        print(f"  🔧 [{block.name}] {str(code)[:120]}...")

            elif block.type in ("bash_code_execution_result", "text_editor_code_execution_result"):
                # Execution result — show stdout/stderr
                if verbose:
                    result = block
                    if hasattr(result, "stdout") and result.stdout:
                        print(f"  📤 stdout: {result.stdout[:200]}")
                    if hasattr(result, "stderr") and result.stderr:
                        stderr_preview = result.stderr[:200]
                        if "error" in stderr_preview.lower():
                            print(f"  ⚠ stderr: {stderr_preview}")
                    if hasattr(result, "return_code") and result.return_code != 0:
                        print(f"  ✗ return_code: {result.return_code}")

            elif block.type == "code_execution_result":
                # Legacy format — handle for compatibility
                if verbose and hasattr(block, "content"):
                    for item in block.content:
                        if item.type == "text":
                            print(f"  📤 {item.text[:200]}")

        # ── Stop conditions ──────────────────────────────────────────────────
        if stop_reason == "end_turn":
            print(f"\n  ✅ Agent completed in {turn} turns")
            break

        elif stop_reason == "pause_turn":
            # Long-running code (notebook execution) — continue without modification
            if verbose:
                print("  ⏳ pause_turn: long operation running, continuing...")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "continue"}],
            })
            continue

        elif stop_reason == "tool_use":
            # Standard tool use (shouldn't happen with code_execution, but handle it)
            messages.append({"role": "assistant", "content": response.content})
            # For code execution, results are handled automatically by the API
            # Just continue the loop
            continue

        else:
            # max_tokens or unexpected stop
            messages.append({"role": "assistant", "content": response.content})
            if stop_reason == "max_tokens":
                if verbose:
                    print(f"  ⚠ max_tokens reached on turn {turn}, continuing...")
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "Please continue."}],
                })
                continue
            break

    # Final assistant message for return
    messages.append({"role": "assistant", "content": response.content})

    return {
        "container_id": container_id,
        "turns": turn,
        "messages": messages,
        "final_response": next(
            (b.text for b in response.content if b.type == "text"), ""
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DOWNLOAD GENERATED ARTIFACTS
# Files created inside the sandbox are accessible via Files API download.
# Pattern: list files → identify generated ones → download to local disk.
# ═══════════════════════════════════════════════════════════════════════════════

def download_generated_artifacts(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """
    Download all files generated by the code execution agent.

    IMPORTANT: Only files CREATED by code execution can be downloaded.
    Files you uploaded cannot be downloaded back (Files API limitation).

    Returns: list of local paths where files were saved.
    """
    print(f"\n[FILES API] Downloading generated artifacts → {output_dir}/")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    saved_paths = []
    download_extensions = {".ipynb", ".html", ".png", ".svg", ".pdf", ".csv"}

    try:
        files = list(client.beta.files.list())
        print(f"  Found {len(files)} files in workspace")

        for file_info in files:
            fname = file_info.filename or f"file_{file_info.id}"
            ext = Path(fname).suffix.lower()

            if ext not in download_extensions:
                continue

            # Determine local save path
            if ext == ".png":
                local_path = output_dir / "figures" / fname
            else:
                local_path = output_dir / fname

            try:
                content = client.beta.files.download(file_info.id)
                local_path.write_bytes(content)
                saved_paths.append(local_path)
                print(f"  ✓ {fname:40s} → {local_path}")
            except Exception as e:
                # Uploaded files (not generated) cannot be downloaded
                if "cannot be downloaded" not in str(e).lower():
                    print(f"  ✗ {fname}: {e}")

    except Exception as e:
        print(f"  ✗ Error listing files: {e}")

    print(f"  Downloaded {len(saved_paths)} artifact(s)")
    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MULTI-TURN FOLLOWUP
# Reuse the container to run follow-up analysis (e.g. "compare channels").
# Demonstrates container_id persistence — filesystem state is preserved.
# ═══════════════════════════════════════════════════════════════════════════════

def run_followup_analysis(
    container_id: str,
    spec_file_id: str,
    followup_question: str,
) -> str:
    """
    Demonstrate container reuse: ask a follow-up question using the same
    sandbox container. Files from the previous turn (analysis.ipynb, figures/)
    are still present — the agent can reference and extend them.

    Use case: "Now apply a bandpass filter and replot the spectrum"
              "Add a statistical summary table to the notebook"
              "Compute the same indices for channel CH_B"
    """
    print(f"\n[FOLLOWUP] Reusing container: {container_id}")
    print(f"  Question: {followup_question}")

    response = client.beta.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        betas=["code-execution-2025-08-25", "files-api-2025-04-14"],
        system=SYSTEM_PROMPT,
        tools=[{"type": "code_execution_20250825"}],
        container={"type": "persistent", "id": container_id},  # ← reuse same container
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": spec_file_id},
                    },
                    {"type": "text", "text": followup_question},
                ],
            }
        ],
    )

    answer = next((b.text for b in response.content if b.type == "text"), "")
    print(f"  Answer preview: {answer[:300]}")
    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: JUPYTER NOTEBOOK PATTERN REFERENCE
# This is the Python code the agent writes INSIDE the sandbox.
# Shown here as a reference so you understand what the agent generates.
# ═══════════════════════════════════════════════════════════════════════════════

NOTEBOOK_TEMPLATE_REFERENCE = '''
# The agent writes code like this inside the sandbox:

# ── Step 1: Setup ──
import subprocess
subprocess.run(["pip", "install", "-q", "nbformat", "nbconvert", "jupyter"], check=True)

# ── Step 2: Build notebook ──
import nbformat, json
from pathlib import Path
import textwrap

Path("figures").mkdir(exist_ok=True)

nb = nbformat.v4.new_notebook()

# Cell 0: Imports and spec constants
nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent("""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal, stats
    import warnings; warnings.filterwarnings("ignore")

    # Physical constants from SM-2040 spec
    V_MIN, V_MAX   = -5.0, 5.0          # V  — valid voltage range
    V_CLIP_THRESH  = 4.9                 # V  — clipping detection threshold
    TEMP_MIN       = 0.0                 # °C
    TEMP_MAX       = 85.0               # °C
    FS             = 1000.0             # Hz — sampling rate
    SNR_PASS       = 60.0              # dB
    THD_PASS       = 0.5               # %
    DR_PASS        = 80.0              # dB
    LIN_PASS       = 0.1               # %
    plt.rcParams["figure.dpi"] = 120
""")))

# Cell 1: Data loading and quality flagging
nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent("""
    df = pd.read_csv("sensor_data.csv")
    df["clipped"]  = df["voltage_V"].abs() > V_CLIP_THRESH
    df["temp_bad"] = ~df["temperature_C"].between(TEMP_MIN, TEMP_MAX)
    df["delta"]    = df["voltage_V"].diff().abs()
    df["spike"]    = df["delta"] > 2.0

    df["valid"] = ~(df["clipped"] | df["temp_bad"] | df["spike"])
    valid = df[df["valid"]]
    print(f"Total samples : {len(df)}")
    print(f"Clipped       : {df['clipped'].sum()}")
    print(f"Temp bad      : {df['temp_bad'].sum()}")
    print(f"Spikes        : {df['spike'].sum()}")
    print(f"Valid samples : {len(valid)} ({len(valid)/len(df)*100:.1f}%)")
""")))

# Cell 2: Compute performance indices per spec
nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent("""
    v = valid["voltage_V"].values
    t = valid["timestamp_ms"].values / 1000.0

    # SNR (signal power vs noise power via Welch PSD)
    f_psd, psd = signal.welch(v, fs=FS, nperseg=512)
    fundamental_idx = np.argmax(psd[1:50]) + 1
    signal_power = psd[fundamental_idx]
    noise_mask   = np.ones(len(psd), dtype=bool)
    noise_mask[fundamental_idx-2:fundamental_idx+3] = False
    noise_power  = np.mean(psd[noise_mask])
    snr_db = 10 * np.log10(signal_power / noise_power)

    # THD (from FFT harmonics)
    N = len(v)
    fft_amp = np.abs(np.fft.rfft(v)) / N * 2
    freqs   = np.fft.rfftfreq(N, d=1/FS)
    f0_idx  = np.argmax(fft_amp[1:50]) + 1
    V1 = fft_amp[f0_idx]
    harmonic_amps = [fft_amp[min(f0_idx*h, len(fft_amp)-1)] for h in range(2, 5)]
    thd_pct = np.sqrt(sum(a**2 for a in harmonic_amps)) / V1 * 100

    # Dynamic Range
    v_max    = np.max(np.abs(v))
    v_noise  = np.sqrt(noise_power)
    dr_db    = 20 * np.log10(v_max / v_noise)

    # Linearity error (vs ideal sine at fundamental frequency)
    f0   = freqs[f0_idx]
    A0   = V1
    phi0 = np.angle(np.fft.rfft(v)[f0_idx])
    ideal = A0 * np.sin(2 * np.pi * f0 * t + phi0)
    lin_error_pct = np.max(np.abs(v[:len(ideal)] - ideal)) / (V_MAX - V_MIN) * 100

    results = {
        "SNR (dB)"           : (snr_db,     SNR_PASS, ">="),
        "THD (%)"            : (thd_pct,    THD_PASS,  "<="),
        "Dynamic Range (dB)" : (dr_db,      DR_PASS,   ">="),
        "Linearity Error (%)": (lin_error_pct, LIN_PASS, "<="),
    }
    for name, (val, limit, cmp) in results.items():
        passed = val >= limit if cmp == ">=" else val <= limit
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{name:25s}: {val:8.3f}   limit {cmp}{limit}   [{status}]")
""")))

# Cell 3: Plot 1 — Time-series
nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent("""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp_ms"], df["voltage_V"], lw=0.5, color="steelblue",
            alpha=0.6, label="All samples")
    ax.scatter(df[df["clipped"]]["timestamp_ms"],
               df[df["clipped"]]["voltage_V"],
               color="red", s=20, zorder=5, label="Clipped")
    ax.axhline(V_CLIP_THRESH,  color="red",  ls="--", lw=0.8, label=f"Clip threshold ±{V_CLIP_THRESH}V")
    ax.axhline(-V_CLIP_THRESH, color="red",  ls="--", lw=0.8)
    ax.axhline(V_MAX,  color="orange", ls=":", lw=0.8, label=f"Spec max ±{V_MAX}V")
    ax.axhline(-V_MAX, color="orange", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("SM-2040 Voltage Time-Series")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig_timeseries.png", dpi=150)
    plt.show()
    print("Saved: figures/fig_timeseries.png")
""")))

# Cell 4: Plot 2 — Amplitude histogram
nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent("""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bins, _ = ax.hist(valid["voltage_V"], bins=80, density=True,
                               color="steelblue", alpha=0.7, label="Valid samples")
    mu, sigma = stats.norm.fit(valid["voltage_V"])
    x_fit = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma), "r-", lw=2,
            label=f"Normal fit (μ={mu:.3f}, σ={sigma:.3f})")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Amplitude Histogram with Normal Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig_histogram.png", dpi=150)
    plt.show()
    print("Saved: figures/fig_histogram.png")
""")))

# Cell 5: Plot 3 — FFT power spectrum
nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent("""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_psd, psd, color="steelblue", lw=1, label="PSD")
    for h, label in [(1, "Fund."), (2, "2nd H."), (3, "3rd H.")]:
        hf = freqs[f0_idx] * h
        if hf < FS / 2:
            hp = np.interp(hf, f_psd, psd)
            ax.axvline(hf, color="red", ls="--", lw=0.8, alpha=0.7)
            ax.annotate(f"{label}\\n{hf:.0f}Hz", xy=(hf, hp),
                        xytext=(hf+10, hp*3), fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (V²/Hz)")
    ax.set_title(f"FFT Power Spectrum  |  SNR = {snr_db:.1f} dB  |  THD = {thd_pct:.3f}%")
    ax.set_xlim(0, 500)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig_fft_spectrum.png", dpi=150)
    plt.show()
    print("Saved: figures/fig_fft_spectrum.png")
""")))

with open("analysis.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Notebook written: analysis.ipynb")

# ── Step 3: Execute notebook ──
result = subprocess.run(
    ["jupyter", "nbconvert", "--to", "notebook", "--execute",
     "analysis.ipynb", "--output", "executed.ipynb"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])

# ── Step 4: Export to HTML ──
subprocess.run(
    ["jupyter", "nbconvert", "--to", "html", "executed.ipynb"],
    capture_output=True
)
print("Execution complete. Artifacts: analysis.ipynb, executed.ipynb, executed.html")
print("Figures:", list(Path("figures").glob("*.png")))
'''


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   STEP 5: DATA ANALYST AGENT (Harness ACI)                   ║")
    print("║   Spec-driven analysis → Jupyter Notebook → Figures          ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    uploaded_file_ids = []

    try:
        # ── Phase 1: Upload artifacts via Files API ──────────────────────────
        spec_file_id, data_file_id = upload_artifacts()
        uploaded_file_ids = [spec_file_id, data_file_id]

        # ── Phase 2: Run the main analysis agent ────────────────────────────
        result = run_analyst_agent(
            spec_file_id=spec_file_id,
            data_file_id=data_file_id,
            task=(
                "Perform a complete spec-driven analysis of the SM-2040 sensor data. "
                "Read the specification first, then load sensor_data.csv, apply all "
                "data quality flags, compute all required performance indices "
                "(SNR, THD, dynamic range, linearity error), create and execute a "
                "Jupyter Notebook with the three required visualizations, and "
                "provide a pass/fail summary table."
            ),
            max_turns=15,
            verbose=True,
        )

        container_id = result["container_id"]
        print(f"\n{'─'*65}")
        print("FINAL ANALYSIS SUMMARY:")
        print(result["final_response"][:1000])

        # ── Phase 3: Download generated artifacts ────────────────────────────
        saved = download_generated_artifacts(OUTPUT_DIR)
        if saved:
            print(f"\n📁 Artifacts saved to: {OUTPUT_DIR.resolve()}/")
            for p in saved:
                size_kb = p.stat().st_size / 1024
                print(f"   {p.name:40s} ({size_kb:.1f} KB)")

        # ── Phase 4 (optional): Follow-up analysis on same container ────────
        if container_id:
            run_followup_analysis(
                container_id=container_id,
                spec_file_id=spec_file_id,
                followup_question=(
                    "The analysis is done. Now add a 4th cell to the executed notebook "
                    "that applies a 45-55 Hz bandpass filter (Butterworth, order=4) to the "
                    "valid voltage samples and plots the filtered signal on top of the original. "
                    "Save the new figure as figures/fig_bandpass.png and re-execute the notebook."
                ),
            )
            # Download the new figure
            download_generated_artifacts(OUTPUT_DIR)

    finally:
        # ── Cleanup: delete uploaded files (generated files persist 30 days) ─
        print(f"\n[CLEANUP] Removing uploaded files from workspace...")
        cleanup_files(uploaded_file_ids)

    print("\n✅ Data analyst agent demo complete.")
    print("\nKEY PATTERNS DEMONSTRATED:")
    print("  ① Files API upload  : spec doc + data CSV uploaded once, reused by file_id")
    print("  ② container_upload  : data loaded directly into sandbox filesystem")
    print("  ③ document block    : spec read as in-context text / PDF visually")
    print("  ④ Code Execution    : pandas, scipy, matplotlib all pre-installed")
    print("  ⑤ Jupyter notebook  : agent creates + executes .ipynb inside sandbox")
    print("  ⑥ pause_turn        : long notebook execution handled gracefully")
    print("  ⑦ Container reuse   : container_id passed back for follow-up analysis")
    print("  ⑧ Files API download: generated PNG/ipynb downloaded to local disk")


if __name__ == "__main__":
    main()
