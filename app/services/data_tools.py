"""Data/execution tools — Group B (6 tools).

These tools require an AnalysisSession to track state (figures, etc.).
They are NOT pure functions — they mutate session state.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.analysis_models import AnalysisSession
    from app.infrastructure.code_runner import CodeRunner

logger = logging.getLogger(__name__)


def execute_python_code(
    code: str,
    session: "AnalysisSession",
    runner: "CodeRunner",
) -> str:
    """Execute Python code and return combined output + figure IDs.

    pandas, numpy, matplotlib, seaborn are pre-imported.
    Figures are captured by patching plt.show().
    Returns JSON with stdout, figures list, and success flag.
    """
    result = runner.execute(code)

    # Register newly generated figures in the session
    for fig_id in result.figures:
        b64 = runner.get_figure_b64(fig_id)
        if b64:
            # Map runner's figure_id to session's figure_id
            session_fig_id = session.next_figure_id
            session.register_figure(session_fig_id, b64)
            logger.debug(
                "Registered figure %s → session %s (session=%s)",
                fig_id, session_fig_id, session.session_id,
            )

    return json.dumps({
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr if not result.success else "",
        "figures": list(session.figures.keys())[-len(result.figures):] if result.figures else [],
        "execution_time_ms": result.execution_time_ms,
    })


def get_execution_variables(runner: "CodeRunner") -> str:
    """Return JSON snapshot of Python variables from the last execution.

    Returns JSON dict {variable_name: type_name}.
    """
    state = runner.get_state()
    return json.dumps(state)


def get_figure(figure_id: str, session: "AnalysisSession") -> str:
    """Return metadata and retrieval URL for a figure generated in this session.

    Returns JSON with figure_id, format, approximate size, and retrieval_url.
    The figure image itself is served by GET /api/v1/analysis/{session_id}/figures/{figure_id}
    to keep tool responses small and avoid LLM context overflow (Gap 7).
    """
    b64 = session.figures.get(figure_id)
    if b64 is None:
        available = list(session.figures.keys())
        return json.dumps({
            "error": f"Figure '{figure_id}' not found.",
            "available_figures": available,
        })
    # Approximate decoded size: base64 chars × 0.75
    approx_bytes = int(len(b64) * 0.75)
    return json.dumps({
        "figure_id": figure_id,
        "format": "png",
        "size_bytes": approx_bytes,
        "retrieval_url": f"/api/v1/analysis/{session.session_id}/figures/{figure_id}",
        "note": (
            "Figure has been saved in the session. "
            "Download it via the retrieval_url or call save_figure to write it to disk."
        ),
    })


def list_figures(session: "AnalysisSession") -> str:
    """Return JSON list of all figure IDs generated in this session."""
    fig_ids = list(session.figures.keys())
    return json.dumps({"count": len(fig_ids), "figure_ids": fig_ids})


def export_notebook_tool(session: "AnalysisSession", title: str) -> str:
    """Export the analysis session as a Jupyter notebook.

    Delegates to infrastructure/notebook_exporter.py.
    """
    from app.infrastructure.notebook_exporter import export_notebook
    return export_notebook(session, title)


def save_figure_tool(
    session: "AnalysisSession",
    figure_id: str,
    filename: str,
) -> str:
    """Save a session figure to disk as a PNG file.

    Delegates to infrastructure/notebook_exporter.py.
    """
    from app.infrastructure.notebook_exporter import save_figure
    return save_figure(session, figure_id, filename)
