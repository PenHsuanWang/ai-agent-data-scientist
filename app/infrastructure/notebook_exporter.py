"""Jupyter notebook exporter.

Converts an AnalysisSession's react_trace and notebook_cells into a
standard .ipynb file using nbformat.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, NotebookCell

logger = logging.getLogger(__name__)


def export_notebook(session: AnalysisSession, title: str) -> str:
    """Export the session as a Jupyter notebook.

    Returns JSON with notebook_path, cell_count, and download_url.
    """
    try:
        import nbformat

        nb = nbformat.v4.new_notebook()
        cells = []

        # Title cell
        cells.append(nbformat.v4.new_markdown_cell(f"# {title}\n\n*Session ID: {session.session_id}*"))

        # ReAct trace → cells
        for i, step in enumerate(session.react_trace):
            thought = step.get("thought", "")
            action = step.get("action", "")
            observation = step.get("observation", "")

            # Thought as markdown
            if thought:
                cells.append(nbformat.v4.new_markdown_cell(f"**Thought {i+1}:** {thought}"))

            # Action — if it was execute_python_code, try to get the code
            if action.startswith("execute_python_code"):
                try:
                    # Extract code from action string
                    import re
                    match = re.search(r'\{"code":\s*"(.*?)"\}', action, re.DOTALL)
                    if match:
                        code = match.group(1).replace("\\n", "\n").replace('\\"', '"')
                        code_cell = nbformat.v4.new_code_cell(code)
                        if observation and observation != "(no output)":
                            code_cell.outputs = [
                                nbformat.v4.new_output(
                                    output_type="stream",
                                    name="stdout",
                                    text=observation,
                                )
                            ]
                        cells.append(code_cell)
                    else:
                        cells.append(nbformat.v4.new_markdown_cell(
                            f"**Action:** `{action}`\n\n**Observation:**\n```\n{observation}\n```"
                        ))
                except Exception:
                    cells.append(nbformat.v4.new_markdown_cell(
                        f"**Action:** `{action}`\n\n**Observation:**\n```\n{observation}\n```"
                    ))
            else:
                cells.append(nbformat.v4.new_markdown_cell(
                    f"**Action:** `{action}`\n\n**Observation:**\n```\n{observation}\n```"
                ))

        # Append any explicit notebook_cells stored in the session
        for nc in session.notebook_cells:
            if nc.cell_type == "code":
                cell = nbformat.v4.new_code_cell(nc.source)
            else:
                cell = nbformat.v4.new_markdown_cell(nc.source)
            cells.append(cell)

        nb.cells = cells
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }

        # Save
        settings.notebooks_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        safe_title = safe_title[:50].strip()
        filename = f"{session.session_id[:8]}_{safe_title}.ipynb"
        nb_path = settings.notebooks_dir / filename

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        session.notebook_path = str(nb_path)
        logger.info("Notebook exported: %s (%d cells)", nb_path, len(cells))

        return json.dumps({
            "notebook_path": str(nb_path),
            "filename": filename,
            "cell_count": len(cells),
            "download_url": f"/api/v1/analysis/{session.session_id}/notebook",
        })

    except ImportError:
        return json.dumps({
            "error": "nbformat is not installed. Run: pip install nbformat",
            "notebook_path": None,
        })
    except Exception as exc:
        logger.error("export_notebook error: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "notebook_path": None})


def save_figure(session: AnalysisSession, figure_id: str, filename: str) -> str:
    """Save a figure from the session to disk as PNG.

    Returns JSON with saved_to, figure_id, and size_bytes.
    """
    import base64

    b64_png = session.figures.get(figure_id)
    if b64_png is None:
        return json.dumps({
            "error": f"Figure '{figure_id}' not found. Available: {list(session.figures.keys())}",
        })

    try:
        # Sanitize filename
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in filename)
        if not safe:
            safe = figure_id
        out_path = settings.figures_dir / f"{safe}.png"
        settings.figures_dir.mkdir(parents=True, exist_ok=True)

        img_bytes = base64.b64decode(b64_png)
        out_path.write_bytes(img_bytes)

        return json.dumps({
            "saved_to": str(out_path),
            "figure_id": figure_id,
            "filename": out_path.name,
            "size_bytes": len(img_bytes),
        })
    except Exception as exc:
        logger.error("save_figure error: %s", exc)
        return json.dumps({"error": str(exc)})
