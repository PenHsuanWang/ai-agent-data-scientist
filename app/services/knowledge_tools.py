"""Knowledge tools — Group A (6 tools).

All functions:
- Never raise exceptions (return "Error: ..." strings on failure)
- Are path-safe (resolve against known base directories)
- Are read-only (no filesystem mutations)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_DOC_BYTES = 51_200   # 50 KB
MAX_DATASET_ROWS = 5


# ──────────────────────────────────────────────────────────────────── #
# Path safety helper                                                    #
# ──────────────────────────────────────────────────────────────────── #


def _safe_resolve(base_dir: Path, file_name: str) -> Path | None:
    """Return resolved Path if safe, None if traversal detected."""
    if ".." in file_name or file_name.startswith("/") or file_name.startswith("\\"):
        return None
    target = (base_dir / file_name).resolve()
    if base_dir.resolve() not in target.parents and target.resolve() != base_dir.resolve():
        return None
    return target


# ──────────────────────────────────────────────────────────────────── #
# Tool 1: list_domain_documents                                         #
# ──────────────────────────────────────────────────────────────────── #


def list_domain_documents() -> str:
    """Lists all domain knowledge documents available for reading.

    Returns JSON array of filenames.
    """
    docs_dir = settings.domain_docs_dir
    if not docs_dir.exists():
        return json.dumps([])

    files = sorted(
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".md", ".txt", ".rst"}
    )
    result = [{"file_name": f.name, "size_bytes": f.stat().st_size} for f in files]
    return json.dumps(result)


# ──────────────────────────────────────────────────────────────────── #
# Tool 2: read_domain_document                                          #
# ──────────────────────────────────────────────────────────────────── #


def read_domain_document(file_name: str) -> str:
    """Reads and returns the full content of a domain knowledge document.

    Truncates at 50 KB with notice.
    """
    docs_dir = settings.domain_docs_dir
    target = _safe_resolve(docs_dir, file_name)
    if target is None:
        return f"Error: '{file_name}' contains invalid path components."
    if not target.exists():
        return (
            f"Error: Document '{file_name}' not found. "
            "Call list_domain_documents to see available files."
        )
    if not target.is_file():
        return f"Error: '{file_name}' is not a file."

    try:
        raw = target.read_bytes()
        if len(raw) > MAX_DOC_BYTES:
            text = raw[:MAX_DOC_BYTES].decode("utf-8", errors="replace")
            return text + f"\n\n[...truncated at {MAX_DOC_BYTES} bytes...]"
        return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.error("read_domain_document('%s'): %s", file_name, exc)
        return f"Error: Could not read '{file_name}' — {exc}"


# ──────────────────────────────────────────────────────────────────── #
# Tool 3: search_domain_knowledge                                       #
# ──────────────────────────────────────────────────────────────────── #


def search_domain_knowledge(query: str, top_k: int = 3) -> str:
    """Keyword search across all domain documents.

    Returns JSON array of {file, score, snippet} sorted by relevance.
    """
    top_k = min(max(1, top_k), 10)
    docs_dir = settings.domain_docs_dir
    if not docs_dir.exists():
        return json.dumps([])

    query_terms = set(query.lower().split())
    results: list[dict[str, Any]] = []

    for doc in sorted(docs_dir.glob("*.md")) + sorted(docs_dir.glob("*.txt")):
        try:
            text = doc.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        text_lower = text.lower()
        score = sum(1 for term in query_terms if term in text_lower)
        if score == 0:
            continue

        # Find best matching snippet
        best_pos = 0
        best_score = 0
        for term in query_terms:
            pos = text_lower.find(term)
            if pos >= 0:
                local_score = sum(1 for t in query_terms if t in text_lower[max(0, pos-200):pos+200])
                if local_score > best_score:
                    best_score = local_score
                    best_pos = pos

        snippet_start = max(0, best_pos - 150)
        snippet_end = min(len(text), best_pos + 300)
        snippet = text[snippet_start:snippet_end].strip()
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(text):
            snippet = snippet + "..."

        results.append({"file": doc.name, "score": score, "snippet": snippet})

    results.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(results[:top_k])


# ──────────────────────────────────────────────────────────────────── #
# Tool 4: list_datasets                                                 #
# ──────────────────────────────────────────────────────────────────── #

_DATASET_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".h5", ".hdf5", ".json"}


def list_datasets() -> str:
    """Lists all available dataset files with format and size.

    Returns JSON array of {file_name, format, size_bytes}.
    """
    datasets_dir = settings.datasets_dir
    if not datasets_dir.exists():
        return json.dumps([])

    files = sorted(
        f for f in datasets_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _DATASET_EXTENSIONS
    )

    def _fmt(f: Path) -> str:
        ext = f.suffix.lower()
        mapping = {".csv": "csv", ".parquet": "parquet", ".xlsx": "excel",
                   ".xls": "excel", ".h5": "hdf5", ".hdf5": "hdf5", ".json": "json"}
        return mapping.get(ext, ext.lstrip("."))

    result = [
        {"file_name": f.name, "format": _fmt(f), "size_bytes": f.stat().st_size}
        for f in files
    ]
    return json.dumps(result)


# ──────────────────────────────────────────────────────────────────── #
# Tool 5: inspect_dataset                                               #
# ──────────────────────────────────────────────────────────────────── #


def inspect_dataset(file_name: str) -> str:
    """Load a dataset and return schema, shape, stats, and sample rows.

    Returns JSON with file_name, format, rows, columns, column_names,
    dtypes, numeric_stats, sample_rows.
    """
    datasets_dir = settings.datasets_dir
    target = _safe_resolve(datasets_dir, file_name)
    if target is None:
        return f"Error: '{file_name}' contains invalid path components."
    if not target.exists():
        return (
            f"Error: Dataset '{file_name}' not found. "
            "Call list_datasets to see available files."
        )

    try:
        import pandas as pd
        ext = target.suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(target)
            fmt = "csv"
        elif ext == ".parquet":
            df = pd.read_parquet(target)
            fmt = "parquet"
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(target)
            fmt = "excel"
        elif ext in (".h5", ".hdf5"):
            df = pd.read_hdf(target)
            fmt = "hdf5"
        elif ext == ".json":
            df = pd.read_json(target)
            fmt = "json"
        else:
            return f"Error: Unsupported format '{ext}'. Supported: csv, parquet, xlsx, h5, json"

        # Numeric statistics
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_stats: dict[str, Any] = {}
        for col in numeric_cols:
            s = df[col]
            numeric_stats[col] = {
                "min": round(float(s.min()), 4) if not s.isna().all() else None,
                "max": round(float(s.max()), 4) if not s.isna().all() else None,
                "mean": round(float(s.mean()), 4) if not s.isna().all() else None,
                "std": round(float(s.std()), 4) if not s.isna().all() else None,
                "null_count": int(s.isna().sum()),
            }

        sample = df.head(MAX_DATASET_ROWS).fillna("").astype(str).to_dict(orient="records")

        return json.dumps({
            "file_name": file_name,
            "format": fmt,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_stats": numeric_stats,
            "sample_rows": sample,
        })

    except Exception as exc:
        logger.error("inspect_dataset('%s'): %s", file_name, exc, exc_info=True)
        return f"Error: Could not inspect '{file_name}' — {exc}"


# ──────────────────────────────────────────────────────────────────── #
# Tool 6: describe_columns                                              #
# ──────────────────────────────────────────────────────────────────── #


def describe_columns(file_name: str, columns: list[str]) -> str:
    """Return detailed per-column statistics for specified columns.

    Returns JSON dict {column_name: {stats}}.
    """
    datasets_dir = settings.datasets_dir
    target = _safe_resolve(datasets_dir, file_name)
    if target is None:
        return f"Error: '{file_name}' contains invalid path components."
    if not target.exists():
        return f"Error: Dataset '{file_name}' not found."

    try:
        import pandas as pd

        ext = target.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(target)
        elif ext == ".parquet":
            df = pd.read_parquet(target)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(target)
        elif ext in (".h5", ".hdf5"):
            df = pd.read_hdf(target)
        elif ext == ".json":
            df = pd.read_json(target)
        else:
            return f"Error: Unsupported format '{ext}'"

        missing = [c for c in columns if c not in df.columns]
        if missing:
            return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

        result: dict[str, Any] = {}
        for col in columns:
            s = df[col]
            if s.dtype.kind in "iuf":  # numeric
                result[col] = {
                    "type": "numeric",
                    "count": int(s.count()),
                    "null_count": int(s.isna().sum()),
                    "min": round(float(s.min()), 6),
                    "max": round(float(s.max()), 6),
                    "mean": round(float(s.mean()), 6),
                    "median": round(float(s.median()), 6),
                    "std": round(float(s.std()), 6),
                    "q25": round(float(s.quantile(0.25)), 6),
                    "q75": round(float(s.quantile(0.75)), 6),
                    "skewness": round(float(s.skew()), 4),
                }
            elif s.dtype.kind == "M":  # datetime
                result[col] = {
                    "type": "datetime",
                    "count": int(s.count()),
                    "null_count": int(s.isna().sum()),
                    "min": str(s.min()),
                    "max": str(s.max()),
                }
            else:  # categorical / object
                vc = s.value_counts().head(10)
                result[col] = {
                    "type": "categorical",
                    "count": int(s.count()),
                    "null_count": int(s.isna().sum()),
                    "unique_count": int(s.nunique()),
                    "top_values": {str(k): int(v) for k, v in vc.items()},
                }

        return json.dumps(result)

    except Exception as exc:
        logger.error("describe_columns('%s', %s): %s", file_name, columns, exc)
        return f"Error: {exc}"


# ──────────────────────────────────────────────────────────────────── #
# Tool 7: get_coding_standards                                          #
# ──────────────────────────────────────────────────────────────────── #

_STANDARDS_FILE = "coding_standards.md"


def get_coding_standards() -> str:
    """Return the project's coding style, analysis strategy, and visualization spec.

    Reads coding_standards.md from the domain_docs directory.
    Always call this at the start of any analysis task to ensure
    your generated code follows the project's conventions.
    """
    return read_domain_document(_STANDARDS_FILE)
