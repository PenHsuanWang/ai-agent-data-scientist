# API Design — FastAPI Endpoints and OpenAPI Spec

**Document Version:** 1.0  
**Status:** Approved  
**Bounded Context:** API Layer — HTTP Request/Response Handling  

---

## 1. URL Structure

All routes grouped by concern:

```
/api/v1/                               ← Existing MVP (UNCHANGED)
├── chat                               POST  — MVP conversation
├── documents/                         POST  — MVP document upload
└── documents/{filename}               DELETE — MVP document delete

/api/v1/analysis/                      ← NEW: Data Scientist Agent
├── chat                               POST  — ReAct analysis conversation
└── {session_id}/
    ├── figures/{figure_id}            GET   — PNG figure retrieval
    └── notebook                       GET   — .ipynb download

/api/v1/datasets/                      ← NEW: Dataset management
├──                                    GET   — List all datasets
└── {name}/
    └── schema                         GET   — Dataset schema + sample
```

**Router prefix structure in `main.py`:**

```python
from app.api.v1 import chat, documents, analysis, datasets

def create_app() -> FastAPI:
    app = FastAPI(...)
    app.include_router(chat.router,      prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(analysis.router,  prefix="/api/v1/analysis")
    app.include_router(datasets.router,  prefix="/api/v1/datasets")
    return app
```

---

## 2. AnalysisRequest Schema

```python
# api/v1/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator


class AnalysisRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Natural language analysis request",
        examples=["Analyze power_plant_data.csv and compute thermal efficiency"],
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID to continue. If null, a new session is created.",
        examples=["a3f2b1c4-d5e6-7890-abcd-ef1234567890"],
    )
    dataset_hint: str | None = Field(
        default=None,
        description="Optional: pre-load this dataset before the agent starts",
        examples=["power_plant_data.csv"],
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank or whitespace-only")
        return v.strip()

    @field_validator("session_id")
    @classmethod
    def session_id_safe(cls, v: str | None) -> str | None:
        if v is None:
            return None
        import re
        if not re.fullmatch(r"[a-zA-Z0-9\-]{1,64}", v):
            raise ValueError("session_id must be alphanumeric with hyphens, max 64 chars")
        return v
```

---

## 3. AnalysisResponse Schema

```python
class ReActStep(BaseModel):
    thought: str = Field(description="Claude's reasoning for this step")
    action: str = Field(description="Tool called or 'Final Answer'")
    observation: str = Field(description="Tool result or final answer text")


class FigureRef(BaseModel):
    figure_id: str = Field(description="Unique figure identifier within the session")
    retrieval_url: str = Field(
        description="URL to GET the PNG: /api/v1/analysis/{session_id}/figures/{figure_id}"
    )


class AnalysisResponse(BaseModel):
    response: str = Field(description="Agent's final answer")
    session_id: str = Field(description="Session identifier for follow-up requests")
    react_trace: list[ReActStep] = Field(default_factory=list)
    figures: list[FigureRef] = Field(default_factory=list)
    notebook_available: bool = Field(default=False)
    unit_validations: list[dict] = Field(default_factory=list)
    iterations_used: int = Field(default=0)
    model: str = Field(default="")

    model_config = {"json_schema_extra": {
        "example": {
            "response": "The thermal efficiency averages 42.3% (range: 38%-47%).",
            "session_id": "a3f2b1c4-d5e6-7890-abcd-ef1234567890",
            "react_trace": [
                {
                    "thought": "First I should inspect the dataset.",
                    "action": "inspect_dataset({\"file_name\": \"power_plant_data.csv\"})",
                    "observation": "{\"shape\": [1000, 8]}",
                }
            ],
            "figures": [
                {"figure_id": "fig_000",
                 "retrieval_url": "/api/v1/analysis/.../figures/fig_000"}
            ],
            "notebook_available": False,
            "unit_validations": [{"quantity": "thermal_efficiency", "is_valid": True}],
            "iterations_used": 3,
            "model": "claude-sonnet-4-6",
        }
    }}
```

---

## 4. Figure Retrieval

Figures are returned as raw PNG bytes. The endpoint returns `Response` with `Content-Type: image/png`. No Pydantic schema needed for binary responses.

```
GET /api/v1/analysis/{session_id}/figures/{figure_id}
→ Content-Type: image/png
→ Body: raw PNG bytes (base64-decoded from session.figures[figure_id])
```

---

## 5. Dataset Schemas

```python
class DatasetInfo(BaseModel):
    file_name: str
    format: str = Field(description="File format: csv, xlsx, parquet, etc.")
    file_size_bytes: int
    shape: list[int] | None = Field(default=None, description="[rows, cols] if available")


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo]
    total: int = Field(description="Total number of datasets")


class DatasetSchemaResponse(BaseModel):
    file_name: str
    path: str
    shape: list[int] = Field(description="[rows, cols]")
    dtypes: dict[str, str] = Field(description="Column name to pandas dtype string")
    sample_rows: list[dict] = Field(description="First 5 rows as list of row dicts")
    null_counts: dict[str, int] = Field(description="Column name to null value count")
    file_size_bytes: int
```

---

## 6. ErrorDetail Schema

```python
class ErrorDetail(BaseModel):
    error: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    session_id: str | None = Field(default=None)
    react_trace: list[ReActStep] = Field(default_factory=list)
    details: dict | None = Field(default=None)
```

**HTTP Status Code Mapping:**

| Error Condition                   | HTTP Status | `error` Code           |
|-----------------------------------|-------------|------------------------|
| Invalid request body              | 422         | `validation_error`     |
| Session not found                 | 404         | `session_not_found`    |
| Figure not found                  | 404         | `figure_not_found`     |
| Dataset not found                 | 404         | `dataset_not_found`    |
| Notebook not generated            | 404         | `notebook_not_available` |
| ReAct max iterations exceeded     | 500         | `react_loop_exhausted` |
| Claude API error                  | 502         | `llm_api_error`        |
| Code execution timeout            | 500         | `code_timeout`         |
| Generic server error              | 500         | `internal_error`       |

> **Design note**: ReAct errors (exhausted iterations, malformed output) return 500, not 200. This
> differs from some systems that embed domain errors in HTTP 200 bodies. Here, HTTP status reflects
> transport + application-level success; agent reasoning failures are not "expected" outcomes.

---

## 7. POST /api/v1/analysis/chat — Full Handler

```python
# api/v1/analysis.py
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, DatasetMeta
from app.domain.exceptions import ReActLoopError, ReActMaxIterationsError
from app.services.data_agent import DataScienceAgentService
from app.services.memory import InMemorySessionStore
from .schemas import (
    AnalysisRequest, AnalysisResponse, ReActStep, FigureRef, ErrorDetail,
)

router = APIRouter(tags=["analysis"])
_agent_service = DataScienceAgentService()
_session_store: InMemorySessionStore = InMemorySessionStore()


@router.post(
    "/chat",
    response_model=AnalysisResponse,
    summary="Run data analysis via ReAct agent",
    description=(
        "Sends a natural-language analysis request to the Data Scientist Agent. "
        "The agent runs a ReAct reasoning loop (up to 20 iterations), using 14 tools "
        "to inspect datasets, execute Python code, validate physical units, and "
        "generate figures. Returns the agent final answer plus a complete reasoning trace."
    ),
    responses={
        200: {"description": "Analysis complete"},
        422: {"model": ErrorDetail, "description": "Invalid request"},
        500: {"model": ErrorDetail, "description": "Agent loop error"},
        502: {"model": ErrorDetail, "description": "Claude API error"},
    },
)
async def analysis_chat(request: AnalysisRequest) -> AnalysisResponse:
    session_id = request.session_id or str(uuid.uuid4())
    session = _session_store.get(session_id)

    if session is None:
        session = AnalysisSession(session_id=session_id)
        _session_store.set(session_id, session)
    elif not isinstance(session, AnalysisSession):
        raise HTTPException(
            status_code=409,
            detail=ErrorDetail(
                error="session_type_mismatch",
                message=f"Session '{session_id}' is not an analysis session.",
            ).model_dump(),
        )

    session.add_user_message(request.message)

    if request.dataset_hint:
        _try_preload_dataset(session, request.dataset_hint)

    try:
        response_text = await _agent_service.run(session)
    except ReActMaxIterationsError as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                error="react_loop_exhausted",
                message=str(e),
                session_id=session_id,
                react_trace=_build_trace(session),
            ).model_dump(),
        )
    except ReActLoopError as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                error="react_loop_error",
                message=str(e),
                session_id=session_id,
                react_trace=_build_trace(session),
            ).model_dump(),
        )
    except Exception as e:
        code = 502 if "API" in type(e).__name__ else 500
        raise HTTPException(
            status_code=code,
            detail=ErrorDetail(
                error="llm_api_error" if code == 502 else "internal_error",
                message=str(e),
                session_id=session_id,
            ).model_dump(),
        )

    return AnalysisResponse(
        response=response_text,
        session_id=session_id,
        react_trace=_build_trace(session),
        figures=_build_figure_refs(session, session_id),
        notebook_available=_notebook_exists(session_id),
        unit_validations=[pu.to_json_dict() for pu in session.unit_context],
        iterations_used=len(session.react_trace),
        model=settings.claude_model,
    )
```

---

## 8. GET /api/v1/analysis/{session\_id}/figures/{figure\_id}

```python
@router.get(
    "/{session_id}/figures/{figure_id}",
    summary="Retrieve a generated figure as PNG",
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG image data"},
        404: {"model": ErrorDetail, "description": "Session or figure not found"},
    },
)
async def get_figure(session_id: str, figure_id: str) -> Response:
    session = _get_analysis_session(session_id)

    if figure_id not in session.figures:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                error="figure_not_found",
                message=f"Figure '{figure_id}' not found.",
                details={"available": list(session.figures.keys())},
            ).model_dump(),
        )

    try:
        png_bytes = base64.b64decode(session.figures[figure_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{figure_id}.png"',
            "Cache-Control": "private, max-age=3600",
        },
    )
```

---

## 9. GET /api/v1/analysis/{session\_id}/notebook

```python
@router.get(
    "/{session_id}/notebook",
    summary="Download analysis as Jupyter notebook",
    responses={
        200: {"content": {"application/octet-stream": {}}, "description": ".ipynb file"},
        404: {"model": ErrorDetail, "description": "Notebook not generated"},
    },
)
async def get_notebook(session_id: str) -> FileResponse:
    _get_analysis_session(session_id)

    nb_dir = settings.notebooks_dir
    matches = list(nb_dir.glob(f"{session_id}_*.ipynb")) if nb_dir.exists() else []

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                error="notebook_not_available",
                message=(
                    f"No notebook for session '{session_id}'. "
                    "Ask the agent: 'Export this analysis as a notebook'."
                ),
            ).model_dump(),
        )

    nb_path = max(matches, key=lambda p: p.stat().st_mtime)
    return FileResponse(
        path=str(nb_path),
        media_type="application/octet-stream",
        filename=nb_path.name,
        headers={"Content-Disposition": f'attachment; filename="{nb_path.name}"'},
    )
```

---

## 10. GET /api/v1/datasets

```python
# api/v1/datasets.py
from __future__ import annotations
import json
from fastapi import APIRouter, HTTPException
from app.core.config import settings
from .schemas import DatasetInfo, DatasetListResponse, DatasetSchemaResponse, ErrorDetail

router = APIRouter(tags=["datasets"])
_SUPPORTED = {".csv", ".xlsx", ".xls", ".parquet", ".hdf5", ".h5", ".json"}


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List all available datasets",
    description=(
        "Returns all dataset files in the datasets directory. "
        "Supported formats: CSV, Excel, Parquet, HDF5, JSON."
    ),
)
async def list_datasets() -> DatasetListResponse:
    d = settings.datasets_dir
    if not d.exists():
        return DatasetListResponse(datasets=[], total=0)

    items = [
        DatasetInfo(
            file_name=f.name,
            format=f.suffix.lower().lstrip("."),
            file_size_bytes=f.stat().st_size,
            shape=None,
        )
        for f in sorted(d.glob("*"))
        if f.is_file() and f.suffix.lower() in _SUPPORTED
    ]
    return DatasetListResponse(datasets=items, total=len(items))
```

---

## 11. GET /api/v1/datasets/{name}/schema

```python
@router.get(
    "/{name}/schema",
    response_model=DatasetSchemaResponse,
    summary="Get schema and sample data for a dataset",
    description=(
        "Inspects a dataset: returns column names, dtypes, shape, "
        "sample rows (first 5), and null counts per column."
    ),
    responses={
        200: {"description": "Dataset schema"},
        403: {"model": ErrorDetail, "description": "Path traversal attempt"},
        404: {"model": ErrorDetail, "description": "Dataset not found"},
        422: {"model": ErrorDetail, "description": "Unsupported format"},
    },
)
async def get_dataset_schema(name: str) -> DatasetSchemaResponse:
    if ".." in name or name.startswith("/"):
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(
                error="invalid_name",
                message=f"Invalid dataset name: {name!r}",
            ).model_dump(),
        )

    d = settings.datasets_dir
    path = (d / name).resolve()

    if not str(path).startswith(str(d.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not path.exists():
        avail = [f.name for f in d.glob("*") if f.is_file()] if d.exists() else []
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                error="dataset_not_found",
                message=f"Dataset '{name}' not found.",
                details={"available": avail},
            ).model_dump(),
        )

    from app.services.knowledge_tools import inspect_dataset
    result = inspect_dataset(name)

    if result.startswith("Error:"):
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(
                error="inspection_failed",
                message=result,
            ).model_dump(),
        )

    try:
        data = json.loads(result)
        return DatasetSchemaResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(error="parse_error", message=str(e)).model_dump(),
        )
```

---

## 12. OpenAPI Configuration in `main.py`

```python
# app/main.py
from fastapi import FastAPI
from app.api.v1.analysis import router as analysis_router
from app.api.v1.datasets import router as datasets_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Data Scientist AI Agent",
        description=(
            "Conversational AI for data analysis powered by Claude claude-sonnet-4-6. "
            "Combines ReAct reasoning, Python code execution, physical unit validation, "
            "and Jupyter notebook export."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {
                "name": "chat",
                "description": "MVP conversational agent — existing endpoints, unchanged.",
            },
            {
                "name": "analysis",
                "description": (
                    "Data Scientist Agent. Runs a ReAct loop with 14 tools. "
                    "Multi-turn sessions via session_id."
                ),
            },
            {
                "name": "datasets",
                "description": "Dataset listing and schema inspection.",
            },
        ],
    )

    from app.api.v1 import chat, documents
    app.include_router(chat.router,      prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(analysis_router,  prefix="/api/v1/analysis")
    app.include_router(datasets_router,  prefix="/api/v1/datasets")
    return app
```

---

## 13. Complete `analysis.py`

Full implementation with all helpers:

```python
# api/v1/analysis.py
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from app.core.config import settings
from app.domain.analysis_models import AnalysisSession, DatasetMeta
from app.domain.exceptions import ReActLoopError, ReActMaxIterationsError
from app.services.data_agent import DataScienceAgentService
from app.services.memory import InMemorySessionStore
from .schemas import (
    AnalysisRequest, AnalysisResponse, ReActStep, FigureRef, ErrorDetail,
)

router = APIRouter(tags=["analysis"])
_agent_service = DataScienceAgentService()
_session_store: InMemorySessionStore = InMemorySessionStore()


@router.post("/chat", response_model=AnalysisResponse, summary="Run data analysis via ReAct agent")
async def analysis_chat(request: AnalysisRequest) -> AnalysisResponse:
    session_id = request.session_id or str(uuid.uuid4())
    session = _session_store.get(session_id)

    if session is None:
        session = AnalysisSession(session_id=session_id)
        _session_store.set(session_id, session)
    elif not isinstance(session, AnalysisSession):
        raise HTTPException(
            status_code=409,
            detail=ErrorDetail(
                error="session_type_mismatch",
                message=f"Session '{session_id}' is not an analysis session.",
            ).model_dump(),
        )

    session.add_user_message(request.message)

    if request.dataset_hint:
        _try_preload_dataset(session, request.dataset_hint)

    try:
        response_text = await _agent_service.run(session)
    except ReActMaxIterationsError as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                error="react_loop_exhausted", message=str(e),
                session_id=session_id, react_trace=_build_trace(session),
            ).model_dump(),
        )
    except ReActLoopError as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                error="react_loop_error", message=str(e),
                session_id=session_id, react_trace=_build_trace(session),
            ).model_dump(),
        )
    except Exception as e:
        code = 502 if "API" in type(e).__name__ else 500
        raise HTTPException(
            status_code=code,
            detail=ErrorDetail(
                error="llm_api_error" if code == 502 else "internal_error",
                message=str(e), session_id=session_id,
            ).model_dump(),
        )

    return AnalysisResponse(
        response=response_text,
        session_id=session_id,
        react_trace=_build_trace(session),
        figures=_build_figure_refs(session, session_id),
        notebook_available=_notebook_exists(session_id),
        unit_validations=[pu.to_json_dict() for pu in session.unit_context],
        iterations_used=len(session.react_trace),
        model=settings.claude_model,
    )


@router.get("/{session_id}/figures/{figure_id}", summary="Retrieve a figure as PNG")
async def get_figure(session_id: str, figure_id: str) -> Response:
    session = _get_analysis_session(session_id)
    if figure_id not in session.figures:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                error="figure_not_found",
                message=f"Figure '{figure_id}' not found.",
                details={"available": list(session.figures.keys())},
            ).model_dump(),
        )
    try:
        png = base64.b64decode(session.figures[figure_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{figure_id}.png"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/{session_id}/notebook", summary="Download analysis as Jupyter notebook")
async def get_notebook(session_id: str) -> FileResponse:
    _get_analysis_session(session_id)
    nb_dir = settings.notebooks_dir
    matches = list(nb_dir.glob(f"{session_id}_*.ipynb")) if nb_dir.exists() else []
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                error="notebook_not_available",
                message="No notebook found. Ask the agent to export_notebook first.",
            ).model_dump(),
        )
    nb_path = max(matches, key=lambda p: p.stat().st_mtime)
    return FileResponse(
        path=str(nb_path),
        media_type="application/octet-stream",
        filename=nb_path.name,
        headers={"Content-Disposition": f'attachment; filename="{nb_path.name}"'},
    )


# ─── Private helpers ──────────────────────────────────────────────────────────

def _get_analysis_session(session_id: str) -> AnalysisSession:
    s = _session_store.get(session_id)
    if s is None or not isinstance(s, AnalysisSession):
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                error="session_not_found",
                message=f"Analysis session '{session_id}' not found.",
            ).model_dump(),
        )
    return s


def _build_trace(session: AnalysisSession) -> list[ReActStep]:
    return [
        ReActStep(
            thought=s.get("thought", ""),
            action=s.get("action", ""),
            observation=s.get("observation", ""),
        )
        for s in session.react_trace
    ]


def _build_figure_refs(session: AnalysisSession, session_id: str) -> list[FigureRef]:
    return [
        FigureRef(
            figure_id=fid,
            retrieval_url=f"/api/v1/analysis/{session_id}/figures/{fid}",
        )
        for fid in session.figures
    ]


def _notebook_exists(session_id: str) -> bool:
    d = settings.notebooks_dir
    return d.exists() and any(d.glob(f"{session_id}_*.ipynb"))


def _try_preload_dataset(session: AnalysisSession, file_name: str) -> None:
    from app.services.knowledge_tools import inspect_dataset
    result = inspect_dataset(file_name)
    if result.startswith("Error:"):
        return
    try:
        data = json.loads(result)
        meta = DatasetMeta(
            file_name=data["file_name"],
            path=Path(data["path"]),
            shape=tuple(data["shape"]),
            dtypes=data["dtypes"],
            sample_rows=data["sample_rows"],
            null_counts=data["null_counts"],
            file_size_bytes=data["file_size_bytes"],
        )
        session.register_dataset(meta)
    except Exception:
        pass
```

---

## 14. Complete `datasets.py`

```python
# api/v1/datasets.py
from __future__ import annotations
import json
from fastapi import APIRouter, HTTPException
from app.core.config import settings
from .schemas import DatasetInfo, DatasetListResponse, DatasetSchemaResponse, ErrorDetail

router = APIRouter(tags=["datasets"])
_SUPPORTED = {".csv", ".xlsx", ".xls", ".parquet", ".hdf5", ".h5", ".json"}


@router.get("", response_model=DatasetListResponse, summary="List all available datasets")
async def list_datasets() -> DatasetListResponse:
    d = settings.datasets_dir
    if not d.exists():
        return DatasetListResponse(datasets=[], total=0)
    items = [
        DatasetInfo(
            file_name=f.name,
            format=f.suffix.lower().lstrip("."),
            file_size_bytes=f.stat().st_size,
            shape=None,
        )
        for f in sorted(d.glob("*"))
        if f.is_file() and f.suffix.lower() in _SUPPORTED
    ]
    return DatasetListResponse(datasets=items, total=len(items))


@router.get("/{name}/schema", response_model=DatasetSchemaResponse, summary="Get dataset schema")
async def get_dataset_schema(name: str) -> DatasetSchemaResponse:
    if ".." in name or name.startswith("/"):
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(error="invalid_name", message=f"Invalid: {name!r}").model_dump(),
        )

    d = settings.datasets_dir
    path = (d / name).resolve()

    if not str(path).startswith(str(d.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not path.exists():
        avail = [f.name for f in d.glob("*") if f.is_file()] if d.exists() else []
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                error="dataset_not_found",
                message=f"Dataset '{name}' not found.",
                details={"available": avail},
            ).model_dump(),
        )

    from app.services.knowledge_tools import inspect_dataset
    result = inspect_dataset(name)

    if result.startswith("Error:"):
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(error="inspection_failed", message=result).model_dump(),
        )

    try:
        data = json.loads(result)
        return DatasetSchemaResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(error="parse_error", message=str(e)).model_dump(),
        )
```

---

## Example curl Requests

```bash
# Start a new analysis session
curl -X POST http://localhost:8000/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze power_plant_data.csv and check if efficiency values are physically correct",
    "dataset_hint": "power_plant_data.csv"
  }'

# Continue the same session (multi-turn)
curl -X POST http://localhost:8000/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "a3f2b1c4-d5e6-7890-abcd-ef1234567890",
    "message": "Now plot efficiency over time and export a Jupyter notebook"
  }'

# Download a generated figure as PNG
curl "http://localhost:8000/api/v1/analysis/a3f2b1c4-d5e6-7890-abcd-ef1234567890/figures/fig_000" \
  --output efficiency_plot.png

# Download the Jupyter notebook
curl "http://localhost:8000/api/v1/analysis/a3f2b1c4-d5e6-7890-abcd-ef1234567890/notebook" \
  --output analysis.ipynb

# List all datasets
curl http://localhost:8000/api/v1/datasets

# Get schema for a specific dataset
curl http://localhost:8000/api/v1/datasets/power_plant_data.csv/schema
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| ReAct errors → HTTP 5xx | Agent reasoning failure is not a "successful but empty" response; client should not show a 200 to the user when the analysis failed. |
| Figure stored as base64 in session | Avoids disk I/O per figure; retrieved on demand and decoded to raw bytes only at retrieval time. |
| `dataset_hint` preloads via `inspect_dataset` | Saves one ReAct iteration; agent sees dataset schema in the first prompt injection. Non-fatal if it fails. |
| `session_id` alphanumeric + hyphens only | Prevents path-injection in notebook glob pattern (`f"{session_id}_*.ipynb"`). |
| `notebooks_dir` glob sorted by mtime | Handles re-export of same session (overwrites or appends timestamp); always returns the latest .ipynb. |
| Thin handler pattern | All ReAct logic, tool dispatch, and validation live in `DataScienceAgentService`. The handler's only jobs are: deserialise, session CRUD, delegate, serialise. |
