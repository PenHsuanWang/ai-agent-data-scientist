# Phase 6 — FastAPI Integration

## Overview

Phase 6 wires the DataScienceAgentService into the FastAPI application with two new routers:
- `/api/v1/analysis` — chat-style analysis, continue session, download notebook
- `/api/v1/datasets` — list and inspect datasets

---

## 1. Pydantic Schemas

```python
# app/schemas/analysis.py
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Existing session ID to continue. If None, a new session is created.",
    )
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The analysis question or instruction for the agent.",
    )
    data_dir: str | None = Field(
        default=None,
        description="Override the default data directory. Defaults to settings.data_dir.",
    )
    enable_jupyter: bool = Field(
        default=False,
        description="If True, use Jupyter kernel backend for stateful execution.",
    )


class ReactTraceStep(BaseModel):
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str
    final_answer: str = ""


class PhysicalUnitResult(BaseModel):
    name: str
    value: float
    unit: str
    is_valid: bool
    warning: str
    expected_range: list[float] | None


class AnalysisResponse(BaseModel):
    response: str = Field(description="Final Answer from the agent")
    session_id: str
    status: str = Field(description="completed | error | timeout")
    react_trace: list[ReactTraceStep]
    figures: list[str] = Field(description="List of figure IDs (fetch via /figures/{id})")
    notebook_available: bool
    validated_units: list[PhysicalUnitResult] = []
    iterations: int = Field(description="Number of ReAct iterations used")


class FigureResponse(BaseModel):
    figure_id: str
    format: str = "png"
    encoding: str = "base64"
    data: str = Field(description="Base64-encoded PNG data")


class DatasetListItem(BaseModel):
    file_name: str
    format: str
    size_bytes: int


class DatasetInspectResponse(BaseModel):
    file_name: str
    format: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    numeric_stats: dict[str, Any]
    sample_rows: list[dict[str, Any]]


class SessionSummary(BaseModel):
    session_id: str
    status: str
    iterations: int
    figures: list[str]
    datasets: list[str]
    notebook_available: bool
    created_at: str
```

---

## 2. Session Store

If you already implemented `app/services/memory.py` from the base tutorial, keep that file and extend the existing `InMemorySessionStore` with `get_or_create_analysis(...)` rather than creating a parallel store module.

```python
# app/services/memory.py (extend existing MVP store)
"""In-memory session store for AgentSession + AnalysisSession objects."""
from __future__ import annotations
import threading
from app.domain.analysis_models import AnalysisSession
from app.domain.exceptions import SessionNotFoundError


class InMemorySessionStore:
    """Thread-safe in-memory store for AgentSession and AnalysisSession objects."""

    def __init__(self):
        self._sessions: dict[str, AnalysisSession] = {}
        self._lock = threading.RLock()

    def create(self, user_message: str, data_dir: str = "data") -> AnalysisSession:
        session = AnalysisSession.new(user_message, data_dir=data_dir)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> AnalysisSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def get_or_create_analysis(
        self,
        user_message: str,
        session_id: str | None = None,
        data_dir: str = "data",
    ) -> AnalysisSession:
        if session_id:
            session = self.get(session_id)
            session.add_user_message(user_message)
            return session
        return self.create(user_message, data_dir=data_dir)

    def list_all(self) -> list[dict]:
        with self._lock:
            return [s.to_summary_dict() for s in self._sessions.values()]

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


session_store = InMemorySessionStore()
```

---

## 3. `api/v1/analysis.py` — Complete Router

```python
# app/api/v1/analysis.py
from __future__ import annotations
import base64
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from app.core.config import settings
from app.domain.exceptions import ReActLoopError, SessionNotFoundError
from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    FigureResponse,
    ReactTraceStep,
    PhysicalUnitResult,
    SessionSummary,
)
from app.services.data_agent import DataScienceAgentService
from app.services.memory import session_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])

# One service instance per application (stateless)
_agent_service = DataScienceAgentService()


@router.post("/chat", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """
    Run a data science analysis using the ReAct agent.

    Creates a new session or continues an existing one.
    Returns the Final Answer plus the full ReAct trace and figure list.
    """
    data_dir = request.data_dir or str(settings.data_dir)

    # Override backend if Jupyter requested
    if request.enable_jupyter and not settings.enable_jupyter_bridge:
        raise HTTPException(
            status_code=400,
            detail="Jupyter bridge is disabled. Set ENABLE_JUPYTER_BRIDGE=true in .env",
        )

    try:
        session = session_store.get_or_create_analysis(
            user_message=request.user_message,
            session_id=request.session_id,
            data_dir=data_dir,
        )
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        agent_response = await _agent_service.run(session)
    except ReActLoopError as e:
        logger.warning("ReAct loop failed for session %s: %s", session.session_id, e)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ReAct loop failed to converge",
                "iterations": e.iterations,
                "last_thought": e.last_thought,
                "session_id": session.session_id,
            },
        )
    except Exception as e:
        logger.exception("Unexpected error in session %s", session.session_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Build response
    trace = [
        ReactTraceStep(
            thought=step.get("thought", ""),
            action=step.get("action", ""),
            action_input=step.get("action_input", {}),
            observation=step.get("observation", ""),
            final_answer=step.get("final_answer", ""),
        )
        for step in agent_response.react_trace
    ]
    units = [PhysicalUnitResult(**u) for u in agent_response.validated_units]

    return AnalysisResponse(
        response=agent_response.response,
        session_id=agent_response.session_id,
        status=agent_response.status,
        react_trace=trace,
        figures=agent_response.figures,
        notebook_available=agent_response.notebook_available,
        validated_units=units,
        iterations=len(trace),
    )


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    """List all analysis sessions (in-memory store)."""
    summaries = session_store.list_all()
    return [SessionSummary(**s) for s in summaries]


@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    """Get summary of a specific session."""
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SessionSummary(**session.to_summary_dict())


@router.get("/sessions/{session_id}/figures/{figure_id}", response_model=FigureResponse)
async def get_figure(session_id: str, figure_id: str) -> FigureResponse:
    """Retrieve a base64-encoded PNG figure from a session."""
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if figure_id not in session.figures:
        raise HTTPException(
            status_code=404,
            detail=f"Figure '{figure_id}' not found. Available: {session.figure_ids}",
        )
    return FigureResponse(
        figure_id=figure_id,
        data=session.figures[figure_id],
    )


@router.get("/sessions/{session_id}/figures/{figure_id}/png")
async def get_figure_png(session_id: str, figure_id: str) -> Response:
    """Return the raw PNG bytes for a figure (for browser display)."""
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if figure_id not in session.figures:
        raise HTTPException(status_code=404, detail=f"Figure '{figure_id}' not found.")

    png_bytes = base64.b64decode(session.figures[figure_id])
    return Response(content=png_bytes, media_type="image/png")


@router.get("/sessions/{session_id}/notebook")
async def download_notebook(session_id: str) -> FileResponse:
    """Download the exported Jupyter notebook for a session."""
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not session.notebook_available:
        raise HTTPException(
            status_code=404,
            detail=(
                "No notebook available for this session. "
                "Call export_notebook tool first, or ask the agent to export the notebook."
            ),
        )
    return FileResponse(
        path=str(session.notebook_path),
        media_type="application/octet-stream",
        filename=session.notebook_path.name,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    """Delete a session from the in-memory store."""
    session_store.delete(session_id)
```

---

## 4. `api/v1/datasets.py` — Complete Router

```python
# app/api/v1/datasets.py
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from app.core.config import settings
from app.schemas.analysis import DatasetListItem, DatasetInspectResponse
from app.services.knowledge_tools import (
    list_datasets,
    inspect_dataset,
    describe_columns,
)
from app.domain.exceptions import DatasetNotFoundError

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/", response_model=list[DatasetListItem])
async def list_available_datasets(
    data_dir: str | None = Query(default=None, description="Override default data directory"),
) -> list[DatasetListItem]:
    """List all dataset files available in the data directory."""
    raw = list_datasets(data_dir=data_dir or settings.data_dir)
    items = json.loads(raw)
    return [DatasetListItem(**item) for item in items]


@router.get("/{file_name}/inspect", response_model=DatasetInspectResponse)
async def inspect_dataset_endpoint(
    file_name: str,
    data_dir: str | None = Query(default=None),
) -> DatasetInspectResponse:
    """Inspect a dataset: schema, shape, numeric stats, sample rows."""
    try:
        raw = inspect_dataset(file_name, data_dir=data_dir or settings.data_dir)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    data = json.loads(raw)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return DatasetInspectResponse(**data)


@router.get("/{file_name}/columns")
async def describe_columns_endpoint(
    file_name: str,
    columns: str = Query(..., description="Comma-separated column names"),
    data_dir: str | None = Query(default=None),
) -> dict:
    """Get detailed statistics for specific columns."""
    col_list = [c.strip() for c in columns.split(",") if c.strip()]
    if not col_list:
        raise HTTPException(status_code=400, detail="At least one column name required.")
    raw = describe_columns(file_name, col_list, data_dir=data_dir or settings.data_dir)
    data = json.loads(raw)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return data
```

---

## 5. `main.py` Updates

```python
# app/main.py
from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1 import chat, analysis, datasets

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting %s", settings.app_name)
    settings.ensure_output_dirs()

    # Start idle kernel cleanup task (if Jupyter enabled)
    cleanup_task = None
    if settings.enable_jupyter_bridge:
        try:
            from app.infrastructure.jupyter_bridge import jupyter_manager
            async def _cleanup_loop():
                while True:
                    await asyncio.sleep(600)
                    if jupyter_manager:
                        count = await jupyter_manager.shutdown_idle_kernels()
                        if count:
                            logger.info("Shut down %d idle Jupyter kernels", count)
            cleanup_task = asyncio.create_task(_cleanup_loop())
        except ImportError:
            logger.warning("jupyter_client not installed; Jupyter bridge unavailable")

    yield

    # Shutdown
    if cleanup_task:
        cleanup_task.cancel()
    if settings.enable_jupyter_bridge:
        try:
            from app.infrastructure.jupyter_bridge import jupyter_manager
            if jupyter_manager:
                for sid in list(jupyter_manager._kernels.keys()):
                    await jupyter_manager.shutdown_kernel(sid)
                logger.info("All Jupyter kernels shut down")
        except Exception:
            pass
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing router
app.include_router(chat.router, prefix="/api/v1")

# New routers
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.claude_model,
        "backend": settings.code_execution_backend,
    }
```

---

## 6. Complete curl Examples

```bash
BASE="http://localhost:8000/api/v1"

# Start the server
uvicorn app.main:app --reload --port 8000

# Health check
curl -s $BASE/../health | python3 -m json.tool

# List datasets
curl -s "$BASE/datasets/" | python3 -m json.tool

# Inspect a dataset
curl -s "$BASE/datasets/power_plant_data.csv/inspect" | python3 -m json.tool

# Describe specific columns
curl -s "$BASE/datasets/power_plant_data.csv/columns?columns=efficiency_pct,gross_power_MW" \
  | python3 -m json.tool

# Run analysis (new session)
curl -s -X POST "$BASE/analysis/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "What is the mean thermal efficiency and is it physically plausible?",
    "data_dir": "data"
  }' | python3 -m json.tool

# Continue existing session
SESSION_ID="<session_id from previous response>"
curl -s -X POST "$BASE/analysis/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"user_message\": \"Now plot efficiency vs power output.\"
  }" | python3 -m json.tool

# List sessions
curl -s "$BASE/analysis/sessions" | python3 -m json.tool

# Get session summary
curl -s "$BASE/analysis/sessions/$SESSION_ID" | python3 -m json.tool

# Get figure as base64
curl -s "$BASE/analysis/sessions/$SESSION_ID/figures/fig_000" | python3 -m json.tool

# Get figure as raw PNG (pipe to file)
curl -s "$BASE/analysis/sessions/$SESSION_ID/figures/fig_000/png" --output analysis_fig.png
open analysis_fig.png   # macOS

# Download notebook
curl -s "$BASE/analysis/sessions/$SESSION_ID/notebook" \
  --output analysis.ipynb
jupyter notebook analysis.ipynb

# Delete session
curl -s -X DELETE "$BASE/analysis/sessions/$SESSION_ID"
```

---

## 7. Integration Test

```python
# tests/test_phase6_api.py
"""
FastAPI integration tests using TestClient (synchronous).
Mocks the Anthropic API so no real API key is needed.
"""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_mock_claude_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data


class TestDatasetEndpoints:
    def test_list_datasets_empty(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("app.api.v1.datasets.settings.data_dir", tmp_path)
        (tmp_path / "datasets").mkdir()
        response = client.get("/api/v1/datasets/")
        assert response.status_code == 200
        assert response.json() == []

    def test_inspect_nonexistent_dataset(self, client):
        response = client.get("/api/v1/datasets/nonexistent.csv/inspect")
        assert response.status_code == 404


class TestAnalysisEndpoints:

    def _mock_final_answer(self, text: str = "The answer is 42 percent."):
        return "Thought: I know the answer.\nFinal Answer: " + text

    def test_run_analysis_creates_session(self, client):
        final_text = self._mock_final_answer()
        mock_response = _make_mock_claude_response(final_text)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with patch(
                "app.services.context_injector.PhysicalContextInjector.build_system_prompt",
                new=AsyncMock(return_value="test system prompt")
            ):
                response = client.post(
                    "/api/v1/analysis/chat",
                    json={"user_message": "What is the efficiency?"},
                )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "completed"
        assert "42 percent" in data["response"]

    def test_list_sessions_returns_list(self, client):
        response = client.get("/api/v1/analysis/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_nonexistent_session_404(self, client):
        response = client.get("/api/v1/analysis/sessions/does-not-exist")
        assert response.status_code == 404

    def test_continue_session(self, client):
        """Create a session, then continue it with a follow-up question."""
        final_text1 = self._mock_final_answer("The mean efficiency is 36.7 percent.")
        final_text2 = self._mock_final_answer("The correlation coefficient is 0.82.")

        responses = [
            _make_mock_claude_response(final_text1),
            _make_mock_claude_response(final_text2),
        ]

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(side_effect=responses)
            with patch(
                "app.services.context_injector.PhysicalContextInjector.build_system_prompt",
                new=AsyncMock(return_value="test prompt")
            ):
                # First request
                r1 = client.post(
                    "/api/v1/analysis/chat",
                    json={"user_message": "What is the mean efficiency?"},
                )
                assert r1.status_code == 200
                session_id = r1.json()["session_id"]

                # Follow-up
                r2 = client.post(
                    "/api/v1/analysis/chat",
                    json={
                        "session_id": session_id,
                        "user_message": "What is the correlation between efficiency and power?",
                    },
                )

        assert r2.status_code == 200
        assert "0.82" in r2.json()["response"]

    def test_delete_session(self, client):
        """Create and then delete a session."""
        final_text = self._mock_final_answer()
        mock_response = _make_mock_claude_response(final_text)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with patch(
                "app.services.context_injector.PhysicalContextInjector.build_system_prompt",
                new=AsyncMock(return_value="prompt")
            ):
                r = client.post(
                    "/api/v1/analysis/chat",
                    json={"user_message": "Test."},
                )
        session_id = r.json()["session_id"]

        # Delete
        del_r = client.delete(f"/api/v1/analysis/sessions/{session_id}")
        assert del_r.status_code == 204

        # Should be gone
        get_r = client.get(f"/api/v1/analysis/sessions/{session_id}")
        assert get_r.status_code == 404

    def test_figure_not_found_404(self, client):
        """Requesting a figure that doesn't exist returns 404."""
        final_text = self._mock_final_answer()
        mock_response = _make_mock_claude_response(final_text)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with patch(
                "app.services.context_injector.PhysicalContextInjector.build_system_prompt",
                new=AsyncMock(return_value="prompt")
            ):
                r = client.post(
                    "/api/v1/analysis/chat",
                    json={"user_message": "Test."},
                )
        session_id = r.json()["session_id"]

        fig_r = client.get(f"/api/v1/analysis/sessions/{session_id}/figures/fig_999")
        assert fig_r.status_code == 404

    def test_notebook_not_available_404(self, client):
        """Downloading notebook before it's exported returns 404."""
        final_text = self._mock_final_answer()
        mock_response = _make_mock_claude_response(final_text)

        with patch("app.services.data_agent._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with patch(
                "app.services.context_injector.PhysicalContextInjector.build_system_prompt",
                new=AsyncMock(return_value="prompt")
            ):
                r = client.post(
                    "/api/v1/analysis/chat",
                    json={"user_message": "Test."},
                )
        session_id = r.json()["session_id"]

        nb_r = client.get(f"/api/v1/analysis/sessions/{session_id}/notebook")
        assert nb_r.status_code == 404
```

```bash
pytest tests/test_phase6_api.py -v
```

---

## Checkpoint

After Phase 6, the complete system is running:

```
app/schemas/analysis.py          -- Pydantic request/response models
app/services/memory.py           -- Extended InMemorySessionStore with get_or_create_analysis()
app/api/v1/analysis.py           -- 7 analysis endpoints
app/api/v1/datasets.py           -- 3 dataset endpoints
app/main.py                      -- Updated with new routers + lifespan
tests/test_phase6_api.py         -- FastAPI integration tests passing

Total: 14 tools, full ReAct loop, physical validation, figure capture,
       notebook export, stateful sessions via FastAPI.
```

All 6 phases complete. The Data Scientist AI-Agent is fully operational.
