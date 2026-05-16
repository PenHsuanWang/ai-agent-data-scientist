"""Analysis API — POST /api/v1/analysis/chat + figure/notebook retrieval."""
from __future__ import annotations

import base64
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, Response

from app.domain.analysis_models import AnalysisSession, DatasetMeta
from app.domain.exceptions import (
    AgentError,
    LLMAPIError,
    LLMAuthenticationError,
    LLMContextOverflowError,
    ReActLoopError,
)
from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    FigureRef,
    ReActStep,
)
from app.core.config import settings
from app.services.data_agent import data_science_agent
from app.services.memory import analysis_session_store

router = APIRouter()
logger = logging.getLogger(__name__)


def _try_preload_dataset(session: AnalysisSession, dataset_hint: str | None) -> None:
    """Pre-load dataset metadata into the session if a hint is provided (Gap 12).

    Calls ``inspect_dataset`` and registers the resulting ``DatasetMeta`` so
    Claude can reference it without a round-trip tool call.  Failures are
    logged and silently swallowed so they never block the main request.
    """
    if not dataset_hint:
        return
    try:
        from app.services.knowledge_tools import inspect_dataset
        raw = inspect_dataset(dataset_hint)
        data = json.loads(raw)
        if "error" in data:
            logger.warning(
                "dataset_hint pre-load failed for '%s': %s", dataset_hint, data["error"]
            )
            return
        meta = DatasetMeta(
            file_name=data["file_name"],
            format=data["format"],
            rows=data["rows"],
            columns=data["columns"],
            column_names=data["column_names"],
            dtypes=data["dtypes"],
            numeric_stats=data["numeric_stats"],
            size_bytes=0,
        )
        session.register_dataset(meta)
        logger.info(
            "Pre-loaded dataset '%s' (%d rows) for session %s",
            dataset_hint, meta.rows, session.session_id,
        )
    except Exception as exc:
        logger.warning(
            "dataset_hint pre-load raised for '%s': %s", dataset_hint, exc
        )


@router.post(
    "/chat",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a data science analysis request",
    description=(
        "Submit a natural language analysis request. "
        "The agent uses a ReAct reasoning loop with 15 tools to analyse datasets, "
        "validate physical units, and return a fully-reasoned answer with trace."
    ),
)
async def analysis_chat(request: AnalysisRequest) -> AnalysisResponse:
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(
        "Analysis request | session='%s' | message='%.80s...'",
        session_id, request.message,
    )

    try:
        session = analysis_session_store.get_or_create(session_id)
        _try_preload_dataset(session, request.dataset_hint)
        answer = await data_science_agent.run(session, request.message)
        analysis_session_store.save(session)

        # Build figure refs
        figure_refs = [
            FigureRef(
                figure_id=fig_id,
                retrieval_url=f"/api/v1/analysis/{session_id}/figures/{fig_id}",
            )
            for fig_id in session.figures
        ]

        # Build ReAct trace for response
        react_steps = [
            ReActStep(
                thought=step.get("thought", ""),
                action=step.get("action", ""),
                observation=step.get("observation", ""),
            )
            for step in session.react_trace
        ]

        return AnalysisResponse(
            response=answer,
            session_id=session_id,
            react_trace=react_steps,
            figures=figure_refs,
            notebook_available=session.notebook_path is not None,
            unit_validations=[u.__dict__ for u in session.unit_context],
            iterations_used=len(session.react_trace),
            model=settings.claude_model,
            status="completed",
        )

    except LLMContextOverflowError as exc:
        logger.warning("Context overflow (session=%s): %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "context_overflow",
                "message": (
                    "The conversation history is too long for the model. "
                    "Please start a new session."
                ),
            },
        )
    except LLMAuthenticationError as exc:
        logger.error("LLM auth error (session=%s): %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "llm_auth_error",
                "message": "The AI service authentication failed. Please contact support.",
            },
        )
    except LLMAPIError as exc:
        logger.error("LLM API error (session=%s): %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "llm_api_error",
                "message": "The AI service is temporarily unavailable. Please retry.",
            },
        )
    except ReActLoopError as exc:
        logger.warning("ReAct loop error (session=%s): %s", session_id, exc)
        return AnalysisResponse(
            response=(
                "The analysis could not be completed within the allowed reasoning steps. "
                f"Last thought: {exc.last_thought}"
            ),
            session_id=session_id,
            status="error",
            iterations_used=exc.iterations,
        )
    except AgentError as exc:
        logger.warning("Agent error (session=%s): %s", session_id, exc)
        return AnalysisResponse(
            response=f"Agent error: {exc}",
            session_id=session_id,
            status="error",
        )
    except Exception as exc:
        logger.error("Unhandled error (session=%s): %s", session_id, exc, exc_info=True)
        return AnalysisResponse(
            response="An unexpected error occurred. Please try again.",
            session_id=session_id,
            status="error",
        )


@router.get(
    "/{session_id}/figures/{figure_id}",
    summary="Retrieve a figure as PNG",
    response_class=Response,
)
async def get_figure(session_id: str, figure_id: str) -> Response:
    session = analysis_session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    b64 = session.figures.get(figure_id)
    if b64 is None:
        raise HTTPException(
            status_code=404,
            detail=f"Figure '{figure_id}' not found. Available: {list(session.figures.keys())}",
        )

    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        logger.error(
            "Figure '%s' in session '%s' has corrupted base64 data",
            figure_id, session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "corrupted_figure",
                "message": f"Figure '{figure_id}' data is corrupted and cannot be decoded.",
            },
        )
    return Response(content=img_bytes, media_type="image/png")


@router.get(
    "/{session_id}/notebook",
    summary="Download the session's Jupyter notebook",
)
async def get_notebook(session_id: str) -> FileResponse:
    session = analysis_session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if not session.notebook_path:
        raise HTTPException(
            status_code=404,
            detail="No notebook has been exported for this session. Call export_notebook first.",
        )
    return FileResponse(
        path=session.notebook_path,
        media_type="application/x-ipynb+json",
        filename=f"analysis_{session_id[:8]}.ipynb",
    )
