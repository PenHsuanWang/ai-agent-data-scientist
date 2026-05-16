"""FastAPI application entry point.

Registers all routers including new analysis and datasets routers.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.analysis import router as analysis_router
from app.api.v1.datasets import router as datasets_router
from app.core.config import settings


def _configure_logging() -> None:
    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def _gc_sessions_task() -> None:
    """Background task: evict stale sessions and release their CodeRunners (Gaps 5, 15).

    Runs every 5 minutes.  Calls ``shutdown_session`` on the agent service to
    free any CodeRunner resources (subprocess handles, Jupyter kernel sockets)
    before removing the session from the store.
    """
    from app.services.data_agent import data_science_agent
    from app.services.memory import analysis_session_store

    logger = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(300)  # 5-minute cadence
        try:
            expired = analysis_session_store.get_expired_ids()
            for sid in expired:
                data_science_agent.shutdown_session(sid)
                analysis_session_store.delete(sid)
            if expired:
                logger.info(
                    "GC evicted %d stale session(s): %s",
                    len(expired),
                    expired,
                )
        except Exception as exc:
            logger.warning("Session GC task error: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _configure_logging()
    logger = logging.getLogger(__name__)
    settings.ensure_directories()

    # Validate code execution backend at startup (Gap 9).
    # Exit immediately if misconfigured rather than failing silently at runtime.
    try:
        from app.infrastructure.code_runner import CodeRunnerFactory
        probe = CodeRunnerFactory.create(session_id="__startup_probe__")
        probe.shutdown()
        logger.info("Code execution backend '%s' validated OK", settings.code_execution_backend)
    except ValueError as exc:
        logger.critical(
            "Invalid CODE_EXECUTION_BACKEND=%r: %s — shutting down.",
            settings.code_execution_backend,
            exc,
        )
        sys.exit(1)
    except Exception as exc:
        # Non-fatal (e.g. Jupyter not installed but backend=subprocess): log and continue
        logger.warning("Backend probe warning: %s", exc)

    logger.info(
        "Starting Data Scientist Agent | env=%s | model=%s | backend=%s",
        settings.app_env,
        settings.claude_model,
        settings.code_execution_backend,
    )

    gc_task = asyncio.create_task(_gc_sessions_task())
    try:
        yield
    finally:
        gc_task.cancel()
        logger.info("Shutting down Data Scientist Agent.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Data Scientist AI Agent",
        description=(
            "Domain-aware AI agent for data science: ReAct reasoning, Python code execution, "
            "physical unit validation, and Jupyter notebook export. "
            "Powered by Claude + Anthropic Python SDK."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
    app.include_router(datasets_router, prefix="/api/v1/datasets", tags=["datasets"])

    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health():
        return {
            "status": "ok",
            "env": settings.app_env,
            "model": settings.claude_model,
            "code_backend": settings.code_execution_backend,
        }

    return app


app = create_app()
