"""FastAPI application entry point.

Registers all routers including new analysis and datasets routers.
"""
from __future__ import annotations

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _configure_logging()
    logger = logging.getLogger(__name__)
    settings.ensure_directories()
    logger.info(
        "Starting Data Scientist Agent | env=%s | model=%s | backend=%s",
        settings.app_env,
        settings.claude_model,
        settings.code_execution_backend,
    )
    yield
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
        allow_origins=["*"],
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
