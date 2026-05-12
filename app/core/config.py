"""Core configuration — pydantic-settings backed by .env.

SecretStr prevents ANTHROPIC_API_KEY from appearing in logs or repr().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────────────── #
    anthropic_api_key: SecretStr = Field(..., alias="ANTHROPIC_API_KEY")
    anthropic_base_url: Optional[str] = Field(None, alias="ANTHROPIC_BASE_URL")

    # ── Model ──────────────────────────────────────────────────────── #
    claude_model: str = Field("claude-sonnet-4-6", alias="CLAUDE_MODEL")
    max_tokens: int = Field(8192, alias="MAX_TOKENS")
    max_retries: int = Field(2, alias="MAX_RETRIES")

    # ── Application ────────────────────────────────────────────────── #
    app_env: str = Field("development", alias="APP_ENV")
    debug: bool = Field(False, alias="DEBUG")

    # ── Data directories ───────────────────────────────────────────── #
    data_dir: Path = Field(Path("data"), alias="DATA_DIR")
    domain_docs_dir: Path = Field(Path("data/domain_docs"), alias="DOMAIN_DOCS_DIR")
    datasets_dir: Path = Field(Path("data/datasets"), alias="DATASETS_DIR")

    # ── Code execution ─────────────────────────────────────────────── #
    # "subprocess" | "jupyter" | "anthropic"
    code_execution_backend: str = Field("subprocess", alias="CODE_EXECUTION_BACKEND")
    code_execution_timeout: int = Field(30, alias="CODE_EXECUTION_TIMEOUT")
    enable_jupyter_bridge: bool = Field(False, alias="ENABLE_JUPYTER_BRIDGE")

    # ── ReAct loop ─────────────────────────────────────────────────── #
    max_react_iterations: int = Field(20, alias="MAX_REACT_ITERATIONS")

    # ── Output directories ─────────────────────────────────────────── #
    figures_dir: Path = Field(Path("outputs/figures"), alias="FIGURES_DIR")
    notebooks_dir: Path = Field(Path("outputs/notebooks"), alias="NOTEBOOKS_DIR")

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for d in (
            self.data_dir,
            self.domain_docs_dir,
            self.datasets_dir,
            self.figures_dir,
            self.notebooks_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
