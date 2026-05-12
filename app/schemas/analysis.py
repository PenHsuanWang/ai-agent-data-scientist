"""Pydantic request/response models for the analysis API."""
from __future__ import annotations

import re
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
    )
    dataset_hint: str | None = Field(
        default=None,
        description="Optional: pre-load this dataset before starting",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v.strip()

    @field_validator("session_id")
    @classmethod
    def session_id_safe(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not re.fullmatch(r"[a-zA-Z0-9\-]{1,64}", v):
            raise ValueError("session_id must be alphanumeric with hyphens, max 64 chars")
        return v


class ReActStep(BaseModel):
    thought: str = Field(description="Claude's reasoning")
    action: str = Field(description="Tool called or 'Final Answer'")
    observation: str = Field(description="Tool result or final answer")


class FigureRef(BaseModel):
    figure_id: str
    retrieval_url: str


class AnalysisResponse(BaseModel):
    response: str = Field(description="Agent's final answer")
    session_id: str = Field(description="Session ID for follow-up requests")
    react_trace: list[ReActStep] = Field(default_factory=list)
    figures: list[FigureRef] = Field(default_factory=list)
    notebook_available: bool = Field(default=False)
    unit_validations: list[dict] = Field(default_factory=list)
    iterations_used: int = Field(default=0)
    model: str = Field(default="")
    status: str = Field(default="completed")
