"""Pydantic models for the datasets API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    file_name: str
    format: str
    size_bytes: int


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo]
    total: int


class DatasetSchemaResponse(BaseModel):
    file_name: str
    format: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    numeric_stats: dict[str, Any]
    sample_rows: list[dict[str, Any]]
