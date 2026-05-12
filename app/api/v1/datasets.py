"""Datasets API — GET /api/v1/datasets + GET /api/v1/datasets/{name}/schema."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.datasets import DatasetInfo, DatasetListResponse, DatasetSchemaResponse
from app.services.knowledge_tools import inspect_dataset, list_datasets

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List all available datasets",
)
async def list_all_datasets() -> DatasetListResponse:
    raw = list_datasets()
    items = json.loads(raw)
    datasets = [DatasetInfo(**item) for item in items]
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get(
    "/{name}/schema",
    response_model=DatasetSchemaResponse,
    summary="Get dataset schema and sample rows",
)
async def get_dataset_schema(name: str) -> DatasetSchemaResponse:
    raw = inspect_dataset(name)
    if raw.startswith("Error:"):
        raise HTTPException(status_code=404, detail=raw)
    data = json.loads(raw)
    return DatasetSchemaResponse(**data)
