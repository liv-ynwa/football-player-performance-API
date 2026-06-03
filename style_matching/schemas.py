from __future__ import annotations

from pydantic import BaseModel, Field


class CountResponse(BaseModel):
    count: int
    items: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    matching_db_exists: bool
    model_quality_exists: bool
    sample_mode: bool

