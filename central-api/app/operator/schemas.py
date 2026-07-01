"""Operator review schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.policy.rules import Sensitivity, Status


class ReviewItem(BaseModel):
    memory_id: str
    content: str
    sensitivity: Sensitivity
    status: Status
    reason: str | None = None
    provenance: dict = Field(default_factory=dict)


class ReviewResponse(BaseModel):
    count: int
    items: list[ReviewItem]


class ApproveRequest(BaseModel):
    memory_id: str
    note: str | None = None


class RejectRequest(BaseModel):
    memory_id: str
    reason: str | None = None


class OperatorActionResponse(BaseModel):
    memory_id: str
    status: Status
    trace_id: str
