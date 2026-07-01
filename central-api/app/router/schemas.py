"""Router decision schemas."""
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


RequestType = Literal["chat", "reason", "tool", "retrieval", "voice", "other"]
DecisionStatus = Literal["selected", "fallback", "failed", "no_selection"]


class RouterDecision(BaseModel):
    """A single router decision record."""

    id: str
    timestamp: datetime
    tenant_id: str
    actor_id: str
    request_type: RequestType
    selected_model: str
    candidate_models: List[str] = Field(default_factory=list)
    selection_reason: Optional[str] = None
    latency_ms: Optional[int] = None
    estimated_cost: Optional[float] = None
    fallback_chain: List[str] = Field(default_factory=list)
    status: DecisionStatus
    trace_id: Optional[str] = None


class RouterDecisionsResponse(BaseModel):
    """Response containing router decisions."""

    decisions: List[RouterDecision] = Field(default_factory=list)
    count: int = 0
