"""Pydantic models for directives."""

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field


class Directive(BaseModel):
    """A directive is a hard rule injected into prompts.

    Directives are user-defined rules that guide agent behavior. Unlike mental models
    which are automatically consolidated from memories, directives are explicit
    instructions that are always included in relevant prompts.

    Examples:
    - "Always respond in formal English"
    - "Never share personal data with third parties"
    - "Prefer conservative investment recommendations"
    """

    id: UUID = Field(description="Unique identifier")
    bank_id: str = Field(description="Bank this directive belongs to")
    name: str = Field(description="Human-readable name")
    content: str = Field(description="The directive text to inject into prompts")
    priority: int = Field(default=0, description="Higher priority directives are injected first")
    is_active: bool = Field(default=True, description="Whether this directive is currently active")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When this directive was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When this directive was last updated"
    )

    class Config:
        from_attributes = True
