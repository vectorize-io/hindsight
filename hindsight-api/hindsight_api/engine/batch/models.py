"""
Pydantic models for Groq Batch API requests and responses.

These models match the Groq Batch API specification:
https://console.groq.com/docs/batch
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BatchStatus(str, Enum):
    """Status values for a Groq batch job."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this status is terminal (no further transitions)."""
        return self in (
            BatchStatus.COMPLETED,
            BatchStatus.EXPIRED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
        )

    def is_success(self) -> bool:
        """Check if this status indicates successful completion."""
        return self == BatchStatus.COMPLETED


class BatchRequestBody(BaseModel):
    """Body of a single request within a batch JSONL file."""

    model: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    max_completion_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None


class BatchRequest(BaseModel):
    """A single line in the batch JSONL input file.

    Each request maps to one chat completion call.
    """

    custom_id: str = Field(description="Unique ID for matching results back to requests")
    method: str = Field(default="POST", description="HTTP method (always POST)")
    url: str = Field(default="/v1/chat/completions", description="API endpoint path")
    body: BatchRequestBody


class BatchResponseChoice(BaseModel):
    """A choice in the batch response."""

    index: int = 0
    message: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class BatchResponseUsage(BaseModel):
    """Token usage in the batch response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BatchResponseBody(BaseModel):
    """Body of a successful batch response."""

    id: str = ""
    object: str = "chat.completion"
    model: str = ""
    choices: list[BatchResponseChoice] = Field(default_factory=list)
    usage: BatchResponseUsage = Field(default_factory=BatchResponseUsage)


class BatchResponseHTTP(BaseModel):
    """HTTP-level response for a batch request."""

    status_code: int = 200
    body: BatchResponseBody = Field(default_factory=BatchResponseBody)


class BatchResponse(BaseModel):
    """A single line in the batch JSONL output file.

    Contains the result for one request, matched via custom_id.
    """

    custom_id: str = Field(description="Matches the custom_id from the request")
    id: str = Field(default="", description="Batch request ID")
    response: BatchResponseHTTP = Field(default_factory=BatchResponseHTTP)
    error: dict[str, Any] | None = Field(default=None, description="Error info if request failed")

    def is_success(self) -> bool:
        """Check if this individual response succeeded."""
        return self.error is None and self.response.status_code == 200

    def get_content(self) -> str | None:
        """Extract the message content from the response."""
        if not self.is_success():
            return None
        choices = self.response.body.choices
        if not choices:
            return None
        return choices[0].message.get("content")


class BatchRequestCounts(BaseModel):
    """Counts of requests in various states within a batch."""

    total: int = 0
    completed: int = 0
    failed: int = 0


class BatchJob(BaseModel):
    """Represents a Groq batch job.

    This model tracks the state of a submitted batch,
    including its status and output file references.
    """

    id: str = Field(description="Batch job ID")
    object: str = "batch"
    endpoint: str = "/v1/chat/completions"
    input_file_id: str = Field(description="ID of the uploaded JSONL input file")
    completion_window: str = "24h"
    status: BatchStatus = BatchStatus.VALIDATING
    output_file_id: str | None = Field(default=None, description="ID of the output file (set when completed)")
    error_file_id: str | None = Field(default=None, description="ID of the error file (set on failures)")
    request_counts: BatchRequestCounts = Field(default_factory=BatchRequestCounts)
    created_at: int = 0
    completed_at: int | None = None
    expired_at: int | None = None
    failed_at: int | None = None
    cancelled_at: int | None = None


class FileUploadResponse(BaseModel):
    """Response from uploading a file to the Groq Files API."""

    id: str = Field(description="File ID for use in batch creation")
    object: str = "file"
    bytes: int = 0
    created_at: int = 0
    filename: str = ""
    purpose: str = "batch"
