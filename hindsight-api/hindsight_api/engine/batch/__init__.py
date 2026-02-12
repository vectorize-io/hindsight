"""
Groq Batch API support for async LLM processing.

This module provides batch processing capabilities that submit LLM requests
via the Groq Batch API for a 50% cost reduction on retain and consolidation
operations.

Usage:
    Enable with HINDSIGHT_API_BATCH_MODE=groq

Components:
    - models: Pydantic models for batch requests/responses
    - client: Groq Batch API client (file upload, batch create, poll, download)
    - queue: Async batch request queue with Future-based result delivery
    - processor: Routes completed batch results back to callers
"""

from .client import GroqBatchClient
from .models import BatchJob, BatchRequest, BatchRequestBody, BatchResponse, BatchStatus
from .queue import BatchQueue

__all__ = [
    "BatchJob",
    "BatchQueue",
    "BatchRequest",
    "BatchRequestBody",
    "BatchResponse",
    "BatchStatus",
    "GroqBatchClient",
]
