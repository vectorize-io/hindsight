"""
Batch result processor.

Routes completed batch results back through the same parsing path
as synchronous LLM responses, ensuring parity between batch and sync modes.
"""

import json
import logging
from typing import Any

from .models import BatchResponse

logger = logging.getLogger(__name__)


def parse_batch_response_content(
    response: BatchResponse,
    response_format: Any | None = None,
    skip_validation: bool = False,
) -> Any:
    """Parse a batch response using the same logic as synchronous LLM calls.

    This ensures batch results are processed identically to sync responses,
    including JSON parsing and Pydantic validation.

    Args:
        response: BatchResponse from the completed batch
        response_format: Optional Pydantic model for structured output
        skip_validation: Return raw JSON without Pydantic validation

    Returns:
        Parsed response matching what the sync call() would return

    Raises:
        RuntimeError: If the response indicates an error
        json.JSONDecodeError: If JSON parsing fails
        ValidationError: If Pydantic validation fails
    """
    if not response.is_success():
        error_info = response.error or {"message": "Unknown batch error"}
        raise RuntimeError(
            f"Batch request {response.custom_id} failed: status={response.response.status_code}, error={error_info}"
        )

    content = response.get_content()
    if content is None:
        raise RuntimeError(f"Batch request {response.custom_id} returned no content")

    if response_format is None:
        # Plain text response
        return content

    # Parse JSON response (same as OpenAICompatibleLLM.call)
    json_data = json.loads(content)

    if skip_validation:
        return json_data

    return response_format.model_validate(json_data)


def extract_token_usage(response: BatchResponse) -> tuple[int, int, int]:
    """Extract token usage from a batch response.

    Args:
        response: BatchResponse to extract usage from

    Returns:
        Tuple of (input_tokens, output_tokens, total_tokens)
    """
    if not response.is_success():
        return 0, 0, 0

    usage = response.response.body.usage
    return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
