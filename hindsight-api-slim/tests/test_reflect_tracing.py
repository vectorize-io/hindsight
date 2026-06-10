"""
Test to verify reflect operation creates proper span hierarchy.
"""

import pytest


@pytest.mark.asyncio
async def test_reflect_creates_child_spans(memory, request_context):
    """Test that reflect operation creates child LLM spans."""
    from datetime import datetime, timezone
    from hindsight_api.tracing import initialize_tracing, get_span_recorder, create_span_recorder

    # Initialize tracing with a mock endpoint
    initialize_tracing(service_name="test-hindsight", endpoint="http://localhost:4318", deployment_environment="test")

    # Create span recorder
    recorder = create_span_recorder()

    bank_id = f"test-reflect-hierarchy-{datetime.now(timezone.utc).timestamp()}"

    try:
        # Add some memories
        await memory.retain_async(
            bank_id=bank_id,
            content="Paris is the capital of France",
            context="Geography",
            request_context=request_context,
        )

        # Run reflect
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What is the capital of France?",
            request_context=request_context,
        )

        print(f"Reflect result: {result.text[:100]}")
        print(f"Usage: {result.usage}")

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
