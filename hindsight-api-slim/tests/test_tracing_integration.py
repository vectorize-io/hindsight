"""
Integration tests for OpenTelemetry tracing with memory engine operations.

Tests that parent spans are correctly created for retain, consolidation, reflect,
and mental_model_refresh operations.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
@patch("hindsight_api.engine.memory_engine.create_operation_span")
async def test_retain_creates_parent_span(mock_create_span, memory, request_context):
    """Test that retain operation creates a parent span."""
    # Setup
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_create_span.return_value = mock_span

    bank_id = f"test-retain-{datetime.now(timezone.utc).timestamp()}"

    try:
        # Execute retain (automatically creates bank if needed)
        await memory.retain_async(
            bank_id=bank_id,
            content="Test memory for tracing",
            context="Test context",
            request_context=request_context,
        )

        # Verify parent span was created
        mock_create_span.assert_called()
        call_args = mock_create_span.call_args
        assert call_args[0][0] == "retain"  # operation name
        assert call_args[0][1] == bank_id  # bank_id

        # Verify span was used as context manager
        mock_span.__enter__.assert_called()
        mock_span.__exit__.assert_called()
    finally:
        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@patch("hindsight_api.engine.memory_engine.create_operation_span")
async def test_consolidation_creates_parent_span(mock_create_span, memory, request_context):
    """Test that consolidation operation creates a parent span."""
    # Setup
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_create_span.return_value = mock_span

    bank_id = f"test-consolidation-{datetime.now(timezone.utc).timestamp()}"

    try:
        # Execute consolidation (bank will be created automatically)
        await memory.run_consolidation(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Verify parent span was created
        mock_create_span.assert_called()
        call_args = mock_create_span.call_args
        assert call_args[0][0] == "consolidation"
        assert call_args[0][1] == bank_id

        # Verify span was used as context manager
        mock_span.__enter__.assert_called()
        mock_span.__exit__.assert_called()
    finally:
        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@patch("hindsight_api.engine.memory_engine.create_operation_span")
async def test_reflect_creates_parent_span(mock_create_span, memory, request_context):
    """Test that reflect operation creates a parent span."""
    # Setup
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_create_span.return_value = mock_span

    bank_id = f"test-reflect-{datetime.now(timezone.utc).timestamp()}"

    try:
        # Add some memories first
        await memory.retain_async(
            bank_id=bank_id,
            content="Paris is the capital of France",
            context="Geography fact",
            request_context=request_context,
        )

        # Reset mock to clear retain call
        mock_create_span.reset_mock()

        # Execute reflect
        await memory.reflect_async(
            bank_id=bank_id,
            query="What is the capital of France?",
            request_context=request_context,
        )

        # Verify parent span was created
        mock_create_span.assert_called()
        call_args = mock_create_span.call_args
        assert call_args[0][0] == "reflect"
        assert call_args[0][1] == bank_id

        # Verify span was used as context manager
        mock_span.__enter__.assert_called()
        mock_span.__exit__.assert_called()
    finally:
        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@patch("hindsight_api.engine.memory_engine.create_operation_span")
async def test_retain_batch_creates_single_parent_span(mock_create_span, memory, request_context):
    """Test that batch retain creates one parent span for the entire batch."""
    # Setup
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_create_span.return_value = mock_span

    bank_id = f"test-batch-{datetime.now(timezone.utc).timestamp()}"

    try:
        # Execute batch retain with multiple items
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Memory 1", "context": "Context 1"},
                {"content": "Memory 2", "context": "Context 2"},
                {"content": "Memory 3", "context": "Context 3"},
            ],
            request_context=request_context,
        )

        # Verify parent span was created only once for the entire batch
        assert mock_create_span.call_count == 1
        call_args = mock_create_span.call_args
        assert call_args[0][0] == "retain"
        assert call_args[0][1] == bank_id
    finally:
        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@patch("hindsight_api.tracing._tracing_enabled", False)
@patch("hindsight_api.engine.memory_engine.create_operation_span")
async def test_operations_work_when_tracing_disabled(mock_create_span, memory, request_context):
    """Test that operations work correctly when tracing is disabled."""
    # Setup - create_operation_span should return a no-op context manager
    from contextlib import nullcontext

    mock_create_span.return_value = nullcontext()

    bank_id = f"test-no-trace-{datetime.now(timezone.utc).timestamp()}"

    try:
        # All operations should work without errors
        await memory.retain_async(
            bank_id=bank_id,
            content="Test memory",
            request_context=request_context,
        )

        await memory.run_consolidation(
            bank_id=bank_id,
            request_context=request_context,
        )

        await memory.reflect_async(
            bank_id=bank_id,
            query="Test query",
            request_context=request_context,
        )

        # Verify no errors occurred and spans were attempted to be created
        assert mock_create_span.call_count >= 3
    finally:
        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)
