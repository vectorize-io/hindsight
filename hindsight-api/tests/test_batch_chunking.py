"""Test automatic batch chunking based on character count."""
import asyncio
import pytest
from hindsight_api import MemoryEngine
import os


@pytest.mark.asyncio
async def test_large_batch_auto_chunks(memory, request_context):
    bank_id = "test_chunking_agent"
    # Create a large batch that should trigger chunking
    # Each item is ~2000 chars, so 30 items = 60k chars (exceeds 50k threshold)
    large_content = "Alice met with Bob at the coffee shop. " * 50  # ~2000 chars
    contents = [
        {"content": large_content, "context": f"conversation_{i}"}
        for i in range(30)
    ]

    # Calculate total chars
    total_chars = sum(len(item["content"]) for item in contents)
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Should trigger chunking: {total_chars > 50_000}")

    # Ingest the large batch (should auto-chunk)
    result = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    # Verify we got results back
    assert len(result) == 30, f"Expected 30 results, got {len(result)}"
    print(f"Successfully ingested {len(result)} items (auto-chunked)")


@pytest.mark.asyncio
async def test_small_batch_no_chunking(memory, request_context):
    bank_id = "test_no_chunking_agent"

    # Create a small batch that should NOT trigger chunking
    contents = [
        {"content": "Alice works at Google", "context": "conversation_1"},
        {"content": "Bob loves Python", "context": "conversation_2"}
    ]

    # Calculate total chars
    total_chars = sum(len(item["content"]) for item in contents)
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Should NOT trigger chunking: {total_chars <= 50_000}")

    # Ingest the small batch (should NOT auto-chunk)
    result = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    # Verify we got results back
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    print(f"Successfully ingested {len(result)} items (no chunking)")
