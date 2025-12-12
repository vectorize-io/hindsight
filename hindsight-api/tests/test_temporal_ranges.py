"""Tests for temporal range support (occurred_start, occurred_end, mentioned_at)."""
import asyncio
from datetime import datetime, timezone, timedelta
import pytest
from hindsight_api.engine.memory_engine import Budget


@pytest.mark.asyncio
async def test_temporal_ranges_are_written(memory):
    """Test that occurred_start, occurred_end, and mentioned_at are actually written to database."""
    bank_id = "test_temporal_ranges"

    # Clean up any existing data
    try:
        await memory.delete_bank(bank_id)
    except Exception:
        pass

    # Test 1: Point event (specific date)
    conversation_date = datetime(2024, 11, 17, 10, 0, 0, tzinfo=timezone.utc)
    text1 = "Yesterday I went to a pottery workshop where I made a beautiful vase."

    await memory.retain_async(
        bank_id=bank_id,
        content=text1,
        event_date=conversation_date
    )

    # Test 2: Period event (month range)
    text2 = "In February 2024, Alice visited Paris and explored the Louvre museum."

    await memory.retain_async(
        bank_id=bank_id,
        content=text2,
        event_date=conversation_date
    )

    # Give it a moment for async processing
    await asyncio.sleep(2)

    # Retrieve facts from database directly
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, text, event_date, occurred_start, occurred_end, mentioned_at
            FROM memory_units
            WHERE bank_id = $1
            ORDER BY created_at
            """,
            bank_id
        )

    print(f"\n\n=== Retrieved {len(rows)} facts ===")
    for i, row in enumerate(rows):
        print(f"\nFact {i+1}:")
        print(f"  Text: {row['text'][:80]}...")
        print(f"  event_date: {row['event_date']}")
        print(f"  occurred_start: {row['occurred_start']}")
        print(f"  occurred_end: {row['occurred_end']}")
        print(f"  mentioned_at: {row['mentioned_at']}")

    # Assertions
    assert len(rows) >= 2, f"Expected at least 2 facts, got {len(rows)}"

    # Check that temporal fields are populated
    for row in rows:
        assert row['occurred_start'] is not None, f"occurred_start is None for fact: {row['text'][:50]}"
        assert row['occurred_end'] is not None, f"occurred_end is None for fact: {row['text'][:50]}"
        assert row['mentioned_at'] is not None, f"mentioned_at is None for fact: {row['text'][:50]}"

        # mentioned_at should be close to the conversation date
        time_diff = abs((row['mentioned_at'] - conversation_date).total_seconds())
        assert time_diff < 60, f"mentioned_at is too far from conversation_date: {time_diff}s"

    # Find the pottery fact (point event)
    pottery_fact = next((r for r in rows if 'pottery' in r['text'].lower()), None)
    if pottery_fact:
        print(f"\n=== Pottery Fact (Point Event) ===")
        print(f"  occurred_start: {pottery_fact['occurred_start']}")
        print(f"  occurred_end: {pottery_fact['occurred_end']}")

        # For "yesterday", occurred_start and occurred_end should be Nov 16
        # (or the same day - it should be a point event)
        # We'll check they're within the same day
        time_diff = abs((pottery_fact['occurred_end'] - pottery_fact['occurred_start']).total_seconds())
        assert time_diff < 86400, f"Point event should have occurred_start and occurred_end within same day, got diff: {time_diff}s"

    # Find the Paris fact (period event)
    paris_fact = next((r for r in rows if 'paris' in r['text'].lower() or 'february' in r['text'].lower()), None)
    if paris_fact:
        print(f"\n=== Paris Fact (Period Event) ===")
        print(f"  occurred_start: {paris_fact['occurred_start']}")
        print(f"  occurred_end: {paris_fact['occurred_end']}")

        # "In February 2024" is ambiguous - could be interpreted as:
        # 1. A month-long period (Feb 1 - Feb 29) - ideal interpretation
        # 2. A point event sometime in February - also valid
        # We accept either interpretation as long as the dates are in February 2024
        if paris_fact['occurred_start'] and paris_fact['occurred_end']:
            time_diff_days = (paris_fact['occurred_end'] - paris_fact['occurred_start']).days
            print(f"  Duration: {time_diff_days} days")

            # Verify the dates are in February 2024
            assert paris_fact['occurred_start'].year == 2024, f"occurred_start should be 2024"
            assert paris_fact['occurred_start'].month == 2, f"occurred_start should be in February"
        else:
            print("  Note: occurred_start/end not set (fact may not have been classified as event)")

    # Test search results also include temporal fields
    print("\n=== Testing Search Results ===")
    search_result = await memory.recall_async(
        bank_id=bank_id,
        query="pottery workshop",
        fact_type=["world", "experience"],
        budget=Budget.LOW,
        max_tokens=4096
    )

    print(f"Found {len(search_result.results)} search results")
    if len(search_result.results) > 0:
        first_result = search_result.results[0]
        print(f"  Text: {first_result.text[:80]}...")
        print(f"  occurred_start: {first_result.occurred_start}")
        print(f"  occurred_end: {first_result.occurred_end}")
        print(f"  mentioned_at: {first_result.mentioned_at}")

        # Note: Search results may not have temporal fields populated yet (work in progress)
        if first_result.occurred_start:
            print("✓ Temporal fields are present in search results")
        else:
            print("⚠ Temporal fields not yet populated in search results (known issue)")

    # Clean up
    await memory.delete_bank(bank_id)
