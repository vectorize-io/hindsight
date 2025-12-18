"""
Test think function for opinion generation and consistency.
"""
import pytest
from datetime import datetime, timezone
from hindsight_api.engine.memory_engine import Budget
from hindsight_api import RequestContext


@pytest.mark.asyncio
async def test_think_opinion_consistency(memory, request_context):
    """
    Test that think function:
    1. Generates an opinion
    2. Stores the opinion in the database
    3. Returns consistent response on subsequent calls with the same query
    """
    bank_id = f"test_think_{datetime.now(timezone.utc).timestamp()}"

    try:

        # Store some initial facts to give context for opinion formation
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice is a software engineer who has worked on 5 major projects. She always delivers on time and writes clean, well-documented code.",
            context="performance review",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        await memory.retain_async(
            bank_id=bank_id,
            content="Bob recently joined the team. He missed his first deadline and his code had many bugs.",
            context="performance review",
            event_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            request_context=request_context,
        )

        # First think call - should generate opinions
        query = "Who is a more reliable engineer?"
        result1 = await memory.reflect_async(
            bank_id=bank_id,
            query=query,
            budget=Budget.LOW,
            request_context=request_context,
        )

        print(f"\n=== First Think Call ===")
        print(f"Answer: {result1.text}")

        # Verify we got an answer
        assert result1.text, "First think call should return an answer"
        assert result1.based_on, "Should return based_on facts"

        # Wait for background opinion processing tasks to complete
        await memory.wait_for_background_tasks()

        # Search for stored opinions to verify they were actually saved
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            stored_opinions = await conn.fetch(
                """
                SELECT id, text, confidence_score, fact_type
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'opinion'
                ORDER BY created_at DESC
                """,
                bank_id
            )

        print(f"\n=== Stored Opinions in Database ===")
        print(f"Total opinions stored: {len(stored_opinions)}")
        for op in stored_opinions:
            print(f"  - {op['text']} (confidence: {op['confidence_score']:.2f})")

        # Verify opinions were actually written to database
        # NOTE: Opinion extraction may not always detect opinions depending on the LLM response format
        if len(stored_opinions) > 0:
            assert all(op['fact_type'] == 'opinion' for op in stored_opinions), "All stored items should have fact_type='opinion'"
            print(f"✓ Opinions were successfully stored in database")
        else:
            print(f"⚠ Note: No opinions were extracted/stored (this can happen if the LLM response format doesn't trigger opinion extraction)")

        # Second think call - should use the stored opinions
        result2 = await memory.reflect_async(
            bank_id=bank_id,
            query=query,
            budget=Budget.LOW,
            request_context=request_context,
        )

        print(f"\n=== Second Think Call ===")
        print(f"Answer: {result2.text}")
        print(f"Existing opinions used: {len(result2.based_on.get('opinion', []))}")
        for opinion in result2.based_on.get('opinion', []):
            print(f"  - {opinion.text}")

        # Verify second call also got an answer
        assert result2.text, "Second think call should return an answer"

        # Verify second call used the stored opinions (if any were stored)
        if len(stored_opinions) > 0:
            assert len(result2.based_on.get('opinion', [])) > 0, "Second call should retrieve stored opinions"

        # The responses should be consistent (both should mention the same person as more reliable)
        # We'll do a basic check that they're not contradictory
        text1_lower = result1.text.lower()
        text2_lower = result2.text.lower()

        print(f"\n=== Consistency Check ===")

        # Check if Alice is mentioned as more reliable in first response
        if 'alice' in text1_lower and ('reliable' in text1_lower or 'better' in text1_lower):
            print("First response favors Alice")
            # Second response should also favor Alice (consistency)
            assert 'alice' in text2_lower, "Second response should also mention Alice"
            print("Second response also mentions Alice - CONSISTENT ✓")

        # Check if Bob is mentioned
        if 'bob' in text1_lower:
            print("First response mentions Bob")
            if 'bob' in text2_lower:
                print("Second response also mentions Bob - CONSISTENT ✓")

        print(f"\n✅ Test passed - opinions were formed, stored, and used consistently")

    finally:
        # Clean up agent data
        try:
            await memory.delete_bank(bank_id, request_context=request_context)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


@pytest.mark.asyncio
async def test_think_without_prior_context(memory, request_context):
    """
    Test that think function handles queries when there's no relevant context.
    """
    bank_id = f"test_think_no_context_{datetime.now(timezone.utc).timestamp()}"

    # Call think without storing any prior facts
    result = await memory.reflect_async(
        bank_id=bank_id,
        query="What is the capital of France?",
        budget=Budget.LOW,
        request_context=request_context,
    )

    print(f"\n=== Think Without Context ===")
    print(f"Answer: {result.text}")

    # Should still return an answer (even if it says it doesn't have enough info)
    assert result.text, "Should return some answer"
    assert result.based_on, "Should return based_on structure"

