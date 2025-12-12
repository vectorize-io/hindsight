"""
Test that facts from the same conversation maintain temporal ordering.

This ensures that when multiple facts are extracted from a long conversation,
their relative order is preserved via time offsets, allowing retrieval to
distinguish between things said earlier vs later.
"""
import pytest
from datetime import datetime, timezone
from hindsight_api import MemoryEngine
from hindsight_api.engine.memory_engine import Budget
import os


@pytest.mark.asyncio
async def test_fact_ordering_within_conversation(memory):
    bank_id = "test_ordering_agent"

    # Get/create agent (auto-creates with defaults)
    await memory.get_bank_profile(bank_id)

    # Update disposition to match Marcus
    await memory.update_bank_disposition(bank_id, {
        "skepticism": 3,
        "literalism": 3,
        "empathy": 3
    })

    # A conversation where Marcus changes his position
    conversation = """
Marcus: I think the Rams will win 27-24. Their defense is really strong.
Jamie: I disagree, I think Niners will win.
Marcus: Actually, after thinking about it more, I'm changing my prediction to Rams by 3 points only.
Jamie: That's more reasonable.
Marcus: Yeah, I realized I was being too optimistic about their defense.
"""

    base_event_date = datetime(2024, 11, 14, 10, 0, 0, tzinfo=timezone.utc)

    # Store the conversation
    await memory.retain_async(
        bank_id=bank_id,
        content=conversation,
        context="podcast discussion about NFL game",
        event_date=base_event_date,
        document_id="test_conv_1"
    )

    # Search for all facts about Marcus's predictions
    results = await memory.recall_async(
        bank_id=bank_id,
        query="Marcus prediction Rams",
        fact_type=['opinion', 'experience', 'world'],
        budget=Budget.LOW,
        max_tokens=8192
    )

    print(f"\n=== Retrieved {len(results.results)} facts ===")
    for i, result in enumerate(results.results):
        print(f"{i+1}. [{result.mentioned_at}] {result.text[:100]}")

    # Get all opinion facts (Marcus's predictions/statements)
    agent_facts = [r for r in results.results if r.fact_type == 'opinion']

    print(f"\n=== Agent facts (Marcus's statements) ===")
    for i, fact in enumerate(agent_facts):
        print(f"{i+1}. [{fact.mentioned_at}] {fact.text}")

    # Check that agent facts have different timestamps
    if len(agent_facts) >= 2:
        timestamps = [datetime.fromisoformat(f.mentioned_at.replace('Z', '+00:00')) for f in agent_facts]

        # Verify timestamps are different (have time offsets)
        unique_timestamps = set(timestamps)
        assert len(unique_timestamps) == len(timestamps), \
            f"Expected unique timestamps for each fact, but got duplicates: {timestamps}"

        # Verify timestamps are in order (ascending)
        for i in range(len(timestamps) - 1):
            assert timestamps[i] < timestamps[i + 1], \
                f"Facts should be ordered by time. Fact {i} ({timestamps[i]}) >= Fact {i+1} ({timestamps[i+1]})"

        # Verify reasonable time spacing (should be ~10 seconds apart)
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
        print(f"\n=== Time differences between facts: {time_diffs} seconds ===")

        # Each fact should be 10+ seconds apart (allowing for some flexibility)
        for diff in time_diffs:
            assert diff >= 5, f"Expected at least 5 seconds between facts, got {diff}"

        print(f"\n✅ All {len(agent_facts)} agent facts have properly ordered timestamps")

    # Verify that retrieval returns facts in chronological order
    # The first prediction should come before the changed prediction
    agent_texts = [f.text.lower() for f in agent_facts]

    # Look for evidence of the sequence
    has_first_prediction = any('27' in text and '24' in text for text in agent_texts)
    has_changed_prediction = any('chang' in text or 'by 3' in text or 'realized' in text for text in agent_texts)

    if has_first_prediction and has_changed_prediction:
        # Find indices
        first_idx = next(i for i, text in enumerate(agent_texts) if '27' in text and '24' in text)
        changed_idx = next(i for i, text in enumerate(agent_texts) if 'chang' in text or 'by 3' in text or 'realized' in text)

        print(f"\nFirst prediction at index {first_idx}: {agent_facts[first_idx].text[:100]}")
        print(f"Changed prediction at index {changed_idx}: {agent_facts[changed_idx].text[:100]}")

        # The original prediction should come before the changed one
        assert timestamps[first_idx] < timestamps[changed_idx], \
            "Original prediction should have earlier timestamp than changed prediction"

        print(f"\n✅ Temporal ordering preserved: First prediction came before changed prediction")

    # Cleanup
    await memory.delete_bank(bank_id)

    print(f"\n✅ Test passed: Fact ordering within conversation is preserved")


@pytest.mark.asyncio
async def test_multiple_documents_ordering(memory):

    bank_id = "test_multi_doc_agent"

    await memory.get_bank_profile(bank_id)  # Auto-creates with defaults

    # Two separate conversations with same base time
    base_time = datetime(2024, 11, 14, 10, 0, 0, tzinfo=timezone.utc)

    conv1 = """
Alice: I prefer React for this project.
Bob: Why React?
Alice: It has better tooling and I'm more familiar with it.
"""

    conv2 = """
Alice: Actually, I'm thinking Vue might be better.
Bob: What changed your mind?
Alice: I reconsidered the team's experience level.
"""

    # Store both conversations with batch
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {"content": conv1, "context": "project discussion 1", "event_date": base_time},
            {"content": conv2, "context": "project discussion 2", "event_date": base_time}
        ]
    )

    # Search for Alice's preferences
    results = await memory.recall_async(
        bank_id=bank_id,
        query="Alice preference React Vue",
        fact_type=['opinion', 'experience'],
        budget=Budget.LOW,
        max_tokens=8192
    )

    print(f"\n=== Retrieved {len(results.results)} agent facts ===")
    agent_facts = [r for r in results.results if r.fact_type in ('opinion', 'experience')]

    for i, fact in enumerate(agent_facts):
        print(f"{i+1}. [{fact.mentioned_at}] {fact.text[:80]}")

    # Each conversation's facts should have different timestamps
    if len(agent_facts) >= 2:
        timestamps = [datetime.fromisoformat(f.mentioned_at.replace('Z', '+00:00')) for f in agent_facts]
        unique_timestamps = set(timestamps)

        assert len(unique_timestamps) >= 2, \
            f"Expected multiple unique timestamps across conversations, got: {len(unique_timestamps)}"

        print(f"\n✅ Facts from {len(agent_facts)} statements have {len(unique_timestamps)} unique timestamps")

    # Cleanup
    await memory.delete_bank(bank_id)

    print(f"\n✅ Test passed: Multiple documents maintain separate ordering")
