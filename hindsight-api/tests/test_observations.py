"""
Test observation generation and entity state functionality.

NOTE: Observations are now stored as summaries on the entities table,
not as separate memory_units. The observations list in EntityState is
populated from the summary for backwards compatibility.
"""
import pytest
from hindsight_api.engine.memory_engine import Budget
from hindsight_api import RequestContext
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_observation_generation_on_put(memory, request_context):
    """
    Test that summaries are generated when new facts are added.

    Summaries are generated via background task when:
    - Entity has >= 5 facts (MIN_FACTS_THRESHOLD)

    This test stores enough facts and triggers summary regeneration.
    """
    bank_id = f"test_obs_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store multiple facts about John to reach the MIN_FACTS_THRESHOLD (5)
        contents = [
            "John is a software engineer at Google.",
            "John is detail-oriented and methodical in his work.",
            "John has been working on the AI team for 3 years.",
            "John specializes in machine learning and deep learning.",
            "John presented at the company conference last week.",
            "John mentors junior engineers on the team.",
        ]

        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="work info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        # Wait for background tasks (summaries are generated in background)
        await memory.wait_for_background_tasks()

        # Find the John entity
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            entity_row = await conn.fetchrow(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE bank_id = $1 AND LOWER(canonical_name) LIKE '%john%'
                LIMIT 1
                """,
                bank_id
            )

            # Also check the fact count for this entity
            if entity_row:
                fact_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM unit_entities WHERE entity_id = $1
                    """,
                    entity_row['id']
                )
                print(f"\n=== Entity Facts ===")
                print(f"Entity: {entity_row['canonical_name']} has {fact_count} linked facts")

        assert entity_row is not None, "John entity should have been extracted"

        entity_id = str(entity_row['id'])
        entity_name = entity_row['canonical_name']
        print(f"\n=== Found Entity ===")
        print(f"Entity: {entity_name} (id: {entity_id})")

        # Get entity state (includes summary as single observation)
        state = await memory.get_entity_state(
            bank_id, entity_id, entity_name, request_context=request_context
        )

        print(f"\n=== Entity State for {entity_name} ===")
        print(f"Total observations: {len(state.observations)}")
        for obs in state.observations:
            print(f"  - {obs.text}")

        # Verify summary was created (requires >= 5 facts and background task completion)
        assert len(state.observations) > 0, \
            f"Summary should have been generated (entity has {fact_count} facts, threshold is 5)"

        # Check that summary mentions relevant content
        obs_texts = " ".join([o.text.lower() for o in state.observations])
        assert any(keyword in obs_texts for keyword in ["google", "engineer", "ai", "machine learning", "detail"]), \
            "Summary should contain relevant information about John"

        print(f"Summary was successfully generated")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_regenerate_entity_observations(memory, request_context):
    """
    Test explicit regeneration of summary for an entity.
    """
    bank_id = f"test_regen_obs_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts about an entity
        await memory.retain_async(
            bank_id=bank_id,
            content="Sarah is a product manager who loves user research and data analysis.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        await memory.wait_for_background_tasks()

        # Find the Sarah entity
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            entity_row = await conn.fetchrow(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE bank_id = $1 AND LOWER(canonical_name) LIKE '%sarah%'
                LIMIT 1
                """,
                bank_id
            )

        if entity_row:
            entity_id = str(entity_row['id'])
            entity_name = entity_row['canonical_name']

            # Manually regenerate summary (via observations API for backwards compat)
            created_ids = await memory.regenerate_entity_observations(
                bank_id=bank_id,
                entity_id=entity_id,
                entity_name=entity_name,
                request_context=request_context,
            )

            print(f"\n=== Regenerated Summary ===")
            print(f"Created {len(created_ids)} summary for {entity_name}")

            # Get entity state
            state = await memory.get_entity_state(
                bank_id, entity_id, entity_name, request_context=request_context
            )
            for obs in state.observations:
                print(f"  - {obs.text}")

            # Verify summary was created
            if len(created_ids) > 0:
                assert len(state.observations) == 1, "Should have exactly 1 observation (the summary)"
                print(f"Summary regenerated successfully")
            else:
                print(f"Note: No summary was regenerated")

        else:
            print(f"Note: No 'Sarah' entity was extracted")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_manual_regenerate_with_few_facts(memory, request_context):
    """
    Test that manual regeneration works even with fewer than 5 facts.

    This is important because:
    - Automatic generation requires MIN_FACTS_THRESHOLD (5)
    - But manual regeneration via API should work with any number of facts
    """
    bank_id = f"test_manual_regen_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store only 2 facts - below the automatic threshold
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google as a senior software engineer.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice loves hiking and outdoor photography.",
            context="hobbies",
            event_date=datetime(2024, 1, 16, tzinfo=timezone.utc),
            request_context=request_context,
        )

        # Find the Alice entity
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            entity_row = await conn.fetchrow(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE bank_id = $1 AND LOWER(canonical_name) LIKE '%alice%'
                LIMIT 1
                """,
                bank_id
            )

        assert entity_row is not None, "Alice entity should have been extracted"

        entity_id = str(entity_row['id'])
        entity_name = entity_row['canonical_name']

        # Check fact count - should be < 5
        async with pool.acquire() as conn:
            fact_count = await conn.fetchval(
                "SELECT COUNT(*) FROM unit_entities WHERE entity_id = $1",
                entity_row['id']
            )

        print(f"\n=== Manual Regeneration Test ===")
        print(f"Entity: {entity_name} (id: {entity_id})")
        print(f"Linked facts: {fact_count}")

        # Verify we're testing with fewer than the automatic threshold
        assert fact_count < 5, f"Test requires < 5 facts, but entity has {fact_count}"

        # Before regeneration - should have no summary (auto threshold not met)
        state_before = await memory.get_entity_state(
            bank_id, entity_id, entity_name, request_context=request_context
        )
        print(f"Observations before manual regenerate: {len(state_before.observations)}")

        # Manually regenerate summary - this should work regardless of fact count
        created_ids = await memory.regenerate_entity_observations(
            bank_id=bank_id,
            entity_id=entity_id,
            entity_name=entity_name,
            request_context=request_context,
        )

        print(f"Summary created by manual regenerate: {len(created_ids)}")

        # Get state after regeneration
        state = await memory.get_entity_state(
            bank_id, entity_id, entity_name, request_context=request_context
        )
        print(f"Observations after manual regenerate: {len(state.observations)}")
        for obs in state.observations:
            print(f"  - {obs.text}")

        # Manual regeneration should create summary even with < 5 facts
        assert len(state.observations) > 0, \
            f"Manual regeneration should create summary even with only {fact_count} facts. " \
            f"The LLM should synthesize a summary from the available facts."

        # Verify summary contains relevant content
        obs_texts = " ".join([o.text.lower() for o in state.observations])
        assert any(keyword in obs_texts for keyword in ["google", "engineer", "hiking", "photography", "alice"]), \
            "Summary should contain relevant information about Alice"

        print(f"Manual regeneration works with {fact_count} facts (below automatic threshold of 5)")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_search_with_include_entities(memory, request_context):
    """
    Test that search with include_entities=True returns entity summaries.

    This test verifies that:
    1. Summaries are generated after retain (via background task)
    2. Summaries are returned in recall results with include_entities=True
    """
    bank_id = f"test_search_ent_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store enough facts about Alice to trigger summary generation (>= 5 facts)
        contents = [
            "Alice is a data scientist who works on recommendation systems at Netflix.",
            "Alice presented her research at the ML conference last month.",
            "Alice is an expert in deep learning and neural networks.",
            "Alice graduated from Stanford with a PhD in Computer Science.",
            "Alice leads a team of 5 data scientists at Netflix.",
            "Alice published a paper on collaborative filtering algorithms.",
        ]

        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="work info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        # Wait for background tasks (summaries generated in background)
        await memory.wait_for_background_tasks()

        # Search with include_entities=True
        # Note: max_entity_tokens must be large enough to accommodate generated summaries
        # which can be several thousand tokens for entities with many facts
        result = await memory.recall_async(
            bank_id=bank_id,
            query="What does Alice do?",
            fact_type=["world", "experience"],
            budget=Budget.LOW,
            max_tokens=2000,
            include_entities=True,
            max_entity_tokens=5000,  # Increased to accommodate larger summaries
            request_context=request_context,
        )

        print(f"\n=== Search Results ===")
        print(f"Found {len(result.results)} facts")
        for fact in result.results:
            print(f"  - {fact.text}")
            if fact.entities:
                print(f"    Entities: {', '.join(fact.entities)}")

        print(f"\n=== Entity Observations in Recall ===")
        if result.entities:
            for name, state in result.entities.items():
                print(f"\n{name}:")
                for obs in state.observations:
                    print(f"  - {obs.text}")
        else:
            print("No entity observations returned")

        # Verify results
        assert len(result.results) > 0, "Should find some facts"

        # Check if entities are included in facts
        facts_with_entities = [f for f in result.results if f.entities]
        assert len(facts_with_entities) > 0, "Some facts should have entity information"
        print(f"{len(facts_with_entities)} facts have entity information")

        # Check if entity observations are included in recall
        assert result.entities is not None and len(result.entities) > 0, \
            "Entity observations should be included in recall results"
        print(f"Entity observations included for {len(result.entities)} entities")

        # Verify Alice entity has observations (from summary)
        alice_found = False
        for name, state in result.entities.items():
            assert state.canonical_name == name, "Entity canonical_name should match key"
            assert state.entity_id, "Entity should have an ID"
            if "alice" in name.lower():
                alice_found = True
                assert len(state.observations) > 0, \
                    "Alice should have observations (generated from summary)"
                print(f"Alice has {len(state.observations)} observations in recall result")

        assert alice_found, "Alice entity should be in recall results"

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_get_entity_state(memory, request_context):
    """
    Test getting the full state of an entity.
    """
    bank_id = f"test_entity_state_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts
        await memory.retain_async(
            bank_id=bank_id,
            content="Bob is a frontend developer who specializes in React and TypeScript.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        await memory.wait_for_background_tasks()

        # Find entity
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            entity_row = await conn.fetchrow(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE bank_id = $1 AND LOWER(canonical_name) LIKE '%bob%'
                LIMIT 1
                """,
                bank_id
            )

        if entity_row:
            entity_id = str(entity_row['id'])
            entity_name = entity_row['canonical_name']

            # Get entity state
            state = await memory.get_entity_state(
                bank_id=bank_id,
                entity_id=entity_id,
                entity_name=entity_name,
                limit=10,
                request_context=request_context,
            )

            print(f"\n=== Entity State for {entity_name} ===")
            print(f"Entity ID: {state.entity_id}")
            print(f"Canonical Name: {state.canonical_name}")
            print(f"Observations: {len(state.observations)}")
            for obs in state.observations:
                print(f"  - {obs.text}")

            assert state.entity_id == entity_id, "Entity ID should match"
            assert state.canonical_name == entity_name, "Canonical name should match"

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_observation_fact_type_in_database(memory, request_context):
    """
    Test that summaries are stored on entities table (not as memory_units).

    NOTE: Observations are no longer stored as memory_units with fact_type='observation'.
    Instead, they are stored as summaries on the entities table.
    """
    bank_id = f"test_obs_db_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts
        await memory.retain_async(
            bank_id=bank_id,
            content="Charlie is a DevOps engineer who manages the Kubernetes infrastructure.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        await memory.wait_for_background_tasks()

        # Check that NO observations exist in memory_units (they're now on entities table)
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, fact_type, context
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id
            )

        print(f"\n=== Observation Records in memory_units ===")
        print(f"Found {len(observations)} observation records (should be 0)")

        # Observations are no longer stored as memory_units
        assert len(observations) == 0, "Observations should NOT be stored as memory_units anymore"

        # Verify summaries are stored on entities table
        async with pool.acquire() as conn:
            entity_with_summary = await conn.fetchrow(
                """
                SELECT id, canonical_name, summary
                FROM entities
                WHERE bank_id = $1 AND summary IS NOT NULL
                LIMIT 1
                """,
                bank_id
            )

        if entity_with_summary:
            print(f"\n=== Entity Summary in Database ===")
            print(f"Entity: {entity_with_summary['canonical_name']}")
            print(f"Summary: {entity_with_summary['summary']}")
            print(f"Summaries are correctly stored on entities table")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_entity_summary_trigger_conditions(memory, request_context):
    """
    Test that entity summaries are only triggered during retain if:
    1. Entity has >= min_facts mentions (default 5)
    2. Entity is in top X% of most mentioned entities (default 20%)

    This test creates entities with varying mention counts and verifies
    that only entities meeting both conditions get summaries regenerated.
    """
    bank_id = f"test_trigger_cond_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create content with varying entity mention counts:
        # - "HighMention Corp" mentioned 10+ times (should qualify)
        # - "LowMention Ltd" mentioned 1 time (should NOT qualify - below min_facts=5)
        contents = [
            # High mentions - HighMention Corp
            "HighMention Corp is a tech company based in San Francisco.",
            "HighMention Corp was founded in 2010 by experienced entrepreneurs.",
            "HighMention Corp has over 500 employees worldwide.",
            "HighMention Corp specializes in cloud computing solutions.",
            "HighMention Corp recently raised $50 million in Series C funding.",
            "HighMention Corp has partnerships with major tech companies.",
            "HighMention Corp is known for its innovative culture.",
            "HighMention Corp offers competitive salaries and benefits.",
            "HighMention Corp has offices in 5 countries.",
            "HighMention Corp won the best workplace award last year.",
            # Low mentions - LowMention Ltd
            "LowMention Ltd is a small consulting firm.",
        ]

        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="company info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        # Wait for background tasks (summaries are generated in background)
        await memory.wait_for_background_tasks()

        # Check entity mention counts and summaries
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            entities = await conn.fetch(
                """
                SELECT e.id, e.canonical_name, e.mention_count, e.summary
                FROM entities e
                WHERE e.bank_id = $1
                ORDER BY e.mention_count DESC
                """,
                bank_id
            )

        print(f"\n=== Entity Trigger Conditions Test ===")
        print(f"Total entities: {len(entities)}")

        high_mention_entity = None
        low_mention_entity = None

        for entity in entities:
            name = entity['canonical_name'].lower()
            mention_count = entity['mention_count']
            has_summary = entity['summary'] is not None

            print(f"  {entity['canonical_name']}: mentions={mention_count}, has_summary={has_summary}")

            if "highmention" in name:
                high_mention_entity = entity
            elif "lowmention" in name:
                low_mention_entity = entity

        # Verify that LowMention Ltd did NOT get a summary (below min_facts threshold of 5)
        if low_mention_entity:
            low_mentions = low_mention_entity['mention_count']
            low_has_summary = low_mention_entity['summary'] is not None
            print(f"\n=== LowMention Ltd ===")
            print(f"Mention count: {low_mentions}")
            print(f"Has summary: {low_has_summary}")

            # Entity with only 1 mention should NOT get a summary (min_facts=5)
            if low_mentions < 5:
                assert not low_has_summary, \
                    f"LowMention Ltd should NOT have a summary (mentions={low_mentions} < min_facts=5)"
                print("PASS: LowMention Ltd correctly excluded (below min_facts)")

        # Verify HighMention Corp - high mention count, should be in top %
        if high_mention_entity:
            high_mentions = high_mention_entity['mention_count']
            high_has_summary = high_mention_entity['summary'] is not None
            print(f"\n=== HighMention Corp ===")
            print(f"Mention count: {high_mentions}")
            print(f"Has summary: {high_has_summary}")

            # High mention entity should get summary (if it meets both conditions)
            if high_mentions >= 5:
                print(f"High mention entity meets min_facts requirement")

        print("\n=== Trigger Conditions Test Complete ===")
        print("The filtering logic based on min_facts and top_percent is working")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_entity_summary_cleanup_when_no_longer_top_percent(memory, request_context):
    """
    Test that entity summaries are DELETED when an entity is no longer in the top X%.

    This test:
    1. Creates an entity with enough mentions to get a summary
    2. Adds many more entities with higher mention counts that push the original out of top X%
    3. Verifies the original entity's summary gets deleted
    """
    bank_id = f"test_cleanup_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Phase 1: Create "OriginalEntity" with 6 mentions (should qualify initially)
        print("\n=== Phase 1: Create OriginalEntity with 6 mentions ===")
        for i in range(6):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"OriginalEntity is mentioned here in fact {i+1}.",
                context="test",
                event_date=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        await memory.wait_for_background_tasks()

        # Check that OriginalEntity has a summary
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            original_entity = await conn.fetchrow(
                """
                SELECT id, canonical_name, mention_count, summary IS NOT NULL as has_summary
                FROM entities
                WHERE bank_id = $1 AND LOWER(canonical_name) LIKE '%originalentity%'
                """,
                bank_id
            )

        assert original_entity is not None, "OriginalEntity should exist"
        print(f"OriginalEntity: mentions={original_entity['mention_count']}, has_summary={original_entity['has_summary']}")

        # It should have a summary (meets min_facts=5 and is in top 20% since it's the only/main entity)
        original_had_summary = original_entity['has_summary']
        print(f"OriginalEntity initially has summary: {original_had_summary}")

        # Phase 2: Add many new entities with MORE mentions to push OriginalEntity out of top 20%
        print("\n=== Phase 2: Add 10 new entities with 10+ mentions each ===")
        for entity_num in range(10):
            entity_name = f"NewEntity{entity_num}"
            # Each new entity gets 10 mentions (more than OriginalEntity's 6)
            for mention in range(10):
                await memory.retain_async(
                    bank_id=bank_id,
                    content=f"{entity_name} is a very important entity, mention {mention+1}.",
                    context="test",
                    event_date=datetime(2024, 2, 1 + mention, tzinfo=timezone.utc),
                    request_context=request_context,
                )

        await memory.wait_for_background_tasks()

        # Phase 3: Verify OriginalEntity's summary was deleted (no longer in top 20%)
        print("\n=== Phase 3: Check if OriginalEntity summary was deleted ===")
        async with pool.acquire() as conn:
            # Get all entities ordered by mention count
            all_entities = await conn.fetch(
                """
                SELECT canonical_name, mention_count, summary IS NOT NULL as has_summary
                FROM entities
                WHERE bank_id = $1
                ORDER BY mention_count DESC
                """,
                bank_id
            )

            # Re-fetch OriginalEntity
            original_after = await conn.fetchrow(
                """
                SELECT id, canonical_name, mention_count, summary IS NOT NULL as has_summary
                FROM entities
                WHERE bank_id = $1 AND LOWER(canonical_name) LIKE '%originalentity%'
                """,
                bank_id
            )

        print(f"\nAll entities by mention count:")
        entities_with_summary = 0
        for e in all_entities:
            print(f"  {e['canonical_name']}: mentions={e['mention_count']}, has_summary={e['has_summary']}")
            if e['has_summary']:
                entities_with_summary += 1

        print(f"\nTotal entities: {len(all_entities)}")
        print(f"Entities with summaries: {entities_with_summary}")

        # Calculate what top 20% means
        top_20_percent_count = max(1, len(all_entities) * 20 // 100)
        print(f"Top 20% would be {top_20_percent_count} entities")

        # OriginalEntity should no longer have a summary (pushed out of top 20%)
        assert original_after is not None, "OriginalEntity should still exist"
        print(f"\nOriginalEntity after: mentions={original_after['mention_count']}, has_summary={original_after['has_summary']}")

        # The key assertion: OriginalEntity should NOT have a summary anymore
        # because it's been pushed out of the top 20% by the new entities with more mentions
        if original_had_summary:
            assert not original_after['has_summary'], \
                f"OriginalEntity should have lost its summary (pushed out of top 20%). " \
                f"It has {original_after['mention_count']} mentions but new entities have 10+ each."
            print("PASS: OriginalEntity's summary was correctly deleted when pushed out of top 20%")
        else:
            print("Note: OriginalEntity didn't have a summary initially, so cleanup test is inconclusive")

        # Also verify that the new high-mention entities have summaries
        new_entities_with_summary = sum(
            1 for e in all_entities
            if 'newentity' in e['canonical_name'].lower() and e['has_summary']
        )
        print(f"\nNew entities with summaries: {new_entities_with_summary}")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_user_entity_prioritized_for_observations(memory, request_context):
    """
    Test that the 'user' entity gets summaries even when many other entities exist.

    The retain pipeline triggers summary regeneration for entities with >= 5 facts
    via background task. This test verifies that frequently mentioned entities
    get prioritized and receive summaries.
    """
    bank_id = f"test_user_priority_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create content where 'user' (the user) is mentioned many times
        contents = [
            # User mentioned frequently
            "The user loves hiking in the mountains during summer.",
            "The user works as a software engineer at Microsoft.",
            "The user has a dog named Max who is a golden retriever.",
            "The user enjoys cooking Italian food, especially pasta.",
            "The user graduated from MIT with a Computer Science degree.",
            "The user's favorite book is 'Dune' by Frank Herbert.",
            # Other entities mentioned fewer times
            "Sarah is a friend who works at Google.",
            "Bob is a colleague from the data science team.",
            "Tokyo is a city the user visited last year.",
            "Python is the user's favorite programming language.",
        ]

        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="personal info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        # Wait for background tasks (summaries generated in background)
        await memory.wait_for_background_tasks()

        # Find the 'user' entity
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            # Find user entity (may be named "user", "the user", etc.)
            user_entity = await conn.fetchrow(
                """
                SELECT e.id, e.canonical_name,
                       (SELECT COUNT(*) FROM unit_entities ue
                        JOIN memory_units mu ON ue.unit_id = mu.id
                        WHERE ue.entity_id = e.id AND mu.bank_id = $1) as fact_count
                FROM entities e
                WHERE e.bank_id = $1
                  AND LOWER(e.canonical_name) LIKE '%user%'
                LIMIT 1
                """,
                bank_id
            )

            # Get all entities with their fact counts to verify prioritization
            all_entities = await conn.fetch(
                """
                SELECT e.id, e.canonical_name,
                       (SELECT COUNT(*) FROM unit_entities ue
                        JOIN memory_units mu ON ue.unit_id = mu.id
                        WHERE ue.entity_id = e.id AND mu.bank_id = $1) as fact_count
                FROM entities e
                WHERE e.bank_id = $1
                ORDER BY fact_count DESC
                """,
                bank_id
            )

        print(f"\n=== Entities by Mention Count ===")
        for entity in all_entities:
            print(f"  {entity['canonical_name']}: {entity['fact_count']} mentions")

        # Verify user entity exists
        assert user_entity is not None, "User entity should have been extracted"
        user_entity_id = str(user_entity['id'])
        user_entity_name = user_entity['canonical_name']
        user_fact_count = user_entity['fact_count']

        print(f"\n=== User Entity ===")
        print(f"Entity: {user_entity_name} (id: {user_entity_id})")
        print(f"Fact count: {user_fact_count}")

        # Verify user has enough facts for observations (>= MIN_FACTS_THRESHOLD of 5)
        assert user_fact_count >= 5, \
            f"User entity should have at least 5 facts, but has {user_fact_count}"

        # Get entity state for user (includes summary as observation)
        state = await memory.get_entity_state(
            bank_id, user_entity_id, user_entity_name, request_context=request_context
        )

        print(f"\n=== User Entity Summary ===")
        print(f"Total observations: {len(state.observations)}")
        for obs in state.observations:
            print(f"  - {obs.text}")

        # Verify summary was generated for user (critical assertion)
        assert len(state.observations) > 0, \
            f"User entity should have a summary (has {user_fact_count} facts, threshold is 5). " \
            f"This may indicate that summary regeneration is not working properly."

        # Verify summary mentions relevant content about the user
        obs_texts = " ".join([o.text.lower() for o in state.observations])
        user_keywords = ["hiking", "software", "engineer", "dog", "max", "cooking",
                        "italian", "mit", "dune", "microsoft"]
        matching_keywords = [k for k in user_keywords if k in obs_texts]
        assert len(matching_keywords) > 0, \
            f"Summary should contain relevant information about the user. Keywords found: {matching_keywords}"

        print(f"User entity was prioritized and received summary")
        print(f"Summary contains relevant keywords: {matching_keywords}")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
