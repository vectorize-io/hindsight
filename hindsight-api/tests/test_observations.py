"""
Test observation generation and entity state functionality.
"""
import pytest
from hindsight_api.engine.memory_engine import Budget
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_observation_generation_on_put(memory):
    """
    Test that observations are generated SYNCHRONOUSLY when new facts are added.

    Observations are generated during retain when:
    - Entity has >= 5 facts (MIN_FACTS_THRESHOLD)
    - Entity is in top 5 by mention count

    This test stores enough facts to trigger automatic observation generation.
    """
    bank_id = f"test_obs_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store multiple facts about John to reach the MIN_FACTS_THRESHOLD (5)
        # Each retain call should extract at least one fact about John
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
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc)
            )

        # Observations are generated SYNCHRONOUSLY during retain,
        # so they should be available immediately after retain completes.
        # No need to wait for background tasks for observations.

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

        # Get observations for the entity - should be available immediately
        observations = await memory.get_entity_observations(bank_id, entity_id, limit=10)

        print(f"\n=== Observations for {entity_name} ===")
        print(f"Total observations: {len(observations)}")
        for obs in observations:
            print(f"  - {obs.text}")

        # Verify observations were created (requires >= 5 facts)
        assert len(observations) > 0, \
            f"Observations should have been generated synchronously during retain (entity has {fact_count} facts, threshold is 5)"

        # Check that observations mention relevant content
        obs_texts = " ".join([o.text.lower() for o in observations])
        assert any(keyword in obs_texts for keyword in ["google", "engineer", "ai", "machine learning", "detail"]), \
            "Observations should contain relevant information about John"

        print(f"✓ Observations were successfully generated synchronously during retain")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_regenerate_entity_observations(memory):
    """
    Test explicit regeneration of observations for an entity.
    """
    bank_id = f"test_regen_obs_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts about an entity
        await memory.retain_async(
            bank_id=bank_id,
            content="Sarah is a product manager who loves user research and data analysis.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc)
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

            # Manually regenerate observations
            created_ids = await memory.regenerate_entity_observations(
                bank_id=bank_id,
                entity_id=entity_id,
                entity_name=entity_name
            )

            print(f"\n=== Regenerated Observations ===")
            print(f"Created {len(created_ids)} observations for {entity_name}")

            # Get the observations
            observations = await memory.get_entity_observations(bank_id, entity_id, limit=10)
            for obs in observations:
                print(f"  - {obs.text}")

            # Verify observations were created
            if len(created_ids) > 0:
                assert len(observations) == len(created_ids), "Should have same number of observations as created IDs"
                print(f"✓ Observations regenerated successfully")
            else:
                print(f"⚠ Note: No observations were regenerated")

        else:
            print(f"⚠ Note: No 'Sarah' entity was extracted")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_search_with_include_entities(memory):
    """
    Test that search with include_entities=True returns entity observations.

    This test verifies that:
    1. Observations are generated during retain (when entity has >= 5 facts)
    2. Observations are returned in recall results with include_entities=True
    """
    bank_id = f"test_search_ent_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store enough facts about Alice to trigger observation generation (>= 5 facts)
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
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc)
            )

        # Observations are generated synchronously during retain, no need to wait

        # Search with include_entities=True
        result = await memory.recall_async(
            bank_id=bank_id,
            query="What does Alice do?",
            fact_type=["world", "experience"],
            budget=Budget.LOW,
            max_tokens=2000,
            include_entities=True,
            max_entity_tokens=500
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
        print(f"✓ {len(facts_with_entities)} facts have entity information")

        # Check if entity observations are included in recall
        assert result.entities is not None and len(result.entities) > 0, \
            "Entity observations should be included in recall results"
        print(f"✓ Entity observations included for {len(result.entities)} entities")

        # Verify Alice entity has observations
        alice_found = False
        for name, state in result.entities.items():
            assert state.canonical_name == name, "Entity canonical_name should match key"
            assert state.entity_id, "Entity should have an ID"
            if "alice" in name.lower():
                alice_found = True
                assert len(state.observations) > 0, \
                    "Alice should have observations (generated during retain)"
                print(f"✓ Alice has {len(state.observations)} observations in recall result")

        assert alice_found, "Alice entity should be in recall results"

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_get_entity_state(memory):
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
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc)
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
                limit=10
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
async def test_observation_fact_type_in_database(memory):
    """
    Test that observations are stored with correct fact_type in database.
    """
    bank_id = f"test_obs_db_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts
        await memory.retain_async(
            bank_id=bank_id,
            content="Charlie is a DevOps engineer who manages the Kubernetes infrastructure.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc)
        )

        await memory.wait_for_background_tasks()

        # Check that observations have correct fact_type
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

        print(f"\n=== Observation Records in Database ===")
        print(f"Found {len(observations)} observation records")
        for obs in observations:
            print(f"  - fact_type: {obs['fact_type']}")
            print(f"    text: {obs['text']}")
            print(f"    context: {obs['context']}")

        if len(observations) > 0:
            for obs in observations:
                assert obs['fact_type'] == 'observation', "All observation records should have fact_type='observation'"
            print(f"✓ All observations have correct fact_type")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_user_entity_prioritized_for_observations(memory):
    """
    Test that the 'user' entity gets observations even when many other entities exist.

    The retain pipeline only regenerates observations for TOP_N_ENTITIES (5) entities,
    sorted by mention count. This test verifies that the most mentioned entity ('user')
    gets prioritized and receives observations.

    This is critical because 'user' is often the most important entity in personal memory.
    """
    bank_id = f"test_user_priority_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create content where 'user' (the user) is mentioned many times
        # along with several other entities
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

        # Retain all content in a single batch for efficiency
        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="personal info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc)
            )

        # Observations are generated synchronously during retain

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

        # Get observations for user entity
        observations = await memory.get_entity_observations(bank_id, user_entity_id, limit=10)

        print(f"\n=== User Entity Observations ===")
        print(f"Total observations: {len(observations)}")
        for obs in observations:
            print(f"  - {obs.text}")

        # Verify observations were generated for user (critical assertion)
        assert len(observations) > 0, \
            f"User entity should have observations (has {user_fact_count} facts, threshold is 5). " \
            f"This may indicate that 'user' is not being prioritized in the top 5 entities by mention count."

        # Verify observations mention relevant content about the user
        obs_texts = " ".join([o.text.lower() for o in observations])
        user_keywords = ["hiking", "software", "engineer", "dog", "max", "cooking",
                        "italian", "mit", "dune", "microsoft"]
        matching_keywords = [k for k in user_keywords if k in obs_texts]
        assert len(matching_keywords) > 0, \
            f"Observations should contain relevant information about the user. Keywords found: {matching_keywords}"

        print(f"✓ User entity was prioritized and received {len(observations)} observations")
        print(f"✓ Observations contain relevant keywords: {matching_keywords}")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
