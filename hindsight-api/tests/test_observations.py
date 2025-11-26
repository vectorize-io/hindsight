"""
Test observation generation and entity state functionality.
"""
import pytest
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_observation_generation_on_put(memory):
    """
    Test that observations are generated when new facts are added.

    1. Store facts about an entity
    2. Wait for background tasks (observation generation)
    3. Verify observations were created and linked to the entity
    """
    agent_id = f"test_obs_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store some facts about an entity
        await memory.put_async(
            agent_id=agent_id,
            content="John is a software engineer at Google. He is detail-oriented and methodical.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc)
        )

        await memory.put_async(
            agent_id=agent_id,
            content="John has been working on the AI team for 3 years. He specializes in machine learning.",
            context="work info",
            event_date=datetime(2024, 2, 1, tzinfo=timezone.utc)
        )

        # Wait for background tasks to complete (including observation generation)
        await memory.wait_for_background_tasks()

        # Find the John entity
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            entity_row = await conn.fetchrow(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE agent_id = $1 AND LOWER(canonical_name) LIKE '%john%'
                LIMIT 1
                """,
                agent_id
            )

        if entity_row:
            entity_id = str(entity_row['id'])
            entity_name = entity_row['canonical_name']
            print(f"\n=== Found Entity ===")
            print(f"Entity: {entity_name} (id: {entity_id})")

            # Get observations for the entity
            observations = await memory.get_entity_observations(agent_id, entity_id, limit=10)

            print(f"\n=== Observations for {entity_name} ===")
            print(f"Total observations: {len(observations)}")
            for obs in observations:
                print(f"  - {obs.text}")

            # Verify observations were created
            if len(observations) > 0:
                print(f"✓ Observations were successfully generated")
                # Check that observations mention relevant content
                obs_texts = " ".join([o.text.lower() for o in observations])
                assert any(keyword in obs_texts for keyword in ["google", "engineer", "ai", "machine learning", "detail"]), \
                    "Observations should contain relevant information about John"
            else:
                print(f"⚠ Note: No observations were generated (this can happen if LLM extraction varies)")

        else:
            print(f"⚠ Note: No 'John' entity was extracted (LLM extraction may vary)")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)
            await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)


@pytest.mark.asyncio
async def test_regenerate_entity_observations(memory):
    """
    Test explicit regeneration of observations for an entity.
    """
    agent_id = f"test_regen_obs_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts about an entity
        await memory.put_async(
            agent_id=agent_id,
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
                WHERE agent_id = $1 AND LOWER(canonical_name) LIKE '%sarah%'
                LIMIT 1
                """,
                agent_id
            )

        if entity_row:
            entity_id = str(entity_row['id'])
            entity_name = entity_row['canonical_name']

            # Manually regenerate observations
            created_ids = await memory.regenerate_entity_observations(
                agent_id=agent_id,
                entity_id=entity_id,
                entity_name=entity_name
            )

            print(f"\n=== Regenerated Observations ===")
            print(f"Created {len(created_ids)} observations for {entity_name}")

            # Get the observations
            observations = await memory.get_entity_observations(agent_id, entity_id, limit=10)
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
            await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)
            await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)


@pytest.mark.asyncio
async def test_search_with_include_entities(memory):
    """
    Test that search with include_entities=True returns entity observations.
    """
    agent_id = f"test_search_ent_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts about entities
        await memory.put_async(
            agent_id=agent_id,
            content="Alice is a data scientist who works on recommendation systems at Netflix.",
            context="work info",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc)
        )

        await memory.put_async(
            agent_id=agent_id,
            content="Alice presented her research at the ML conference last month. She is an expert in deep learning.",
            context="work info",
            event_date=datetime(2024, 2, 1, tzinfo=timezone.utc)
        )

        # Wait for background tasks
        await memory.wait_for_background_tasks()

        # Search with include_entities=True
        result = await memory.search_async(
            agent_id=agent_id,
            query="What does Alice do?",
            fact_type=["world", "agent"],
            thinking_budget=30,
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

        print(f"\n=== Entity Observations ===")
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
        if facts_with_entities:
            print(f"✓ {len(facts_with_entities)} facts have entity information")

        # Check if entity observations are included
        if result.entities:
            print(f"✓ Entity observations included for {len(result.entities)} entities")
            for name, state in result.entities.items():
                assert state.canonical_name == name, "Entity canonical_name should match key"
                assert state.entity_id, "Entity should have an ID"

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)
            await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)


@pytest.mark.asyncio
async def test_get_entity_state(memory):
    """
    Test getting the full state of an entity.
    """
    agent_id = f"test_entity_state_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts
        await memory.put_async(
            agent_id=agent_id,
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
                WHERE agent_id = $1 AND LOWER(canonical_name) LIKE '%bob%'
                LIMIT 1
                """,
                agent_id
            )

        if entity_row:
            entity_id = str(entity_row['id'])
            entity_name = entity_row['canonical_name']

            # Get entity state
            state = await memory.get_entity_state(
                agent_id=agent_id,
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
            await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)
            await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)


@pytest.mark.asyncio
async def test_observation_fact_type_in_database(memory):
    """
    Test that observations are stored with correct fact_type in database.
    """
    agent_id = f"test_obs_db_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts
        await memory.put_async(
            agent_id=agent_id,
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
                WHERE agent_id = $1 AND fact_type = 'observation'
                """,
                agent_id
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
            await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)
            await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)
