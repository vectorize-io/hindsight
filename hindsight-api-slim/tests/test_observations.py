"""
Test observation generation and entity state functionality.

NOTE: Observations are now stored as summaries on the entities table,
not as separate memory_units. The observations list in EntityState is
populated from the summary for backwards compatibility.
"""
import pytest
from hindsight_api.engine.memory_engine import Budget
from hindsight_api import RequestContext
from hindsight_api.config import _get_raw_config
from datetime import datetime, timezone


@pytest.fixture
def disable_observations():
    """Disable observations for a specific test."""
    config = _get_raw_config()
    original_value = config.enable_observations
    config.enable_observations = False
    yield
    config.enable_observations = original_value


@pytest.mark.asyncio
async def test_entity_extraction_on_retain(memory, request_context):
    """
    Test that entities are extracted when new facts are added.

    This test stores multiple facts and verifies entities are extracted.
    """
    bank_id = f"test_entity_extraction_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store multiple facts about John
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

        # Wait for background tasks
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

            # Check the fact count for this entity
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
        print(f"\n=== Found Entity ===")
        print(f"Entity: {entity_row['canonical_name']} (id: {entity_row['id']})")
        print(f"Entity was successfully extracted")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_search_with_include_entities(memory, request_context):
    """
    Test that recall accepts include_entities parameter for backwards compatibility.

    Note: Entity observations have been deprecated. This test verifies the parameter
    is still accepted without errors.
    """
    bank_id = f"test_search_ent_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store facts about Alice
        contents = [
            "Alice is a data scientist who works on recommendation systems at Netflix.",
            "Alice presented her research at the ML conference last month.",
        ]

        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="work info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        # Wait for background tasks
        await memory.wait_for_background_tasks()

        # Search with include_entities=True (should be accepted for backwards compatibility)
        result = await memory.recall_async(
            bank_id=bank_id,
            query="What does Alice do?",
            fact_type=["world", "experience"],
            budget=Budget.LOW,
            max_tokens=2000,
            include_entities=True,
            max_entity_tokens=5000,
            request_context=request_context,
        )

        # Verify recall works
        assert len(result.results) > 0, "Should find some facts"
        print(f"Found {len(result.results)} facts")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_observation_fact_type_in_database(memory, request_context, disable_observations):
    """
    Test that when observations are disabled, no observation records are created.

    When enable_observations=False, consolidation does not run and no
    memory_units with fact_type='observation' should exist.
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

        # Check that NO observations exist in memory_units
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
        assert len(observations) == 0, "Observations should NOT be stored as memory_units"

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_entity_mention_counts(memory, request_context):
    """
    Test that entity mention counts are tracked correctly.

    This test creates entities with varying mention counts and verifies
    that the counts are accurate.
    """
    bank_id = f"test_mention_counts_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create content with varying entity mention counts:
        # - "Nexora" mentioned 10 times
        # - "Trivex" mentioned 1 time
        contents = [
            "Nexora is a tech company based in San Francisco.",
            "Nexora was founded in 2010 by experienced entrepreneurs.",
            "Nexora has over 500 employees worldwide.",
            "Nexora specializes in cloud computing solutions.",
            "Nexora recently raised $50 million in Series C funding.",
            "Nexora has partnerships with major tech companies.",
            "Nexora is known for its innovative culture.",
            "Nexora offers competitive salaries and benefits.",
            "Nexora has offices in 5 countries.",
            "Nexora won the best workplace award last year.",
            # Low mentions
            "Trivex is a small consulting firm.",
        ]

        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="company info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        # Wait for background tasks
        await memory.wait_for_background_tasks()

        # Check entity mention counts
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            entities = await conn.fetch(
                """
                SELECT e.id, e.canonical_name, e.mention_count
                FROM entities e
                WHERE e.bank_id = $1
                ORDER BY e.mention_count DESC
                """,
                bank_id
            )

        print(f"\n=== Entity Mention Counts Test ===")
        print(f"Total entities: {len(entities)}")

        high_mention_entity = None
        low_mention_entity = None

        for entity in entities:
            name = entity['canonical_name'].lower()
            mention_count = entity['mention_count']

            print(f"  {entity['canonical_name']}: mentions={mention_count}")

            if "nexora" in name:
                high_mention_entity = entity
            elif "trivex" in name:
                low_mention_entity = entity

        # Both entities must exist
        assert high_mention_entity is not None, "Nexora entity should exist"
        assert low_mention_entity is not None, "Trivex entity should exist"

        # Nexora (10 mentions) must rank higher than Trivex (1 mention)
        assert high_mention_entity['mention_count'] > low_mention_entity['mention_count'], \
            f"Nexora ({high_mention_entity['mention_count']} mentions) should have more than Trivex ({low_mention_entity['mention_count']} mentions)"

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
@pytest.mark.timeout(1200)
async def test_entity_mention_ranking(memory, request_context):
    """
    Test that entity mention counts correctly rank entities.

    This test:
    1. Creates an entity with 6 mentions
    2. Adds more entities with higher mention counts
    3. Verifies entities are ranked correctly by mention count
    """
    bank_id = f"test_ranking_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Phase 1: Create "Alice" with 6 mentions
        for i in range(6):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Alice is mentioned here in fact {i+1}.",
                context="test",
                event_date=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        await memory.wait_for_background_tasks()

        # Phase 2: Add entities with MORE mentions (10 each)
        for entity_num, entity_name in enumerate(["Bruno", "Carlos", "Diana"]):
            for mention in range(10):
                await memory.retain_async(
                    bank_id=bank_id,
                    content=f"{entity_name} is a very important entity, mention {mention+1}.",
                    context="test",
                    event_date=datetime(2024, 2, 1 + mention, tzinfo=timezone.utc),
                    request_context=request_context,
                )

        await memory.wait_for_background_tasks()

        # Phase 3: Verify entities are ranked by mention count
        print("\n=== Phase 3: Check entity ranking ===")
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            all_entities = await conn.fetch(
                """
                SELECT canonical_name, mention_count
                FROM entities
                WHERE bank_id = $1
                ORDER BY mention_count DESC
                """,
                bank_id
            )

        print(f"\nAll entities by mention count:")
        for e in all_entities:
            print(f"  {e['canonical_name']}: mentions={e['mention_count']}")

        # Verify high-mention entities rank higher than Alice (6 mentions)
        alice = next((e for e in all_entities if 'alice' in e['canonical_name'].lower()), None)
        high_mention = [e for e in all_entities if e['canonical_name'].lower() in ('bruno', 'carlos', 'diana')]

        assert alice is not None, "Alice entity should exist"
        assert len(high_mention) > 0, "High-mention entities should exist"

        # Entities with 10 mentions each should rank higher than Alice (6 mentions)
        for entity in high_mention:
            assert entity['mention_count'] > alice['mention_count'], (
                f"{entity['canonical_name']} ({entity['mention_count']} mentions) "
                f"should rank higher than Alice ({alice['mention_count']} mentions)"
            )

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
@pytest.mark.hs_llm_core
async def test_user_entity_extraction(memory_real_llm, request_context):
    """
    Test that the 'user' entity is correctly extracted when mentioned frequently.
    """
    memory = memory_real_llm
    bank_id = f"test_user_entity_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create content where 'user' is mentioned many times
        contents = [
            "The user loves hiking in the mountains during summer.",
            "The user works as a software engineer at Microsoft.",
            "The user has a dog named Max who is a golden retriever.",
            "The user enjoys cooking Italian food, especially pasta.",
            "The user graduated from MIT with a Computer Science degree.",
            "The user's favorite book is 'Dune' by Frank Herbert.",
            # Other entities mentioned fewer times
            "Sarah is a friend who works at Google.",
            "Bob is a colleague from the data science team.",
        ]

        for i, content in enumerate(contents):
            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="personal info",
                event_date=datetime(2024, 1, 15 + i, tzinfo=timezone.utc),
                request_context=request_context,
            )

        # Wait for background tasks
        await memory.wait_for_background_tasks()

        # Find the 'user' entity
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
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

            # Get all entities with their fact counts
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
        print(f"\n=== User Entity ===")
        print(f"Entity: {user_entity['canonical_name']} (id: {user_entity['id']})")
        print(f"Fact count: {user_entity['fact_count']}")
        print(f"User entity was successfully extracted")

    finally:
        # Cleanup
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
