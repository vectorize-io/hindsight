"""
Observation regeneration for retain pipeline.

Regenerates entity observations as part of the retain transaction.
"""

import logging
import time
import uuid
from datetime import UTC, datetime

from ...config import get_config
from ..memory_engine import fq_table
from ..search import observation_utils
from . import embedding_utils
from .types import EntityLink

logger = logging.getLogger(__name__)


def utcnow():
    """Get current UTC time."""
    return datetime.now(UTC)


# Simple dataclass-like container for facts (avoid importing from memory_engine)
class MemoryFactForObservation:
    def __init__(self, id: str, text: str, fact_type: str, context: str, occurred_start: str | None):
        self.id = id
        self.text = text
        self.fact_type = fact_type
        self.context = context
        self.occurred_start = occurred_start


async def regenerate_observations_batch(
    conn, embeddings_model, llm_config, bank_id: str, entity_links: list[EntityLink], log_buffer: list[str] = None
) -> None:
    """
    Regenerate observations for top entities in this batch.

    Called INSIDE the retain transaction for atomicity - if observations
    fail, the entire retain batch is rolled back.

    Args:
        conn: Database connection (from the retain transaction)
        embeddings_model: Embeddings model for generating observation embeddings
        llm_config: LLM configuration for observation extraction
        bank_id: Bank identifier
        entity_links: Entity links from this batch
        log_buffer: Optional log buffer for timing
    """
    config = get_config()
    TOP_N_ENTITIES = config.observation_top_entities
    MIN_FACTS_THRESHOLD = config.observation_min_facts

    if not entity_links:
        return

    # Count mentions per entity in this batch
    entity_mention_counts: dict[str, int] = {}
    for link in entity_links:
        if link.entity_id:
            entity_id = str(link.entity_id)
            entity_mention_counts[entity_id] = entity_mention_counts.get(entity_id, 0) + 1

    if not entity_mention_counts:
        return

    # Sort by mention count descending and take top N
    sorted_entities = sorted(entity_mention_counts.items(), key=lambda x: x[1], reverse=True)
    entities_to_process = [e[0] for e in sorted_entities[:TOP_N_ENTITIES]]

    obs_start = time.time()

    # Convert to UUIDs
    entity_uuids = [uuid.UUID(eid) if isinstance(eid, str) else eid for eid in entities_to_process]

    # Batch query for entity names
    entity_rows = await conn.fetch(
        f"""
        SELECT id, canonical_name FROM {fq_table("entities")}
        WHERE id = ANY($1) AND bank_id = $2
        """,
        entity_uuids,
        bank_id,
    )
    entity_names = {row["id"]: row["canonical_name"] for row in entity_rows}

    # Batch query for fact counts
    fact_counts = await conn.fetch(
        f"""
        SELECT ue.entity_id, COUNT(*) as cnt
        FROM {fq_table("unit_entities")} ue
        JOIN {fq_table("memory_units")} mu ON ue.unit_id = mu.id
        WHERE ue.entity_id = ANY($1) AND mu.bank_id = $2
        GROUP BY ue.entity_id
        """,
        entity_uuids,
        bank_id,
    )
    entity_fact_counts = {row["entity_id"]: row["cnt"] for row in fact_counts}

    # Filter entities that meet the threshold
    entities_with_names = []
    for entity_id in entities_to_process:
        entity_uuid = uuid.UUID(entity_id) if isinstance(entity_id, str) else entity_id
        if entity_uuid not in entity_names:
            continue
        fact_count = entity_fact_counts.get(entity_uuid, 0)
        if fact_count >= MIN_FACTS_THRESHOLD:
            entities_with_names.append((entity_id, entity_names[entity_uuid]))

    if not entities_with_names:
        return

    # Process entities SEQUENTIALLY (asyncpg doesn't allow concurrent queries on same connection)
    # We must use the same connection to stay in the retain transaction
    total_observations = 0

    for entity_id, entity_name in entities_with_names:
        try:
            obs_ids = await _regenerate_entity_observations(
                conn, embeddings_model, llm_config, bank_id, entity_id, entity_name
            )
            total_observations += len(obs_ids)
        except Exception as e:
            logger.error(f"[OBSERVATIONS] Error processing entity {entity_id}: {e}")

    obs_time = time.time() - obs_start
    if log_buffer is not None:
        log_buffer.append(
            f"[11] Observations: {total_observations} observations for {len(entities_with_names)} entities in {obs_time:.3f}s"
        )


async def _regenerate_entity_observations(
    conn, embeddings_model, llm_config, bank_id: str, entity_id: str, entity_name: str
) -> list[str]:
    """
    Regenerate observations for a single entity.

    Uses the provided connection (part of retain transaction).

    Args:
        conn: Database connection (from the retain transaction)
        embeddings_model: Embeddings model
        llm_config: LLM configuration
        bank_id: Bank identifier
        entity_id: Entity UUID
        entity_name: Canonical name of the entity

    Returns:
        List of created observation IDs
    """
    entity_uuid = uuid.UUID(entity_id) if isinstance(entity_id, str) else entity_id

    # Get all facts mentioning this entity (exclude observations themselves)
    rows = await conn.fetch(
        f"""
        SELECT mu.id, mu.text, mu.context, mu.occurred_start, mu.fact_type
        FROM {fq_table("memory_units")} mu
        JOIN {fq_table("unit_entities")} ue ON mu.id = ue.unit_id
        WHERE mu.bank_id = $1
          AND ue.entity_id = $2
          AND mu.fact_type IN ('world', 'experience')
        ORDER BY mu.occurred_start DESC
        LIMIT 50
        """,
        bank_id,
        entity_uuid,
    )

    if not rows:
        return []

    # Convert to fact objects for observation extraction
    facts = []
    for row in rows:
        occurred_start = row["occurred_start"].isoformat() if row["occurred_start"] else None
        facts.append(
            MemoryFactForObservation(
                id=str(row["id"]),
                text=row["text"],
                fact_type=row["fact_type"],
                context=row["context"],
                occurred_start=occurred_start,
            )
        )

    # Extract observations using LLM
    observations = await observation_utils.extract_observations_from_facts(llm_config, entity_name, facts)

    if not observations:
        return []

    # Delete old observations for this entity
    await conn.execute(
        f"""
        DELETE FROM {fq_table("memory_units")}
        WHERE id IN (
            SELECT mu.id
            FROM {fq_table("memory_units")} mu
            JOIN {fq_table("unit_entities")} ue ON mu.id = ue.unit_id
            WHERE mu.bank_id = $1
              AND mu.fact_type = 'observation'
              AND ue.entity_id = $2
        )
        """,
        bank_id,
        entity_uuid,
    )

    # Generate embeddings for new observations
    embeddings = await embedding_utils.generate_embeddings_batch(embeddings_model, observations)

    # Insert new observations
    current_time = utcnow()
    created_ids = []

    for obs_text, embedding in zip(observations, embeddings):
        result = await conn.fetchrow(
            f"""
            INSERT INTO {fq_table("memory_units")} (
                bank_id, text, embedding, context, event_date,
                occurred_start, occurred_end, mentioned_at,
                fact_type, access_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'observation', 0)
            RETURNING id
            """,
            bank_id,
            obs_text,
            str(embedding),
            f"observation about {entity_name}",
            current_time,
            current_time,
            current_time,
            current_time,
        )
        obs_id = str(result["id"])
        created_ids.append(obs_id)

        # Link observation to entity
        await conn.execute(
            f"""
            INSERT INTO {fq_table("unit_entities")} (unit_id, entity_id)
            VALUES ($1, $2)
            """,
            uuid.UUID(obs_id),
            entity_uuid,
        )

    return created_ids
