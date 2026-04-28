"""
Tests for EntityResolver edge cases.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.db.result import ResultRow
from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.pg0 import resolve_database_url

# ---------------------------------------------------------------------------
# Unit tests for discard_pending_stats() — no database required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_pending_stats_clears_both_dicts():
    """discard_pending_stats() must remove entries for the current task from
    both _pending_stats and _pending_cooccurrences."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    key = resolver._task_key()

    resolver._pending_stats[key] = [object()]  # type: ignore[list-item]
    resolver._pending_cooccurrences[key] = [object()]  # type: ignore[list-item]

    resolver.discard_pending_stats()

    assert key not in resolver._pending_stats
    assert key not in resolver._pending_cooccurrences


@pytest.mark.asyncio
async def test_discard_pending_stats_is_idempotent():
    """Calling discard_pending_stats() when nothing is pending must not raise."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    resolver.discard_pending_stats()
    resolver.discard_pending_stats()  # second call — still safe


@pytest.mark.asyncio
async def test_discard_pending_stats_does_not_affect_other_task_keys():
    """discard_pending_stats() must only remove the current task's entries,
    leaving entries keyed under other task IDs untouched."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    other_key = -1  # A fake key that can never be a real task id

    resolver._pending_stats[other_key] = [object()]  # type: ignore[list-item]
    resolver._pending_cooccurrences[other_key] = [object()]  # type: ignore[list-item]

    resolver.discard_pending_stats()  # discards current task's key only

    assert other_key in resolver._pending_stats, "other task's stats must be preserved"
    assert other_key in resolver._pending_cooccurrences, "other task's cooccurrences must be preserved"


@pytest.mark.asyncio
async def test_resolve_entities_batch_handles_unicode_lower_conflicts(pg0_db_url):
    """
    Existing entities with PostgreSQL/Python lowercase mismatches should resolve
    to the conflicted row instead of leaving a missing entity_id.
    """
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"test-entity-resolver-{uuid.uuid4().hex[:8]}"
    event_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    resolver = EntityResolver(pool=backend, entity_lookup="full")

    try:
        async with backend.acquire() as conn:
            existing_entity_id = await conn.fetchval(
                """
                INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count)
                VALUES ($1, $2, $3, $3, 1)
                RETURNING id
                """,
                bank_id,
                "İstanbul",
                event_date,
            )

            resolved_ids = await resolver.resolve_entities_batch(
                bank_id=bank_id,
                entities_data=[
                    {
                        "text": "istanbul",
                        "nearby_entities": [],
                        "event_date": event_date,
                    }
                ],
                context="unicode case mismatch",
                unit_event_date=event_date,
                conn=conn,
            )

            entity_rows = await conn.fetch(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE bank_id = $1
                ORDER BY canonical_name
                """,
                bank_id,
            )

        assert resolved_ids == [existing_entity_id]
        assert len(entity_rows) == 1
        assert entity_rows[0]["id"] == existing_entity_id
        assert entity_rows[0]["canonical_name"] == "İstanbul"
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await backend.shutdown()


# ---------------------------------------------------------------------------
# Oracle fuzzy entity resolution — unit tests (mock conn, no live DB)
# ---------------------------------------------------------------------------


class TestOracleFuzzyEntityResolution:
    """Verify _resolve_entities_batch_oracle_fuzzy produces correct Oracle-native
    SQL and correctly transforms input/output data for the entity resolution pipeline."""

    @pytest.fixture()
    def resolver(self):
        return EntityResolver(pool=None, entity_lookup="oracle_fuzzy")  # type: ignore[arg-type]

    @pytest.fixture()
    def mock_conn(self):
        conn = AsyncMock()
        conn.backend_type = "oracle"
        conn.fetch = AsyncMock(return_value=[])
        return conn

    @pytest.mark.asyncio
    async def test_query_is_valid_oracle_sql(self, resolver, mock_conn):
        """The SQL must use Oracle-native JSON_TABLE + UTL_MATCH, not PG-specific
        unnest or pg_trgm. This is the core behavioral change."""
        with patch.object(resolver, "_resolve_from_candidates", new_callable=AsyncMock, return_value=[]):
            await resolver._resolve_entities_batch_oracle_fuzzy(
                conn=mock_conn,
                bank_id="bank-1",
                entities_data=[{"text": "Alice", "nearby_entities": [], "event_date": None}],
                unit_event_date=None,
            )

        mock_conn.fetch.assert_called_once()
        query = mock_conn.fetch.call_args.args[0]

        # Must use Oracle-native constructs
        assert "JSON_TABLE" in query, "Should use JSON_TABLE to expand entity texts into rows"
        assert "UTL_MATCH.JARO_WINKLER_SIMILARITY" in query, "Should use Oracle's UTL_MATCH for fuzzy matching"
        assert "'$[*]'" in query, "JSON_TABLE should use '$[*]' path to expand array elements"

        # Must NOT use PG-specific constructs
        assert "unnest" not in query.lower(), "Must not use PG-only unnest()"
        # pg_trgm uses standalone "similarity(col, val)" — UTL_MATCH.JARO_WINKLER_SIMILARITY is different
        assert "pg_trgm" not in query.lower(), "Must not reference pg_trgm"

    @pytest.mark.asyncio
    async def test_entity_texts_serialized_as_json_array(self, resolver, mock_conn):
        """Entity texts must be JSON-serialized so JSON_TABLE can parse them.

        This is critical — passing a Python list would fail at the Oracle driver level
        because JSON_TABLE expects a string, not an array bind variable.
        """
        with patch.object(resolver, "_resolve_from_candidates", new_callable=AsyncMock, return_value=[]):
            await resolver._resolve_entities_batch_oracle_fuzzy(
                conn=mock_conn,
                bank_id="bank-1",
                entities_data=[
                    {"text": "Alice", "nearby_entities": [], "event_date": None},
                    {"text": "Bob", "nearby_entities": [], "event_date": None},
                ],
                unit_event_date=None,
            )

        call_args = mock_conn.fetch.call_args.args
        bank_id_arg = call_args[1]
        entity_texts_arg = call_args[2]

        assert bank_id_arg == "bank-1", "First bind param ($1) must be bank_id"
        assert isinstance(entity_texts_arg, str), "Second bind param ($2) must be a JSON string"
        parsed = json.loads(entity_texts_arg)
        assert isinstance(parsed, list), "JSON must deserialize to a list"
        assert set(parsed) == {"Alice", "Bob"}

    @pytest.mark.asyncio
    async def test_duplicate_entity_texts_deduplicated(self, resolver, mock_conn):
        """Duplicate entity texts should be sent once to avoid redundant DB work."""
        with patch.object(resolver, "_resolve_from_candidates", new_callable=AsyncMock, return_value=[]):
            await resolver._resolve_entities_batch_oracle_fuzzy(
                conn=mock_conn,
                bank_id="bank-1",
                entities_data=[
                    {"text": "Alice", "nearby_entities": [], "event_date": None},
                    {"text": "Alice", "nearby_entities": [], "event_date": None},
                    {"text": "Bob", "nearby_entities": [], "event_date": None},
                ],
                unit_event_date=None,
            )

        entity_texts_json = mock_conn.fetch.call_args.args[2]
        parsed = json.loads(entity_texts_json)
        assert len(parsed) == 2, "Should deduplicate 'Alice' to a single entry"

    @pytest.mark.asyncio
    async def test_fallback_to_full_strategy_on_utl_match_error(self, resolver, mock_conn):
        """If UTL_MATCH is unavailable (ORA-06550, etc.), must gracefully fall back
        to the 'full' strategy and permanently switch the resolver's strategy."""
        mock_conn.fetch = AsyncMock(side_effect=Exception("ORA-06550: UTL_MATCH not available"))

        with patch.object(resolver, "_resolve_entities_batch_full", new_callable=AsyncMock, return_value=["eid-1"]):
            result = await resolver._resolve_entities_batch_oracle_fuzzy(
                conn=mock_conn,
                bank_id="bank-1",
                entities_data=[{"text": "Alice", "nearby_entities": [], "event_date": None}],
                unit_event_date=None,
            )

        assert result == ["eid-1"], "Should return results from the full strategy fallback"
        assert resolver.entity_lookup == "full", "Strategy must be permanently switched to 'full'"

    @pytest.mark.asyncio
    async def test_candidate_rows_correctly_structured_for_downstream(self, resolver, mock_conn):
        """DB rows must be correctly parsed into the (id, name, metadata, last_seen, count)
        tuple format that _resolve_from_candidates expects.

        A wrong tuple structure here would cause silent scoring bugs or KeyErrors downstream.
        """
        candidate_rows = [
            ResultRow(
                {
                    "id": "eid-1",
                    "canonical_name": "Alice Smith",
                    "metadata": '{"role": "eng"}',
                    "last_seen": None,
                    "mention_count": 5,
                    "query_text": "Alice",
                }
            ),
            ResultRow(
                {
                    "id": "eid-2",
                    "canonical_name": "Robert Jones",
                    "metadata": None,
                    "last_seen": None,
                    "mention_count": 3,
                    "query_text": "Bob",
                }
            ),
        ]
        # First fetch: candidates. Second fetch: co-occurrences (empty).
        mock_conn.fetch = AsyncMock(side_effect=[candidate_rows, []])

        with patch.object(resolver, "_resolve_from_candidates", new_callable=AsyncMock, return_value=[]) as mock_rfc:
            await resolver._resolve_entities_batch_oracle_fuzzy(
                conn=mock_conn,
                bank_id="bank-1",
                entities_data=[
                    {"text": "Alice", "nearby_entities": [], "event_date": None},
                    {"text": "Bob", "nearby_entities": [], "event_date": None},
                ],
                unit_event_date=None,
            )

            # Verify the all_candidates dict passed to _resolve_from_candidates
            all_candidates = mock_rfc.call_args.args[4]

            # Each query_text should have its candidates grouped
            assert set(all_candidates.keys()) == {"Alice", "Bob"}

            # Verify tuple structure: (id, canonical_name, metadata, last_seen, mention_count)
            alice_candidates = all_candidates["Alice"]
            assert len(alice_candidates) == 1
            cand = alice_candidates[0]
            assert cand[0] == "eid-1", "tuple[0] must be entity id"
            assert cand[1] == "Alice Smith", "tuple[1] must be canonical_name"
            assert cand[2] == '{"role": "eng"}', "tuple[2] must be metadata"
            assert cand[3] is None, "tuple[3] must be last_seen"
            assert cand[4] == 5, "tuple[4] must be mention_count"

            bob_candidates = all_candidates["Bob"]
            assert len(bob_candidates) == 1
            assert bob_candidates[0][0] == "eid-2"
            assert bob_candidates[0][1] == "Robert Jones"

    @pytest.mark.asyncio
    async def test_cooccurrence_query_uses_candidate_ids(self, resolver, mock_conn):
        """When candidates are found, the co-occurrence query should only fetch
        relationships for the candidate entity IDs (not all entities in the bank)."""
        candidate_rows = [
            ResultRow(
                {
                    "id": "eid-1",
                    "canonical_name": "Alice",
                    "metadata": None,
                    "last_seen": None,
                    "mention_count": 1,
                    "query_text": "Alice",
                }
            ),
        ]
        # First fetch: candidates. Second fetch: co-occurrences.
        mock_conn.fetch = AsyncMock(side_effect=[candidate_rows, []])

        with patch.object(resolver, "_resolve_from_candidates", new_callable=AsyncMock, return_value=[]):
            await resolver._resolve_entities_batch_oracle_fuzzy(
                conn=mock_conn,
                bank_id="bank-1",
                entities_data=[{"text": "Alice", "nearby_entities": [], "event_date": None}],
                unit_event_date=None,
            )

        # Second fetch call should be the co-occurrence query
        assert mock_conn.fetch.call_count == 2
        cooc_query = mock_conn.fetch.call_args_list[1].args[0]
        assert "entity_cooccurrences" in cooc_query
        # The candidate IDs should be passed as bind parameter
        cooc_bind_args = mock_conn.fetch.call_args_list[1].args[1:]
        assert "eid-1" in cooc_bind_args[0], "Co-occurrence query must receive candidate IDs"
