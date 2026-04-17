"""
Unit tests for EntityResolver pg_trgm auto-detection (PR #626/#649).

These tests verify:
1. When entity_lookup="trigram" and pg_trgm IS available, the trigram path is used.
2. When entity_lookup="trigram" and pg_trgm is NOT available, the resolver falls back
   to entity_lookup="full" and uses the full-scan path.
3. The pg_trgm check is only performed once (_pg_trgm_checked flag prevents re-checking).
4. When entity_lookup="full" from the start, the trgm check is never performed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.entity_resolver import EntityResolver


def _make_conn(pg_trgm_available: bool) -> MagicMock:
    """Create a minimal mock asyncpg connection for the pg_trgm availability check."""
    conn = MagicMock()
    # Must set backend_type explicitly — MagicMock returns a truthy Mock for
    # any attribute, so getattr(conn, "backend_type", ...) would return a Mock
    # instead of the default, causing the Oracle dispatch path to trigger.
    conn.backend_type = "postgresql"
    conn.fetchval = AsyncMock(return_value=pg_trgm_available)
    conn.fetch = AsyncMock(return_value=[])
    conn.executemany = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


def _make_resolver(entity_lookup: str = "trigram") -> EntityResolver:
    """Return an EntityResolver with a mock pool (only ops attribute is needed)."""
    pool = MagicMock()
    return EntityResolver(pool=pool, entity_lookup=entity_lookup)  # type: ignore[arg-type]


class TestPgTrgmAutoDetection:
    """Unit tests for pg_trgm detection logic inside _resolve_entities_batch_impl."""

    @pytest.mark.asyncio
    async def test_falls_back_to_full_when_pg_trgm_unavailable(self):
        """When pg_trgm is absent the resolver switches to 'full' and calls the full-scan path."""
        resolver = _make_resolver(entity_lookup="trigram")
        conn = _make_conn(pg_trgm_available=False)

        with (
            patch.object(resolver, "_resolve_entities_batch_full", new=AsyncMock(return_value=[])) as mock_full,
            patch.object(resolver, "_resolve_entities_batch_trigram", new=AsyncMock(return_value=[])) as mock_trgm,
        ):
            await resolver._resolve_entities_batch_impl(
                conn=conn,
                bank_id="test-bank",
                entities_data=[],
                context="",
                unit_event_date=None,
            )

        # Trigram path must NOT be called
        mock_trgm.assert_not_called()
        # Full-scan path must be called as the fallback
        mock_full.assert_called_once()
        # Strategy is permanently downgraded
        assert resolver.entity_lookup == "full"
        assert resolver._pg_trgm_checked is True

    @pytest.mark.asyncio
    async def test_uses_trigram_when_pg_trgm_available(self):
        """When pg_trgm is present the trigram path is used."""
        resolver = _make_resolver(entity_lookup="trigram")
        conn = _make_conn(pg_trgm_available=True)

        with (
            patch.object(resolver, "_resolve_entities_batch_full", new=AsyncMock(return_value=[])) as mock_full,
            patch.object(resolver, "_resolve_entities_batch_trigram", new=AsyncMock(return_value=[])) as mock_trgm,
        ):
            await resolver._resolve_entities_batch_impl(
                conn=conn,
                bank_id="test-bank",
                entities_data=[],
                context="",
                unit_event_date=None,
            )

        mock_trgm.assert_called_once()
        mock_full.assert_not_called()
        assert resolver.entity_lookup == "trigram"
        assert resolver._pg_trgm_checked is True

    @pytest.mark.asyncio
    async def test_pg_trgm_check_performed_only_once(self):
        """The fetchval check is only issued on the first call; subsequent calls skip it."""
        resolver = _make_resolver(entity_lookup="trigram")
        conn = _make_conn(pg_trgm_available=True)

        with patch.object(resolver, "_resolve_entities_batch_trigram", new=AsyncMock(return_value=[])):
            # First call — check is issued
            await resolver._resolve_entities_batch_impl(
                conn=conn,
                bank_id="test-bank",
                entities_data=[],
                context="",
                unit_event_date=None,
            )
            # Second call — check must NOT be issued again
            await resolver._resolve_entities_batch_impl(
                conn=conn,
                bank_id="test-bank",
                entities_data=[],
                context="",
                unit_event_date=None,
            )

        # fetchval (the pg_trgm availability query) should be called exactly once
        assert conn.fetchval.call_count == 1

    @pytest.mark.asyncio
    async def test_full_strategy_skips_pg_trgm_check(self):
        """When entity_lookup='full' from the start, no pg_trgm check is ever issued."""
        resolver = _make_resolver(entity_lookup="full")
        conn = _make_conn(pg_trgm_available=False)

        with patch.object(resolver, "_resolve_entities_batch_full", new=AsyncMock(return_value=[])):
            await resolver._resolve_entities_batch_impl(
                conn=conn,
                bank_id="test-bank",
                entities_data=[],
                context="",
                unit_event_date=None,
            )

        # fetchval should never be called when entity_lookup is already "full"
        conn.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_is_sticky_across_calls(self):
        """After falling back to 'full', subsequent calls also use the full path."""
        resolver = _make_resolver(entity_lookup="trigram")
        conn = _make_conn(pg_trgm_available=False)

        with (
            patch.object(resolver, "_resolve_entities_batch_full", new=AsyncMock(return_value=[])) as mock_full,
            patch.object(resolver, "_resolve_entities_batch_trigram", new=AsyncMock(return_value=[])) as mock_trgm,
        ):
            # First call triggers the fallback
            await resolver._resolve_entities_batch_impl(
                conn=conn,
                bank_id="b",
                entities_data=[],
                context="",
                unit_event_date=None,
            )
            # Second call — _pg_trgm_checked is True so no re-check; entity_lookup=="full"
            await resolver._resolve_entities_batch_impl(
                conn=conn,
                bank_id="b",
                entities_data=[],
                context="",
                unit_event_date=None,
            )

        # Trigram path is never called
        mock_trgm.assert_not_called()
        # Full-scan path is called both times
        assert mock_full.call_count == 2
        # pg_trgm check was issued exactly once
        assert conn.fetchval.call_count == 1
