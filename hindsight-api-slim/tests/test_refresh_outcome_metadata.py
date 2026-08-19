"""Tests for refresh outcome metadata, specifically the SERVING TIER field.

Why this exists: a two-tier refresh system whose tier is unobservable after the fact
cannot be operated or measured. Before this field, the tier was only readable from
`mental_models.reflect_response.fast_path`, which holds only each model's LATEST
refresh and is overwritten by the next one -- so tier distribution over any window was
unrecoverable from the database, and had to be reconstructed by an external sampler
polling every 5 minutes.

These tests drive the real `_write_refresh_outcome_metadata` with a fake connection and
assert on the JSON it actually persists, rather than on the dataclass in isolation --
the mapping (`fast_path` -> `serving_tier`, with None normalised to "tier2") is the part
that can be wrong.
"""

from __future__ import annotations

import json
import uuid

import pytest

from hindsight_api.engine import memory_engine as me
from hindsight_api.engine.operation_metadata import RefreshMentalModelOutcomeMetadata


class _FakeConn:
    """Captures the parameters of the UPDATE the writer issues."""

    def __init__(self) -> None:
        self.executed: list[tuple] = []

    async def execute(self, sql: str, *params):
        self.executed.append((sql, params))


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        return False


class _StubEngine:
    """Minimal stand-in carrying only what the writer touches."""

    async def _get_backend(self):
        return object()


async def _write(monkeypatch, refreshed: dict) -> dict:
    """Run the real writer against a fake conn; return the metadata JSON it persisted."""
    conn = _FakeConn()
    monkeypatch.setattr(me, "acquire_with_retry", lambda backend: _FakeAcquire(conn))
    op_id = str(uuid.uuid4())
    await me.MemoryEngine._write_refresh_outcome_metadata(_StubEngine(), op_id, refreshed)
    assert conn.executed, "writer issued no UPDATE"
    _sql, params = conn.executed[-1]
    return json.loads(params[1])


@pytest.mark.asyncio
async def test_tier0_is_recorded(monkeypatch):
    meta = await _write(
        monkeypatch,
        {
            "content": "preserved document",
            "reflect_response": {"fast_path": "tier0", "fast_path_fallback_reason": None},
        },
    )
    assert meta["serving_tier"] == "tier0"
    assert meta["fast_path_fallback_reason"] is None


@pytest.mark.asyncio
async def test_tier1_is_recorded(monkeypatch):
    meta = await _write(
        monkeypatch,
        {
            "content": "edited document",
            "reflect_response": {
                "fast_path": "tier1",
                "fast_path_fallback_reason": None,
                "delta_operations_applied": [{"op": "replace_block"}, {"op": "append_block"}],
                "delta_operations_skipped": [{"op": "replace_block"}],
            },
        },
    )
    assert meta["serving_tier"] == "tier1"
    assert meta["delta_ops_applied"] == 2
    assert meta["delta_ops_skipped"] == 1


@pytest.mark.asyncio
async def test_agentic_loop_normalises_to_tier2_with_its_reason(monkeypatch):
    """The load-bearing case: `fast_path` is None on the agentic path.

    Persisting that null verbatim would be ambiguous between "the agentic loop ran"
    and "written by a build predating this field", so it is normalised to "tier2" and
    the hand-back reason is carried alongside.
    """
    meta = await _write(
        monkeypatch,
        {
            "content": "regenerated document",
            "reflect_response": {"fast_path": None, "fast_path_fallback_reason": "needs_full_context"},
        },
    )
    assert meta["serving_tier"] == "tier2"
    assert meta["fast_path_fallback_reason"] == "needs_full_context"


@pytest.mark.asyncio
async def test_missing_fast_path_key_still_yields_a_tier(monkeypatch):
    """A reflect_response from an older build has no `fast_path` key at all."""
    meta = await _write(monkeypatch, {"content": "doc", "reflect_response": {}})
    assert meta["serving_tier"] == "tier2"


@pytest.mark.asyncio
async def test_no_operation_id_is_a_noop(monkeypatch):
    """Guard the early return -- a refresh outside an operation must not raise."""
    conn = _FakeConn()
    monkeypatch.setattr(me, "acquire_with_retry", lambda backend: _FakeAcquire(conn))
    await me.MemoryEngine._write_refresh_outcome_metadata(_StubEngine(), None, {"content": "x"})
    assert conn.executed == []


def test_tier_fields_default_to_none_for_existing_callers():
    """Both fields are optional, so callers constructed before them still work."""
    meta = RefreshMentalModelOutcomeMetadata(content_len=10, populated_content=True)
    assert meta.serving_tier is None
    assert meta.fast_path_fallback_reason is None
    assert meta.to_dict()["serving_tier"] is None
