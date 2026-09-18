"""The ``time_field`` / ``start_date`` / ``end_date`` window on the two list surfaces (#4349).

Three things are worth a test here, and they are the three that would silently
regress:

* ``total`` counts the WINDOW, not the bank. A page filtered after the fact
  reports the unfiltered total and every client paginating on it walks off the
  end — the exact bug the tags filter shipped with once already.
* Rows with no value on the chosen axis are DROPPED, not sorted last. This is the
  semantic the user chose over the ``COALESCE(..., created_at)`` fallback that
  ``memories-timeseries`` uses, and nothing else in the codebase encodes it.
* The invalidated archive reads through the same builder against a different
  table, so the window has to work there too — it is a different ``source_table``,
  not a different predicate, and that is easy to forget when touching the query.

The whitelist/validation half is a pure function and is tested without a database
at the bottom.
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_processing
from hindsight_api.engine.time_filter import (
    DOCUMENT_TIME_FIELDS,
    MEMORY_TIME_FIELDS,
    build_time_clause,
    validate_time_window,
)

# Seeds `memory_units` directly to build the time matrix — see the note in
# test_list_memory_units_filters.py for why an alternative store cannot see it.
pytestmark = pytest.mark.memory_backend_incompatible


async def _seed_unit(
    conn,
    memory: MemoryEngine,
    bank_id: str,
    text: str,
    *,
    created_at: datetime,
    mentioned_at: datetime | None,
) -> str:
    """Insert a live memory unit with explicit ingest and event timestamps.

    ``mentioned_at=None`` is the case the whole feature turns on: a memory nobody
    dated.
    """
    mem_id = uuid.uuid4()
    emb = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    await conn.execute(
        """
        INSERT INTO memory_units (
            id, bank_id, text, fact_type, embedding, event_date,
            mentioned_at, created_at, updated_at, consolidated_at
        )
        VALUES ($1, $2, $3, 'experience', $4::vector, NOW(), $5, $6, $6, $6)
        """,
        mem_id,
        bank_id,
        text,
        str(emb[0]),
        mentioned_at,
        created_at,
    )
    return str(mem_id)


def _ids(result: dict) -> list[str]:
    return [item["id"] for item in result["items"]]


@pytest.mark.asyncio
async def test_memories_window_filters_orders_and_drops_undated(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-window-mem-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    now = datetime.now(UTC)

    try:
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            # All four share an ingest time, so only `mentioned_at` separates them —
            # which is the point: the axis has to be the one the caller named.
            inside_old = await _seed_unit(
                conn, memory, bank_id, "mentioned ten days ago", created_at=now, mentioned_at=now - timedelta(days=10)
            )
            inside_new = await _seed_unit(
                conn, memory, bank_id, "mentioned two days ago", created_at=now, mentioned_at=now - timedelta(days=2)
            )
            await _seed_unit(
                conn, memory, bank_id, "mentioned a year ago", created_at=now, mentioned_at=now - timedelta(days=365)
            )
            undated = await _seed_unit(conn, memory, bank_id, "never dated", created_at=now, mentioned_at=None)

        windowed = await memory.list_memory_units(
            bank_id,
            time_field="mentioned_at",
            start_date=now - timedelta(days=30),
            end_date=now,
            request_context=request_context,
        )
        assert _ids(windowed) == [inside_new, inside_old], "ordered by the named axis, newest first"
        assert windowed["total"] == 2, "total counts the window, not the bank"
        assert undated not in _ids(windowed)

        # `time_field` alone, no bounds, still drops the undated row — ordering by a
        # column a row has no value for cannot place it anywhere honest.
        no_bounds = await memory.list_memory_units(bank_id, time_field="mentioned_at", request_context=request_context)
        assert undated not in _ids(no_bounds)
        assert no_bounds["total"] == 3

        # The default listing is untouched: no time parameter, nothing dropped.
        default = await memory.list_memory_units(bank_id, request_context=request_context)
        assert default["total"] == 4
        assert undated in _ids(default)

        # A bank whose memories carry no value on the axis answers 0 rather than
        # falling back to created_at. Documented on the endpoint; asserted here so
        # nobody "fixes" it into a COALESCE.
        empty_axis = await memory.list_memory_units(
            bank_id, time_field="occurred_start", request_context=request_context
        )
        assert empty_axis["total"] == 0
        assert empty_axis["items"] == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_the_window_composes_with_the_other_filters(memory: MemoryEngine, request_context: RequestContext):
    """Tags, search and the window share one ``$N`` counter across two clause builders.

    Nothing else exercises that handoff: each builder takes ``param_offset`` from
    where the last one stopped, and the count query and the page query bind the
    same list. Get the order wrong and asyncpg fails at runtime — not at import,
    and not in any test that uses the filters one at a time.
    """
    bank_id = f"test-window-combo-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    now = datetime.now(UTC)

    try:
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            wanted = await _seed_unit(
                conn, memory, bank_id, "kangaroo in the window", created_at=now, mentioned_at=now - timedelta(days=2)
            )
            # Right text and tag, wrong window.
            await _seed_unit(
                conn, memory, bank_id, "kangaroo long ago", created_at=now, mentioned_at=now - timedelta(days=400)
            )
            # Right window, wrong text.
            await _seed_unit(
                conn, memory, bank_id, "wombat in the window", created_at=now, mentioned_at=now - timedelta(days=2)
            )
            await conn.execute(
                "UPDATE memory_units SET tags = ARRAY['zoo'] WHERE bank_id = $1 AND text LIKE 'kangaroo%'",
                bank_id,
            )

        combined = await memory.list_memory_units(
            bank_id,
            search_query="kangaroo",
            tags=["zoo"],
            tags_match="any_strict",
            time_field="mentioned_at",
            start_date=now - timedelta(days=30),
            end_date=now,
            request_context=request_context,
        )
        assert _ids(combined) == [wanted]
        assert combined["total"] == 1, "the count query must bind the same params as the page query"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_memories_window_applies_to_invalidated_archive(memory: MemoryEngine, request_context: RequestContext):
    """The archive is a different table read through the same builder."""
    bank_id = f"test-window-arch-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    now = datetime.now(UTC)

    try:
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            recent = await _seed_unit(
                conn, memory, bank_id, "archived, recent", created_at=now, mentioned_at=now - timedelta(days=2)
            )
            stale = await _seed_unit(
                conn, memory, bank_id, "archived, ancient", created_at=now, mentioned_at=now - timedelta(days=400)
            )

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            for mem_id in (recent, stale):
                await memory.update_memory_unit(
                    bank_id, mem_id, state="invalidated", reason="test", request_context=request_context
                )

        archived = await memory.list_memory_units(
            bank_id,
            state="invalidated",
            time_field="mentioned_at",
            start_date=now - timedelta(days=30),
            request_context=request_context,
        )
        assert _ids(archived) == [recent]
        assert archived["total"] == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_documents_window_filters_and_counts(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-window-doc-{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)

    try:
        # Four documents, each excluded by a different one of the three filters below,
        # so no filter can be dropped or mis-bound without changing the result.
        for doc_id in ("report-old", "report-new", "report-untagged", "memo-recent"):
            # Gibberish so fact extraction finds nothing; the document row is what matters.
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[{"content": f"xyzabc123 !@# $$$ {doc_id}"}],
                document_id=doc_id,
                document_tags=None if doc_id == "report-untagged" else ["shelf"],
                request_context=request_context,
            )

        # Backdate one document's ingest time; `created_at` and `updated_at` are
        # separate axes and the filter must honour whichever was asked for.
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET created_at = $1 WHERE bank_id = $2 AND id = 'report-old'",
                now - timedelta(days=90),
                bank_id,
            )

        recent = await memory.list_documents(
            bank_id=bank_id,
            time_field="created_at",
            start_date=now - timedelta(days=7),
            request_context=request_context,
        )
        assert {d["id"] for d in recent["items"]} == {"report-new", "report-untagged", "memo-recent"}
        assert recent["total"] == 3

        # This builder is the trickier of the two: the tags SQL is appended to the WHERE
        # text AFTER the window conditions, while its params go in BEFORE them. That is
        # only correct because asyncpg binds by `$N` rather than by textual position, so
        # search + tags + window together are what prove the handoff. `memo-recent` fails
        # the search, `report-untagged` the tags, `report-old` the window.
        combined = await memory.list_documents(
            bank_id=bank_id,
            search_query="report",
            tags=["shelf"],
            tags_match="any_strict",
            time_field="created_at",
            start_date=now - timedelta(days=7),
            request_context=request_context,
        )
        assert [d["id"] for d in combined["items"]] == ["report-new"]
        assert combined["total"] == 1, "the count query must bind the same params as the page query"

        # All four are still there on the untouched default listing.
        assert (await memory.list_documents(bank_id=bank_id, request_context=request_context))["total"] == 4
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- http


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["memories/list", "documents"])
async def test_http_rejects_bad_dates_and_fields(api_client, memory, request_context, path):
    """Garbage in a query parameter is a 4xx, never a 500.

    Before #802 these parameters were dropped silently; the fix made that visible
    but not usable. Now they work — and the failure modes have to be legible, or
    a caller reading `X-Ignored-Params` swaps one confusing answer for another.
    """
    bank_id = f"test-window-http-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    base = f"/v1/default/banks/{bank_id}/{path}"

    try:
        malformed = await api_client.get(base, params={"start_date": "last tuesday"})
        assert malformed.status_code == 400
        assert "start_date" in malformed.json()["detail"]

        # FastAPI validates the enum from the Literal, so an unknown axis never
        # reaches the SQL builder — and the 422 names the ones that do exist.
        unknown_axis = await api_client.get(base, params={"time_field": "whenever"})
        assert unknown_axis.status_code == 422

        inverted = await api_client.get(
            base, params={"start_date": "2024-02-01T00:00:00Z", "end_date": "2024-01-01T00:00:00Z"}
        )
        assert inverted.status_code == 400

        # A bound with no offset is read as UTC. Left naive it would either compare
        # against the tz-aware bound and raise TypeError — which is not ValueError,
        # so it escaped the 400 path and surfaced as a 500 — or reach a timestamptz
        # column meaning whatever zone the server assumes.
        mixed = await api_client.get(base, params={"start_date": "2024-01-01", "end_date": "2024-06-01T00:00:00Z"})
        assert mixed.status_code == 200, mixed.text
        both_naive = await api_client.get(base, params={"start_date": "2024-01-01T00:00:00"})
        assert both_naive.status_code == 200

        # The parameters are known now, so nothing reports them as ignored.
        ok = await api_client.get(base, params={"start_date": "2024-01-01T00:00:00Z"})
        assert ok.status_code == 200
        assert "start_date" not in ok.headers.get("X-Ignored-Params", "")
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- pure


def test_no_time_parameters_leaves_the_query_alone():
    clause = build_time_clause(
        time_field=None,
        start_date=None,
        end_date=None,
        allowed=MEMORY_TIME_FIELDS,
        default_field="created_at",
    )
    assert clause.conditions == []
    assert clause.params == []
    assert clause.order_by is None, "None is what tells the caller to keep today's ORDER BY"


def test_bounds_without_time_field_use_the_default_axis():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    clause = build_time_clause(
        time_field=None,
        start_date=start,
        end_date=None,
        allowed=DOCUMENT_TIME_FIELDS,
        default_field="updated_at",
        param_offset=3,
    )
    assert clause.conditions == ["updated_at IS NOT NULL", "updated_at >= $3"]
    assert clause.params == [start]
    assert clause.next_param_offset == 4
    assert clause.order_by == "updated_at DESC, id"


def test_unknown_time_field_is_rejected_not_silently_swapped():
    # `time_field` is interpolated into SQL. Rejecting is both the safe answer and
    # the honest one — a list that quietly answers about another column is wrong,
    # not lenient.
    with pytest.raises(ValueError, match="Invalid time_field"):
        build_time_clause(
            time_field="event_date",  # real column, deliberately not exposed
            start_date=None,
            end_date=None,
            allowed=MEMORY_TIME_FIELDS,
            default_field="created_at",
        )
    with pytest.raises(ValueError, match="Invalid time_field"):
        build_time_clause(
            time_field="created_at); DROP TABLE memory_units; --",
            start_date=None,
            end_date=None,
            allowed=MEMORY_TIME_FIELDS,
            default_field="created_at",
        )


def test_the_store_owned_branch_validates_too():
    """`list_documents` returns to a store that owns its documents before any clause
    is built, so validation has to sit in front of that branch — otherwise an
    inverted window is a 400 on Postgres and a silently empty page elsewhere."""
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        validate_time_window(
            time_field=None,
            start_date=datetime(2024, 2, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, tzinfo=UTC),
            allowed=DOCUMENT_TIME_FIELDS,
        )
    with pytest.raises(ValueError, match="Invalid time_field"):
        validate_time_window(
            time_field="mentioned_at",  # a memory axis; documents have no such column
            start_date=None,
            end_date=None,
            allowed=DOCUMENT_TIME_FIELDS,
        )


def test_inverted_window_is_rejected():
    # Silently returning nothing would look like "the bank is empty" to the caller.
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        build_time_clause(
            time_field="created_at",
            start_date=datetime(2024, 2, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, tzinfo=UTC),
            allowed=MEMORY_TIME_FIELDS,
            default_field="created_at",
        )
