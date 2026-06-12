"""Tests for the C4 channel-B backflow endpoint + shared engine primitive.

The endpoint ``POST /v1/default/banks/{bank_id}/graphiti/edge-invalidated``
and the channel-A forwarder both call the same engine method
``MemoryEngine.handle_graphiti_edge_invalidated`` — the tests below
cover both call paths and the HTTP surface that wires them together.

Coverage:

* **HTTP happy path** — valid source_uri + enabled flag → 200 with
  ``GraphitiBackflowResult``.
* **HTTP gating** — flag off → 404 (endpoint is invisible, not 503).
* **HTTP 404 paths** — cross-bank URI, malformed URI, missing memory
  all return 404 per deep-dive 4 §1.3 ("the edge may outlive the
  memory — that's a normal outcome, not an error").
* **HTTP 422** — request body missing required fields is rejected at
  the Pydantic boundary.
* **Engine method** — direct call covers the same 404 paths plus
  the B1 supersession gating (CHECK constraint blocks writes when
  ``occurred_start`` is null; ``supersession_written`` flips to
  False instead of raising).
* **Idempotency** — calling the engine method twice with the same
  edge_uuid is a no-op the second time (step 2's ``consolidated_at
  = NULL`` no-ops when already NULL; step 4's ``valid_until IS NULL``
  guard prevents re-stomp).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.models import RequestContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def backflow_api_client(memory):
    """HTTP test client over the `memory` fixture's app.

    Mirrors ``audit_api_client`` in test_audit_log.py: just the
    FastAPI ASGI surface over a mock-LLM engine, no special audit
    wiring. The backflow endpoint writes its own fire-and-forget
    audit log via ``self.audit_logger``; tests that don't care about
    audit can ignore it.
    """
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def bank_id():
    """Unique bank_id per test to keep DB state isolated."""
    from datetime import datetime as _dt

    return f"backflow_test_{_dt.now().timestamp()}"


async def _create_bank(client: httpx.AsyncClient, bank_id: str) -> None:
    await client.put(f"/v1/default/banks/{bank_id}", json={"name": "Backflow Test Bank"})


async def _enable_backflow(client: httpx.AsyncClient, bank_id: str) -> None:
    """Set ``graphiti_backflow_enabled=true`` and ``enable_auto_consolidation=true``
    for the bank so the engine method's consolidation-submit step actually
    runs. Both flags are hierarchical (per deep-dive 4 §1.4)."""
    response = await client.patch(
        f"/v1/default/banks/{bank_id}/config",
        json={
            "updates": {
                "graphiti_backflow_enabled": True,
                "enable_auto_consolidation": True,
            }
        },
    )
    assert response.status_code == 200, response.text


async def _retain_and_get_memory_id(client: httpx.AsyncClient, bank_id: str, content: str) -> str:
    """Retain one fact and return its memory id (via recall)."""
    response = await client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": content, "context": "backflow test"}]},
    )
    assert response.status_code == 200, response.text
    # The retain response doesn't echo memory ids — recall to find
    # the one we just inserted. The mock LLM extracts the fact text
    # as-is, so a query that includes a unique substring will find it.
    response = await client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": content, "thinking_budget": 50},
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert results, f"Recall found no memories for {content!r}"
    return results[0]["id"]


def _backflow_body(
    memory_id: str, bank_id: str, *, edge_uuid: str | None = None, invalid_at: str | None = None
) -> dict:
    """Build a valid backflow request body."""
    return {
        "edge_uuid": edge_uuid or str(uuid4()),
        "source_uri": f"hindsight://bank/{bank_id}/memory/{memory_id}",
        "invalid_at": invalid_at or "2026-06-01T10:00:00Z",
        "superseded_by_fact": "Replacement fact text",
    }


# ---------------------------------------------------------------------------
# HTTP endpoint — channel B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_happy_path_returns_200_with_result(backflow_api_client, bank_id):
    """Enabled flag + valid source_uri → 200 with the backflow result."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)
    memory_id = await _retain_and_get_memory_id(backflow_api_client, bank_id, "Alice works at Acme Corp")

    response = await backflow_api_client.post(
        f"/v1/default/banks/{bank_id}/graphiti/edge-invalidated",
        json=_backflow_body(memory_id, bank_id),
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["memory_id"] == memory_id
    assert data["not_found"] is False
    # The mock LLM doesn't produce observations, so cleared count is 0;
    # the key invariant is "the call returned a typed result".
    assert data["observations_cleared"] == 0
    assert data["supersession_written"] is False  # flag is off (default)
    assert data["consolidation_submitted"] is False  # nothing to consolidate


@pytest.mark.asyncio
async def test_http_404_when_backflow_flag_disabled(backflow_api_client, bank_id):
    """Per the spec, the endpoint is invisible (404, not 503) when the
    bank hasn't opted into the channel-B wire protocol. The overlay
    should treat this as 'try again later' rather than an error."""
    await _create_bank(backflow_api_client, bank_id)
    # Note: no _enable_backflow() call — flag stays at its default (False).
    memory_id = await _retain_and_get_memory_id(backflow_api_client, bank_id, "Alice works at Acme Corp")

    response = await backflow_api_client.post(
        f"/v1/default/banks/{bank_id}/graphiti/edge-invalidated",
        json=_backflow_body(memory_id, bank_id),
    )

    assert response.status_code == 404
    # The detail mentions the env var name (the operator-facing knob).
    # We don't pin the *exact* wording to avoid being brittle to
    # message edits — just the negative invariant that the message
    # is informative and points at the right env var.
    assert "GRAPHITI_BACKFLOW_ENABLED" in response.json()["detail"]


@pytest.mark.asyncio
async def test_http_404_when_source_uri_points_at_other_bank(backflow_api_client, bank_id):
    """A source_uri whose bank prefix doesn't match the path bank → 404
    (cross-bank). The edge may have been written by another bank
    earlier in the overlay's lifetime; the per-bank handler has no
    authority over a memory that lives in a different tenant."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)
    foreign_memory = uuid4()

    response = await backflow_api_client.post(
        f"/v1/default/banks/{bank_id}/graphiti/edge-invalidated",
        json=_backflow_body(foreign_memory, "some-other-bank"),
    )
    # The Pydantic body has source_uri pointing at a different bank;
    # the handler short-circuits with 404 before touching the engine
    # method. We don't pin the detail wording — just the 404 status.
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_http_404_when_source_uri_malformed(backflow_api_client, bank_id):
    """source_uri that doesn't start with the expected prefix → 404.
    The overlay may be running an old build that used a different
    scheme; we surface that as 404, not 5xx, so the overlay can log
    and move on."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)

    body = {
        "edge_uuid": str(uuid4()),
        "source_uri": f"hindsight://wrong-prefix/bank/{bank_id}/memory/{uuid4()}",
        "invalid_at": "2026-06-01T10:00:00Z",
    }
    response = await backflow_api_client.post(
        f"/v1/default/banks/{bank_id}/graphiti/edge-invalidated",
        json=body,
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_http_404_when_memory_does_not_exist(backflow_api_client, bank_id):
    """source_uri with a well-formed UUID that points at a memory the
    bank has since deleted → 404. The edge may have outlived the
    memory; per deep-dive 4 §1.3 that's a normal outcome."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)
    deleted_memory = uuid4()

    response = await backflow_api_client.post(
        f"/v1/default/banks/{bank_id}/graphiti/edge-invalidated",
        json=_backflow_body(deleted_memory, bank_id),
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_http_422_when_required_fields_missing(backflow_api_client, bank_id):
    """Pydantic enforces the body shape at the boundary: missing
    edge_uuid / source_uri / invalid_at → 422. (We omit source_uri
    here because the schema requires all three.)"""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)

    response = await backflow_api_client.post(
        f"/v1/default/banks/{bank_id}/graphiti/edge-invalidated",
        json={"edge_uuid": str(uuid4())},  # missing source_uri, invalid_at
    )

    assert response.status_code == 422
    body = response.json()
    # FastAPI's validation error format lists missing fields under
    # ``detail[*].loc`` as a path tuple. We don't pin the exact
    # wording — just that both fields are reported.
    missing = {loc[-1] for err in body["detail"] for loc in [err["loc"]]}
    assert "source_uri" in missing
    assert "invalid_at" in missing


@pytest.mark.asyncio
async def test_http_404_when_bank_does_not_exist(backflow_api_client):
    """POSTing to a non-existent bank → 404 from the auth layer
    (no row in the banks table to authenticate against)."""
    response = await backflow_api_client.post(
        f"/v1/default/banks/never-created-{uuid4()}/graphiti/edge-invalidated",
        json={
            "edge_uuid": str(uuid4()),
            "source_uri": f"hindsight://bank/never-created-{uuid4()}/memory/{uuid4()}",
            "invalid_at": "2026-06-01T10:00:00Z",
        },
    )
    # Auth-failure path: either 401 (no token) or 404 (bank not
    # found) — either is a "not authorized / not found" outcome,
    # not a 5xx. Pin the *negative* invariant.
    assert response.status_code in (401, 403, 404)


# ---------------------------------------------------------------------------
# Engine method — direct call (covers both channel A and channel B paths)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_method_happy_path(backflow_api_client, bank_id, memory):
    """Direct call to the shared engine primitive → result with the
    resolved memory_id and not_found=False. Same path channel A and
    channel B both take after the HTTP / forwarder boundary."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)
    memory_id = await _retain_and_get_memory_id(backflow_api_client, bank_id, "Alice works at Acme Corp")

    result = await memory.handle_graphiti_edge_invalidated(
        bank_id=bank_id,
        edge_uuid=str(uuid4()),
        source_uri=f"hindsight://bank/{bank_id}/memory/{memory_id}",
        invalid_at=datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        request_context=RequestContext(internal=True),
    )

    assert result.not_found is False
    assert result.memory_id == memory_id
    assert result.supersession_written is False  # flag off by default


@pytest.mark.asyncio
async def test_engine_method_cross_bank_source_uri_returns_not_found(backflow_api_client, bank_id, memory):
    """A source_uri whose bank prefix doesn't match the bank_id the
    call is being made for → ``not_found=True`` (no 5xx). The edge
    may have been written by another bank earlier in the overlay's
    lifetime; the per-bank handler has no authority over it."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)

    result = await memory.handle_graphiti_edge_invalidated(
        bank_id=bank_id,
        edge_uuid=str(uuid4()),
        source_uri=f"hindsight://bank/some-other-bank/memory/{uuid4()}",
        invalid_at=datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        request_context=RequestContext(internal=True),
    )

    assert result.not_found is True
    assert result.memory_id is None


@pytest.mark.asyncio
async def test_engine_method_malformed_source_uri_returns_not_found(backflow_api_client, bank_id, memory):
    """Empty source_uri, missing prefix, or unparseable UUID → all
    map to ``not_found=True``. The overlay may be running an old
    build that used a different scheme; we don't 5xx on contract
    drift, we just log + skip (the audit log captures the attempt)."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)

    for bad_uri in (
        "",  # empty
        f"hindsight://bank/{bank_id}/memory/not-a-uuid",  # unparseable UUID
        f"http://some-other-scheme/{bank_id}/memory/{uuid4()}",  # wrong scheme
    ):
        result = await memory.handle_graphiti_edge_invalidated(
            bank_id=bank_id,
            edge_uuid=str(uuid4()),
            source_uri=bad_uri,
            invalid_at=datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            request_context=RequestContext(internal=True),
        )
        assert result.not_found is True, f"expected not_found for {bad_uri!r}"


@pytest.mark.asyncio
async def test_engine_method_missing_memory_returns_not_found(backflow_api_client, bank_id, memory):
    """source_uri with a well-formed UUID that points at a memory the
    bank has since deleted → ``not_found=True``. The edge may have
    outlived the memory — that's a normal outcome."""
    await _create_bank(backflow_api_client, bank_id)
    await _enable_backflow(backflow_api_client, bank_id)
    deleted_memory = uuid4()

    result = await memory.handle_graphiti_edge_invalidated(
        bank_id=bank_id,
        edge_uuid=str(uuid4()),
        source_uri=f"hindsight://bank/{bank_id}/memory/{deleted_memory}",
        invalid_at=datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        request_context=RequestContext(internal=True),
    )

    assert result.not_found is True


@pytest.mark.asyncio
async def test_engine_method_idempotent_on_replay(backflow_api_client, bank_id, memory):
    """Calling the same edge twice → second call is a no-op. This
    is the "at-least-once replay" safety net: if a transient DB blip
    causes the forwarder's end-of-drain to retry, or the overlay
    retries after a network hiccup, the second call must not double-
    clear observations or stomp on a fresh supersession row.

    We assert via ``supersession_written``: enable the flag, set up
    a memory with ``occurred_start`` and a derived observation (the
    engine gates step 4 on ``observations_cleared > 0``), call twice,
    expect True once and False the second time (the ``valid_until IS
    NULL`` guard blocked the second write)."""
    from hindsight_api.engine.memory_engine import fq_table

    await _create_bank(backflow_api_client, bank_id)
    # Enable BOTH flags: backflow enabled (gate) AND supersession on
    # (so step 4 actually fires).
    response = await backflow_api_client.patch(
        f"/v1/default/banks/{bank_id}/config",
        json={
            "updates": {
                "graphiti_backflow_enabled": True,
                "graphiti_backflow_supersession": True,
                "enable_auto_consolidation": True,
            }
        },
    )
    assert response.status_code == 200, response.text
    memory_id = await _retain_and_get_memory_id(backflow_api_client, bank_id, "Alice works at Acme Corp")

    backend = await memory._get_backend()
    # Pin occurred_start directly so the B1 CHECK constraint
    # ``chk_mu_valid_until_after_start`` allows the supersession
    # write. The mock LLM doesn't set this, so we set it via SQL.
    # Also insert a derived observation so the engine's gate
    # ``if observations_cleared > 0 and graphiti_backflow_supersession``
    # actually fires — the mock LLM doesn't produce observations.
    # Observations live in ``memory_units`` with ``fact_type =
    # 'observation'`` and ``source_memory_ids`` as a UUID array
    # (mirrors what consolidation writes).
    async with acquire_with_retry(backend) as conn:
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET occurred_start = $1 WHERE id = $2",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            memory_id,
        )
        await conn.execute(
            f"""
            INSERT INTO {fq_table("memory_units")}
                (bank_id, text, fact_type, source_memory_ids, event_date, mentioned_at)
            VALUES ($1, $2, 'observation', $3::uuid[], now(), now())
            """,
            bank_id,
            "Alice works at Acme Corp (derived observation)",
            [UUID(memory_id)],
        )

    edge_uuid = str(uuid4())
    source_uri = f"hindsight://bank/{bank_id}/memory/{memory_id}"
    invalid_at = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    ctx = RequestContext(internal=True)

    first = await memory.handle_graphiti_edge_invalidated(
        bank_id=bank_id,
        edge_uuid=edge_uuid,
        source_uri=source_uri,
        invalid_at=invalid_at,
        request_context=ctx,
    )
    assert first.not_found is False
    assert first.observations_cleared == 1
    assert first.supersession_written is True  # flag is on + memory has occurred_start + obs was cleared

    second = await memory.handle_graphiti_edge_invalidated(
        bank_id=bank_id,
        edge_uuid=edge_uuid,
        source_uri=source_uri,
        invalid_at=invalid_at,
        request_context=ctx,
    )
    # The second call must be a no-op for the supersession write —
    # the row's valid_until is now set, so the ``valid_until IS NULL``
    # guard in the UPDATE blocks the second write. We do NOT want
    # ``supersession_written`` to flip to True again (that would be
    # "we wrote twice", which is exactly the bug idempotency guards
    # against). ``observations_cleared`` is also 0 the second time
    # (the observation was already removed by the first call).
    assert second.supersession_written is False
    assert second.observations_cleared == 0
    # And the memory row's valid_until is set (with the right invalid_at
    # value), but ``superseded_by`` is NULL — the C4 path doesn't write
    # the Graphiti edge UUID there because the B1 FK constrains that
    # column to memory_units.id. The edge_uuid is recoverable from the
    # audit log metadata; the row-level supersession verdict survives
    # with the pointer cleared.
    async with acquire_with_retry(backend) as conn:
        row = await conn.fetchrow(
            f"SELECT superseded_by, valid_until FROM {fq_table('memory_units')} WHERE id = $1",
            memory_id,
        )
    assert row["superseded_by"] is None
    assert row["valid_until"] is not None


@pytest.mark.asyncio
async def test_engine_method_check_constraint_blocks_supersession_without_occurred_start(
    backflow_api_client, bank_id, memory
):
    """Memory without ``occurred_start`` + ``graphiti_backflow_supersession=on``
    → ``supersession_written=False`` (the DB CHECK constraint
    ``chk_mu_valid_until_after_start`` would block the write; the
    engine method pre-checks client-side and flips the flag instead
    of letting the UPDATE error)."""
    await _create_bank(backflow_api_client, bank_id)
    response = await backflow_api_client.patch(
        f"/v1/default/banks/{bank_id}/config",
        json={
            "updates": {
                "graphiti_backflow_enabled": True,
                "graphiti_backflow_supersession": True,
                "enable_auto_consolidation": True,
            }
        },
    )
    assert response.status_code == 200, response.text
    memory_id = await _retain_and_get_memory_id(backflow_api_client, bank_id, "Alice works at Acme Corp")

    # Confirm the memory has no occurred_start (the mock LLM doesn't
    # set it; we want a clean NULL).
    backend = await memory._get_backend()
    from hindsight_api.engine.memory_engine import fq_table

    async with acquire_with_retry(backend) as conn:
        row = await conn.fetchrow(
            f"SELECT occurred_start FROM {fq_table('memory_units')} WHERE id = $1",
            memory_id,
        )
    assert row["occurred_start"] is None

    result = await memory.handle_graphiti_edge_invalidated(
        bank_id=bank_id,
        edge_uuid=str(uuid4()),
        source_uri=f"hindsight://bank/{bank_id}/memory/{memory_id}",
        invalid_at=datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        request_context=RequestContext(internal=True),
    )

    assert result.not_found is False
    # The CHECK constraint blocks the write; the engine flips the flag
    # to False instead of letting the UPDATE error out.
    assert result.supersession_written is False
