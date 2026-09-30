"""Bank aliases: extra ids a bank answers to, so an id migration needs no downtime.

Renaming a bank rewrites ``bank_id`` in every table and requires stopping its
clients first — anything still calling the old id gets a 404, or silently
auto-creates a fresh empty bank under it. An alias is the zero-downtime shape of
the same job: the bank keeps its id and its rows and answers to a new one too, so
callers move over in phases.

The cases below cover the two halves that can go wrong independently:
*routing* (an aliased id reaches the bank's real data, everywhere) and
*uniqueness* (one name can never reach two banks).
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def client(memory):
    import httpx

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def bank(memory):
    """A real bank with one document in it, so recall through an alias has data to find."""
    bank_id = f"alias-src-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=RequestContext())
    return bank_id


def _aliases(resp) -> list[str]:
    assert resp.status_code in (200, 201), resp.text
    return [a["alias"] for a in resp.json()["aliases"]]


def _primary(resp) -> str | None:
    """The alias a bank is presented under, or None when it shows its own id."""
    assert resp.status_code in (200, 201), resp.text
    return next((a["alias"] for a in resp.json()["aliases"] if a["primary"]), None)


@pytest.mark.asyncio
async def test_alias_reaches_the_same_bank(client, bank):
    """The point of the feature: a second id, same bank, same data."""
    alias = f"alias-new-{uuid.uuid4().hex[:8]}"
    assert _aliases(await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})) == [alias]

    # Reached through the alias, the bank reports its own id -- the route class
    # rewrote the path param, so nothing below it ever saw the alias.
    resp = await client.get(f"/v1/default/banks/{alias}/aliases")
    assert resp.status_code == 200
    assert resp.json()["bank_id"] == bank
    assert _aliases(resp) == [alias]


@pytest.mark.asyncio
async def test_alias_sees_the_banks_memories(client, memory, bank):
    """Routing has to hold for the data endpoints, not just the bank's own metadata."""
    await memory.retain_batch_async(
        bank_id=bank,
        contents=[{"content": "The deployment runs on Tuesdays."}],
        document_id="ops",
        request_context=RequestContext(),
    )
    alias = f"alias-new-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    via_alias = await client.get(f"/v1/default/banks/{alias}/memories/list")
    via_real = await client.get(f"/v1/default/banks/{bank}/memories/list")
    assert via_alias.status_code == 200
    assert via_alias.json()["total"] == via_real.json()["total"] > 0


@pytest.mark.asyncio
async def test_a_bank_can_hold_several_aliases(client, bank):
    """A phased migration may run more than one old id at a time."""
    first, second = f"a1-{uuid.uuid4().hex[:8]}", f"a2-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": first})
    assert _aliases(await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": second})) == [first, second]
    # And each one routes.
    for name in (first, second):
        assert (await client.get(f"/v1/default/banks/{name}/aliases")).json()["bank_id"] == bank


@pytest.mark.asyncio
async def test_one_alias_cannot_reach_two_banks(client, memory, bank):
    """The PRIMARY KEY on ``alias`` is what enforces this -- no read-then-write to race."""
    other = f"alias-other-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(other, request_context=RequestContext())
    alias = f"a-{uuid.uuid4().hex[:8]}"

    assert (await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})).status_code == 201
    resp = await client.post(f"/v1/default/banks/{other}/aliases", json={"alias": alias})
    assert resp.status_code == 409
    # The first bank keeps it, so the loser changed nothing.
    assert (await client.get(f"/v1/default/banks/{alias}/aliases")).json()["bank_id"] == bank


@pytest.mark.asyncio
async def test_an_alias_cannot_shadow_an_existing_bank(client, memory, bank):
    """Refused in the same statement that inserts it: the two tables cannot share
    a constraint, so each write path checks the other."""
    occupied = f"alias-taken-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(occupied, request_context=RequestContext())

    resp = await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": occupied})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_a_bank_cannot_be_created_under_an_alias(client, memory, bank):
    """The mirror check. Without it the new bank would be reachable only until the
    next resolve, since a real bank always wins over an alias."""
    from hindsight_api.extensions import OperationValidationError

    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    with pytest.raises(OperationValidationError) as exc:
        await memory.ensure_bank_profile(alias, request_context=RequestContext())
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_removing_an_alias_stops_it_routing_and_keeps_the_bank(client, memory, bank):
    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    assert _aliases(await client.delete(f"/v1/default/banks/{bank}/aliases/{alias}")) == []
    # The alias is now just an unknown id again, and the bank is untouched.
    assert (await client.get(f"/v1/default/banks/{alias}/memories/list")).status_code == 404
    assert (await client.get(f"/v1/default/banks/{bank}/aliases")).status_code == 200


@pytest.mark.asyncio
async def test_deleting_an_alias_of_another_bank_is_a_404(client, memory, bank):
    """Scoped to the bank on purpose: reaching a bank must not let a caller detach
    a name from one it never reached."""
    other = f"alias-other-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(other, request_context=RequestContext())
    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{other}/aliases", json={"alias": alias})

    assert (await client.delete(f"/v1/default/banks/{bank}/aliases/{alias}")).status_code == 404
    assert (await client.get(f"/v1/default/banks/{alias}/aliases")).json()["bank_id"] == other


@pytest.mark.asyncio
async def test_deleting_the_bank_takes_its_aliases_with_it(client, memory, bank):
    """ON DELETE CASCADE, so a deleted bank's names are free again rather than
    left pointing at nothing."""
    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    await memory.delete_bank(bank, request_context=RequestContext())
    assert (await client.get(f"/v1/default/banks/{alias}/aliases")).status_code == 404


@pytest.mark.asyncio
async def test_a_bank_cannot_alias_itself(client, bank):
    assert (await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": bank})).status_code == 400


@pytest.mark.asyncio
async def test_an_alias_obeys_the_bank_id_rules(client, bank):
    """An alias is used exactly like a bank id, so it lives under the same limits."""
    assert (await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": ""})).status_code == 400
    too_long = "x" * 200
    assert (await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": too_long})).status_code == 400


@pytest.mark.asyncio
async def test_the_bank_list_can_be_searched_by_alias(client, memory, bank):
    """Mid-migration the new id may be the only one someone knows, so a picker
    that searches only bank_id cannot find the bank the id already reaches."""
    alias = f"findme-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    found = await client.get("/v1/default/banks", params={"q": alias})
    assert found.status_code == 200
    assert [b["bank_id"] for b in found.json()["banks"]] == [bank]

    # And a name that is nobody's alias still matches nothing, so the new clause
    # widened the search rather than loosening it.
    assert (await client.get("/v1/default/banks", params={"q": f"no-{alias}"})).json()["banks"] == []


@pytest.mark.asyncio
async def test_only_a_bank_found_via_an_alias_is_labelled_with_one(client, memory, bank):
    """``matched_aliases`` explains a surprising hit, so a row that already contains
    the search text must not be annotated — otherwise searching a bank's real id
    tags it with every alias that happens to share the prefix."""
    alias = f"{bank}-extra"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    # Searching the bank's own id: the row speaks for itself.
    by_id = await client.get("/v1/default/banks", params={"q": bank})
    assert [b["matched_aliases"] for b in by_id.json()["banks"]] == [[]]

    # Searching only the alias's distinctive tail: now the label earns its place.
    by_alias = await client.get("/v1/default/banks", params={"q": "-extra"})
    assert [(b["bank_id"], b["matched_aliases"]) for b in by_alias.json()["banks"]] == [(bank, [alias])]


@pytest.mark.asyncio
async def test_extensions_see_the_canonical_id_not_the_alias(client, memory, bank, monkeypatch):
    """Ordering guard: resolution happens before any bank-aware extension runs.

    An operation validator gates access per bank, so handing it the alias would
    have it authorise (or refuse) an id that names no bank — and a deployment
    whose policy is keyed on bank ids would silently diverge from the data being
    touched. The only extension that runs *before* resolution is the tenant
    authentication, which takes no bank id and has to go first: aliases live in
    the tenant's schema, so there is nothing to look one up in until the caller
    is identified.
    """
    from unittest.mock import AsyncMock

    from hindsight_api.extensions import ValidationResult

    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    validator = AsyncMock()
    validator.validate_bank_read = AsyncMock(return_value=ValidationResult(allowed=True))
    validator.precheck = AsyncMock(return_value=ValidationResult(allowed=True))
    monkeypatch.setattr(memory, "_operation_validator", validator)

    assert (await client.get(f"/v1/default/banks/{alias}/memories/list")).status_code == 200

    seen = {c.args[0].bank_id for c in validator.validate_bank_read.await_args_list}
    assert seen == {bank}, f"validator saw {seen}, expected only the bank's own id"
    assert alias not in seen


# ── Primary (display) alias ──────────────────────────────────────────────────
#
# A bank keeps its `bank_id` forever, so after a migration the UI would still
# show the id nobody uses. Promoting an alias changes what the bank is PRESENTED
# as and nothing else — every other part of the system keeps naming the real id.


@pytest.mark.asyncio
async def test_a_bank_shows_its_own_id_until_an_alias_is_promoted(client, bank):
    """Optional by design: the bank's own id is not an alias row, so "no primary"
    is the normal state and there is no implicit one to fall into."""
    alias = f"a-{uuid.uuid4().hex[:8]}"
    resp = await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    assert _primary(resp) is None
    listing = await client.get("/v1/default/banks", params={"q": bank})
    assert [b["display_alias"] for b in listing.json()["banks"]] == [None]


@pytest.mark.asyncio
async def test_promoting_an_alias_changes_only_what_the_bank_is_shown_as(client, bank):
    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias})

    resp = await client.patch(f"/v1/default/banks/{bank}/aliases/{alias}", json={"primary": True})
    assert _primary(resp) == alias
    # The identity is untouched: the response still names the bank by its own id,
    # and the list endpoint reports both, so a caller can show one and use the other.
    assert resp.json()["bank_id"] == bank
    row = next(b for b in (await client.get("/v1/default/banks", params={"q": bank})).json()["banks"])
    assert row["bank_id"] == bank
    assert row["display_alias"] == alias


@pytest.mark.asyncio
async def test_promoting_a_second_alias_demotes_the_first(client, bank):
    """At most one, enforced by a unique index rather than by application logic —
    two concurrent promotions would both read "none yet" and both write one."""
    first, second = f"a1-{uuid.uuid4().hex[:8]}", f"a2-{uuid.uuid4().hex[:8]}"
    for a in (first, second):
        await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": a})

    await client.patch(f"/v1/default/banks/{bank}/aliases/{first}", json={"primary": True})
    resp = await client.patch(f"/v1/default/banks/{bank}/aliases/{second}", json={"primary": True})

    assert _primary(resp) == second
    primaries = [a["alias"] for a in resp.json()["aliases"] if a["primary"]]
    assert primaries == [second], f"expected exactly one primary, got {primaries}"


@pytest.mark.asyncio
async def test_an_alias_can_be_created_already_primary(client, bank):
    alias = f"a-{uuid.uuid4().hex[:8]}"
    resp = await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias, "primary": True})
    assert _primary(resp) == alias


@pytest.mark.asyncio
async def test_demoting_returns_the_bank_to_its_own_id(client, bank):
    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias, "primary": True})

    resp = await client.patch(f"/v1/default/banks/{bank}/aliases/{alias}", json={"primary": False})
    assert _primary(resp) is None
    assert alias in _aliases(resp), "demoting must not remove the alias, only stop showing it"


@pytest.mark.asyncio
async def test_deleting_the_primary_alias_takes_the_display_with_it(client, bank):
    """The flag lives on the alias row, so it cannot outlive what it names — the
    reason it is not a `banks.display_alias` column, which would dangle here."""
    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{bank}/aliases", json={"alias": alias, "primary": True})

    await client.delete(f"/v1/default/banks/{bank}/aliases/{alias}")

    listing = await client.get("/v1/default/banks", params={"q": bank})
    assert [b["display_alias"] for b in listing.json()["banks"]] == [None]


@pytest.mark.asyncio
async def test_promoting_an_alias_of_another_bank_is_a_404(client, memory, bank):
    other = f"alias-other-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(other, request_context=RequestContext())
    alias = f"a-{uuid.uuid4().hex[:8]}"
    await client.post(f"/v1/default/banks/{other}/aliases", json={"alias": alias})

    resp = await client.patch(f"/v1/default/banks/{bank}/aliases/{alias}", json={"primary": True})
    assert resp.status_code == 404
    # And the other bank's alias was not promoted as a side effect.
    assert _primary(await client.get(f"/v1/default/banks/{other}/aliases")) is None
