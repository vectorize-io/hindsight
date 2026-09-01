"""retain_params must round-trip whatever the caller supplied.

reprocess_document rebuilds its retain call entirely from documents.retain_params.
The inclusion list this replaces dropped three fields in turn — `strategy`,
`entities`, `resolve_entities` — each written by api_retain and never captured, so
a reprocess re-extracted under the bank's default strategy with entity resolution
it had been told not to do. Every one was invisible: the reprocess succeeds and
only the resulting facts are wrong.

The rule is now an exclusion list, so a new retain field round-trips unless
somebody deliberately excludes it, and api_retain's content dict is the single
source of truth for what gets replayed.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import orchestrator as orch

API_HTTP = Path(orch.__file__).resolve().parents[2] / "api" / "http.py"
MCP_TOOLS = Path(orch.__file__).resolve().parents[2] / "mcp_tools.py"


def _params(**item):
    retain_params, _tags = orch._build_retain_params([item])
    return retain_params


def test_captures_the_fields_that_used_to_go_missing():
    p = _params(
        content="x",
        strategy="survey",
        entities=[{"text": "Widget", "type": "CONCEPT"}],
        resolve_entities=False,
    )
    assert p["strategy"] == "survey"
    assert p["entities"] == [{"text": "Widget", "type": "CONCEPT"}]
    assert p["resolve_entities"] is False


def test_excludes_what_a_reprocess_supplies_itself():
    """Replaying these would fight the reprocess: content is the stored
    original_text, and document_id/update_mode/tags are set by the reprocess."""
    p = _params(content="x", document_id="d1", update_mode="append", tags=["a"], context="ctx")
    assert set(p) == {"context"}


def test_absent_fields_are_not_invented():
    assert _params(content="x") == {}


def test_event_date_is_normalised_for_json():
    p = _params(content="x", event_date=datetime(2026, 1, 2, 3, 4, 5))
    assert p["event_date"] == "2026-01-02T03:04:05"


def test_a_new_field_round_trips_without_being_listed():
    """The point of inverting the rule: the failure mode is now a deliberate
    exclusion, not a silent omission."""
    assert _params(content="x", some_future_field="v")["some_future_field"] == "v"


def test_every_field_api_retain_sends_can_round_trip():
    """Pairs the writer against the rule so they cannot drift apart.

    api_retain's content dict is the source of truth for what a retain can carry.
    Anything it sets that is neither excluded nor capturable is a field a reprocess
    would silently lose — which is exactly how this bug arose three times.
    """
    written = set(re.findall(r'content_dict\["(\w+)"\]\s*=', API_HTTP.read_text()))
    assert written, "could not locate api_retain's content dict assignments"

    replayable = written - orch._RETAIN_PARAMS_NOT_REPLAYED
    produced = _params(content="x", **{k: "v" for k in replayable})
    missing = sorted(replayable - set(produced))
    assert not missing, f"api_retain sends fields retain_params cannot carry: {missing}"


def test_api_retain_puts_the_strategy_on_the_content_dict():
    """Where the break actually was. api_retain used `strategy` only as the key it
    grouped items by, so it never reached the dict _build_retain_params reads — and
    capturing it in the orchestrator alone changed nothing. The pairing test above
    cannot see this: a strategy that is never assigned simply drops out of the set
    it compares."""
    src = API_HTTP.read_text()
    block = src[src.index("# Group items by strategy") :][:2500]
    assert 'content_dict["strategy"] = item.strategy' in block


def test_mcp_retain_leaves_the_strategy_on_the_item():
    """The MCP tools pass the same dict they hand to retain, so `pop` would strip
    `strategy` off it before the call ran (arguments evaluate left to right) and
    retain_params would never see it — the api_retain bug, one layer over."""
    src = MCP_TOOLS.read_text()
    assert 'content_dict.pop("strategy"' not in src, (
        "popping strategy off the content dict removes it from the item retain stores, "
        "so a later reprocess falls back to the bank default"
    )
    assert src.count('strategy=content_dict.get("strategy")') == 4


class _StubEngine:
    """Just enough MemoryEngine for reprocess_document: it reads a document and
    submits a retain. Everything else the method touches is disabled."""

    _operation_validator = None

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self.submitted: list[tuple[dict[str, Any], str | None]] = []

    async def _authenticate_tenant(self, request_context) -> None:
        return None

    async def get_document(self, document_id, bank_id, request_context=None):
        return self._doc

    async def submit_async_retain(self, bank_id, contents, strategy=None, request_context=None, **kwargs):
        self.submitted.append((contents[0], strategy))
        return {"operation_id": "op-1"}


async def _reprocess(retain_params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Run one reprocess over a document carrying ``retain_params``."""
    engine = _StubEngine({"original_text": "STATUS: survey started", "retain_params": retain_params, "tags": ["t1"]})
    await MemoryEngine.reprocess_document(engine, "bank-1", "doc-1", request_context=None)
    return engine.submitted[0]


@pytest.mark.asyncio
async def test_reprocess_replays_what_the_retain_stored():
    """The reader half: everything the caller supplied comes back on the replayed
    item, and the reprocess's own three fields win over anything stored."""
    stored = _params(
        content="STATUS: survey started",
        strategy="survey",
        context="ctx",
        metadata={"a": "b"},
        entities=[{"text": "Widget", "type": "CONCEPT"}],
        resolve_entities=False,
        observation_scopes="shared",
    )

    item, strategy = await _reprocess(stored)

    assert strategy == "survey"
    assert item["entities"] == [{"text": "Widget", "type": "CONCEPT"}]
    assert item["resolve_entities"] is False
    assert item["context"] == "ctx"
    assert item["metadata"] == {"a": "b"}
    assert item["observation_scopes"] == "shared"
    # The reprocess's own, whatever the stored params say.
    assert item["content"] == "STATUS: survey started"
    assert item["document_id"] == "doc-1"
    assert item["update_mode"] == "replace"
    assert item["tags"] == ["t1"]


@pytest.mark.asyncio
async def test_strategy_survives_more_than_one_reprocess():
    """Excluding `strategy` from the replayed dict survived exactly ONE reprocess.

    reprocess pulls it out as a call argument, so the first re-extraction used the
    right strategy — but _build_retain_params never saw it on the item, retain_params
    came back without it, and a second reprocess fell back to the bank default. Feed
    each reprocess's item back through the capture, as the retain pipeline does.
    """
    stored = _params(content="STATUS: survey started", strategy="survey")

    for _ in range(3):
        item, strategy = await _reprocess(stored)
        assert strategy == "survey"
        stored = _params(**item)

    assert stored["strategy"] == "survey"


@pytest.mark.asyncio
async def test_reprocess_forces_reextraction():
    """A reprocess replays identical content, which the delta and crash-recovery
    paths both read as "nothing to do" — so it has to say it means it (#3899)."""
    item, _strategy = await _reprocess(_params(content="STATUS: survey started"))
    assert item["force_reextract"] is True


@pytest.mark.asyncio
async def test_force_reextract_is_not_stored_in_retain_params():
    """It is the reprocess's own instruction, not something the document carries:
    storing it would put an internal flag on every reprocessed document's
    retain_params (and back onto the next replay) for no added effect."""
    item, _strategy = await _reprocess(_params(content="STATUS: survey started"))
    assert "force_reextract" not in _params(**item)
