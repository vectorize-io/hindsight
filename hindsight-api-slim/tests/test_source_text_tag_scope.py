"""#5030: source text follows its DOCUMENT's tags, not the fact's.

A fact can be visible to a reader through a tag the writer did not give the
document — an entity label with ``tag: true`` turns "Team rule: ..." into a fact
tagged ``kind:rule`` while the note it came from stays tagged ``user:kate``. Every
path that hands back that fact's chunk or document must check the document, or the
reader gets the rest of a note their filter excludes.

Three layers here:

* the matcher (``source_scope``), over every tag mode, against fakes;
* ``tool_expand``, against a fake connection, so each branch is pinned on its own;
* the real pipeline — retain with a label tag, then recall with chunks, recall of an
  observation, expand, reflect and a mental-model refresh — each with a positive
  control proving the private text *would* come back to a reader allowed to see it.
"""

from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.memories.base import MemoriesExtension
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.reflect.tools import tool_expand
from hindsight_api.engine.response_models import Budget, LLMToolCall, LLMToolCallResult
from hindsight_api.engine.search.tags import TagGroupLeaf, TagGroupNot, tag_filter_active
from hindsight_api.engine.source_scope import ids_passing, visible_document_ids

# ---------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------

_DOCS = {
    "kate": ["user:kate"],
    "dan": ["user:dan"],
    "both": ["user:kate", "user:dan"],
    "untagged": None,
    "empty": [],
}


@pytest.mark.parametrize(
    ("tags", "match", "groups", "active"),
    [
        (None, "any", None, False),
        ([], "any_strict", None, False),
        (["a"], "any", None, True),
        (None, "any", [TagGroupLeaf(tags=["a"])], True),
        # The one mode where no tags still filters: it selects the untagged scope.
        (None, "exact", None, True),
        ([], "exact", None, True),
    ],
)
def test_tag_filter_active(tags, match, groups, active):
    assert tag_filter_active(tags, match, groups) is active


@pytest.mark.parametrize(
    ("tags", "match", "expected"),
    [
        (["user:dan", "kind:rule"], "any_strict", {"dan", "both"}),
        (["user:dan", "kind:rule"], "any", {"dan", "both", "untagged", "empty"}),
        (["user:kate", "user:dan"], "all_strict", {"both"}),
        (["user:kate", "user:dan"], "all", {"both", "untagged", "empty"}),
        (["user:kate"], "exact", {"kate"}),
        ([], "exact", {"untagged", "empty"}),
        # A tag only facts carry never opens a document.
        (["kind:rule"], "any_strict", set()),
        (["kind:rule"], "all_strict", set()),
        (None, "any", set(_DOCS)),
    ],
)
def test_ids_passing_matches_documents_like_facts(tags, match, expected):
    assert ids_passing(_DOCS, tags=tags, tags_match=match, tag_groups=None) == expected


def test_ids_passing_applies_tag_groups_and_ands_them_with_tags():
    groups = [TagGroupNot(filter=TagGroupLeaf(tags=["user:kate"], match="any_strict"))]

    assert ids_passing(_DOCS, tags=None, tags_match="any", tag_groups=groups) == {"dan", "untagged", "empty"}
    # tags and tag_groups are AND-ed, as recall does.
    assert ids_passing(_DOCS, tags=["user:dan"], tags_match="any_strict", tag_groups=groups) == {"dan"}


class _RecordingConn:
    """Answers the documents-tags query from a dict; records every query it sees."""

    def __init__(self, doc_tags: dict[str, list[str] | None]) -> None:
        self.doc_tags = doc_tags
        self.queries: list[str] = []

    async def fetch(self, query: str, *args):
        self.queries.append(query)
        ids, _bank = args
        return [{"id": d, "tags": self.doc_tags[d]} for d in ids if d in self.doc_tags]


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_visible_document_ids_fails_closed_on_a_missing_document():
    conn = _RecordingConn({"kate": ["user:kate"]})

    visible = await visible_document_ids(
        conn,
        lambda t: f"public.{t}",
        "bank",
        ["kate", "deleted", None],
        tags=["user:kate"],
        tags_match="any",
        tag_groups=None,
    )

    # "deleted" would pass an "any" filter as untagged if a missing row read as no tags.
    assert visible == {"kate"}


@pytest.mark.asyncio
async def test_visible_document_ids_without_a_filter_reads_nothing():
    conn = MagicMock()
    conn.fetch.side_effect = AssertionError("an unfiltered read must not query documents")

    visible = await visible_document_ids(
        conn, lambda t: t, "bank", ["a", "b"], tags=None, tags_match="any", tag_groups=None
    )

    assert visible == {"a", "b"}


@pytest.mark.asyncio
async def test_visible_document_ids_asks_the_store_for_the_tags(monkeypatch):
    """The document tags come from the memories store, which owns the `documents` rows —
    a store-owned bank answers without SQL, so it is handed no connection."""

    class _Store:
        async def documents_tags(self, *, conn, fq_table, bank_id: str, document_ids: list[str]):
            assert conn is None
            return {d: list(_DOCS[d] or []) for d in document_ids if d in _DOCS}

    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: _Store())

    visible = await visible_document_ids(
        None,
        lambda t: t,
        "bank",
        ["kate", "dan", "missing"],
        tags=["user:dan"],
        tags_match="any_strict",
        tag_groups=None,
    )

    assert visible == {"dan"}


@pytest.mark.asyncio
async def test_the_default_store_read_leaves_out_a_record_without_tags():
    """A record that does not carry ``tags`` is unreadable, not untagged: read as untagged,
    an ``any`` filter would admit it. Left out, the document is hidden."""

    async def get_document_records(*, bank_id: str, document_ids: list[str]) -> dict[str, dict]:
        return {"tagged": {"tags": ["user:kate"]}, "null": {"tags": None}, "unreadable": {"text": "x"}}

    store = SimpleNamespace(get_document_records=get_document_records)

    tags = await MemoriesExtension.documents_tags(
        store, conn=None, fq_table=lambda t: t, bank_id="bank", document_ids=["tagged", "null", "unreadable"]
    )

    assert tags == {"tagged": ["user:kate"], "null": []}


# ---------------------------------------------------------------------------
# tool_expand, branch by branch
# ---------------------------------------------------------------------------

PRIVATE = "Kate is interviewing at another company"
RULE = "Team rule: nobody deploys to production on Fridays."


class _FakeExpandConn:
    """One bank's memory_units / chunks / documents, served to tool_expand."""

    def __init__(self, memories: list[dict], docs: dict[str, list[str] | None]) -> None:
        self.memories = {m["id"]: m for m in memories}
        self.docs = docs
        self.text_reads: list[str] = []

    async def fetch(self, query: str, *args):
        q = re.sub(r"\s+", " ", query)
        if "FROM public.memory_units" in q:
            return [self.memories[i] for i in args[0] if i in self.memories]
        if "FROM public.documents" in q and "original_text" not in q:
            return [{"id": d, "tags": self.docs[d]} for d in args[0] if d in self.docs]
        if "FROM public.chunks" in q:
            self.text_reads.append("chunks")
            return [
                {
                    "chunk_id": c,
                    "chunk_text": f"{RULE} {PRIVATE}",
                    "chunk_index": 0,
                    "document_id": c.removesuffix("_0"),
                }
                for c in args[0]
            ]
        if "FROM public.documents" in q:
            self.text_reads.append("documents")
            return [
                {"id": d, "original_text": f"{RULE} {PRIVATE}", "retain_params": None}
                for d in args[0]
                if d in self.docs
            ]
        raise AssertionError(f"unexpected query: {q}")


def _mem(tags: list[str], document_id: str | None, chunk: bool = True) -> dict:
    return {
        "id": uuid.uuid4(),
        "text": RULE,
        "chunk_id": f"{document_id}_0" if (chunk and document_id) else ("orphan_0" if chunk else None),
        "document_id": document_id,
        "fact_type": "world",
        "context": None,
        "tags": tags,
    }


async def _expand(conn, mems, depth="document", **scope):
    return await tool_expand(conn, "bank", [str(m["id"]) for m in mems], depth, **scope)


DAN = {"tags": ["user:dan", "kind:rule"], "tags_match": "any_strict", "tag_groups": None}
KATE = {"tags": ["user:kate"], "tags_match": "any_strict", "tag_groups": None}
NONE = {"tags": None, "tags_match": "any", "tag_groups": None}


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("depth", ["chunk", "document"])
async def test_expand_withholds_the_source_of_a_fact_shared_through_a_tag(depth):
    rule = _mem(["user:kate", "kind:rule"], "kate-sync")
    conn = _FakeExpandConn([rule], {"kate-sync": ["user:kate"]})

    result = await _expand(conn, [rule], depth, **DAN)

    item = result["results"][0]
    assert item["memory"]["text"] == RULE, "the shared fact itself must still be returned"
    assert "chunk" not in item and "document" not in item
    assert item["source_withheld"]
    assert PRIVATE not in json.dumps(result, default=str)
    assert conn.text_reads == [], "a withheld source must not even be read"


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_expand_returns_the_source_when_the_document_passes():
    rule = _mem(["user:kate", "kind:rule"], "kate-sync")
    conn = _FakeExpandConn([rule], {"kate-sync": ["user:kate"]})

    item = (await _expand(conn, [rule], **KATE))["results"][0]

    assert PRIVATE in item["chunk"]["text"]
    assert PRIVATE in item["document"]["full_text"]
    assert "source_withheld" not in item


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_expand_without_a_filter_is_unchanged():
    rule = _mem(["user:kate", "kind:rule"], "kate-sync")
    conn = _FakeExpandConn([rule], {"kate-sync": ["user:kate"]})

    item = (await _expand(conn, [rule], **NONE))["results"][0]

    assert PRIVATE in item["document"]["full_text"]


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_expand_reports_a_memory_outside_the_filter_as_not_found():
    """Expand takes ids from the model, which can name any memory in the bank —
    one it never recalled, or one quoted in a mental model. Outside the filter it
    must read exactly like an id that does not exist."""
    private = _mem(["user:kate"], "kate-sync")
    private["text"] = PRIVATE
    conn = _FakeExpandConn([private], {"kate-sync": ["user:kate"]})

    result = await _expand(conn, [private], **DAN)

    assert result["results"][0] == {
        "memory_id": str(private["id"]),
        "error": f"Memory not found: {private['id']}",
    }
    assert PRIVATE not in json.dumps(result, default=str)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_expand_fails_closed_on_a_chunk_with_no_document():
    orphan = _mem(["kind:rule"], None, chunk=True)
    conn = _FakeExpandConn([orphan], {})

    item = (await _expand(conn, [orphan], **DAN))["results"][0]

    assert "chunk" not in item and item["source_withheld"]
    assert conn.text_reads == []


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_expand_filters_each_memory_on_its_own_document():
    """A batch mixing a visible and a hidden source: only the hidden one is withheld."""
    rule = _mem(["user:kate", "kind:rule"], "kate-sync")
    mine = _mem(["user:dan"], "dan-notes")
    conn = _FakeExpandConn([rule, mine], {"kate-sync": ["user:kate"], "dan-notes": ["user:dan"]})

    hidden, shown = (await _expand(conn, [rule, mine], **DAN))["results"]

    assert "document" not in hidden and hidden["source_withheld"]
    assert shown["document"]["id"] == "dan-notes" and "source_withheld" not in shown


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_expand_honours_tag_groups():
    rule = _mem(["user:kate", "kind:rule"], "kate-sync")
    conn = _FakeExpandConn([rule], {"kate-sync": ["user:kate"]})
    groups = [TagGroupLeaf(tags=["kind:rule"], match="any_strict")]

    item = (await _expand(conn, [rule], tags=None, tags_match="any", tag_groups=groups))["results"][0]

    assert item["memory"]["text"] == RULE
    assert "document" not in item and item["source_withheld"]


# ---------------------------------------------------------------------------
# The real pipeline: Kate's note, Dan's reads
# ---------------------------------------------------------------------------

KATE_NOTE = (
    "Team rule: every pull request needs two approvals before merge.\n"
    "Team rule: nobody deploys to production on Fridays.\n"
    "Kate is interviewing at another company next week and has told nobody.\n"
)
DAN_NOTE = "Dan is refactoring the billing service this sprint.\n"
PRIVATE_MARK = "interviewing at another company"
QUERY = "Team rule deploys production Fridays pull request approvals"
DAN_SCOPE = {"tags": ["user:dan", "kind:rule"], "tags_match": "any_strict"}
KATE_SCOPE = {"tags": ["user:kate"], "tags_match": "any_strict"}


@pytest.fixture
def label_tagged_extraction(monkeypatch):
    """The mock extractor's facts, with ``kind:rule`` on every "Team rule" sentence —
    the entity a ``tag: true`` label turns into a fact tag."""
    original = MockLLM._build_mock_facts

    def _facts(messages):
        out = original(messages)
        for f in out["facts"]:
            if "Team rule" in f["what"]:
                f["entities"] = ["kind:rule"]
        return out

    monkeypatch.setattr(MockLLM, "_build_mock_facts", staticmethod(_facts))


@pytest.fixture
async def shared_bank(memory: MemoryEngine, request_context, label_tagged_extraction):
    bank_id = f"test-5030-{uuid.uuid4().hex[:8]}"
    await memory._ensure_bank_exists(bank_id, request_context)
    await memory._config_resolver.update_bank_config(
        bank_id,
        {
            "entity_labels": [
                {
                    "key": "kind",
                    "type": "multi-values",
                    "tag": True,
                    "optional": True,
                    "values": [{"value": "rule", "description": "a team rule"}],
                }
            ]
        },
    )
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": KATE_NOTE, "document_id": "kate-sync", "tags": ["user:kate"]}],
        request_context=request_context,
    )
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": DAN_NOTE, "document_id": "dan-notes", "tags": ["user:dan"]}],
        request_context=request_context,
    )
    yield bank_id
    await memory.delete_bank(bank_id, request_context=request_context)


async def _recall(memory, bank_id, request_context, **scope):
    return await memory.recall_async(
        bank_id=bank_id,
        query=QUERY,
        fact_type=["world", "experience"],
        max_tokens=4096,
        include_chunks=True,
        max_chunk_tokens=4000,
        budget=Budget.HIGH,
        request_context=request_context,
        **scope,
    )


def _chunk_texts(result) -> str:
    return "\n".join(c.chunk_text for c in (result.chunks or {}).values())


async def _rule_fact_id(memory, bank_id, request_context) -> str:
    result = await _recall(memory, bank_id, request_context, **DAN_SCOPE)
    rules = [r for r in result.results if "nobody deploys" in r.text]
    assert rules, f"setup: Dan should see the shared rule. Got {[r.text for r in result.results]}"
    return str(rules[0].id)


@pytest.mark.asyncio
async def test_setup_tags_the_facts_as_the_scenario_needs(memory, shared_bank, request_context):
    """Pins the premise: rules carry the label tag, the private fact does not, the
    document carries only the writer's tag. Without this every leak test below could
    pass vacuously."""
    units = (await memory.list_memory_units(shared_bank, document_id="kate-sync", request_context=request_context))[
        "items"
    ]
    tags_by_text = {u["text"]: set(u.get("tags") or []) for u in units}
    rules = [t for text, t in tags_by_text.items() if "Team rule" in text]
    private = [t for text, t in tags_by_text.items() if PRIVATE_MARK in text]
    assert rules and all(t == {"user:kate", "kind:rule"} for t in rules), tags_by_text
    assert private and all(t == {"user:kate"} for t in private), tags_by_text

    doc = await memory.get_document("kate-sync", shared_bank, request_context=request_context)
    assert set(doc["tags"]) == {"user:kate"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        DAN_SCOPE,
        {"tags": ["user:dan", "kind:rule"], "tags_match": "any"},
        {"tags": ["kind:rule"], "tags_match": "all_strict"},
        {"tag_groups": [TagGroupLeaf(tags=["user:dan", "kind:rule"], match="any_strict")]},
    ],
    ids=["any_strict", "any", "all_strict", "tag_groups"],
)
async def test_recall_chunks_do_not_carry_a_hidden_document(memory, shared_bank, request_context, scope):
    result = await _recall(memory, shared_bank, request_context, **scope)

    texts = [r.text for r in result.results]
    assert any("nobody deploys" in t for t in texts), f"the shared rule must still be recalled: {texts}"
    assert not any(PRIVATE_MARK in t for t in texts), "the private fact itself leaked"
    assert PRIVATE_MARK not in _chunk_texts(result)


@pytest.mark.asyncio
async def test_recall_chunks_still_include_the_readers_own_documents(memory, shared_bank, request_context):
    result = await memory.recall_async(
        bank_id=shared_bank,
        query="Dan refactoring billing service sprint",
        max_tokens=4096,
        include_chunks=True,
        max_chunk_tokens=4000,
        request_context=request_context,
        **DAN_SCOPE,
    )

    assert "billing service" in _chunk_texts(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [KATE_SCOPE, {}], ids=["writer", "unfiltered"])
async def test_recall_chunks_reach_a_reader_the_document_allows(memory, shared_bank, request_context, scope):
    """The positive control: the same recall, by a reader the document admits,
    does return the private text — so its absence above is the filter."""
    result = await _recall(memory, shared_bank, request_context, **scope)

    assert PRIVATE_MARK in _chunk_texts(result)


@pytest.mark.asyncio
async def test_observation_chunks_do_not_carry_a_hidden_document(memory, shared_bank, request_context):
    """An observation has no chunk of its own; its chunks are its source facts'. Those
    resolve through a separate query, which must land on the same filter."""
    await memory.run_consolidation(bank_id=shared_bank, request_context=request_context)

    def recall_observations(**scope):
        return memory.recall_async(
            bank_id=shared_bank,
            query=QUERY,
            fact_type=["observation"],
            max_tokens=4096,
            include_chunks=True,
            max_chunk_tokens=4000,
            budget=Budget.HIGH,
            request_context=request_context,
            **scope,
        )

    kate = await recall_observations(**KATE_SCOPE)
    assert kate.results, "setup: consolidation produced no observation in Kate's scope"
    assert PRIVATE_MARK in _chunk_texts(kate), "control: the observation's source chunk should reach Kate"

    dan = await recall_observations(**DAN_SCOPE)
    assert dan.results, "setup: Dan should see the observation built from the shared rules"
    assert PRIVATE_MARK not in _chunk_texts(dan)


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["chunk", "document"])
async def test_expand_over_the_real_store(memory, shared_bank, request_context, depth):
    rule_id = await _rule_fact_id(memory, shared_bank, request_context)

    async with memory._backend.acquire() as conn:
        as_dan = await tool_expand(conn, shared_bank, [rule_id], depth, tag_groups=None, **DAN_SCOPE)
        as_kate = await tool_expand(conn, shared_bank, [rule_id], depth, tag_groups=None, **KATE_SCOPE)

    dan_item = as_dan["results"][0]
    assert "nobody deploys" in dan_item["memory"]["text"]
    assert dan_item["source_withheld"]
    assert PRIVATE_MARK not in json.dumps(as_dan, default=str)

    kate_item = as_kate["results"][0]
    assert PRIVATE_MARK in kate_item["chunk"]["text"]
    if depth == "document":
        assert PRIVATE_MARK in kate_item["document"]["full_text"]


# ---------------------------------------------------------------------------
# Reflect and mental-model refresh: the model asks to expand the rule
# ---------------------------------------------------------------------------


def _install_expanding_llm(memory: MemoryEngine, rule_id: str) -> list[list[dict[str, Any]]]:
    """Drive reflect through recall → expand(rule, document) → done, recording every
    message list the model was shown. The prompts are what a model would read, so a
    leak anywhere in them — a recall chunk or an expand result — is a leak."""
    seen: list[list[dict[str, Any]]] = []
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")

    def callback(messages, scope):
        seen.append(list(messages))
        if scope != "reflect_tool_call":
            return "The no-Friday-deploy rule is a team rule."
        called = {
            tc["function"]["name"] if "function" in tc else tc.get("name")
            for m in messages
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        }
        if "recall" not in called:
            call = LLMToolCall(id="recall-1", name="recall", arguments={"query": QUERY})
        elif "expand" not in called:
            call = LLMToolCall(id="expand-1", name="expand", arguments={"memory_ids": [rule_id], "depth": "document"})
        else:
            call = LLMToolCall(
                id="done-1", name="done", arguments={"answer": "It is a team rule.", "memory_ids": [rule_id]}
            )
        return LLMToolCallResult(tool_calls=[call], finish_reason="tool_calls")

    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    memory._reflect_llm_config = wrapper
    return seen


def _expand_results(seen: list[list[dict[str, Any]]]) -> list[dict]:
    out = []
    for messages in seen:
        for m in messages:
            if m.get("role") != "tool":
                continue
            try:
                payload = json.loads(m.get("content") or "{}")
            except (TypeError, ValueError):
                continue
            # Reflect trims the tool payload it shows the model (no "count"); "results"
            # is unique to expand among the tools this loop calls.
            if isinstance(payload, dict) and "results" in payload:
                out.append(payload)
    return out


def _all_text(seen: list[list[dict[str, Any]]]) -> str:
    return json.dumps(seen, default=str)


@pytest.mark.asyncio
async def test_reflect_expand_does_not_quote_a_hidden_document(memory, shared_bank, request_context):
    rule_id = await _rule_fact_id(memory, shared_bank, request_context)
    seen = _install_expanding_llm(memory, rule_id)

    await memory.reflect_async(
        bank_id=shared_bank,
        query="Where does the no-Friday-deploy rule come from? Expand to the full original note.",
        request_context=request_context,
        **DAN_SCOPE,
    )

    expanded = _expand_results(seen)
    assert expanded, "setup: the reflect loop never ran expand, so nothing was tested"
    item = expanded[-1]["results"][0]
    assert item["memory_id"] == rule_id and item["source_withheld"]
    assert PRIVATE_MARK not in _all_text(seen)


@pytest.mark.asyncio
async def test_reflect_expand_quotes_the_document_to_its_writer(memory, shared_bank, request_context):
    rule_id = await _rule_fact_id(memory, shared_bank, request_context)
    seen = _install_expanding_llm(memory, rule_id)

    await memory.reflect_async(
        bank_id=shared_bank,
        query="Where does the no-Friday-deploy rule come from? Expand to the full original note.",
        request_context=request_context,
        **KATE_SCOPE,
    )

    expanded = _expand_results(seen)
    assert expanded and PRIVATE_MARK in expanded[-1]["results"][0]["document"]["full_text"]


@pytest.mark.asyncio
async def test_a_shared_tag_mental_model_cannot_absorb_a_hidden_document(memory, shared_bank, request_context):
    """Mental models refresh through reflect, so a page scoped to ``kind:rule`` would
    otherwise expand into Kate's note and publish her private paragraph to every rule
    reader."""
    rule_id = await _rule_fact_id(memory, shared_bank, request_context)
    mm = await memory.create_mental_model(
        bank_id=shared_bank,
        name="team-rules",
        source_query="What are the team rules and where do they come from?",
        content="",
        tags=["kind:rule"],
        trigger={"refresh_after_consolidation": False, "tags_match": "any_strict"},
        request_context=request_context,
    )
    seen = _install_expanding_llm(memory, rule_id)

    await memory.refresh_mental_model(bank_id=shared_bank, mental_model_id=mm["id"], request_context=request_context)

    expanded = _expand_results(seen)
    assert expanded, "setup: the refresh never ran expand, so nothing was tested"
    assert expanded[-1]["results"][0]["source_withheld"]
    assert PRIVATE_MARK not in _all_text(seen)
