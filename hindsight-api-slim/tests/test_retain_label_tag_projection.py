"""#4068: what a re-retain may and may not rewrite on a surviving memory unit.

``memory_units.tags`` carries two different things:

* **Document tags** — what the caller hands to retain. Caller-owned, so a re-retain
  replaces them outright: the new array is exactly what this submission said, with no
  merge against the previous one. Same for the document metadata bag.
* **Label tags** — the projection of ``entity_labels`` groups flagged ``tag: true``.
  These are *derived* from the unit's own entities by ``_inject_label_tags``; the
  entities are the ground truth and the tag is a mirror kept so the tags API can filter
  on a label. A unit only acquires them by being extracted, so only a unit the LLM
  actually re-extracted may see them change — and then exactly, to whatever the fresh
  extraction produced. A survivor was not re-extracted, so its projection must stand.

Today ``update_memory_units_metadata_and_tags`` overwrites the whole array with the
document tags alone, which drops a survivor's label tags while the units extracted
beside it in the same retain keep theirs.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.providers.mock_llm import MockLLM


@pytest.fixture(scope="session")
def db_url():
    """Isolated pg0 instance so this module never touches the dev database."""
    return "pg0://hs4068:55468"


def _labels(*values):
    return [
        {
            "key": "category",
            "type": "multi-values",
            "tag": True,
            "optional": True,
            "values": [{"value": v, "description": f"the {v} category"} for v in values],
        }
    ]


class _Label:
    """The label value the mock LLM attaches to every fact it extracts."""

    value = "durable"


@pytest.fixture(autouse=True)
def labelled_extraction(monkeypatch):
    def _facts(messages):
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        return {
            "facts": [
                {
                    "what": f"Fact about {user_text[-40:]}",
                    "when": "N/A",
                    "where": "N/A",
                    "who": "N/A",
                    "why": "N/A",
                    "fact_kind": "conversation",
                    "fact_type": "world",
                    "entities": [f"category:{_Label.value}"],
                }
            ]
        }

    _Label.value = "durable"
    monkeypatch.setattr(MockLLM, "_build_mock_facts", staticmethod(_facts))


async def _units(memory, bank_id, document_id, request_context):
    res = await memory.list_memory_units(bank_id, document_id=document_id, limit=200, request_context=request_context)
    return {u["id"]: {"tags": set(u.get("tags") or []), "metadata": u.get("metadata")} for u in res["items"]}


async def _retain(memory, bank_id, doc, content, tags, metadata, request_context):
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": content, "document_id": doc, "tags": tags, "metadata": metadata}],
        request_context=request_context,
    )


# Two chunks under a 60-char chunk size, so a re-retain that rewrites only the second
# leaves the first one's units in place as survivors.
_A = "Alpha " * 20
_B = "Bravo " * 20
_C = "Charlie " * 20


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_document_tags_override_exactly_and_label_tags_survive(memory, request_context):
    """One re-retain that changes the document tags, the metadata and one chunk."""
    bank_id = f"test_4068_delta_{datetime.now(timezone.utc).timestamp()}"
    doc = "doc-delta"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(
            bank_id, {"entity_labels": _labels("durable"), "retain_chunk_size": 60}
        )

        await _retain(memory, bank_id, doc, _A + "\n\n" + _B, ["t1"], {"src": "v1"}, request_context)
        before = await _units(memory, bank_id, doc, request_context)
        assert before and all(u["tags"] == {"t1", "category:durable"} for u in before.values()), before

        await _retain(memory, bank_id, doc, _A + "\n\n" + _C, ["t2"], {"src": "v2"}, request_context)
        after = await _units(memory, bank_id, doc, request_context)

        survivors = {uid: u for uid, u in after.items() if uid in before}
        fresh = {uid: u for uid, u in after.items() if uid not in before}
        assert survivors, "the first chunk was re-extracted — no survivors to assert on"
        assert fresh, "the second chunk was not re-extracted"

        # Document tags: replaced outright, on every unit. 't1' is gone because this
        # submission did not carry it, not merged forward.
        # Label tags: 'category:durable' stands on the survivors (not re-extracted) and is
        # re-derived on the fresh units.
        for uid, u in after.items():
            assert u["tags"] == {"t2", "category:durable"}, f"{uid}: {u['tags']}"
            assert u["metadata"] == {"src": "v2"}, f"{uid}: {u['metadata']}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_label_tags_are_exact_per_unit_not_merged(memory, request_context):
    """A re-extracted unit takes the FRESH label exactly; a survivor keeps its own.

    The two must not bleed into each other: no unit ends up carrying both labels.
    """
    bank_id = f"test_4068_exact_{datetime.now(timezone.utc).timestamp()}"
    doc = "doc-exact"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(
            bank_id, {"entity_labels": _labels("durable", "provisional"), "retain_chunk_size": 60}
        )

        await _retain(memory, bank_id, doc, _A + "\n\n" + _B, ["t1"], {}, request_context)
        before = await _units(memory, bank_id, doc, request_context)
        assert before and all(u["tags"] == {"t1", "category:durable"} for u in before.values()), before

        # The second retain's extraction yields a different label.
        _Label.value = "provisional"
        await _retain(memory, bank_id, doc, _A + "\n\n" + _C, ["t1"], {}, request_context)
        after = await _units(memory, bank_id, doc, request_context)

        survivors = {uid: u for uid, u in after.items() if uid in before}
        fresh = {uid: u for uid, u in after.items() if uid not in before}
        assert survivors and fresh, (survivors, fresh)

        for uid, u in survivors.items():
            assert u["tags"] == {"t1", "category:durable"}, f"survivor {uid} was relabelled: {u['tags']}"
        for uid, u in fresh.items():
            assert u["tags"] == {"t1", "category:provisional"}, f"fresh unit {uid} kept a stale label: {u['tags']}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_unchanged_content_re_retain_keeps_label_tags(memory, request_context):
    """The metadata-only path: nothing is re-extracted, so every label tag stands."""
    bank_id = f"test_4068_same_{datetime.now(timezone.utc).timestamp()}"
    doc = "doc-same"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(bank_id, {"entity_labels": _labels("durable")})

        content = "Alice works at Google."
        await _retain(memory, bank_id, doc, content, ["t1"], {"src": "v1"}, request_context)
        before = await _units(memory, bank_id, doc, request_context)
        assert before

        # Same bytes, new document tags and metadata: a pure relabelling.
        await _retain(memory, bank_id, doc, content, ["t2"], {"src": "v2"}, request_context)
        after = await _units(memory, bank_id, doc, request_context)

        assert set(after) == set(before), "unchanged content must not replace any unit"
        for uid, u in after.items():
            assert u["tags"] == {"t2", "category:durable"}, f"{uid}: {u['tags']}"
            assert u["metadata"] == {"src": "v2"}, f"{uid}: {u['metadata']}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_document_tags_patch_keeps_label_tags(memory, request_context):
    """The tags PATCH re-processes nothing, so it may not touch the label projection."""
    bank_id = f"test_4068_patch_{datetime.now(timezone.utc).timestamp()}"
    doc = "doc-patch"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(bank_id, {"entity_labels": _labels("durable")})

        await _retain(memory, bank_id, doc, "Alice works at Google.", ["t1"], {}, request_context)
        before = await _units(memory, bank_id, doc, request_context)
        assert before and all(u["tags"] == {"t1", "category:durable"} for u in before.values()), before

        assert await memory.update_document(doc, bank_id, tags=["t2"], request_context=request_context)
        after = await _units(memory, bank_id, doc, request_context)

        assert set(after) == set(before), "a retag must not replace any unit"
        for uid, u in after.items():
            assert u["tags"] == {"t2", "category:durable"}, f"{uid}: {u['tags']}"

        # And the document's own tags are the document tags alone — the projection is a
        # per-memory thing and must not be promoted onto the document row.
        document = await memory.get_document(doc, bank_id, request_context=request_context)
        assert set(document["tags"]) == {"t2"}, document["tags"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_label_tag_also_supplied_as_a_document_tag_is_not_duplicated(memory, request_context):
    """A document tag that collides with the label projection must appear once.

    The two halves of the array are computed separately, so a caller that happens to
    send ``category:durable`` as a document tag is where they would be concatenated
    into a unit carrying it twice.
    """
    bank_id = f"test_4068_dupe_{datetime.now(timezone.utc).timestamp()}"
    doc = "doc-dupe"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(
            bank_id, {"entity_labels": _labels("durable"), "retain_chunk_size": 60}
        )

        # Every unit's projection is 'category:durable', and it is a document tag too.
        await _retain(memory, bank_id, doc, _A + "\n\n" + _B, ["category:durable", "t1"], {}, request_context)
        await _retain(memory, bank_id, doc, _A + "\n\n" + _C, ["category:durable", "t1"], {}, request_context)

        res = await memory.list_memory_units(bank_id, document_id=doc, limit=200, request_context=request_context)
        for unit in res["items"]:
            raw = list(unit.get("tags") or [])
            assert raw.count("category:durable") == 1, f"{unit['id']} carries a duplicated tag: {raw}"
            assert set(raw) == {"category:durable", "t1"}, raw

        # Same through the retag path.
        assert await memory.update_document(
            doc, bank_id, tags=["category:durable", "t2"], request_context=request_context
        )
        res = await memory.list_memory_units(bank_id, document_id=doc, limit=200, request_context=request_context)
        for unit in res["items"]:
            raw = list(unit.get("tags") or [])
            assert raw.count("category:durable") == 1, f"{unit['id']} carries a duplicated tag: {raw}"
            assert set(raw) == {"category:durable", "t2"}, raw
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
