"""A validator's forced tag scope (``resolve_tag_scope``) confines a caller everywhere.

Two people share a bank: Kate writes under ``user:kate``, Dan under ``user:dan``, and one of
Kate's facts is also tagged ``kind:rule`` — the shared scope. Dan is confined to
``user:dan`` OR ``kind:rule``. Every read and by-id write below must keep Kate's private
memories, her document, its text and her mental models out of Dan's reach, while still
showing him the shared rule.
"""

import uuid

import pytest

from hindsight_api.engine.memory_engine import _scope_mental_model_trigger
from hindsight_api.engine.reflect.tools import tool_expand, tool_read_mental_models
from hindsight_api.engine.schema import fq_store_table_explicit
from hindsight_api.engine.search.tags import TagGroupLeaf, tags_satisfy_groups
from hindsight_api.extensions import (
    OperationValidationError,
    OperationValidatorExtension,
    TagScopeContext,
    ValidationResult,
)
from hindsight_api.models import RequestContext

DAN_SCOPE = [TagGroupLeaf(tags=["user:dan", "kind:rule"], match="any_strict")]

KATE_TEXT = "Nobody deploys to production on Fridays. Kate is interviewing at another company next week."
DAN_TEXT = "Dan is working on the search latency regression."


class _ScopeByApiKey(OperationValidatorExtension):
    """Confines a caller by its API key; a request without a key is unrestricted."""

    def __init__(self, scopes):
        super().__init__({})
        self.scopes = scopes

    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def resolve_tag_scope(self, ctx: TagScopeContext):
        return self.scopes.get(ctx.request_context.api_key)


def test_tags_satisfy_groups():
    assert tags_satisfy_groups(["user:dan"], DAN_SCOPE)
    assert tags_satisfy_groups(["user:kate", "kind:rule"], DAN_SCOPE)
    assert not tags_satisfy_groups(["user:kate"], DAN_SCOPE)
    assert not tags_satisfy_groups([], DAN_SCOPE), "strict: untagged rows are outside the scope"
    assert tags_satisfy_groups(["anything"], None)


def test_scope_mental_model_trigger_ands_the_scope_into_the_refresh_filter():
    # Flat tags become a leaf under their resolved mode (all_strict by default), then the scope.
    scoped = _scope_mental_model_trigger(["user:kate"], {"mode": "delta"}, DAN_SCOPE)
    assert scoped == {
        "mode": "delta",
        "tag_groups": [
            {"tags": ["user:kate"], "match": "all_strict", "resolve": "exact"},
            {"tags": ["user:dan", "kind:rule"], "match": "any_strict", "resolve": "exact"},
        ],
    }
    # Re-applying the same scope does not stack it.
    assert _scope_mental_model_trigger(["user:kate"], scoped, DAN_SCOPE) == scoped
    # An untagged model (the whole bank) is narrowed to the scope alone.
    assert _scope_mental_model_trigger([], None, DAN_SCOPE) == {
        "tag_groups": [{"tags": ["user:dan", "kind:rule"], "match": "any_strict", "resolve": "exact"}]
    }
    # No scope: the trigger is left exactly as it was.
    assert _scope_mental_model_trigger(["user:kate"], {"mode": "delta"}, None) == {"mode": "delta"}


@pytest.fixture
async def scoped_bank(memory):
    """A bank holding Kate's note (one fact forged into the shared ``kind:rule`` scope) and Dan's."""
    bank_id = f"tag-scope-{uuid.uuid4().hex[:8]}"
    admin = RequestContext()
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": KATE_TEXT, "document_id": "kate-sync", "tags": ["user:kate"]}],
        request_context=admin,
    )
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": DAN_TEXT, "document_id": "dan-notes", "tags": ["user:dan"]}],
        request_context=admin,
    )
    # In production the rule tag comes from an entity label with `tag: true`, which needs a real
    # LLM to extract; the mock cannot. Forging it on the stored fact is the only way to put a
    # memory and its document in different scopes here, which is exactly the case under test.
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        await conn.execute(
            f"UPDATE {fq_store_table_explicit('memory_units')} SET tags = ARRAY['user:kate', 'kind:rule'] "
            "WHERE bank_id = $1 AND fact_type = 'world' AND text ILIKE '%Fridays%'",
            bank_id,
        )
    validator = _ScopeByApiKey({"dan": DAN_SCOPE})
    original = memory._operation_validator
    memory._operation_validator = validator
    try:
        yield bank_id
    finally:
        memory._operation_validator = original
        await memory.delete_bank(bank_id, request_context=admin)


def _texts(items) -> str:
    return " | ".join(i["text"] for i in items)


@pytest.mark.asyncio
async def test_memory_reads_are_confined(memory, scoped_bank):
    dan = RequestContext(api_key="dan")
    listed = await memory.list_memory_units(scoped_bank, request_context=dan)
    texts = _texts(listed["items"])
    assert "Fridays" in texts and "latency" in texts
    assert "interviewing" not in texts
    assert all(tags_satisfy_groups(i["tags"], DAN_SCOPE) for i in listed["items"])

    # An unscoped caller (no key) still sees everything.
    everything = await memory.list_memory_units(scoped_bank, request_context=RequestContext())
    assert "interviewing" in _texts(everything["items"])

    private = next(i for i in everything["items"] if "interviewing" in i["text"])
    rule = next(i for i in everything["items"] if "Fridays" in i["text"] and i["fact_type"] == "world")
    assert await memory.get_memory_unit(scoped_bank, private["id"], request_context=dan) is None
    assert await memory.get_memory_unit(scoped_bank, rule["id"], request_context=dan) is not None
    assert await memory.get_observation_history(scoped_bank, private["id"], request_context=dan) is None

    recalled = await memory.recall_async(
        bank_id=scoped_bank, query="What is Kate doing and what are the rules?", request_context=dan
    )
    recalled_texts = " | ".join(r.text for r in recalled.results)
    assert "interviewing" not in recalled_texts
    assert all(tags_satisfy_groups(r.tags, DAN_SCOPE) for r in recalled.results)

    tags = await memory.list_tags(scoped_bank, request_context=dan)
    # `user:kate` is still listed: the shared rule fact carries it. The count is that fact alone.
    counts = {t["tag"]: t["count"] for t in tags["items"]}
    assert counts["kind:rule"] == counts["user:kate"]

    scopes = await memory.list_observation_scopes(scoped_bank, request_context=dan)
    assert all(tags_satisfy_groups(s["tags"], DAN_SCOPE) for s in scopes["scopes"])


@pytest.mark.asyncio
async def test_document_text_follows_the_document(memory, scoped_bank):
    dan = RequestContext(api_key="dan")
    docs = await memory.list_documents(scoped_bank, request_context=dan)
    assert [d["id"] for d in docs["items"]] == ["dan-notes"]
    assert await memory.get_document("kate-sync", scoped_bank, request_context=dan) is None
    assert await memory.list_document_chunks(scoped_bank, "kate-sync", request_context=dan) is None

    admin_chunks = await memory.list_document_chunks(scoped_bank, "kate-sync", request_context=RequestContext())
    assert admin_chunks is not None
    kate_chunk = admin_chunks["items"][0]["chunk_id"]
    assert await memory.get_chunk(kate_chunk, request_context=dan) is None

    # Reflect's expand tool: the rule fact is visible, the note it came from is not.
    everything = await memory.list_memory_units(scoped_bank, request_context=RequestContext())
    rule = next(i for i in everything["items"] if "Fridays" in i["text"] and i["fact_type"] == "world")
    private = next(i for i in everything["items"] if "interviewing" in i["text"])
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        expanded = await tool_expand(
            conn,
            scoped_bank,
            [rule["id"], private["id"]],
            "document",
            tags=None,
            tags_match="any",
            tag_groups=DAN_SCOPE,
        )
        unscoped = await tool_expand(
            conn, scoped_bank, [rule["id"]], "document", tags=None, tags_match="any", tag_groups=None
        )
    by_id = {r["memory_id"]: r for r in expanded["results"]}
    assert "Fridays" in by_id[rule["id"]]["memory"]["text"]
    assert "chunk" not in by_id[rule["id"]] and "document" not in by_id[rule["id"]]
    assert "error" in by_id[private["id"]]
    assert "interviewing" in unscoped["results"][0]["document"]["full_text"]


@pytest.mark.asyncio
async def test_writes_outside_the_scope_are_refused(memory, scoped_bank):
    dan = RequestContext(api_key="dan")
    # Appending to (or replacing) Kate's document would fold her text into what Dan reads.
    with pytest.raises(OperationValidationError) as refused:
        await memory.retain_batch_async(
            bank_id=scoped_bank,
            contents=[{"content": "Dan's addition.", "document_id": "kate-sync", "tags": ["user:dan"]}],
            request_context=dan,
        )
    assert refused.value.status_code == 403
    with pytest.raises(OperationValidationError) as missing:
        await memory.delete_document("kate-sync", scoped_bank, request_context=dan)
    assert missing.value.status_code == 404
    assert await memory.get_document("kate-sync", scoped_bank, request_context=RequestContext()) is not None


@pytest.mark.asyncio
async def test_mental_models_are_confined(memory, scoped_bank):
    admin = RequestContext()
    dan = RequestContext(api_key="dan")
    await memory.create_mental_model(
        scoped_bank,
        "Team rules",
        "What are the rules?",
        "No Friday deploys.",
        tags=["kind:rule"],
        request_context=admin,
    )
    private = await memory.create_mental_model(
        scoped_bank,
        "Kate career",
        "What is Kate planning?",
        "Kate is interviewing.",
        tags=["user:kate"],
        request_context=admin,
    )

    page = await memory.list_mental_models(scoped_bank, request_context=dan)
    assert [m["name"] for m in page.items] == ["Team rules"]
    assert await memory.get_mental_model(scoped_bank, private["id"], request_context=dan) is None
    assert await memory.get_mental_model_history(scoped_bank, private["id"], request_context=dan) is None
    assert await memory.update_mental_model(scoped_bank, private["id"], name="x", request_context=dan) is None
    assert await memory.delete_mental_model(scoped_bank, private["id"], request_context=dan) is False
    mm_tags = await memory.list_mental_model_tags(scoped_bank, request_context=dan)
    assert [t["tag"] for t in mm_tags["items"]] == ["kind:rule"]

    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        read = await tool_read_mental_models(conn, scoped_bank, [private["id"]], tag_scope=DAN_SCOPE)
    assert read["mental_models"] == [] and read["not_read"] == [private["id"]]

    # A model Dan could not see once made is refused up front, not written and then 404'd.
    with pytest.raises(OperationValidationError) as refused:
        await memory.create_mental_model(
            scoped_bank, "Kate summary", "Summarize Kate", "", tags=["user:kate"], request_context=dan
        )
    assert refused.value.status_code == 403

    # A model he can see stores his scope in its refresh filter: tagged `kind:rule` and matched
    # loosely ("any" also admits untagged memories), it still never reads past `user:dan`/`kind:rule`.
    shared = await memory.create_mental_model(
        scoped_bank,
        "Rules digest",
        "Summarize the rules",
        "",
        tags=["kind:rule"],
        trigger={"tags_match": "any"},
        request_context=dan,
    )
    assert shared["trigger"]["tag_groups"] == [
        {"tags": ["kind:rule"], "match": "any", "resolve": "exact"},
        {"tags": ["user:dan", "kind:rule"], "match": "any_strict", "resolve": "exact"},
    ]


@pytest.mark.asyncio
async def test_reflect_is_confined(memory, scoped_bank):
    """Reflect's tools all run under the forced scope (the mock drives one recall, then done)."""
    dan = RequestContext(api_key="dan")
    result = await memory.reflect_async(
        bank_id=scoped_bank, query="What is Kate doing and what are the rules?", request_context=dan
    )
    seen = [fact for facts in result.based_on.values() for fact in facts]
    seen_texts = " | ".join(f.text for f in seen)
    assert seen, "the forced recall should have gathered Dan-visible evidence"
    assert "interviewing" not in seen_texts
    assert all(tags_satisfy_groups(f.tags, DAN_SCOPE) for f in seen)

    # The same reflect unscoped does reach Kate's private fact, so the check above is not vacuous.
    unscoped = await memory.reflect_async(
        bank_id=scoped_bank, query="What is Kate doing and what are the rules?", request_context=RequestContext()
    )
    assert "interviewing" in " | ".join(f.text for facts in unscoped.based_on.values() for f in facts)


@pytest.mark.asyncio
async def test_recall_chunks_follow_the_document_without_a_validator(memory, scoped_bank):
    """#5030: with no extension at all, a reader's own tag filter gates source text too."""
    memory._operation_validator = None  # the fixture restores the original afterwards
    filtered = await memory.recall_async(
        bank_id=scoped_bank,
        query="Can I deploy on Fridays?",
        tags=["user:dan", "kind:rule"],
        tags_match="any_strict",
        include_chunks=True,
        request_context=RequestContext(),
    )
    assert any("Fridays" in r.text for r in filtered.results), "the shared rule is still recalled"
    filtered_chunks = " | ".join(c.chunk_text for c in (filtered.chunks or {}).values())
    assert "interviewing" not in filtered_chunks

    unfiltered = await memory.recall_async(
        bank_id=scoped_bank, query="Can I deploy on Fridays?", include_chunks=True, request_context=RequestContext()
    )
    assert "interviewing" in " | ".join(c.chunk_text for c in (unfiltered.chunks or {}).values())


@pytest.mark.asyncio
async def test_entities_are_confined(memory, scoped_bank):
    """#5031: entities exist for a reader only through memories it can see."""
    dan = RequestContext(api_key="dan")
    admin = RequestContext()
    everything = await memory.list_entities(scoped_bank, request_context=admin)
    dan_view = await memory.list_entities(scoped_bank, request_context=dan)

    # Which entities the mock extracts is its business; what matters is that Dan's view is a
    # strict subset, and that an entity only Kate's private fact mentions is not in it.
    assert {e["canonical_name"] for e in dan_view["items"]} < {e["canonical_name"] for e in everything["items"]}
    assert dan_view["total"] == len(dan_view["items"])

    hidden = [
        e for e in everything["items"] if e["canonical_name"] not in {d["canonical_name"] for d in dan_view["items"]}
    ]
    assert hidden, "Kate's private fact mentions an entity Dan must not see"
    assert await memory.get_entity(scoped_bank, hidden[0]["id"], request_context=dan) is None
    assert await memory.get_entity(scoped_bank, hidden[0]["id"], request_context=admin) is not None

    # The same filter, asked for directly (no validator): what an OSS caller passes as `tags`.
    memory._operation_validator = None
    by_tags = await memory.list_entities(
        scoped_bank, tags=["user:dan", "kind:rule"], tags_match="any_strict", request_context=admin
    )
    assert by_tags["items"] == dan_view["items"]
    graph = await memory.get_entity_graph(
        scoped_bank, tags=["user:dan", "kind:rule"], tags_match="any_strict", request_context=admin
    )
    graph_names = {n["data"]["label"] for n in graph["nodes"]}
    assert graph_names <= {d["canonical_name"] for d in dan_view["items"]}


@pytest.mark.asyncio
async def test_knowledge_base_is_confined(memory, scoped_bank):
    admin = RequestContext()
    dan = RequestContext(api_key="dan")
    kate_folder = await memory.create_knowledge_folder(scoped_bank, "Kate HR", request_context=admin)
    kate_page = await memory.create_knowledge_page(
        scoped_bank,
        "Kate career",
        "What is Kate planning?",
        "Kate is interviewing.",
        parent_id=kate_folder["id"],
        tags=["user:kate"],
        request_context=admin,
    )
    rules_page = await memory.create_knowledge_page(
        scoped_bank,
        "Team rules",
        "What are the rules?",
        "No Friday deploys.",
        tags=["kind:rule"],
        request_context=admin,
    )
    empty_folder = await memory.create_knowledge_folder(scoped_bank, "Drafts", request_context=admin)

    tree = await memory.list_knowledge_nodes(scoped_bank, request_context=dan)
    assert {n["id"] for n in tree} == {rules_page["id"], empty_folder["id"]}
    assert await memory.get_knowledge_page(scoped_bank, kate_page["id"], request_context=dan) is None
    assert await memory.get_knowledge_page(scoped_bank, rules_page["id"], request_context=dan) is not None

    found = await memory.search_knowledge_pages(scoped_bank, "Kate interviewing rules", request_context=dan)
    assert [p["id"] for p in found] == [rules_page["id"]]
    exported = await memory.export_knowledge_base(scoped_bank, request_context=dan)
    assert "interviewing" not in str(exported)

    # Writes on what Dan cannot see read as missing; a page in someone else's folder is refused.
    assert await memory.delete_knowledge_node(scoped_bank, kate_folder["id"], request_context=dan) is False
    with pytest.raises(OperationValidationError) as refused:
        await memory.create_knowledge_page(
            scoped_bank, "Peek", "q", "", parent_id=kate_folder["id"], request_context=dan
        )
    assert refused.value.status_code == 404
    with pytest.raises(OperationValidationError):
        await memory.update_knowledge_node(scoped_bank, kate_page["id"], name="mine", request_context=dan)
    with pytest.raises(OperationValidationError) as hidden_tags:
        await memory.create_knowledge_page(scoped_bank, "Kate notes", "q", "", tags=["user:kate"], request_context=dan)
    assert hidden_tags.value.status_code == 403
