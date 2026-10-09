"""A validator's forced tag scope (``resolve_tag_scope``) confines a caller everywhere.

Two people share a bank: Kate writes under ``user:kate``, Dan under ``user:dan``, and one of
Kate's facts is also tagged ``kind:rule`` — the shared scope. Dan is confined to
``user:dan`` OR ``kind:rule``. Every read and by-id write below must keep Kate's private
memories, her document, its text and her mental models out of Dan's reach, while still
showing him the shared rule.
"""

import ast
import uuid
from pathlib import Path

import pytest

from hindsight_api.engine.memory_engine import _resolve_refresh_tag_filtering, _scope_mental_model_trigger
from hindsight_api.engine.reflect.tools import tool_expand, tool_read_mental_models
from hindsight_api.engine.schema import fq_store_table_explicit
from hindsight_api.engine.search.tags import TagGroupLeaf, tags_satisfy_groups, tags_writable
from hindsight_api.extensions import (
    BankReadOperation,
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

    def __init__(self, scopes, writes=None):
        super().__init__({})
        self.scopes = scopes
        self.writes = writes or {}

    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def resolve_tag_scope(self, ctx: TagScopeContext):
        return self.scopes.get(ctx.request_context.api_key)

    async def resolve_write_tag_scope(self, ctx: TagScopeContext):
        return self.writes.get(ctx.request_context.api_key)


def test_tags_satisfy_groups():
    assert tags_satisfy_groups(["user:dan"], DAN_SCOPE)
    assert tags_satisfy_groups(["user:kate", "kind:rule"], DAN_SCOPE)
    assert not tags_satisfy_groups(["user:kate"], DAN_SCOPE)
    assert not tags_satisfy_groups([], DAN_SCOPE), "strict: untagged rows are outside the scope"
    assert tags_satisfy_groups(["anything"], None)


def test_scope_mental_model_trigger_ands_the_scope_into_the_refresh_filter():
    # The scope is recorded on its own, and the refresh AND-s it onto the model's own filter.
    scoped = _scope_mental_model_trigger({"mode": "delta"}, DAN_SCOPE)
    assert scoped == {
        "mode": "delta",
        "scope_tag_groups": [{"tags": ["user:dan", "kind:rule"], "match": "any_strict", "resolve": "exact"}],
    }
    # Re-applying the same scope does not stack it.
    assert _scope_mental_model_trigger(scoped, DAN_SCOPE) == scoped
    # No scope: the trigger is left exactly as it was.
    assert _scope_mental_model_trigger({"mode": "delta"}, None) == {"mode": "delta"}

    # Flat tags become a leaf under their resolved mode (all_strict by default), then the scope.
    resolved = _resolve_refresh_tag_filtering(["kind:rule"], scoped)
    assert resolved.tags is None
    assert resolved.tag_groups == [TagGroupLeaf(tags=["kind:rule"], match="all_strict"), *DAN_SCOPE]
    # Retagging the model changes what it reads: the scope did not freeze the old tags.
    retagged = _resolve_refresh_tag_filtering(["user:dan"], scoped)
    assert retagged.tag_groups == [TagGroupLeaf(tags=["user:dan"], match="all_strict"), *DAN_SCOPE]
    # An untagged model (the whole bank) is narrowed to the scope alone.
    assert _resolve_refresh_tag_filtering([], scoped).tag_groups == DAN_SCOPE


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
    assert _resolve_refresh_tag_filtering(shared["tags"], shared["trigger"]).tag_groups == [
        TagGroupLeaf(tags=["kind:rule"], match="any"),
        *DAN_SCOPE,
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


def test_tags_writable():
    assert tags_writable(["user:dan"], ["user:dan"])
    assert not tags_writable(["user:dan", "kind:rule"], ["user:dan"]), "every tag must be writable"
    assert tags_writable(["user:kate", "kind:rule"], ["user:kate", "kind:*"])
    assert not tags_writable([], ["user:dan"]), "an untagged item belongs to everyone"
    assert tags_writable([], None)


@pytest.mark.asyncio
async def test_writes_are_confined_to_the_write_scope(memory, scoped_bank):
    """The video's half: Dan reads the rules but cannot change them; Kate can.

    Dan writes only `user:dan`; Kate writes `user:kate` and `kind:rule`.
    """
    admin = RequestContext()
    dan = RequestContext(api_key="dan")
    kate = RequestContext(api_key="kate")
    memory._operation_validator.scopes["kate"] = [TagGroupLeaf(tags=["user:kate", "kind:rule"], match="any_strict")]
    memory._operation_validator.writes = {"dan": ["user:dan"], "kate": ["user:kate", "kind:rule"]}
    await memory.update_bank_config(
        scoped_bank,
        {
            "retain_strategies": {
                "rules": {
                    "entity_labels": [{"key": "kind", "type": "value", "tag": True, "values": [{"value": "rule"}]}]
                }
            }
        },
        request_context=admin,
    )

    def refused(exc_info) -> int:
        return exc_info.value.status_code

    # Retain: an explicit kind:rule tag, or a strategy whose label can produce one, is refused.
    with pytest.raises(OperationValidationError) as e:
        await memory.retain_batch_async(
            bank_id=scoped_bank,
            contents=[{"content": "Call it compensation management.", "tags": ["user:dan", "kind:rule"]}],
            request_context=dan,
        )
    assert refused(e) == 403
    with pytest.raises(OperationValidationError) as e:
        await memory.retain_batch_async(
            bank_id=scoped_bank,
            contents=[{"content": "Call it compensation management.", "tags": ["user:dan"], "strategy": "rules"}],
            request_context=dan,
        )
    assert refused(e) == 403 and "kind:rule" in e.value.reason
    # His own note, without the rules strategy, goes through.
    await memory.retain_batch_async(
        bank_id=scoped_bank,
        contents=[{"content": "Dan's ad says compensation management.", "tags": ["user:dan"]}],
        request_context=dan,
    )
    # Kate may use the rules strategy.
    await memory.retain_batch_async(
        bank_id=scoped_bank,
        contents=[{"content": "The product is called payroll software.", "tags": ["user:kate"], "strategy": "rules"}],
        request_context=kate,
    )

    # Curating the shared rule fact: Dan can read it, so it is a 403, not a 404.
    everything = await memory.list_memory_units(scoped_bank, request_context=admin)
    rule = next(i for i in everything["items"] if "Fridays" in i["text"] and i["fact_type"] == "world")
    with pytest.raises(OperationValidationError) as e:
        await memory.update_memory_unit(scoped_bank, rule["id"], text="Deploy whenever.", request_context=dan)
    assert refused(e) == 403

    # The shared rules model: readable by Dan, writable only by Kate.
    rules = await memory.create_mental_model(
        scoped_bank, "Ad rules", "What are the rules?", "Payroll software.", tags=["kind:rule"], request_context=kate
    )
    assert (await memory.get_mental_model(scoped_bank, rules["id"], request_context=dan)) is not None
    for attempt in (
        memory.update_mental_model(scoped_bank, rules["id"], content="Compensation management.", request_context=dan),
        memory.clear_mental_model(scoped_bank, rules["id"], request_context=dan),
        memory.delete_mental_model(scoped_bank, rules["id"], request_context=dan),
        memory.create_mental_model(scoped_bank, "Dan's rules", "q", "", tags=["kind:rule"], request_context=dan),
    ):
        with pytest.raises(OperationValidationError) as e:
            await attempt
        assert refused(e) == 403
    updated = await memory.update_mental_model(
        scoped_bank, rules["id"], content="Payroll software, in every ad.", request_context=kate
    )
    assert updated is not None and "every ad" in updated["content"]

    # Knowledge pages follow the same rule: Dan cannot delete or rename the rules page.
    page = await memory.create_knowledge_page(
        scoped_bank, "Rules page", "What are the rules?", "", tags=["kind:rule"], request_context=kate
    )
    with pytest.raises(OperationValidationError) as e:
        await memory.delete_knowledge_node(scoped_bank, page["id"], request_context=dan)
    assert refused(e) == 403
    with pytest.raises(OperationValidationError):
        await memory.update_knowledge_node(scoped_bank, page["id"], name="Dan's page", request_context=dan)


@pytest.mark.asyncio
async def test_every_other_surface_respects_both_scopes(memory, scoped_bank):
    """The review's checklist: each remaining read and write a scoped caller can reach."""
    admin = RequestContext()
    dan = RequestContext(api_key="dan")
    kate = RequestContext(api_key="kate")
    memory._operation_validator.scopes["kate"] = [TagGroupLeaf(tags=["user:kate", "kind:rule"], match="any_strict")]
    memory._operation_validator.writes = {"dan": ["user:dan", "topic:*"], "kate": ["user:kate", "kind:rule"]}
    everything = await memory.list_memory_units(scoped_bank, request_context=admin)
    rule = next(i for i in everything["items"] if "Fridays" in i["text"] and i["fact_type"] == "world")

    # Graph and timeseries count only what Dan can read.
    graph = await memory.get_graph_data(scoped_bank, request_context=dan)
    graph_text = str(graph)
    assert "interviewing" not in graph_text and "Fridays" in graph_text
    timeseries = await memory.get_memories_timeseries(scoped_bank, period="7d", request_context=dan)
    admin_series = await memory.get_memories_timeseries(scoped_bank, period="7d", request_context=admin)
    assert str(timeseries) != str(admin_series), "Kate's private memories must not be counted for Dan"

    # Documents: a document Dan can read but not write; new tags must be writable too.
    with pytest.raises(OperationValidationError) as e:
        await memory.update_document("dan-notes", scoped_bank, tags=["user:dan", "kind:rule"], request_context=dan)
    assert e.value.status_code == 403
    with pytest.raises(OperationValidationError) as e:
        await memory.reprocess_document(scoped_bank, "kate-sync", request_context=dan)
    assert e.value.status_code == 404
    with pytest.raises(OperationValidationError) as e:
        await memory.clear_observations_for_memory(scoped_bank, rule["id"], request_context=dan)
    assert e.value.status_code == 403

    # The async and file retain paths check before queueing: the worker runs unscoped.
    with pytest.raises(OperationValidationError):
        await memory.submit_async_retain(
            bank_id=scoped_bank, contents=[{"content": "x", "tags": ["kind:rule"]}], request_context=dan
        )
    with pytest.raises(OperationValidationError):
        await memory.submit_async_file_retain(
            bank_id=scoped_bank,
            file_items=[{"file": None, "document_id": "kate-sync", "tags": ["user:dan"]}],
            document_tags=None,
            request_context=dan,
        )
    # Explicit observation scopes are writes too; "shared" writes untagged observations.
    for scopes in ("shared", [["user:kate"]]):
        with pytest.raises(OperationValidationError):
            await memory.retain_batch_async(
                bank_id=scoped_bank,
                contents=[{"content": "x", "tags": ["user:dan"], "observation_scopes": scopes}],
                request_context=dan,
            )

    # The operation payload is the raw request: never shown to a scoped caller.
    op = await memory.submit_async_retain(
        bank_id=scoped_bank,
        contents=[{"content": "Kate's private retain", "tags": ["user:kate"]}],
        request_context=kate,
    )
    status = await memory.get_operation_status(
        scoped_bank, op["operation_id"], request_context=dan, include_payload=True
    )
    assert not status.get("task_payload")

    # Whole-bank operations cannot be narrowed to a scope.
    for attempt in (
        memory.submit_bank_export_async(scoped_bank, request_context=dan),
        memory.delete_bank(scoped_bank, request_context=dan),
        memory.update_bank_config(scoped_bank, {"retain_mission": "x"}, request_context=dan),
        memory.update_bank(scoped_bank, mission="x", request_context=dan),
        memory.update_bank_disposition(
            scoped_bank, {"skepticism": 3, "literalism": 3, "empathy": 3}, request_context=dan
        ),
        memory.submit_async_consolidation(bank_id=scoped_bank, request_context=dan, caller_requested=True),
        memory.retry_failed_consolidation(scoped_bank, request_context=dan),
    ):
        with pytest.raises(OperationValidationError) as e:
            await attempt
        assert e.value.status_code == 403

    # Directives: bank-wide (untagged) ones apply to everyone; tagged ones follow the scopes.
    await memory.create_directive(scoped_bank, "Be brief", "Answer briefly.", request_context=admin)
    secret = await memory.create_directive(
        scoped_bank, "Kate only", "Mention the interview.", tags=["user:kate"], request_context=admin
    )
    names = {d["name"] for d in (await memory.list_directives(scoped_bank, request_context=dan)).items}
    assert names == {"Be brief"}
    assert await memory.get_directive(scoped_bank, secret["id"], request_context=dan) is None
    with pytest.raises(OperationValidationError):
        await memory.create_directive(scoped_bank, "Everyone", "Say hi.", request_context=dan)
    assert await memory.delete_directive(scoped_bank, secret["id"], request_context=dan) is False

    # A shared model Dan can read: he cannot refresh it (a refresh rewrites it), nor retag it away.
    rules = await memory.create_mental_model(
        scoped_bank, "Ad rules", "What are the rules?", "", tags=["kind:rule"], request_context=kate
    )
    with pytest.raises(OperationValidationError) as e:
        await memory.submit_async_refresh_mental_model(scoped_bank, rules["id"], request_context=dan)
    assert e.value.status_code == 403
    # His own model can be retagged, and the refresh then reads the new tags.
    own = await memory.create_mental_model(
        scoped_bank, "Mine", "What am I doing?", "", tags=["user:dan"], request_context=dan
    )
    retagged = await memory.update_mental_model(
        scoped_bank, own["id"], tags=["user:dan", "topic:ads"], request_context=dan
    )
    assert retagged is not None
    resolved = _resolve_refresh_tag_filtering(retagged["tags"], retagged["trigger"])
    assert TagGroupLeaf(tags=["user:dan", "topic:ads"], match="all_strict") in resolved.tag_groups

    # A folder holding a page Dan can only read: renaming or moving into it is refused.
    folder = await memory.create_knowledge_folder(scoped_bank, "Shared", request_context=admin)
    await memory.create_knowledge_page(
        scoped_bank, "Rules page", "q", "", parent_id=folder["id"], tags=["kind:rule"], request_context=kate
    )
    my_page = await memory.create_knowledge_page(
        scoped_bank, "My page", "q", "", tags=["user:dan"], request_context=dan
    )
    with pytest.raises(OperationValidationError) as e:
        await memory.update_knowledge_node(scoped_bank, folder["id"], name="Dan's folder", request_context=dan)
    assert e.value.status_code == 403
    with pytest.raises(OperationValidationError) as e:
        await memory.update_knowledge_node(scoped_bank, my_page["id"], parent_id=folder["id"], request_context=dan)
    assert e.value.status_code == 403


@pytest.mark.asyncio
async def test_the_default_template_applies_whole_when_a_scoped_caller_creates_the_bank(memory, monkeypatch):
    """The template is the server's, not the caller's: its untagged model and directive must not
    be refused because the caller who happened to create the bank is scoped."""
    from hindsight_api.config import _get_raw_config

    monkeypatch.setattr(
        _get_raw_config(),
        "default_bank_template",
        {
            "version": "1",
            "mental_models": [{"id": "team-overview", "name": "Team overview", "source_query": "What is the team?"}],
            "directives": [{"name": "Be brief", "content": "Answer briefly."}],
        },
    )
    bank_id = f"tag-scope-template-{uuid.uuid4().hex[:8]}"
    original = memory._operation_validator
    memory._operation_validator = _ScopeByApiKey({"dan": DAN_SCOPE}, writes={"dan": ["user:dan"]})
    admin = RequestContext()
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Dan's first note.", "tags": ["user:dan"]}],
            request_context=RequestContext(api_key="dan"),
        )
        assert await memory.get_mental_model(bank_id, "team-overview", request_context=admin) is not None
        names = {d["name"] for d in (await memory.list_directives(bank_id, request_context=admin)).items}
        assert names == {"Be brief"}
    finally:
        memory._operation_validator = original
        await memory.delete_bank(bank_id, request_context=admin)


@pytest.mark.asyncio
async def test_bank_wide_reads_and_writes_are_refused_to_a_scoped_caller(memory, scoped_bank):
    """Operations answered for, or acting on, the whole bank ignore the caller's tag scope, so a
    scoped caller is refused them (403) instead of being shown or allowed bank-wide effects:
    statistics, operations, webhooks and their deliveries (which carry memory content), aliases,
    the audit log and LLM request log (every caller's payloads), the mission, a
    synchronous consolidation run, and bank template import."""
    dan = RequestContext(api_key="dan")
    admin = RequestContext()
    some_id = str(uuid.uuid4())
    hook = uuid.uuid4()
    attempts = {
        "stats": lambda: memory.get_bank_stats(scoped_bank, request_context=dan),
        "freshness": lambda: memory.get_bank_freshness(scoped_bank, request_context=dan),
        "operations": lambda: memory.list_operations(scoped_bank, request_context=dan),
        "webhooks": lambda: memory.list_webhooks(scoped_bank, request_context=dan),
        "webhook deliveries": lambda: memory.list_webhook_deliveries(
            scoped_bank, hook, limit=10, cursor=None, request_context=dan
        ),
        "create webhook": lambda: memory.create_webhook(
            scoped_bank,
            webhook_id=hook,
            url="https://example.com/hook",
            secret=None,
            event_types=["retain.completed"],
            enabled=True,
            http_config_json="{}",
            request_context=dan,
        ),
        "update webhook": lambda: memory.update_webhook(
            scoped_bank, hook, set_clauses=["url = $1"], params=["https://example.org/x"], request_context=dan
        ),
        "delete webhook": lambda: memory.delete_webhook(scoped_bank, hook, request_context=dan),
        "create alias": lambda: memory.create_bank_alias(scoped_bank, "dans-alias", request_context=dan),
        "alias primary": lambda: memory.set_bank_alias_primary(scoped_bank, "dans-alias", True, request_context=dan),
        "delete alias": lambda: memory.delete_bank_alias(scoped_bank, "dans-alias", request_context=dan),
        "cancel operation": lambda: memory.cancel_operation(scoped_bank, some_id, request_context=dan),
        "retry operation": lambda: memory.retry_operation(scoped_bank, some_id, request_context=dan),
        "delete operation": lambda: memory.delete_operation(scoped_bank, some_id, request_context=dan),
        "audit log": lambda: memory.list_audit_logs(scoped_bank, request_context=dan),
        "audit log stats": lambda: memory.audit_log_stats(scoped_bank, request_context=dan),
        "llm requests": lambda: memory.list_llm_requests(scoped_bank, request_context=dan),
        "llm request stats": lambda: memory.llm_request_stats(scoped_bank, request_context=dan),
        "set mission": lambda: memory.set_bank_mission(scoped_bank, "x", request_context=dan),
        "merge mission": lambda: memory.merge_bank_mission(scoped_bank, "x", request_context=dan),
        "run consolidation": lambda: memory.run_consolidation(scoped_bank, request_context=dan),
    }
    not_refused = {}
    for name, attempt in attempts.items():
        try:
            await attempt()
            not_refused[name] = "allowed"
        except OperationValidationError as e:
            if e.status_code != 403 or "tag-scoped caller" not in e.reason:
                not_refused[name] = f"{e.status_code}: {e.reason}"
    assert not_refused == {}

    with pytest.raises(OperationValidationError) as e:
        async with memory.bank_template_import_authorization(
            scoped_bank,
            config_updates={},
            bank_writes=[],
            mental_model_ids=[],
            bank_exists=True,
            request_context=dan,
        ):
            pass
    assert e.value.status_code == 403

    # An unscoped caller is not affected.
    await memory.get_bank_stats(scoped_bank, request_context=admin)
    assert await memory.list_audit_logs(scoped_bank, request_context=admin) is not None
    assert await memory.list_llm_requests(scoped_bank, request_context=admin) is not None


@pytest.mark.asyncio
async def test_reflect_works_where_bank_statistics_are_refused(memory, scoped_bank):
    """Reflect reads the bank's freshness only for itself, under its own authorization, so a
    deployment that refuses bank-wide statistics to scoped callers must not break their
    reflect. (It used to: reflect went through the public freshness read.)"""
    validator = memory._operation_validator
    seen: list = []

    async def validate_bank_read(ctx):
        seen.append(ctx.operation)
        if ctx.operation == BankReadOperation.GET_BANK_STATS and ctx.request_context.api_key in validator.scopes:
            return ValidationResult.reject("bank-wide statistics refused", status_code=403)
        return ValidationResult.accept()

    validator.validate_bank_read = validate_bank_read
    dan = RequestContext(api_key="dan")
    result = await memory.reflect_async(bank_id=scoped_bank, query="What are the rules?", request_context=dan)
    assert result.text is not None
    assert BankReadOperation.GET_BANK_STATS not in seen


@pytest.mark.asyncio
async def test_dry_run_extract_refuses_labels_the_caller_cannot_write(memory, scoped_bank):
    """Previewing extraction with entity labels that could add an unwritable tag is refused,
    exactly as retaining with them is: nothing is stored either way, but the preview would
    spend the bank's LLM budget on a path retain refuses."""
    memory._operation_validator.writes = {"dan": ["user:dan", "topic:*"]}
    labels = [{"key": "kind", "type": "value", "tag": True, "values": [{"value": "rule"}]}]
    dan = RequestContext(api_key="dan")
    with pytest.raises(OperationValidationError) as e:
        await memory.extract_dry_run(
            scoped_bank, "Every ad says payroll software.", overrides={"entity_labels": labels}, request_context=dan
        )
    assert e.value.status_code == 403 and "kind:rule" in e.value.reason

    # Labels within the write scope preview fine, and an unscoped caller is not affected.
    own = [{"key": "topic", "type": "value", "tag": True, "values": [{"value": "ads"}]}]
    await memory.extract_dry_run(scoped_bank, "x", overrides={"entity_labels": own}, request_context=dan)
    await memory.extract_dry_run(
        scoped_bank, "x", overrides={"entity_labels": labels}, request_context=RequestContext()
    )


# Methods that validate a bank read or write but need no scope handling of their own:
# bank settings that carry no memory content, or helpers whose callers apply the scope.
_UNSCOPED_BY_DESIGN = {
    "_authorize_bank_profile_read": "bank profile (name, mission, disposition): settings, no memory content",
    "_authorize_bank_config_read": "bank config: settings, no memory content",
    "list_bank_aliases": "alias names: settings, no memory content",
    "check_bank_llm": "probes the bank's LLM connectivity; returns status only",
    "get_entity_state": "returns the entity with no observations",
    "export_knowledge_base": "built from list_knowledge_nodes / get_knowledge_page, which apply the scope",
    "_knowledge_read_filter": "computes the knowledge-tree scope filter itself",
    "authorize_bank_template_import_write": (
        "only inside bank_template_import_authorization, which refuses scoped callers"
    ),
    # Queued on the caller's behalf after every retain and delete (failures are only logged),
    # so refusing scoped callers would silently skip maintenance after their writes. They
    # drain the bank's own queues and return nothing.
    "submit_async_graph_maintenance": "queued after every write, including a scoped caller's; returns no data",
    "submit_async_vector_index_maintenance": "queued after every write, including a scoped caller's; returns no data",
}
_SCOPE_HANDLING = {
    "_tag_scope",
    "_write_tag_scope",
    "_authorize_bank_read",
    "_refuse_bank_wide_if_scoped",
    "_refuse_unwritable",
    "_refuse_unwritable_label_tags",
    "_require_writable",
    "_check_retain_writes",
    "_memory_in_tag_scope",
    "_document_in_tag_scope",
    "_require_knowledge_subtree_writable",
    "_require_knowledge_nodes_in_tag_scope",
    "_directive_writable",
    "_knowledge_read_filter",
}


def test_every_validated_bank_operation_takes_a_stance_on_scoped_callers():
    """Deny by default: every MemoryEngine method that validates a bank read or write either
    handles the caller's tag scope (narrows to it, or refuses scoped callers) or is listed in
    _UNSCOPED_BY_DESIGN with a reason. A new bank-wide operation that forgets both fails here,
    instead of silently answering a scoped caller for the whole bank."""
    import hindsight_api.engine.memory_engine as engine_module

    tree = ast.parse(Path(engine_module.__file__).read_text())
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MemoryEngine")
    unhandled = []
    for fn in engine.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        if not attrs & {"validate_bank_read", "validate_bank_write"}:
            continue
        if attrs & _SCOPE_HANDLING or fn.name in _UNSCOPED_BY_DESIGN:
            continue
        unhandled.append(fn.name)
    assert unhandled == [], (
        "These validate a bank operation without handling a tag-scoped caller; narrow to the "
        f"scope, refuse with _refuse_bank_wide_if_scoped, or justify in _UNSCOPED_BY_DESIGN: {unhandled}"
    )
    stale = [
        name
        for name in _UNSCOPED_BY_DESIGN
        if not any(isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == name for fn in engine.body)
    ]
    assert stale == [], f"_UNSCOPED_BY_DESIGN names methods that no longer exist: {stale}"
