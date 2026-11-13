"""Knowledge banks and the extension points: the operation validator's knowledge-bank hooks, and
ingestion strategies.

Uses the fake embedder and helpers from test_knowledge_banks: what is under test is which hooks
the validator sees and what a strategy's output becomes, not retrieval quality.
"""

import json
import uuid

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.extensions import (
    BankListResult,
    BankListScope,
    KnowledgeBankReadOperation,
    KnowledgeBankWriteOperation,
    OperationValidatorExtension,
    ValidationResult,
)
from hindsight_api.knowledge import ingest_strategy
from hindsight_api.knowledge.ingest_strategy import (
    IngestExtension,
    IngestFile,
    IngestOutput,
    OutputFile,
    OutputPassage,
)
from tests.test_knowledge_banks import KB, _png, _search, _upload, fake_embedder, kb  # noqa: F401  (fixtures)

# --- access control (operation validator) ---------------------------------------------------------


class _Accepting(OperationValidatorExtension):
    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()


class HidesSecretKnowledgeBanks(_Accepting):
    """Rejects every knowledge-bank operation on a bank whose id contains "secret", and records them."""

    def __init__(self, scope: BankListScope | None = None) -> None:
        super().__init__({})
        self.scope = scope
        self.seen: list[tuple[str, str]] = []

    def _verdict(self, bank_id: str, operation: str) -> ValidationResult:
        self.seen.append((bank_id, operation))
        return ValidationResult.reject("not yours", 403) if "secret" in bank_id else ValidationResult.accept()

    async def validate_knowledge_bank_read(self, ctx):
        return self._verdict(ctx.bank_id, ctx.operation)

    async def validate_knowledge_bank_write(self, ctx):
        return self._verdict(ctx.bank_id, ctx.operation)

    async def validate_create_knowledge_bank(self, ctx):
        return self._verdict(ctx.bank_id, "create")

    async def knowledge_bank_list_scope(self, request_context):
        return self.scope

    async def filter_knowledge_bank_list(self, ctx):
        return BankListResult(banks=[b for b in ctx.banks if "secret" not in b["bank_id"]])


class RejectsEveryMemoryBankHook(_Accepting):
    """A validator written for memory banks only: it must never decide a knowledge-bank call."""

    async def validate_bank_read(self, ctx):
        return ValidationResult.reject("memory hook", 403)

    async def validate_bank_write(self, ctx):
        return ValidationResult.reject("memory hook", 403)

    async def validate_create_bank(self, ctx):
        return ValidationResult.reject("memory hook", 403)

    async def bank_list_scope(self, request_context):
        return BankListScope(bank_ids=[])


@pytest.fixture
def install_validator(memory):
    original = memory._operation_validator

    def install(validator: OperationValidatorExtension | None) -> OperationValidatorExtension | None:
        memory._operation_validator = validator
        return validator

    yield install
    memory._operation_validator = original


async def test_every_knowledge_bank_operation_goes_through_its_own_hook(api_client, kb, install_validator):  # noqa: F811
    secret = f"secret-{uuid.uuid4().hex[:8]}"
    assert (await api_client.post(KB, json={"bank_id": secret})).status_code == 200
    try:
        await _upload(api_client, kb, [("red.png", _png((255, 0, 0)), "image/png")], [{"document_id": "red"}])
        validator = install_validator(HidesSecretKnowledgeBanks())
        assert isinstance(validator, HidesSecretKnowledgeBanks)

        # An allowed bank: each call is checked as its own operation.
        assert (await api_client.get(f"{KB}/{kb}")).status_code == 200
        assert (await api_client.get(f"{KB}/{kb}/documents")).status_code == 200
        assert (await api_client.get(f"{KB}/{kb}/documents/red")).status_code == 200
        await _search(api_client, kb, query="red")
        r = await _upload(api_client, kb, [("blue.png", _png((0, 0, 255)), "image/png")], [{"document_id": "b"}])
        assert r.is_success, r.text
        assert (await api_client.delete(f"{KB}/{kb}/documents/b")).status_code == 200
        assert (await api_client.get(f"{KB}/{kb}/config")).status_code == 200
        r = await api_client.patch(f"{KB}/{kb}/config", json={"updates": {"text_passage_tokens": 64}})
        assert r.status_code == 200, r.text
        assert (await api_client.delete(f"{KB}/{kb}/config")).status_code == 200
        assert [op for bank, op in validator.seen if bank == kb] == [
            KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK,
            KnowledgeBankReadOperation.LIST_KNOWLEDGE_BANK_DOCUMENTS,
            KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK_DOCUMENT,
            KnowledgeBankReadOperation.SEARCH_KNOWLEDGE_BANK,
            KnowledgeBankWriteOperation.INGEST_KNOWLEDGE_BANK,
            KnowledgeBankWriteOperation.DELETE_KNOWLEDGE_BANK_DOCUMENT,
            KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK_CONFIG,
            KnowledgeBankWriteOperation.UPDATE_KNOWLEDGE_BANK_CONFIG,
            KnowledgeBankWriteOperation.RESET_KNOWLEDGE_BANK_CONFIG,
        ]

        # A rejected bank answers the validator's status, never a 404 that would say it exists.
        for call in (
            api_client.get(f"{KB}/{secret}"),
            api_client.get(f"{KB}/{secret}/documents"),
            api_client.post(f"{KB}/{secret}/search", json={"query": "red"}),
            api_client.post(f"{KB}/{secret}/ingest", json={"items": [{"text": "x"}]}),
            api_client.get(f"{KB}/{secret}/documents/nope/file"),
            api_client.get(f"{KB}/{secret}/config"),
            api_client.patch(f"{KB}/{secret}/config", json={"updates": {"text_passage_tokens": 64}}),
            api_client.delete(f"{KB}/{secret}/config"),
            api_client.delete(f"{KB}/{secret}"),
            api_client.post(KB, json={"bank_id": f"secret-{uuid.uuid4().hex[:8]}"}),
        ):
            r = await call
            assert r.status_code == 403 and r.json()["detail"] == "not yours", r.text
        assert (secret, KnowledgeBankWriteOperation.DELETE_KNOWLEDGE_BANK) in validator.seen
    finally:
        install_validator(None)
        await api_client.delete(f"{KB}/{secret}")


async def test_memory_bank_hooks_never_apply_to_a_knowledge_bank(api_client, install_validator, fake_embedder):  # noqa: F811
    install_validator(RejectsEveryMemoryBankHook({}))
    bank_id = f"kb-{uuid.uuid4().hex[:8]}"
    try:
        assert (await api_client.post(KB, json={"bank_id": bank_id})).status_code == 200
        r = await api_client.post(f"{KB}/{bank_id}/ingest", json={"items": [{"document_id": "d", "text": "red"}]})
        assert r.is_success, r.text
        assert (await api_client.get(f"{KB}/{bank_id}/documents/d")).status_code == 200
        assert bank_id in [b["bank_id"] for b in (await api_client.get(KB)).json()["banks"]]
        # Through the memory-bank config routes, a caller the validator rejects is rejected (403), and
        # does not learn from a 400 that the bank is a knowledge bank.
        memory_config = f"/v1/default/banks/{bank_id}/config"
        assert (await api_client.patch(memory_config, json={"updates": {"retain_mission": "x"}})).status_code == 403
        assert (await api_client.delete(memory_config)).status_code == 403
        assert (await api_client.delete(f"{KB}/{bank_id}")).status_code == 200
    finally:
        install_validator(None)
        await api_client.delete(f"{KB}/{bank_id}")


@pytest.mark.parametrize("declares_scope", [False, True])
async def test_the_bank_list_shows_only_what_the_validator_allows(
    api_client,
    install_validator,
    declares_scope,
    fake_embedder,  # noqa: F811
):
    tag = uuid.uuid4().hex[:8]
    ids = [f"kb-{tag}-open", f"kb-{tag}-secret"]
    for bank_id in ids:
        assert (await api_client.post(KB, json={"bank_id": bank_id})).status_code == 200
    try:
        # A validator either declares the allowed ids up front, or filters the whole list.
        install_validator(HidesSecretKnowledgeBanks(BankListScope(bank_ids=[ids[0]]) if declares_scope else None))
        page = (await api_client.get(KB, params={"q": tag})).json()
        assert [b["bank_id"] for b in page["banks"]] == [ids[0]] and page["total"] == 1
        assert (await api_client.get(KB, params={"q": tag, "offset": 1})).json() == {
            **page,
            "banks": [],
            "offset": 1,
        }
    finally:
        install_validator(None)
        for bank_id in ids:
            await api_client.delete(f"{KB}/{bank_id}")


async def test_reflect_leaves_out_a_linked_bank_the_caller_cannot_read(
    api_client,
    memory,
    kb,  # noqa: F811
    request_context,
    install_validator,
):
    from dataclasses import replace

    from hindsight_api.engine.reflect.tools import tool_recall

    secret = f"secret-{uuid.uuid4().hex[:8]}"
    assert (await api_client.post(KB, json={"bank_id": secret})).status_code == 200
    mem_bank = f"mem-{uuid.uuid4().hex[:8]}"
    try:
        for bank_id, doc in ((kb, "open"), (secret, "hidden")):
            await _upload(api_client, bank_id, [(f"{doc}.png", _png((255, 0, 0)), "image/png")], [{"document_id": doc}])
        await api_client.post(f"/v1/default/banks/{mem_bank}/memories", json={"items": [{"content": "Red cars."}]})
        r = await api_client.patch(
            f"/v1/default/banks/{mem_bank}/config", json={"updates": {"knowledge_banks": [kb, secret]}}
        )
        assert r.status_code == 200, r.text
        validator = install_validator(HidesSecretKnowledgeBanks())
        assert isinstance(validator, HidesSecretKnowledgeBanks)

        # Reflect's recall tool runs internally, but the linked banks are checked as the caller.
        internal = replace(request_context, internal=True)
        linked = await tool_recall(memory, mem_bank, "red", internal, include_documents=True)
        assert [d["document_id"] for d in linked["documents"]] == ["open"]
        assert (secret, KnowledgeBankReadOperation.SEARCH_KNOWLEDGE_BANK) in validator.seen
    finally:
        install_validator(None)
        await memory.delete_bank(mem_bank, request_context=request_context)
        await api_client.delete(f"{KB}/{secret}")


# --- ingestion strategies ------------------------------------------------------------------------


class Unbundle(IngestExtension):
    """A JSON file of {name: text} becomes one raw Markdown file per entry."""

    async def ingest(self, file: IngestFile) -> IngestOutput:
        return IngestOutput(
            files=[
                OutputFile(name=name, data=text.encode(), filename=f"{name}.md", content_type="text/markdown")
                for name, text in json.loads(file.data).items()
            ]
        )


class LinePassages(IngestExtension):
    """Already chunked: one text passage per non-empty line, prefixed by its ``prefix`` setting."""

    version = "test-1"

    async def ingest(self, file: IngestFile) -> IngestOutput:
        lines = [line for line in file.data.decode().splitlines() if line.strip()]
        prefix = self.settings["prefix"]
        return IngestOutput(passages=[OutputPassage(modality="text", text=f"{prefix} {x}") for x in lines])


class EmitsAnUnreadableFile(IngestExtension):
    async def ingest(self, file: IngestFile) -> IngestOutput:
        return IngestOutput(files=[OutputFile(name="x", data=b"\x00", filename="x.bin")])


def _entry(cls: type, **settings) -> dict:
    return {"extension": f"{__name__}:{cls.__name__}", "settings": settings}


STRATEGIES = {
    # .bundle files are unpacked; .log files are chunked by the extension, by extension or content type.
    "assets": {".bundle": _entry(Unbundle), "text/x-log": _entry(LinePassages, prefix="LOG")},
    "other-prefix": {".log": _entry(LinePassages, prefix="OTHER")},
    "broken": {".bundle": _entry(EmitsAnUnreadableFile)},
}


@pytest.fixture
def strategies(monkeypatch):
    def configure(default: str | None = None) -> None:
        monkeypatch.setenv("HINDSIGHT_API_INGEST_STRATEGIES", json.dumps(STRATEGIES))
        if default:
            monkeypatch.setenv("HINDSIGHT_API_INGEST_DEFAULT_STRATEGY", default)
        else:
            monkeypatch.delenv("HINDSIGHT_API_INGEST_DEFAULT_STRATEGY", raising=False)
        clear_config_cache()

    configure()
    yield configure
    monkeypatch.delenv("HINDSIGHT_API_INGEST_STRATEGIES")
    monkeypatch.delenv("HINDSIGHT_API_INGEST_DEFAULT_STRATEGY", raising=False)
    clear_config_cache()


def _passages(detail: dict) -> list[str]:
    return [p["text"] for p in detail["passages"]]


async def test_a_strategy_applies_only_when_chosen(api_client, kb, strategies):  # noqa: F811
    log = b"boot ok\n\nred alert raised\n"
    # Nothing chosen: we chunk the file, whatever the strategies map.
    await _upload(api_client, kb, [("app.log", log, "text/x-log")], [{"document_id": "plain"}])
    plain = (await api_client.get(f"{KB}/{kb}/documents/plain")).json()
    assert plain["passages"] and not any(t.startswith("LOG") for t in _passages(plain))

    # Chosen on the item: the extension its type maps to chunks it, with that entry's settings.
    await _upload(api_client, kb, [("app.log", log, "text/x-log")], [{"document_id": "log", "strategy": "assets"}])
    assert _passages((await api_client.get(f"{KB}/{kb}/documents/log")).json()) == [
        "LOG boot ok",
        "LOG red alert raised",
    ]

    # A type the chosen strategy does not map is chunked by us, as with no strategy.
    r = await api_client.post(
        f"{KB}/{kb}/ingest", json={"items": [{"document_id": "t", "text": "red", "strategy": "assets"}]}
    )
    assert r.is_success, r.text
    assert _passages((await api_client.get(f"{KB}/{kb}/documents/t")).json()) == ["red"]

    # An unknown name is refused before anything is stored.
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"text": "x", "strategy": "nope"}]})
    assert r.status_code == 400 and "unknown ingestion strategy 'nope'" in r.json()["detail"]


async def test_the_server_default_strategy_applies_unless_the_item_names_another(api_client, kb, strategies):  # noqa: F811
    strategies(default="assets")
    log = b"one\ntwo\n"
    await _upload(api_client, kb, [("a.log", log, "text/x-log")], [{"document_id": "default"}])
    assert _passages((await api_client.get(f"{KB}/{kb}/documents/default")).json()) == ["LOG one", "LOG two"]
    await _upload(
        api_client, kb, [("a.log", log, "text/x-log")], [{"document_id": "named", "strategy": "other-prefix"}]
    )
    assert _passages((await api_client.get(f"{KB}/{kb}/documents/named")).json()) == ["OTHER one", "OTHER two"]


async def test_an_extension_unpacks_a_file_into_documents_we_chunk(api_client, kb, strategies):  # noqa: F811
    bundle = json.dumps({"keys": "API keys expire after 90 days.", "red": "The red car is fast."}).encode()
    opts = [{"document_id": "pack", "strategy": "assets"}]
    r = await _upload(api_client, kb, [("pack.bundle", bundle, "application/octet-stream")], opts)
    assert r.is_success, r.text  # a type only the extension reads is accepted
    docs = {d["document_id"]: d for d in (await api_client.get(f"{KB}/{kb}/documents")).json()["documents"]}
    assert set(docs) == {"pack", "pack/keys", "pack/red"}
    assert docs["pack"]["status"] == "ready" and docs["pack"]["passage_count"] == 0
    assert docs["pack/keys"]["status"] == "ready" and docs["pack/keys"]["filename"] == "keys.md"
    assert docs["pack/keys"]["source_document_id"] == "pack" and docs["pack"]["source_document_id"] is None
    hits = await _search(api_client, kb, query="red")
    assert hits[0]["document_id"] == "pack/red"

    # Ingested again with one entry fewer, the document it no longer holds is gone.
    smaller = json.dumps({"keys": "API keys expire after 90 days."}).encode()
    await _upload(api_client, kb, [("pack.bundle", smaller, "application/octet-stream")], opts)
    docs = (await api_client.get(f"{KB}/{kb}/documents")).json()["documents"]
    assert {d["document_id"] for d in docs} == {"pack", "pack/keys"}

    # Deleting the source deletes what was split out of it.
    assert (await api_client.delete(f"{KB}/{kb}/documents/pack")).status_code == 200
    assert (await api_client.get(f"{KB}/{kb}/documents")).json()["documents"] == []


async def test_a_raw_file_we_cannot_chunk_fails_its_source_with_the_reason(api_client, kb, strategies):  # noqa: F811
    opts = [{"document_id": "pack", "strategy": "broken"}]
    await _upload(api_client, kb, [("pack.bundle", b"{}", "application/octet-stream")], opts)
    doc = (await api_client.get(f"{KB}/{kb}/documents/pack")).json()
    assert doc["status"] == "failed" and "unsupported file type" in doc["error"]


def test_strategies_are_parsed_and_matched_by_extension_or_content_type():
    parsed = ingest_strategy.parse_strategies(
        {"zip": {".ZIP": {"extension": "m:C"}, "application/zip": {"extension": "m:C", "settings": {"a": 1}}}}
    )["zip"]
    assert parsed.handler_for("application/octet-stream", "assets.zip").settings == {}
    assert parsed.handler_for("application/zip; charset=binary", None).settings == {"a": 1}
    assert parsed.handler_for("text/plain", "notes.txt") is None
    with pytest.raises(ValueError, match="needs an 'extension'"):
        ingest_strategy.parse_strategies({"zip": {".zip": {}}})
    with pytest.raises(ValueError, match="either files or passages"):
        IngestOutput()


async def test_a_split_document_stays_ready_when_another_document_of_its_batch_fails(
    api_client,
    kb,  # noqa: F811
    strategies,
    fake_embedder,  # noqa: F811
    monkeypatch,
):
    import aiohttp

    real = fake_embedder.embed_documents

    async def down_for_b(items):
        if any("bravo" in (i.text or "") for i in items):
            raise aiohttp.ClientConnectionError("embedder down")
        return await real(items)

    monkeypatch.setattr(fake_embedder, "embed_documents", down_for_b)
    items = [
        {
            "document_id": "pack",
            "text": json.dumps({"a": "alpha red"}),
            "filename": "pack.bundle",
            "strategy": "assets",
        },
        {"document_id": "b", "text": "bravo blue"},
    ]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.status_code == 503, r.text
    # The split committed before the embedder went down on "b": it is not reset by that failure.
    statuses = {d: (await api_client.get(f"{KB}/{kb}/documents/{d}")).json()["status"] for d in ("pack", "b")}
    assert statuses == {"pack": "ready", "b": "failed"}
