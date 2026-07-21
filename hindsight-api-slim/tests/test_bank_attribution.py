"""
Tests for per-bank provider cost attribution.

Covers the opt-in `HINDSIGHT_API_LLM_SEND_BANK_AS_USER` plumbing that tags
outbound OpenAI-compatible LLM and embedding calls with `user=<bank_id>`, the
`_current_bank_id` engine ContextVar that carries the bank across the async call
chain, and its propagation into the embedding executor thread.

All deterministic — no network, stdlib/pytest only.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.bank_attribution import apply_bank_attribution
from hindsight_api.engine.cross_encoder import LiteLLMCrossEncoder
from hindsight_api.engine.embeddings import OpenAIEmbeddings
from hindsight_api.engine.memory_engine import (
    MemoryEngine,
    _bind_bank_id,
    _current_bank_id,
    get_current_bank_id,
)
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.engine.retain.embedding_utils import generate_embeddings_batch
from hindsight_api.models import RequestContext


@pytest.fixture(autouse=True)
def restore_send_bank_env():
    """Save/restore the attribution env var and clear the cached config."""
    from hindsight_api.config import clear_config_cache

    original = os.environ.get("HINDSIGHT_API_LLM_SEND_BANK_AS_USER")
    clear_config_cache()
    yield
    if original is None:
        os.environ.pop("HINDSIGHT_API_LLM_SEND_BANK_AS_USER", None)
    else:
        os.environ["HINDSIGHT_API_LLM_SEND_BANK_AS_USER"] = original
    clear_config_cache()


def _set_flag(enabled: bool) -> None:
    from hindsight_api.config import clear_config_cache

    os.environ["HINDSIGHT_API_LLM_SEND_BANK_AS_USER"] = "true" if enabled else "false"
    clear_config_cache()


# ── ContextVar lifecycle ──────────────────────────────────────────────────────


class TestBankContextVar:
    def test_default_is_none(self):
        assert get_current_bank_id() is None


class TestBindBankIdDecorator:
    """The engine binds the bank via @_bind_bank_id on recall/retain/batch/reflect/task methods."""

    async def test_binds_named_arg_positional_and_keyword(self):
        @_bind_bank_id()
        async def op(bank_id: str, query: str) -> str | None:
            return get_current_bank_id()

        assert await op("user-pos", "q") == "user-pos"
        assert await op(bank_id="user-kw", query="q") == "user-kw"
        assert get_current_bank_id() is None

    async def test_extracts_dict_key(self):
        @_bind_bank_id("task_dict", key="bank_id")
        async def op(task_dict: dict) -> str | None:
            return get_current_bank_id()

        assert await op({"bank_id": "user-task", "type": "consolidation"}) == "user-task"
        assert await op({"type": "consolidation"}) is None
        assert get_current_bank_id() is None

    async def test_resets_on_exception(self):
        @_bind_bank_id()
        async def op(bank_id: str) -> None:
            assert get_current_bank_id() == "user-boom"
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await op("user-boom")
        assert get_current_bank_id() is None

    async def test_non_string_value_binds_none(self):
        @_bind_bank_id()
        async def op(bank_id: object) -> str | None:
            return get_current_bank_id()

        assert await op(12345) is None

    async def test_engine_provider_paths_bind_and_reset_their_bank_arguments(self):
        engine = object.__new__(MemoryEngine)
        engine._reflect_llm_config = None
        engine._operation_validator = None
        observed_bank_ids: list[str | None] = []

        with patch(
            "hindsight_api.engine.memory_engine.sanitize_text",
            side_effect=lambda value: observed_bank_ids.append(get_current_bank_id()) or value,
        ):
            with pytest.raises(ValueError, match="Memory LLM API key not set"):
                await engine.reflect_async("user-reflect", "question", request_context=RequestContext())

        assert observed_bank_ids == ["user-reflect", "user-reflect"]
        assert get_current_bank_id() is None

        with (
            patch.object(
                engine,
                "_authenticate_tenant",
                AsyncMock(side_effect=lambda _context: observed_bank_ids.append(get_current_bank_id())),
            ),
            patch.object(engine, "_get_backend", AsyncMock(side_effect=RuntimeError("stop after authentication"))),
        ):
            with pytest.raises(RuntimeError, match="stop after authentication"):
                await engine.update_memory_unit(
                    "user-update",
                    "54a647e5-0a22-4e5d-8504-b8bfca2a6142",
                    text="corrected",
                    request_context=RequestContext(),
                )

        assert observed_bank_ids[-1] == "user-update"
        assert get_current_bank_id() is None


# ── LLM provider: user injection ──────────────────────────────────────────────


class _SimpleJson(BaseModel):
    ok: bool


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
    )


def _chat_response(content: str = '{"ok": true}'):
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
    )
    return SimpleNamespace(choices=[choice], usage=None, error=None)


async def _call(llm: OpenAICompatibleLLM, create: AsyncMock):
    llm._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        return await llm.call(
            messages=[{"role": "user", "content": "ping"}],
            max_retries=0,
        )


async def test_user_injected_when_flag_on_and_bank_set():
    _set_flag(True)
    llm = _llm()
    create = AsyncMock(return_value=_chat_response())
    token = _current_bank_id.set("user-7")
    try:
        await _call(llm, create)
    finally:
        _current_bank_id.reset(token)
    assert create.call_args.kwargs["user"] == "user-7"


async def test_user_not_injected_when_flag_off():
    _set_flag(False)
    llm = _llm()
    create = AsyncMock(return_value=_chat_response())
    token = _current_bank_id.set("user-7")
    try:
        await _call(llm, create)
    finally:
        _current_bank_id.reset(token)
    assert "user" not in create.call_args.kwargs


async def test_user_not_injected_when_bank_unset():
    _set_flag(True)
    llm = _llm()
    create = AsyncMock(return_value=_chat_response())
    # No bank bound in context.
    assert get_current_bank_id() is None
    await _call(llm, create)
    assert "user" not in create.call_args.kwargs


async def test_caller_set_user_is_not_overridden():
    """The helper never clobbers a `user` the caller already placed in call_params."""
    _set_flag(True)
    # Simulate a caller-provided user via the centralized helper directly.
    params = {"user": "explicit-user"}
    token = _current_bank_id.set("user-7")
    try:
        apply_bank_attribution(params)
    finally:
        _current_bank_id.reset(token)
    assert params["user"] == "explicit-user"


async def test_user_injected_in_tool_calling_path():
    """call_with_tools() builds its own call_params; attribution must reach it too."""
    _set_flag(True)
    llm = _llm()
    tool_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="done", tool_calls=None, refusal=None, reasoning_content=None),
            )
        ],
        usage=None,
        error=None,
    )
    create = AsyncMock(return_value=tool_response)
    llm._client.chat.completions.create = create
    token = _current_bank_id.set("user-tools")
    try:
        with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "ping"}],
                tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
                max_retries=0,
            )
    finally:
        _current_bank_id.reset(token)
    assert create.call_args.kwargs["user"] == "user-tools"


# ── Embeddings: user injection ─────────────────────────────────────────────────


def _openai_embeddings() -> OpenAIEmbeddings:
    emb = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small", batch_size=100)
    emb._dimension = 1536
    return emb


def _fake_embed_client(captured: list[dict]):
    def fake_create(**kwargs):
        captured.append(kwargs)
        n = len(kwargs["input"])
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.0] * 1536) for i in range(n)])

    return SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))


def test_embeddings_user_injected_when_flag_on_and_bank_set():
    _set_flag(True)
    emb = _openai_embeddings()
    captured: list[dict] = []
    emb._client = _fake_embed_client(captured)
    token = _current_bank_id.set("user-emb")
    try:
        emb.encode(["hello"])
    finally:
        _current_bank_id.reset(token)
    assert captured[0]["user"] == "user-emb"


async def test_litellm_proxy_sends_bank_header():
    encoder = LiteLLMCrossEncoder(api_base="https://rerank.example", model="will-memory-rerank")
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"results": [{"index": 0, "relevance_score": 0.91}]},
    )
    encoder._async_client = SimpleNamespace(post=AsyncMock(return_value=response))

    with patch(
        "hindsight_api.engine.cross_encoder.reranker_bank_attribution_headers",
        return_value={"X-Hindsight-Bank-Id": "bank-litellm-proxy"},
    ):
        scores = await encoder.predict([("query", "document")])

    assert scores == [0.91]
    assert encoder._async_client.post.call_args.kwargs["headers"] == {"X-Hindsight-Bank-Id": "bank-litellm-proxy"}


# ── Executor context propagation ──────────────────────────────────────────────


class _BankCapturingBackend:
    """Embeddings backend whose encode records the bank id visible at call time.

    The real `generate_embeddings_batch` offloads encode to a thread via
    run_in_executor; this verifies the bank ContextVar survives that thread hop.
    """

    dimension = 1

    def __init__(self) -> None:
        self.seen_bank_id: str | None = "UNSET"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.seen_bank_id = get_current_bank_id()
        return [[0.0] for _ in texts]

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        return self.encode_documents(texts)


async def test_executor_propagates_bank_contextvar_into_worker_thread():
    backend = _BankCapturingBackend()
    token = _current_bank_id.set("user-thread")
    try:
        vectors = await generate_embeddings_batch(backend, ["a", "b"], input_type="document")
    finally:
        _current_bank_id.reset(token)
    assert backend.seen_bank_id == "user-thread"
    assert len(vectors) == 2


async def test_executor_length_validation_preserved():
    """The 1:1 alignment guard must still fire after the context-aware offload."""

    class _ShortBackend:
        dimension = 1

        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0]]  # one vector for two inputs

        def encode_query(self, texts: list[str]) -> list[list[float]]:
            return self.encode_documents(texts)

    with pytest.raises(Exception, match="expected exact 1:1 alignment"):
        await generate_embeddings_batch(_ShortBackend(), ["a", "b"], input_type="document")
