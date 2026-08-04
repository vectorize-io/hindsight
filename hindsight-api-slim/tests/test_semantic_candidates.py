import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.api.http import SemanticCandidatesResponse
from hindsight_api.cancellation import CancellationToken, OperationCancelledError
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.search import retrieval as retrieval_mod
from hindsight_api.engine.search.types import SemanticCandidate, SemanticCandidatesResult
from hindsight_api.extensions import OperationValidationError, ValidationResult
from hindsight_api.models import RequestContext


@pytest.mark.asyncio
async def test_semantic_candidates_endpoint_returns_bounded_full_bank_provenance(
    api_client,
    memory,
    monkeypatch,
):
    candidate_id = "123e4567-e89b-12d3-a456-426614174000"
    semantic_candidates = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[SimpleNamespace(id=candidate_id, fact_type="world", score=0.82)],
            limit_reached=False,
            min_similarity=0.45,
        )
    )
    monkeypatch.setattr(memory, "semantic_candidates_async", semantic_candidates, raising=False)

    response = await api_client.post(
        "/v1/default/banks/test-bank/memories/semantic-candidates",
        json={
            "query": "software deployment",
            "types": ["world", "experience"],
            "limit": 25,
            "min_similarity": 0.45,
            "document_id": "document-1",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "candidates": [{"id": candidate_id, "type": "world", "score": 0.82}],
        "limit": 25,
        "returned": 1,
        "limit_reached": False,
        "exhaustive": False,
        "total_relation": "unknown",
        "min_similarity": 0.45,
        "score": {"name": "cosine_similarity", "approximate": True},
        "corpus_scope": "full_bank",
        "scope": "valid_memory_units",
    }
    semantic_candidates.assert_awaited_once()
    call = semantic_candidates.await_args
    assert call.kwargs["bank_id"] == "test-bank"
    assert call.kwargs["query"] == "software deployment"
    assert call.kwargs["fact_types"] == ["world", "experience"]
    assert call.kwargs["limit"] == 25
    assert call.kwargs["min_similarity"] == 0.45
    assert call.kwargs["document_id"] == "document-1"
    assert call.kwargs["request_context"] is not None


@pytest.mark.asyncio
async def test_semantic_candidates_endpoint_preserves_policy_rejection_status(
    api_client,
    memory,
    monkeypatch,
):
    semantic_candidates = AsyncMock(side_effect=OperationValidationError("semantic search denied", status_code=429))
    monkeypatch.setattr(memory, "semantic_candidates_async", semantic_candidates)

    response = await api_client.post(
        "/v1/default/banks/test-bank/memories/semantic-candidates",
        json={"query": "alpha"},
    )

    assert response.status_code == 429
    assert response.json() == {"detail": "semantic search denied"}


@pytest.mark.asyncio
async def test_engine_semantic_candidates_observes_cancellation_after_retrieval(
    memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hindsight_api.engine import memory_engine as engine_mod

    token = CancellationToken()
    request_context = RequestContext(cancellation=token)
    memory._authenticate_tenant = AsyncMock(return_value="tenant")
    memory._operation_validator = None
    memory._search_semaphore = asyncio.Semaphore(1)
    memory.embeddings = object()
    read_backend = object()
    memory._get_read_backend = AsyncMock(return_value=read_backend)

    @asynccontextmanager
    async def fake_acquire(backend):
        assert backend is read_backend
        yield object()

    monkeypatch.setattr(engine_mod, "acquire_with_retry", fake_acquire)
    monkeypatch.setattr(
        engine_mod.embedding_utils,
        "generate_embeddings_batch",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )

    async def retrieve_then_cancel(*args: object, **kwargs: object) -> SemanticCandidatesResult:
        del args, kwargs
        token.cancel("client disconnected")
        return SemanticCandidatesResult(candidates=[], min_similarity=0.4, limit_reached=False)

    monkeypatch.setattr(
        retrieval_mod,
        "retrieve_semantic_candidates",
        retrieve_then_cancel,
    )

    with pytest.raises(OperationCancelledError):
        await memory.semantic_candidates_async(
            bank_id="bank",
            query="alpha",
            fact_types=["world"],
            limit=10,
            request_context=request_context,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_update",
    [
        {"query": "   ---   "},
        {"limit": 0},
        {"limit": 501},
        {"min_similarity": -1.01},
        {"min_similarity": 1.01},
        {"types": []},
        {"types": ["world", "world"]},
        {"types": ["world", "invalid"]},
        {
            "tags": ["tenant:one"],
            "tag_groups": [{"tags": ["tenant:one"], "match": "all_strict"}],
        },
    ],
    ids=[
        "normalized-empty-query",
        "zero-limit",
        "excessive-limit",
        "similarity-below-cosine-range",
        "similarity-above-cosine-range",
        "empty-types",
        "duplicate-types",
        "invalid-type",
        "ambiguous-tag-filters",
    ],
)
async def test_semantic_candidates_endpoint_rejects_invalid_requests_before_retrieval(
    api_client,
    memory,
    monkeypatch,
    request_update,
):
    semantic_candidates = AsyncMock()
    monkeypatch.setattr(memory, "semantic_candidates_async", semantic_candidates, raising=False)
    request = {"query": "software deployment", "types": ["world"], "limit": 25}
    request.update(request_update)

    response = await api_client.post(
        "/v1/default/banks/test-bank/memories/semantic-candidates",
        json=request,
    )

    assert response.status_code == 422
    semantic_candidates.assert_not_awaited()


def test_semantic_candidates_contract_requires_completeness_provenance() -> None:
    required = set(SemanticCandidatesResponse.model_json_schema()["required"])

    assert {
        "candidates",
        "limit",
        "returned",
        "limit_reached",
        "exhaustive",
        "total_relation",
        "min_similarity",
        "score",
        "corpus_scope",
        "scope",
    } <= required


@pytest.mark.asyncio
async def test_native_semantic_retrieval_is_semantic_only_filtered_and_globally_bounded(monkeypatch):
    class FakeDialect:
        def __init__(self) -> None:
            self.semantic_arms: list[dict] = []

        def build_semantic_arm(self, **kwargs):
            self.semantic_arms.append(kwargs)
            return f"SELECT '{kwargs['fact_type']}' AS fact_type"

        def build_bm25_arm(self, **kwargs):
            raise AssertionError("semantic candidates must not execute BM25")

    class FakeConnection:
        backend_type = "postgresql"

        def __init__(self) -> None:
            self.params = None

        async def fetch(self, query, *params):
            self.params = params
            return [
                {"id": "world-1", "text": "w1", "fact_type": "world", "source": "semantic", "similarity": 0.8},
                {
                    "id": "experience-1",
                    "text": "e1",
                    "fact_type": "experience",
                    "source": "semantic",
                    "similarity": 0.9,
                },
                {"id": "world-2", "text": "w2", "fact_type": "world", "source": "semantic", "similarity": 0.7},
            ]

    dialect = FakeDialect()
    connection = FakeConnection()
    config = SimpleNamespace(semantic_min_similarity=0.1, bm25_min_score=0.0)
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: config)
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: dialect)

    result = await retrieval_mod.retrieve_semantic_candidates(
        connection,
        "[0.0]",
        "bank-1",
        ["world", "experience"],
        2,
        min_similarity=0.45,
        document_id="document-1",
    )

    assert [(candidate.id, candidate.score) for candidate in result.candidates] == [
        ("experience-1", 0.9),
        ("world-1", 0.8),
    ]
    assert result.limit_reached is True
    assert connection.params == ("[0.0]", "bank-1", "document-1")
    assert len(dialect.semantic_arms) == 2
    assert all(arm["extra_where"] == " AND document_id = $3" for arm in dialect.semantic_arms)


@pytest.mark.asyncio
async def test_engine_semantic_candidates_authenticates_embeds_and_uses_read_backend(monkeypatch):
    from hindsight_api.engine import memory_engine as engine_mod

    engine = object.__new__(engine_mod.MemoryEngine)
    engine.embeddings = object()
    engine._operation_validator = None
    engine._search_semaphore = asyncio.Semaphore(1)
    engine._authenticate_tenant = AsyncMock()
    read_backend = object()
    engine._get_read_backend = AsyncMock(return_value=read_backend)
    connection = object()

    @asynccontextmanager
    async def fake_acquire(backend):
        assert backend is read_backend
        yield connection

    generate_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
    retrieve_candidates = AsyncMock(
        return_value=SemanticCandidatesResult(
            candidates=[SemanticCandidate(id="candidate-1", fact_type="world", score=0.75)],
            limit_reached=False,
            min_similarity=0.4,
        )
    )
    monkeypatch.setattr(engine_mod, "acquire_with_retry", fake_acquire)
    monkeypatch.setattr(engine_mod.embedding_utils, "generate_embeddings_batch", generate_embeddings)
    monkeypatch.setattr(retrieval_mod, "retrieve_semantic_candidates", retrieve_candidates)
    request_context = RequestContext(tenant_id="default")

    result = await engine_mod.MemoryEngine.semantic_candidates_async(
        engine,
        bank_id="bank-1",
        query="software deployment",
        fact_types=["world"],
        limit=20,
        min_similarity=0.4,
        document_id="document-1",
        request_context=request_context,
    )

    assert result.candidates[0].id == "candidate-1"
    engine._authenticate_tenant.assert_awaited_once_with(request_context)
    generate_embeddings.assert_awaited_once_with(engine.embeddings, ["software deployment"], input_type="query")
    retrieve_candidates.assert_awaited_once_with(
        connection,
        "[0.1, 0.2]",
        "bank-1",
        ["world"],
        20,
        tags=None,
        tags_match="any",
        tag_groups=None,
        document_id="document-1",
        min_similarity=0.4,
    )


@pytest.mark.asyncio
async def test_engine_semantic_candidates_applies_recall_visibility_policy(monkeypatch):
    from hindsight_api.engine import memory_engine as engine_mod

    captured_contexts = []

    class Validator:
        async def validate_recall(self, context):
            captured_contexts.append(context)
            return ValidationResult.accept_with(tags=["tenant:visible"], tags_match="all_strict")

    engine = object.__new__(engine_mod.MemoryEngine)
    engine.embeddings = object()
    engine._operation_validator = Validator()
    engine._search_semaphore = asyncio.Semaphore(1)
    engine._authenticate_tenant = AsyncMock()
    engine._get_read_backend = AsyncMock(return_value=object())

    @asynccontextmanager
    async def fake_acquire(backend):
        del backend
        yield object()

    retrieve_candidates = AsyncMock(
        return_value=SemanticCandidatesResult(candidates=[], limit_reached=False, min_similarity=0.4)
    )
    monkeypatch.setattr(engine_mod, "acquire_with_retry", fake_acquire)
    monkeypatch.setattr(
        engine_mod.embedding_utils,
        "generate_embeddings_batch",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )
    monkeypatch.setattr(retrieval_mod, "retrieve_semantic_candidates", retrieve_candidates)

    await engine_mod.MemoryEngine.semantic_candidates_async(
        engine,
        bank_id="bank-1",
        query="software deployment",
        fact_types=["world"],
        limit=20,
        min_similarity=0.4,
        tags=["caller:tag"],
        request_context=RequestContext(tenant_id="default"),
    )

    assert len(captured_contexts) == 1
    assert captured_contexts[0].bank_id == "bank-1"
    assert captured_contexts[0].tags == ["caller:tag"]
    assert retrieve_candidates.await_args.kwargs["tags"] == ["tenant:visible"]
    assert retrieve_candidates.await_args.kwargs["tags_match"] == "all_strict"


@pytest.mark.asyncio
async def test_semantic_candidates_endpoint_queries_native_index_with_document_filter(api_client, memory):
    bank_id = f"semantic-candidates-{uuid.uuid4().hex}"
    included_id = uuid.uuid4()
    excluded_id = uuid.uuid4()
    query = "software deployment"
    await memory.get_bank_profile(bank_id=bank_id, request_context=RequestContext())
    embeddings = await embedding_utils.generate_embeddings_batch(
        memory.embeddings,
        [query, query],
        input_type="document",
    )
    pool = await memory._get_pool()
    async with pool.acquire() as connection:
        await connection.executemany(
            "INSERT INTO documents (id, bank_id) VALUES ($1, $2)",
            [("document-1", bank_id), ("document-2", bank_id)],
        )
        await connection.executemany(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, document_id)
            VALUES ($1, $2, $3, 'world', $4::vector, $5)
            """,
            [
                (included_id, bank_id, "included", str(embeddings[0]), "document-1"),
                (excluded_id, bank_id, "excluded", str(embeddings[1]), "document-2"),
            ],
        )

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/semantic-candidates",
        json={
            "query": query,
            "types": ["world"],
            "limit": 10,
            "min_similarity": -1,
            "document_id": "document-1",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [candidate["id"] for candidate in payload["candidates"]] == [str(included_id)]
    assert payload["candidates"][0]["type"] == "world"
    assert payload["score"] == {"name": "cosine_similarity", "approximate": True}
    assert payload["corpus_scope"] == "full_bank"
    assert payload["scope"] == "valid_memory_units"
    assert payload["total_relation"] == "unknown"
    assert payload["returned"] == 1
    assert payload["exhaustive"] is False
    assert payload["min_similarity"] == -1
