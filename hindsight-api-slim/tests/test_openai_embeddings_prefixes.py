"""
Tests for asymmetric query/passage prefixes on the OpenAI-compatible embeddings provider.

Issue #3514: an OpenAI-compatible endpoint (llama-server, infinity-emb, ...) only ever
receives the raw input text, so an asymmetric model served behind one (e.g.
google/embeddinggemma-300m) needs Hindsight to apply the model's query/document
instruction client-side. Unset prefixes must leave the existing behavior byte-identical.
"""

import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Save/restore env vars touched by these tests."""
    from hindsight_api.config import clear_config_cache

    env_vars_to_save = [
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX",
        "HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY",
        "HINDSIGHT_API_LLM_API_KEY",
        "HINDSIGHT_API_LLM_PROVIDER",
    ]

    original_values = {key: os.environ.get(key) for key in env_vars_to_save}

    clear_config_cache()

    yield

    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def _stub_client(embeddings) -> list[list[str]]:
    """Point the provider at a fake OpenAI client, returning the recorded input batches."""
    sent: list[list[str]] = []

    def fake_create(*, model, input, **kwargs):
        sent.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.0, 1.0]) for i in range(len(input))])

    embeddings._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    embeddings._dimension = 2
    return sent


def test_prefixes_default_to_empty():
    """Unset env vars keep the existing (symmetric) OpenAI behavior."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX", None)
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX", None)

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_query_prefix == ""
    assert config.embeddings_openai_passage_prefix == ""


def test_prefix_env_vars_are_read_verbatim():
    """Trailing whitespace is significant in these prefixes and must survive config load."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX"] = "task: search result | query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX"] = "title: none | text: "

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_query_prefix == "task: search result | query: "
    assert config.embeddings_openai_passage_prefix == "title: none | text: "


def test_openai_provider_receives_configured_prefixes():
    """create_embeddings_from_env() propagates the prefixes to the 'openai' provider."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openai"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"] = "sk-test"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX"] = "query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX"] = "passage: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.query_prefix == "query: "
    assert embeddings.passage_prefix == "passage: "


def test_openrouter_provider_receives_configured_prefixes():
    """'openrouter' shares the OPENAI_* knobs (batch size, dimensions) — prefixes too."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openrouter"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY"] = "sk-or-test"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX"] = "query: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.query_prefix == "query: "


def test_requesty_provider_receives_configured_prefixes():
    """'requesty' is the third branch building an OpenAIEmbeddings — it must not drop them."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "requesty"
    os.environ["HINDSIGHT_API_LLM_API_KEY"] = "sk-req-test"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX"] = "passage: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.passage_prefix == "passage: "


def test_openai_codex_provider_receives_configured_prefixes(tmp_path, monkeypatch):
    """The Codex OAuth path wraps the same endpoint, so it must forward the prefixes too."""
    import json

    from hindsight_api.engine.embeddings import CodexOAuthEmbeddings, create_embeddings_from_env

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "codex-token", "account_id": "acct"}})
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    # Codex auth resolves CODEX_HOME first, so pin resolution to the patched HOME.
    monkeypatch.delenv("CODEX_HOME", raising=False)

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openai-codex"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX"] = "query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX"] = "passage: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, CodexOAuthEmbeddings)
    assert embeddings.query_prefix == "query: "
    assert embeddings.passage_prefix == "passage: "


def test_query_and_document_inputs_are_prefixed_asymmetrically():
    """The prefix is applied to the text actually sent to the /embeddings endpoint."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(
        api_key="sk-test",
        model="embeddinggemma-300m",
        query_prefix="task: search result | query: ",
        passage_prefix="title: none | text: ",
    )
    sent = _stub_client(emb)

    emb.encode_query(["refund policy?"])
    emb.encode_documents(["We refund within 30 days."])

    assert sent == [
        ["task: search result | query: refund policy?"],
        ["title: none | text: We refund within 30 days."],
    ]


def test_unset_prefixes_leave_inputs_untouched():
    """Default construction must send exactly what the caller passed (no empty-prefix concat)."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small")
    sent = _stub_client(emb)

    emb.encode_query(["refund policy?"])
    emb.encode_documents(["We refund within 30 days."])
    emb.encode(["plain"])

    assert sent == [["refund policy?"], ["We refund within 30 days."], ["plain"]]


def test_encode_stays_unprefixed_when_prefixes_are_configured():
    """encode() is the raw entry point — only the asymmetric wrappers prefix."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test", model="embeddinggemma-300m", query_prefix="query: ")
    sent = _stub_client(emb)

    emb.encode(["plain"])

    assert sent == [["plain"]]


def test_prefixed_inputs_still_respect_batch_size():
    """Prefixing happens before batching, so oversized calls are still split."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test", model="embeddinggemma-300m", batch_size=2, passage_prefix="passage: ")
    sent = _stub_client(emb)

    emb.encode_documents(["a", "b", "c"])

    assert sent == [["passage: a", "passage: b"], ["passage: c"]]
