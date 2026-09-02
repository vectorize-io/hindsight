"""
Tests for the provider-agnostic asymmetric query/passage embedding prefixes.

Issue #3514: a plain text-in/vector-out endpoint (llama-server, infinity-emb, TEI,
a LiteLLM proxy, ...) only ever receives the raw input text, so an asymmetric model
served behind one (e.g. google/embeddinggemma-300m) needs Hindsight to apply the
model's query/document instruction client-side. Providers that carry the distinction
natively (`local`, `zeroentropy`) must be unaffected, and unset prefixes must leave
every provider's payload byte-identical.
"""

import json
import os
from types import SimpleNamespace

import httpx
import pytest

from hindsight_api.engine.embeddings import Embeddings

# Providers that are plain text-in/vector-out and therefore honour the generic prefixes.
# Each entry is (provider, extra env, expected concrete class name).
TEXT_IN_PROVIDERS = [
    ("tei", {"HINDSIGHT_API_EMBEDDINGS_TEI_URL": "http://localhost:8080"}, "RemoteTEIEmbeddings"),
    ("openai", {"HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": "sk-test"}, "OpenAIEmbeddings"),
    ("openrouter", {"HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY": "sk-or-test"}, "OpenAIEmbeddings"),
    ("requesty", {"HINDSIGHT_API_EMBEDDINGS_REQUESTY_API_KEY": "sk-req-test"}, "OpenAIEmbeddings"),
    ("litellm", {}, "LiteLLMEmbeddings"),
    ("litellm-sdk", {}, "LiteLLMSDKEmbeddings"),
]


@pytest.fixture(autouse=True)
def setup_test_env():
    """Save/restore env vars touched by these tests."""
    from hindsight_api.config import clear_config_cache

    env_vars_to_save = [
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX",
        "HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX",
        "HINDSIGHT_API_EMBEDDINGS_TEI_URL",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
        "HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY",
        "HINDSIGHT_API_EMBEDDINGS_REQUESTY_API_KEY",
        "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_API_KEY",
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


class _RecordingEmbeddings(Embeddings):
    """Concrete Embeddings that records the texts each call handed to encode().

    Exercises the base-class prefixing itself, so the behavior is pinned for every
    provider that inherits it rather than for one provider's transport.
    """

    def __init__(self, query_prefix: str = "", passage_prefix: str = ""):
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.sent: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return "recording"

    @property
    def dimension(self) -> int:
        return 2

    async def initialize(self) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.sent.append(list(texts))
        return [[0.0, 1.0] for _ in texts]


def test_prefixes_default_to_empty():
    """Unset env vars keep the existing (symmetric) behavior for every provider."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX", None)
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX", None)

    config = HindsightConfig.from_env()
    assert config.embeddings_query_prefix == ""
    assert config.embeddings_passage_prefix == ""


def test_prefix_env_vars_are_read_verbatim():
    """Trailing whitespace is significant in these prefixes and must survive config load."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX"] = "task: search result | query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX"] = "title: none | text: "

    config = HindsightConfig.from_env()
    assert config.embeddings_query_prefix == "task: search result | query: "
    assert config.embeddings_passage_prefix == "title: none | text: "


@pytest.mark.parametrize("provider,extra_env,expected_class", TEXT_IN_PROVIDERS)
def test_every_text_in_provider_receives_the_prefixes(provider, extra_env, expected_class):
    """Family guard: no text-in provider may silently drop the configured prefixes.

    These providers have no other channel for the query/document distinction, so a
    branch that forgets to forward the config is a silent retrieval-quality
    regression rather than a visible failure.
    """
    from hindsight_api.engine import embeddings as embeddings_module

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = provider
    os.environ["HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX"] = "query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX"] = "passage: "
    for key, value in extra_env.items():
        os.environ[key] = value

    embeddings = embeddings_module.create_embeddings_from_env()

    assert isinstance(embeddings, getattr(embeddings_module, expected_class))
    assert embeddings.query_prefix == "query: "
    assert embeddings.passage_prefix == "passage: "


def test_openai_codex_provider_receives_configured_prefixes(tmp_path, monkeypatch):
    """The Codex OAuth path wraps the same endpoint, so it must forward the prefixes too."""
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
    os.environ["HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX"] = "query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX"] = "passage: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, CodexOAuthEmbeddings)
    assert embeddings.query_prefix == "query: "
    assert embeddings.passage_prefix == "passage: "


def test_local_provider_ignores_the_prefixes():
    """`local` applies the model's own prompts, so a client-side prefix would double up."""
    from hindsight_api.engine.embeddings import LocalSTEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "local"
    os.environ["HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX"] = "query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX"] = "passage: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, LocalSTEmbeddings)
    assert embeddings.query_prefix == ""
    assert embeddings.passage_prefix == ""


def test_zeroentropy_provider_ignores_the_prefixes():
    """`zeroentropy` sends a native input_type, so it must not also be prefixed."""
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "zeroentropy"
    os.environ["HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_API_KEY"] = "ze-test"
    os.environ["HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX"] = "query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX"] = "passage: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, ZeroEntropyEmbeddings)
    assert embeddings.query_prefix == ""
    assert embeddings.passage_prefix == ""


def test_onnx_keeps_its_own_prefix_config():
    """`onnx` predates the generic pair and keeps its E5 defaults via the ONNX_ vars."""
    from hindsight_api.engine.embeddings import OnnxEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "onnx"
    os.environ["HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX"] = "ignored: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OnnxEmbeddings)
    assert embeddings.query_prefix == "query: "
    assert embeddings.passage_prefix == "passage: "


def test_query_and_document_inputs_are_prefixed_asymmetrically():
    """The prefix is applied to the text actually handed to encode()."""
    recorder = _RecordingEmbeddings(
        query_prefix="task: search result | query: ",
        passage_prefix="title: none | text: ",
    )

    recorder.encode_query(["refund policy?"])
    recorder.encode_documents(["We refund within 30 days."])

    assert recorder.sent == [
        ["task: search result | query: refund policy?"],
        ["title: none | text: We refund within 30 days."],
    ]


def test_unset_prefixes_leave_inputs_untouched():
    """Default construction must pass through exactly what the caller passed."""
    recorder = _RecordingEmbeddings()

    recorder.encode_query(["refund policy?"])
    recorder.encode_documents(["We refund within 30 days."])
    recorder.encode(["plain"])

    assert recorder.sent == [["refund policy?"], ["We refund within 30 days."], ["plain"]]


def test_encode_stays_unprefixed_when_prefixes_are_configured():
    """encode() is the raw entry point — only the asymmetric wrappers prefix."""
    recorder = _RecordingEmbeddings(query_prefix="query: ", passage_prefix="passage: ")

    recorder.encode(["plain"])

    assert recorder.sent == [["plain"]]


def test_prefixed_inputs_still_respect_batch_size():
    """Prefixing happens before batching, so oversized calls are still split."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test", model="embeddinggemma-300m", batch_size=2, passage_prefix="passage: ")

    sent: list[list[str]] = []

    def fake_create(*, model, input, **kwargs):
        sent.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.0, 1.0]) for i in range(len(input))])

    emb._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    emb._dimension = 2

    emb.encode_documents(["a", "b", "c"])

    assert sent == [["passage: a", "passage: b"], ["passage: c"]]


def test_tei_sends_prefixed_inputs_to_the_embed_endpoint():
    """End-to-end payload check for TEI, which had no asymmetry mechanism at all before."""
    from hindsight_api.engine.embeddings import RemoteTEIEmbeddings

    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["inputs"]
        sent.append(list(inputs))
        return httpx.Response(200, json=[[0.0, 1.0] for _ in inputs])

    emb = RemoteTEIEmbeddings(base_url="http://localhost:8080", query_prefix="query: ", passage_prefix="passage: ")
    emb._client = httpx.Client(transport=httpx.MockTransport(handler))

    emb.encode_query(["where?"])
    emb.encode_documents(["here."])

    assert sent == [["query: where?"], ["passage: here."]]
