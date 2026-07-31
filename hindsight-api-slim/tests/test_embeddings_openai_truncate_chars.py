"""HINDSIGHT_API_EMBEDDINGS_OPENAI_TRUNCATE_CHARS config wiring + behaviour.

OpenAI-compatible local backends (e.g. llama.cpp ``/v1/embeddings``) hard-
reject inputs beyond the model context instead of truncating server-side the
way the SentenceTransformers provider does — one oversized memory then fails
the whole retain/recall batch. The optional client-side cap keeps the batch
alive; 0 (the default) preserves current behaviour exactly.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Save/restore env vars touched by these tests."""
    from hindsight_api.config import clear_config_cache

    env_vars_to_save = [
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_TRUNCATE_CHARS",
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


def test_default_truncate_chars_is_disabled():
    """Default is 0 (no truncation) — current behaviour is preserved."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_OPENAI_TRUNCATE_CHARS", None)

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_truncate_chars == 0


def test_truncate_chars_env_var_is_read():
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_TRUNCATE_CHARS"] = "3500"

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_truncate_chars == 3500


def _fake_client(sent):
    class _FakeEmbeddings:
        def create(self, *, model, input, **kw):
            sent["input"] = list(input)

            class _Item:
                def __init__(self, i):
                    self.index = i
                    self.embedding = [0.0]

            class _Resp:
                data = [_Item(i) for i, _ in enumerate(input)]

            return _Resp()

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    return _FakeClient()


def test_encode_truncates_oversized_inputs_only():
    """Inputs above the cap are cut to it; shorter inputs pass unchanged."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="k", truncate_chars=10)
    sent = {}
    emb._client = _fake_client(sent)
    emb._dimension = 1

    emb.encode(["short", "x" * 25])
    assert sent["input"] == ["short", "x" * 10]


def test_encode_cap_disabled_leaves_inputs_untouched():
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="k", truncate_chars=0)
    sent = {}
    emb._client = _fake_client(sent)
    emb._dimension = 1

    long = "x" * 25
    emb.encode([long])
    assert sent["input"] == [long]
