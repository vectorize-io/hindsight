"""Tests for multi-LLM env parsing and engine-chain resolution helpers.

Covers indexed-member discovery, strategy JSON parsing/validation, and the
per-slot resolution + wrapping logic (``_build_llm``)
without constructing a full MemoryEngine.
"""

import pytest

from hindsight_api.config import (
    HindsightConfig,
    LLMMemberConfig,
    LLMStrategyConfig,
    _parse_llm_members,
    _parse_llm_strategy,
)
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.memory_engine import _build_llm
from hindsight_api.engine.multi_llm import MultiLLMProvider


@pytest.fixture
def clean_llm_env(monkeypatch):
    """Strip all HINDSIGHT_API_*LLM* env so each test sets only what it needs."""
    import os

    for key in list(os.environ):
        if key.startswith("HINDSIGHT_API_") and "LLM" in key:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "openai")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-primary")
    return monkeypatch


# ── member parsing ─────────────────────────────────────────────────────────────


def test_parse_members_empty_when_unset(clean_llm_env):
    assert _parse_llm_members("") == []
    assert _parse_llm_members("RETAIN_") == []


def test_parse_members_indexed(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "groq")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_API_KEY", "gsk-1")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_MODEL", "llama-x")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_PROVIDER", "anthropic")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_API_KEY", "ak-2")

    members = _parse_llm_members("")
    assert [(m.provider, m.model) for m in members] == [
        ("groq", "llama-x"),
        ("anthropic", "claude-haiku-4-5"),  # model defaulted from provider
    ]
    assert members[0].api_key == "gsk-1"


def test_parse_members_stops_at_first_gap(clean_llm_env):
    # Index 2 present but 1 missing → scan stops at 1, member 2 ignored.
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_PROVIDER", "groq")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_API_KEY", "gsk-2")
    assert _parse_llm_members("") == []


def test_parse_members_per_op_prefix(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_RETAIN_LLM_1_PROVIDER", "groq")
    clean_llm_env.setenv("HINDSIGHT_API_RETAIN_LLM_1_API_KEY", "gsk-1")
    retain = _parse_llm_members("RETAIN_")
    assert [m.provider for m in retain] == ["groq"]
    assert _parse_llm_members("") == []  # global still empty


def test_parse_members_missing_api_key_raises(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "openai")  # requires key
    with pytest.raises(ValueError, match="API_KEY is required"):
        _parse_llm_members("")


def test_parse_members_no_key_provider_ok(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "ollama")  # no key needed
    members = _parse_llm_members("")
    assert members[0].provider == "ollama"
    assert members[0].api_key is None


def test_parse_members_vertexai_project_and_region(clean_llm_env):
    # A vertexai member can carry its own project/region so it can be used as a
    # member of a failover/round-robin chain.
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "vertexai")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_VERTEXAI_PROJECT_ID", "p")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_VERTEXAI_REGION", "us-central1")
    members = _parse_llm_members("")
    assert members[0].provider == "vertexai"
    assert members[0].vertexai_project_id == "p"
    assert members[0].vertexai_region == "us-central1"


def test_parse_members_vertexai_per_op_prefix(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_PROVIDER", "vertexai")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_VERTEXAI_PROJECT_ID", "reflect-proj")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_VERTEXAI_REGION", "europe-west1")
    reflect = _parse_llm_members("REFLECT_")
    assert reflect[0].vertexai_project_id == "reflect-proj"
    assert reflect[0].vertexai_region == "europe-west1"
    # The vertex fields are scoped to the prefix; the global chain is unaffected.
    assert _parse_llm_members("") == []


def test_parse_members_without_vertexai_fields_default_none(clean_llm_env):
    # Non-vertex members (and members that omit the vars) leave the fields None —
    # no regression for existing providers.
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "ollama")
    members = _parse_llm_members("")
    assert members[0].vertexai_project_id is None
    assert members[0].vertexai_region is None
    assert members[0].vertexai_service_account_key is None
    assert members[0].litellmrouter_config is None


def test_parse_members_vertexai_service_account_key(clean_llm_env):
    # A vertexai member can carry its own service-account key for cross-project
    # failover (credentials are otherwise global).
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "vertexai")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_VERTEXAI_PROJECT_ID", "p")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_VERTEXAI_SERVICE_ACCOUNT_KEY", "/keys/sa.json")
    members = _parse_llm_members("")
    assert members[0].vertexai_service_account_key == "/keys/sa.json"


def test_parse_members_litellmrouter_config(clean_llm_env):
    # A litellmrouter member can carry its own router config so a chain can fail
    # over between differently-routed LiteLLM routers.
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "litellmrouter")
    clean_llm_env.setenv(
        "HINDSIGHT_API_LLM_1_LITELLMROUTER_CONFIG",
        '{"model_list": [{"model_name": "m"}]}',
    )
    members = _parse_llm_members("")
    assert members[0].litellmrouter_config == {"model_list": [{"model_name": "m"}]}


def test_parse_members_litellmrouter_config_per_op_prefix(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_PROVIDER", "litellmrouter")
    clean_llm_env.setenv(
        "HINDSIGHT_API_REFLECT_LLM_1_LITELLMROUTER_CONFIG",
        '{"model_list": [{"model_name": "reflect-m"}]}',
    )
    reflect = _parse_llm_members("REFLECT_")
    assert reflect[0].litellmrouter_config == {"model_list": [{"model_name": "reflect-m"}]}
    # Scoped to the prefix; the global chain is unaffected.
    assert _parse_llm_members("") == []


def test_parse_members_litellmrouter_config_invalid_json_raises(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "litellmrouter")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_LITELLMROUTER_CONFIG", "{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_llm_members("")


# ── strategy parsing ───────────────────────────────────────────────────────────


def test_parse_strategy_none_when_unset():
    assert _parse_llm_strategy(None) is None
    assert _parse_llm_strategy("") is None
    assert _parse_llm_strategy("   ") is None


def test_parse_strategy_failover():
    s = _parse_llm_strategy('{"mode": "failover"}')
    assert s == LLMStrategyConfig(mode="failover", weights=None)


def test_parse_strategy_weighted_round_robin():
    s = _parse_llm_strategy('{"mode": "round-robin", "weights": [3, 1]}')
    assert s.mode == "round-robin"
    assert s.weights == [3, 1]


def test_parse_strategy_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_llm_strategy("{not json")


def test_parse_strategy_invalid_mode():
    with pytest.raises(ValueError, match="Invalid LLM strategy mode"):
        _parse_llm_strategy('{"mode": "magic"}')


def test_parse_strategy_weights_require_round_robin():
    with pytest.raises(ValueError, match="only valid with mode"):
        _parse_llm_strategy('{"mode": "failover", "weights": [1, 1]}')


def test_parse_strategy_weights_must_be_positive_ints():
    with pytest.raises(ValueError, match="positive integers"):
        _parse_llm_strategy('{"mode": "round-robin", "weights": [1, 0]}')
    with pytest.raises(ValueError, match="positive integers"):
        _parse_llm_strategy('{"mode": "round-robin", "weights": []}')


# ── from_env integration ────────────────────────────────────────────────────────


def test_from_env_no_chain_is_empty(clean_llm_env):
    config = HindsightConfig.from_env()
    assert config.llm_members == []
    assert config.llm_strategy is None


def test_from_env_global_chain(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "ollama")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_STRATEGY", '{"mode": "failover"}')
    config = HindsightConfig.from_env()
    assert [m.provider for m in config.llm_members] == ["ollama"]
    assert config.llm_strategy.mode == "failover"


# ── chain resolution + wrapping (_build_llm) ────────────────────────────────────


def _empty_config(**overrides) -> HindsightConfig:
    """A minimal HindsightConfig from env, then patched with the given fields."""
    import dataclasses

    base = HindsightConfig.from_env()
    return dataclasses.replace(base, **overrides)


def _member(provider="ollama"):
    return LLMMemberConfig(
        provider=provider,
        api_key=None,
        model="m",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
    )


def _base_llm() -> LLMProvider:
    return LLMProvider(provider="mock", api_key="", base_url="", model="m0")


def test_build_llm_no_chain_returns_plain_provider(clean_llm_env):
    base = _base_llm()
    result = _build_llm(base, _empty_config(), "")
    assert result is base
    assert not isinstance(result, MultiLLMProvider)


def test_build_llm_global_chain_wraps(clean_llm_env):
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
    )
    base = _base_llm()
    result = _build_llm(base, config, "")
    assert isinstance(result, MultiLLMProvider)
    assert result.members[0] is base  # primary stays index 0
    assert result.members[1].provider == "ollama"


def test_build_llm_per_op_inherits_global(clean_llm_env):
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
        retain_llm_members=[],
        retain_llm_strategy=None,
    )
    result = _build_llm(_base_llm(), config, "retain_")
    assert isinstance(result, MultiLLMProvider)
    assert [m.provider for m in result.members[1:]] == ["ollama"]  # inherited


def test_build_llm_per_op_overrides_global(clean_llm_env):
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
        retain_llm_members=[_member("lmstudio")],
        retain_llm_strategy=LLMStrategyConfig(mode="round-robin"),
    )
    result = _build_llm(_base_llm(), config, "retain_")
    assert isinstance(result, MultiLLMProvider)
    assert [m.provider for m in result.members[1:]] == ["lmstudio"]
    assert result._strategy.mode == "round-robin"


def test_build_llm_members_without_strategy_stays_plain(clean_llm_env):
    # Members configured but no strategy → no wrapping (strategy is required).
    config = _empty_config(llm_members=[_member("ollama")], llm_strategy=None)
    base = _base_llm()
    assert _build_llm(base, config, "") is base


# ── vertexai member build path (_member_to_llm) ─────────────────────────────────


def test_member_to_llm_passes_vertexai_project_and_region(clean_llm_env, monkeypatch):
    """A vertexai member's own project/region reach the provider build.

    The global config has no Vertex project, so this also proves the member's
    values are used (no "VERTEXAI_PROJECT_ID is required") instead of the global
    fallback. The Vertex SDK client is patched so no live LLM/network call runs.
    """
    import hindsight_api.engine.providers.gemini_llm as gemini_llm
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import _member_to_llm

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gemini_llm.genai, "Client", _FakeClient)
    clear_config_cache()  # rebuild from the (vertex-less) test env

    member = LLMMemberConfig(
        provider="vertexai",
        api_key=None,
        model="gemini-2.0-flash",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
        vertexai_project_id="member-proj",
        vertexai_region="europe-west1",
    )
    provider = _member_to_llm(member, _empty_config())

    assert provider.provider == "vertexai"
    # Project/region flowed all the way to the Vertex AI SDK client.
    assert captured["project"] == "member-proj"
    assert captured["location"] == "europe-west1"


def test_member_to_llm_passes_vertexai_service_account_key(clean_llm_env, monkeypatch):
    """A vertexai member's own service-account key reaches credential loading.

    The key path flows to ``service_account.Credentials.from_service_account_file``
    and the resulting credentials reach the Vertex AI SDK client — proving the
    member value is used rather than the (unset) global one. Both the credential
    loader and the SDK client are patched so no file or network is touched.
    """
    import hindsight_api.engine.llm_wrapper as llm_wrapper
    import hindsight_api.engine.providers.gemini_llm as gemini_llm
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import _member_to_llm

    captured: dict = {}
    sentinel_creds = object()

    class _FakeServiceAccount:
        class Credentials:
            @staticmethod
            def from_service_account_file(path, scopes=None):
                captured["key_path"] = path
                return sentinel_creds

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_wrapper, "VERTEXAI_AVAILABLE", True)
    monkeypatch.setattr(llm_wrapper, "service_account", _FakeServiceAccount)
    monkeypatch.setattr(gemini_llm.genai, "Client", _FakeClient)
    clear_config_cache()

    member = LLMMemberConfig(
        provider="vertexai",
        api_key=None,
        model="gemini-2.0-flash",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
        vertexai_project_id="member-proj",
        vertexai_service_account_key="/keys/member-sa.json",
    )
    provider = _member_to_llm(member, _empty_config())

    assert provider.provider == "vertexai"
    # The member's key path was loaded, and the credentials reached the SDK client.
    assert captured["key_path"] == "/keys/member-sa.json"
    assert captured["credentials"] is sentinel_creds


# ── litellmrouter member build path (_member_to_llm) ─────────────────────────────


def test_member_to_llm_passes_litellmrouter_config(clean_llm_env, monkeypatch):
    """A litellmrouter member's own router config reaches the LiteLLM router build.

    The global config has no router config, so this also proves the member's
    config is used (no "requires a config object" error) instead of the global
    fallback. The LiteLLM router provider is patched so no router is constructed.
    """
    import hindsight_api.engine.providers as providers
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import _member_to_llm

    captured: dict = {}

    class _FakeRouterLLM:
        provider = "litellmrouter"

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(providers, "LiteLLMRouterLLM", _FakeRouterLLM)
    clear_config_cache()

    router_cfg = {"model_list": [{"model_name": "m"}]}
    member = LLMMemberConfig(
        provider="litellmrouter",
        api_key=None,
        model="m",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
        litellmrouter_config=router_cfg,
    )
    provider = _member_to_llm(member, _empty_config())

    assert provider.provider == "litellmrouter"
    # The member's own router config flowed to the LiteLLM router build.
    assert captured["config"] == router_cfg
