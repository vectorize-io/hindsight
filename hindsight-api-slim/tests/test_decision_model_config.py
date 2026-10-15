"""The decision model is set up once and each operation may override it."""

import pytest

from hindsight_api.config import DecisionModelConfig, HindsightConfig
from hindsight_api.engine.decision_model import DecisionModel

_GLOBAL = {
    "HINDSIGHT_API_DECISION_MODEL_PROVIDER": "typesafe",
    "HINDSIGHT_API_DECISION_MODEL_API_KEY": "global-key",
}


def _config(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> HindsightConfig:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return HindsightConfig.from_env()


def test_no_decision_model_by_default(monkeypatch):
    config = _config(monkeypatch, {})
    config.validate()
    assert config.decision_model_config("recall") is None
    assert config.decision_model_config("reflect") is None
    assert config.decision_model_config("consolidation") is None


def test_global_setup_reaches_every_operation(monkeypatch):
    config = _config(monkeypatch, _GLOBAL)
    for operation in ("recall", "reflect", "consolidation"):
        assert config.decision_model_config(operation) == DecisionModelConfig(
            provider="typesafe",
            api_key="global-key",
            model="jev-latest",
            base_url="https://api.typesafe.ai",
            timeout=60.0,
            max_concurrent=24,
        )


def test_operation_override_wins_and_falls_back_per_setting(monkeypatch):
    config = _config(
        monkeypatch,
        {
            **_GLOBAL,
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_MODEL": "jev-preview",
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_API_KEY": "reflect-key",
        },
    )
    reflect = config.decision_model_config("reflect")
    assert reflect is not None
    assert (reflect.model, reflect.api_key, reflect.base_url) == (
        "jev-preview",
        "reflect-key",
        "https://api.typesafe.ai",
    )
    recall = config.decision_model_config("recall")
    assert recall is not None
    assert (recall.model, recall.api_key) == ("jev-latest", "global-key")


def test_operation_can_opt_out(monkeypatch):
    config = _config(monkeypatch, {**_GLOBAL, "HINDSIGHT_API_CONSOLIDATION_DECISION_MODEL_PROVIDER": "none"})
    assert config.decision_model_config("consolidation") is None
    assert config.decision_model_config("recall") is not None


def test_operation_can_have_one_without_a_global_setup(monkeypatch):
    config = _config(
        monkeypatch,
        {
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_PROVIDER": "typesafe",
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_API_KEY": "reflect-key",
        },
    )
    config.validate()
    assert config.decision_model_config("reflect") is not None
    assert config.decision_model_config("recall") is None


def test_missing_key_fails_at_startup(monkeypatch):
    with pytest.raises(ValueError, match="HINDSIGHT_API_DECISION_MODEL_API_KEY"):
        _config(monkeypatch, {"HINDSIGHT_API_DECISION_MODEL_PROVIDER": "typesafe"}).validate()


def test_unknown_provider_fails_at_startup(monkeypatch):
    with pytest.raises(ValueError, match="Unknown decision model provider for recall"):
        _config(monkeypatch, {**_GLOBAL, "HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER": "jevv"}).validate()


def test_keys_and_endpoints_are_credentials():
    credentials = HindsightConfig.get_credential_fields()
    for prefix in ("", "recall_", "reflect_", "consolidation_"):
        assert f"{prefix}decision_model_api_key" in credentials
        assert f"{prefix}decision_model_base_url" in credentials


class _Response:
    def __init__(self, body: dict):
        self._body = body
        self.status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self._body


class _Session:
    def __init__(self):
        self.posted: list[tuple[str, dict]] = []

    def get(self):
        return self

    def post(self, url, headers=None, json=None):
        self.posted.append((url, json))
        return _Response({"answers": {"q": {"noul": 0.9}}, "usage": {"input_tokens": 12, "output_tokens": 1}})


async def test_client_asks_with_the_configured_model():
    client = DecisionModel.from_config(
        DecisionModelConfig(
            provider="typesafe",
            api_key="k",
            model="jev-preview",
            base_url="https://example.test/",
            timeout=5.0,
            max_concurrent=2,
        )
    )
    session = _Session()
    client._session = session  # type: ignore[assignment]
    questions = {"q": {"type": "noul", "instructions": "Is it?"}}

    answer = await client.ask({"fact": "x"}, questions)

    assert answer.answers["q"].noul == 0.9
    assert answer.usage.input_tokens == 12
    assert session.posted == [
        (
            "https://example.test/v1/systemone",
            {"model": "jev-preview", "state": {"fact": "x"}, "questions": questions},
        )
    ]
