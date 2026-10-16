"""The decision model is set up once and each operation may override it."""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web

from hindsight_api.config import DecisionModelConfig, HindsightConfig
from hindsight_api.engine.decision_model import (
    DecisionAnswer,
    DecisionModel,
    DecisionQuestion,
    DecisionRefusedError,
    DecisionResponse,
    DecisionUsage,
)
from hindsight_api.engine.remote_retry import RetryPolicy
from hindsight_api.engine.token_encoding import count_tokens
from tests.aiohttp_stub import stub_server

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
            max_request_tokens=26_000,
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


def test_each_provider_has_its_own_defaults(monkeypatch):
    config = _config(
        monkeypatch,
        {
            "HINDSIGHT_API_DECISION_MODEL_PROVIDER": "openai",
            "HINDSIGHT_API_DECISION_MODEL_API_KEY": "openai-key",
        },
    )
    recall = config.decision_model_config("recall")
    assert recall is not None
    assert (recall.provider, recall.model, recall.base_url) == ("openai", "gpt-6-luna", "https://api.openai.com")


def test_an_operation_on_another_provider_borrows_nothing_from_the_global_one(monkeypatch):
    config = _config(
        monkeypatch,
        {
            **_GLOBAL,
            "HINDSIGHT_API_DECISION_MODEL_MODEL": "jev-preview",
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_PROVIDER": "openai",
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_API_KEY": "openai-key",
        },
    )
    reflect = config.decision_model_config("reflect")
    assert reflect is not None
    assert (reflect.api_key, reflect.model, reflect.base_url) == ("openai-key", "gpt-6-luna", "https://api.openai.com")


def test_another_provider_without_its_own_key_fails_at_startup(monkeypatch):
    with pytest.raises(ValueError, match="HINDSIGHT_API_REFLECT_DECISION_MODEL_API_KEY"):
        _config(monkeypatch, {**_GLOBAL, "HINDSIGHT_API_REFLECT_DECISION_MODEL_PROVIDER": "openai"}).validate()


def test_cloudflare_needs_its_account_url(monkeypatch):
    env = {"HINDSIGHT_API_DECISION_MODEL_PROVIDER": "cloudflare", "HINDSIGHT_API_DECISION_MODEL_API_KEY": "cf"}
    with pytest.raises(ValueError, match="HINDSIGHT_API_DECISION_MODEL_BASE_URL"):
        _config(monkeypatch, env).validate()
    account_url = "https://api.cloudflare.com/client/v4/accounts/acct/ai/run"
    config = _config(monkeypatch, {**env, "HINDSIGHT_API_DECISION_MODEL_BASE_URL": account_url})
    recall = config.decision_model_config("recall")
    assert recall is not None
    assert (recall.model, recall.base_url) == ("clef", account_url)


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


_QUESTIONS = {
    "urgent": DecisionQuestion(type="noul", instructions="Is it urgent?"),
    "team": DecisionQuestion(type="choice", instructions="Which team?", criteria={"billing": "Payments", "ops": None}),
    "severity": DecisionQuestion(type="score", instructions="How severe?", criteria=["None", "Minor", "Critical"]),
}

#: The System One request body every typesafe/cloudflare call sends for _QUESTIONS.
_SYSTEM_ONE_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "ops": None}},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["None", "Minor", "Critical"]},
}

_SYSTEM_ONE_ANSWERS = {
    "answers": {
        "urgent": {"type": "noul", "noul": 0.9},
        "team": {"type": "choice", "choice": "ops", "probabilities": {"billing": 0.2, "ops": 0.8}, "confidence": 0.7},
        "severity": {"type": "score", "score": 1.6, "confidence": 0.8},
    },
    "usage": {"input_tokens": 12, "output_tokens": 1},
}

_EXPECTED = DecisionResponse(
    answers={
        "urgent": DecisionAnswer(noul=0.9),
        "team": DecisionAnswer(choice="ops", probabilities={"billing": 0.2, "ops": 0.8}, confidence=0.7),
        "severity": DecisionAnswer(score=1.6, confidence=0.8),
    },
    usage=DecisionUsage(input_tokens=12, output_tokens=1),
)


@dataclass
class _Upstream:
    """What the stub server saw, and what it answers."""

    reply: Any
    paths: list[str] = field(default_factory=list)
    bodies: list[Any] = field(default_factory=list)
    auth: list[str] = field(default_factory=list)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.paths.append(request.path)
        self.bodies.append(await request.json())
        self.auth.append(request.headers["Authorization"])
        return web.json_response(self.reply)


def _client(provider: str, base_url: str, model: str) -> DecisionModel:
    return DecisionModel.from_config(
        DecisionModelConfig(
            provider=provider,
            api_key="k",
            model=model,
            base_url=base_url,
            timeout=5.0,
            max_concurrent=2,
            max_request_tokens=26_000,
        )
    )


async def test_typesafe_speaks_system_one():
    upstream = _Upstream(reply=_SYSTEM_ONE_ANSWERS)
    async with stub_server(upstream.handle) as base_url:
        answer = await _client("typesafe", base_url + "/", "jev-preview").ask({"fact": "x"}, _QUESTIONS)

    assert answer == _EXPECTED
    assert upstream.paths == ["/v1/systemone"]
    assert upstream.auth == ["Bearer k"]
    assert upstream.bodies == [{"model": "jev-preview", "state": {"fact": "x"}, "questions": _SYSTEM_ONE_QUESTIONS}]


async def test_cloudflare_posts_system_one_to_the_workers_ai_model_and_unwraps_the_envelope():
    # Workers AI's REST API wraps every answer in Cloudflare's v4 envelope.
    upstream = _Upstream(reply={"result": _SYSTEM_ONE_ANSWERS, "success": True, "errors": [], "messages": []})
    async with stub_server(upstream.handle) as base_url:
        account_url = f"{base_url}/client/v4/accounts/acct/ai/run"
        answer = await _client("cloudflare", account_url, "clef").ask("Checkout is down.", _QUESTIONS)

    assert answer == _EXPECTED
    assert upstream.paths == ["/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"]
    assert upstream.bodies == [{"model": "clef", "state": "Checkout is down.", "questions": _SYSTEM_ONE_QUESTIONS}]


async def test_openai_translates_to_the_decisions_api_and_back():
    # Recorded from a live gpt-6-luna call, reshaped to these questions.
    upstream = _Upstream(
        reply={
            "model": "gpt-6-luna",
            "answers": [
                {"type": "predicate", "name": "urgent", "probability": 0.9},
                {
                    "type": "choice",
                    "name": "team",
                    "choice": "ops",
                    "probabilities": [{"value": "billing", "probability": 0.2}, {"value": "ops", "probability": 0.8}],
                    "confidence": 0.7,
                },
                {
                    "type": "score",
                    "name": "severity",
                    "score": 1.6,
                    "probabilities": [
                        {"value": 0, "label": "None", "probability": 0.0},
                        {"value": 1, "label": "Minor", "probability": 0.4},
                        {"value": 2, "label": "Critical", "probability": 0.6},
                    ],
                    "confidence": 0.8,
                },
            ],
            "usage": {"input_tokens": 12, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1},
        }
    )
    async with stub_server(upstream.handle) as base_url:
        answer = await _client("openai", base_url, "gpt-6-luna").ask({"fact": "x"}, _QUESTIONS)

    assert answer == _EXPECTED
    assert upstream.paths == ["/v1/decisions"]
    assert upstream.bodies == [
        {
            "model": "gpt-6-luna",
            "input": '{"fact": "x"}',
            "questions": [
                {"name": "urgent", "instructions": "Is it urgent?", "type": "predicate"},
                {
                    "name": "team",
                    "instructions": "Which team?",
                    "type": "choice",
                    "choices": [
                        {"value": "billing", "description": "Payments"},
                        {"value": "ops", "description": "ops"},
                    ],
                },
                {
                    "name": "severity",
                    "instructions": "How severe?",
                    "type": "score",
                    "levels": [
                        {"label": "None", "description": "None"},
                        {"label": "Minor", "description": "Minor"},
                        {"label": "Critical", "description": "Critical"},
                    ],
                },
            ],
        }
    ]


async def test_openai_refusal_raises():
    upstream = _Upstream(reply={"answers": [{"type": "refusal", "name": "urgent"}]})
    async with stub_server(upstream.handle) as base_url:
        with pytest.raises(DecisionRefusedError, match="urgent"):
            await _client("openai", base_url, "gpt-6-luna").ask("x", {"urgent": _QUESTIONS["urgent"]})


def test_the_default_request_ceiling_fits_a_32k_window(monkeypatch):
    """The endpoint counted 14% more than we do on the same body."""
    assert _config(monkeypatch, {}).decision_model_max_request_tokens * 1.14 < 32_768


@pytest.mark.parametrize(
    "setting",
    [
        "HINDSIGHT_API_DECISION_MODEL_MAX_REQUEST_TOKENS=500",
        "HINDSIGHT_API_DECISION_MODEL_MAX_CONCURRENT=-1",
        "HINDSIGHT_API_DECISION_MODEL_TIMEOUT=-5",
    ],
)
def test_a_setting_no_request_can_work_with_fails_at_startup(monkeypatch, setting):
    """Not on the first request, as a 422 or a hang."""
    name, value = setting.split("=")
    with pytest.raises(ValueError, match=name):
        _config(monkeypatch, {**_GLOBAL, name: value}).validate()


@dataclass
class _Answering:
    """A System One upstream answering every question it is sent. A request whose
    questions include ``down`` fails with 503; the first ``limited`` requests get 429."""

    down: str | None = None
    limited: int = 0
    bodies: list[Any] = field(default_factory=list)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.bodies.append(body)
        if self.limited:
            self.limited -= 1
            return web.json_response({"detail": "slow down"}, status=429)
        if self.down in body["questions"]:
            return web.json_response({"detail": "down"}, status=503)
        return web.json_response({"answers": {key: {"noul": 0.5} for key in body["questions"]}})


def _splitting_client(base_url: str, max_request_tokens: int = 26_000) -> DecisionModel:
    return DecisionModel(
        provider="typesafe",
        api_key="k",
        model="jev-latest",
        base_url=base_url,
        timeout=5.0,
        max_concurrent=2,
        max_request_tokens=max_request_tokens,
        retry_policy=RetryPolicy(max_retries=2, initial_backoff=0.0, max_backoff=0.0, budget_seconds=5.0),
    )


def _nouls(count: int, text: str) -> dict[str, DecisionQuestion]:
    return {f"m{i}": DecisionQuestion(type="noul", instructions=f"{text} {i}") for i in range(count)}


async def test_every_request_fits_the_ceiling_counting_the_whole_body():
    """The endpoint counts every token of the body against its model's window. A ceiling
    that counted only the candidates was overrun by the state, which goes out with every
    request: 63k-token requests sent to a 32k window came back 422."""
    upstream = _Answering()
    questions = _nouls(60, "memory about the storage layer")
    async with stub_server(upstream.handle) as base_url:
        answers = await _splitting_client(base_url, max_request_tokens=1_500).ask_all(
            "what did we decide about the storage layer " * 40, questions
        )
    assert len(upstream.bodies) > 1
    assert all(count_tokens(json.dumps(body)) <= 1_500 for body in upstream.bodies)
    assert list(answers) == list(questions), "every question asked once, answers keyed as sent"


async def test_a_pool_over_the_question_cap_is_split():
    upstream = _Answering()
    async with stub_server(upstream.handle) as base_url:
        await _splitting_client(base_url).ask_all("q", _nouls(DecisionModel.MAX_QUESTIONS + 5, "x"))
    assert sorted(len(body["questions"]) for body in upstream.bodies) == [5, DecisionModel.MAX_QUESTIONS]


async def test_a_rate_limited_request_is_retried():
    """One replica returns 429 as soon as a few recalls overlap; the request must not be lost."""
    upstream = _Answering(limited=1)
    async with stub_server(upstream.handle) as base_url:
        answers = await _splitting_client(base_url).ask_all("q", _nouls(1, "x"))
    assert len(upstream.bodies) == 2
    assert answers["m0"].noul == 0.5


async def test_a_request_that_keeps_failing_loses_only_its_own_answers():
    upstream = _Answering(down="m0")
    questions = _nouls(4, "word " * 400)
    async with stub_server(upstream.handle) as base_url:
        answers = await _splitting_client(base_url, max_request_tokens=1_000).ask_all("q", questions)
    failed = next(body["questions"] for body in upstream.bodies if "m0" in body["questions"])
    assert len(failed) < len(questions), "the pool must have been split for this to mean anything"
    assert set(answers) == set(questions) - set(failed)
