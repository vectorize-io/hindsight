"""Bounded retry/backoff for remote rerankers (issue #4134).

Rerank sits on the same synchronous recall path as the query embedding, and with no
fallback chain configured — the default — the reranker is used bare: nothing between
`CrossEncoderReranker.rerank` and the caller catches anything. Before this a single
Cohere/Google/SiliconFlow 429 failed the entire recall.

The retry lives on `CrossEncoderModel.predict`, which every backend inherits, so a new
remote reranker gets it without having to remember. `TestEveryRemoteRerankerRetries`
is the guard that keeps that true.
"""

import asyncio
import inspect
import time
from dataclasses import fields
from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine import cross_encoder as cross_encoder_module
from hindsight_api.engine.cross_encoder import (
    _RERANKER_PROVIDERS_WITHOUT_RETRY,
    CrossEncoderModel,
    MultiCrossEncoder,
    create_cross_encoder_from_env,
)
from hindsight_api.engine.remote_retry import RetryPolicy

# Fast policy so these tests exercise the retry logic, not the sleeps.
_FAST_POLICY = RetryPolicy(max_retries=3, initial_backoff=0.01, max_backoff=0.02, budget_seconds=5.0)


class _ApiError(Exception):
    """Stand-in for a provider SDK error, which exposes `status_code`."""

    def __init__(self, status_code: int):
        super().__init__(f"upstream error (status {status_code})")
        self.status_code = status_code


class _FlakyReranker(CrossEncoderModel):
    """A remote-shaped backend that fails `failures` times before answering."""

    def __init__(self, failures: int, status: int = 429, policy: RetryPolicy | None = _FAST_POLICY):
        self.calls = 0
        self._failures = failures
        self._status = status
        self.retry_policy = policy

    @property
    def provider_name(self) -> str:
        return "flaky"

    async def initialize(self) -> None:
        return None

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        if self.calls <= self._failures:
            raise _ApiError(self._status)
        return [1.0] * len(pairs)


PAIRS = [("q", "doc a"), ("q", "doc b")]


class TestRemoteRerankerRetries:
    async def test_transient_status_then_success(self):
        """A 429 is retried and the later success is returned, not raised at the caller."""
        reranker = _FlakyReranker(failures=1)
        assert await reranker.predict(PAIRS) == [1.0, 1.0]
        assert reranker.calls == 2

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408])
    async def test_transient_service_responses_are_retried(self, status):
        reranker = _FlakyReranker(failures=1, status=status)
        assert await reranker.predict(PAIRS) == [1.0, 1.0]
        assert reranker.calls == 2

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_permanent_client_errors_fail_fast(self, status):
        """Auth and validation failures must not be retried — retrying cannot fix them."""
        reranker = _FlakyReranker(failures=99, status=status)
        with pytest.raises(_ApiError):
            await reranker.predict(PAIRS)
        assert reranker.calls == 1

    async def test_exhausted_retries_propagate(self):
        """A sustained outage still surfaces, after a bounded attempt count."""
        reranker = _FlakyReranker(failures=99)
        with pytest.raises(_ApiError):
            await reranker.predict(PAIRS)
        assert reranker.calls == _FAST_POLICY.max_retries + 1

    async def test_no_policy_means_no_retry(self):
        """The in-process backends and TEI opt out — one attempt, straight through."""
        reranker = _FlakyReranker(failures=99, policy=None)
        with pytest.raises(_ApiError):
            await reranker.predict(PAIRS)
        assert reranker.calls == 1

    async def test_budget_bounds_the_added_latency(self):
        """The wall-clock budget caps what one rerank can add to a live recall."""
        policy = RetryPolicy(max_retries=50, initial_backoff=0.05, max_backoff=0.05, budget_seconds=0.1)
        reranker = _FlakyReranker(failures=999, policy=policy)

        started = time.monotonic()
        with pytest.raises(_ApiError):
            await reranker.predict(PAIRS)

        # Without the budget, 50 retries at 0.05s would be ~2.5s.
        assert time.monotonic() - started < 1.0
        assert reranker.calls < policy.max_retries + 1


class TestFailoverChainAdvancesOnlyAfterRetries:
    """`MultiCrossEncoder`'s contract: a member exhausts its own retries first.

    That claim sat in the docstring while only the TEI member could honour it, so an
    ordinary quota blip on the primary burned a fallback — silently degrading ranking
    quality over something a one-second backoff absorbs.
    """

    async def test_primary_recovers_within_its_retries_and_no_fallback_is_touched(self):
        primary = _FlakyReranker(failures=2)
        fallback = _FlakyReranker(failures=0)
        chain = MultiCrossEncoder([primary, fallback])

        assert await chain.predict(PAIRS) == [1.0, 1.0]
        assert primary.calls == 3
        assert fallback.calls == 0

    async def test_chain_advances_once_the_primary_exhausts_its_retries(self):
        primary = _FlakyReranker(failures=99)
        fallback = _FlakyReranker(failures=0)
        chain = MultiCrossEncoder([primary, fallback])

        assert await chain.predict(PAIRS) == [1.0, 1.0]
        assert primary.calls == _FAST_POLICY.max_retries + 1
        assert fallback.calls == 1

    async def test_the_chain_itself_does_not_add_a_second_retry_layer(self):
        """Retry is per member; wrapping the chain too would multiply the attempts."""
        chain = MultiCrossEncoder([_FlakyReranker(failures=0), _FlakyReranker(failures=0)])
        assert chain.retry_policy is None


class TestEveryRemoteRerankerRetries:
    """A structural guard over the whole reranker family.

    The backend that forgets is by definition the one nobody wrote a test for — that
    is how every remote reranker but TEI shipped with no retry at all. These assert
    over the classes and providers enumerated from the module, not over the ones we
    happened to remember.
    """

    def _backend_classes(self) -> list[type[CrossEncoderModel]]:
        return [
            obj
            for obj in vars(cross_encoder_module).values()
            if inspect.isclass(obj)
            and issubclass(obj, CrossEncoderModel)
            and obj is not CrossEncoderModel
            and obj is not MultiCrossEncoder  # a chain, not a backend
        ]

    def test_every_backend_implements_predict_via_the_shared_entry_point(self):
        """A backend that overrides `predict` opts itself out of retry silently."""
        assert self._backend_classes(), "no backends discovered — the guard would vacuously pass"
        overriding = [c.__name__ for c in self._backend_classes() if "predict" in vars(c)]
        assert not overriding, (
            f"{overriding} override predict() and so bypass the shared retry. Implement _predict() instead."
        )

    def test_every_backend_implements_predict(self):
        missing = [c.__name__ for c in self._backend_classes() if "_predict" not in vars(c)]
        assert not missing, f"{missing} implement neither _predict() nor predict()"

    @pytest.mark.parametrize(
        "provider, extra",
        [
            ("cohere", {"reranker_cohere_api_key": "k"}),
            ("openrouter", {"reranker_openrouter_api_key": "k"}),
            ("zeroentropy", {"reranker_zeroentropy_api_key": "k"}),
            (
                "siliconflow",
                {"reranker_siliconflow_api_key": "k", "reranker_siliconflow_base_url": "https://example.invalid"},
            ),
            ("alibaba", {"reranker_alibaba_api_key": "k"}),
            ("google", {"reranker_google_project_id": "p"}),
            ("litellm", {"reranker_litellm_api_base": "https://example.invalid"}),
            ("litellm-sdk", {"reranker_litellm_sdk_api_key": "k"}),
        ],
    )
    def test_factory_gives_every_remote_provider_a_policy(self, provider, extra):
        encoder = self._build(provider, extra)
        assert encoder.retry_policy is not None, f"{provider} was built without a retry policy"
        assert encoder.retry_policy.max_retries == 7
        assert encoder.retry_policy.budget_seconds == 42.5

    @pytest.mark.parametrize("provider", sorted(_RERANKER_PROVIDERS_WITHOUT_RETRY))
    def test_exempt_providers_are_left_alone(self, provider):
        """In-process backends and TEI (own retry loop) must not gain a second layer."""
        extra = (
            {
                "reranker_tei_url": "https://example.invalid",
                "reranker_tei_batch_size": 32,
                "reranker_tei_max_concurrent": 4,
            }
            if provider == "tei"
            else {}
        )
        encoder = self._build(provider, extra)
        assert encoder.retry_policy is None

    def _build(self, provider: str, extra: dict) -> CrossEncoderModel:
        defaults: dict = {}
        for f in fields(HindsightConfig):
            if f.type == "str":
                defaults[f.name] = ""
            elif f.type == "int":
                defaults[f.name] = 0
            elif f.type == "float":
                defaults[f.name] = 0.0
            elif f.type == "bool":
                defaults[f.name] = False
            elif str(f.type).startswith("list["):
                defaults[f.name] = []
            else:
                defaults[f.name] = None
        defaults.update(
            reranker_provider=provider,
            reranker_max_retries=7,
            reranker_initial_backoff=0.25,
            reranker_max_backoff=2.0,
            reranker_retry_budget=42.5,
            **extra,
        )
        config = HindsightConfig(**defaults)
        with patch("hindsight_api.config.get_config", return_value=config):
            return create_cross_encoder_from_env()


def test_retry_runs_on_the_event_loop_without_blocking_it():
    """The backoff must be `await asyncio.sleep`, not `time.sleep` — rerank is awaited.

    A blocking sleep here would stall every other coroutine on the loop, which on the
    recall path means stalling unrelated requests.
    """

    async def scenario() -> bool:
        reranker = _FlakyReranker(
            failures=1,
            policy=RetryPolicy(max_retries=2, initial_backoff=0.2, max_backoff=0.2, budget_seconds=5.0),
        )
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        await reranker.predict(PAIRS)
        tick_task.cancel()
        # The loop kept running other work during the ~0.2s backoff.
        return ticks > 2

    assert asyncio.run(scenario())
