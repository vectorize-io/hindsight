"""Custom LLM providers: an operator's own ``LLMInterface`` class, named by its import path.

``EchoProvider`` below is the reference example for writing one. It is the smallest
complete provider: a synchronous ``call`` and the three batch methods. A real one
replaces the canned answers with requests to its backend.

    HINDSIGHT_API_LLM_PROVIDER=tests.test_custom_llm_provider:EchoProvider
"""

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import Any

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.llm_interface import LLMInterface
from hindsight_api.engine.llm_wrapper import LLMProvider, requires_api_key
from hindsight_api.engine.response_models import (
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchStatus,
    BatchSubmission,
    LLMCallResult,
    TokenUsage,
)
from hindsight_api.engine.retain.fact_extraction import RetainContent, extract_facts_from_contents_batch_api

ECHO_PROVIDER = "tests.test_custom_llm_provider:EchoProvider"

ALICE_FACT = {
    "what": "Alice is a senior software engineer at TechCorp",
    "when": "present",
    "where": "TechCorp",
    "who": "Alice",
    "why": "Professional background information",
    "fact_type": "world",
    "fact_kind": "conversation",
}


class EchoProvider(LLMInterface):
    """Answers every call with one canned fact; batches complete as soon as they are submitted."""

    def __init__(self, **kwargs: Any) -> None:
        # provider, api_key, base_url, model, reasoning_effort and timeout are stored by
        # LLMInterface; extra_body and default_headers arrive in kwargs.
        super().__init__(**kwargs)
        self.extra_body: dict[str, Any] | None = kwargs.get("extra_body")
        self.batches: dict[str, list[BatchRequest]] = {}

    async def verify_connection(self) -> None:
        pass

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        cached_prefix: str | None = None,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMCallResult:
        text = json.dumps({"facts": [ALICE_FACT]})
        content = response_format.model_validate_json(text) if response_format and not skip_validation else text
        return LLMCallResult(content=content, usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15))

    async def supports_batch_api(self) -> bool:
        return True

    async def submit_batch(self, requests: list[BatchRequest]) -> BatchSubmission:
        batch_id = f"echo-{len(self.batches)}"
        self.batches[batch_id] = requests
        return BatchSubmission(batch_id=batch_id, status="validating")

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        total = len(self.batches[batch_id])
        return BatchStatus(
            batch_id=batch_id,
            status="completed",
            request_counts=BatchRequestCounts(total=total, completed=total),
        )

    async def retrieve_batch_results(self, batch_id: str) -> list[BatchResult]:
        return [
            BatchResult(
                custom_id=request.custom_id,
                content=json.dumps({"facts": [ALICE_FACT]}),
                usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
            for request in self.batches[batch_id]
        ]

    async def cleanup(self) -> None:
        pass


class NotAProvider:
    pass


def _echo_llm(**overrides: Any) -> LLMProvider:
    settings: dict[str, Any] = {"provider": ECHO_PROVIDER, "api_key": "", "base_url": "", "model": "echo-1"}
    return LLMProvider(**(settings | overrides))


def test_class_path_builds_the_custom_provider_with_the_configured_settings():
    llm = _echo_llm(base_url="https://echo.example/v1", extra_body={"thinking": {"type": "disabled"}})

    impl = llm._provider_impl
    assert isinstance(impl, EchoProvider)
    assert impl.model == "echo-1"
    assert impl.base_url == "https://echo.example/v1"
    assert impl.extra_body == {"thinking": {"type": "disabled"}}
    # Class names are case-sensitive, so the path is kept as written.
    assert llm.provider == ECHO_PROVIDER


def test_custom_provider_needs_no_api_key():
    assert not requires_api_key(ECHO_PROVIDER)


@pytest.mark.parametrize(
    "path",
    [
        "tests.no_such_module:EchoProvider",
        "tests.test_custom_llm_provider:Missing",
        ECHO_PROVIDER.replace("Echo", "NotA"),
    ],
)
def test_unloadable_class_path_fails_at_construction(path: str):
    with pytest.raises(ValueError, match="Cannot load custom LLM provider"):
        _echo_llm(provider=path)


@pytest.mark.asyncio
async def test_custom_provider_serves_a_call():
    result = await _echo_llm().call(messages=[{"role": "user", "content": "hi"}], scope="test")

    assert json.loads(result.content)["facts"][0]["who"] == "Alice"
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_custom_provider_serves_batch_retain():
    llm = _echo_llm()
    config = HindsightConfig.from_env()
    config.retain_batch_enabled = True
    config.retain_batch_poll_interval_seconds = 0
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False

    extraction = await extract_facts_from_contents_batch_api(
        contents=[
            RetainContent(
                content="Alice is a senior software engineer at TechCorp.",
                event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
                context="team overview",
            )
        ],
        llm_config=llm,
        config=config,
        pool=None,
        operation_id=None,
        schema=None,
    )

    impl = llm._provider_impl
    assert isinstance(impl, EchoProvider)
    [submitted] = impl.batches.values()
    assert [request.custom_id for request in submitted] == ["chunk_0"]
    assert submitted[0].body["model"] == "echo-1"
    assert len(extraction.facts) == 1
    assert "Alice" in extraction.facts[0].fact_text
    assert extraction.usage.total_tokens == 150
