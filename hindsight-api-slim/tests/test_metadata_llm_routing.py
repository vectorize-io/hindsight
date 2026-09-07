"""Retain-only metadata routing across a multi-LLM chain.

``{"mode": "metadata"}`` picks a chain member from each retained item's own
metadata. The whole feature rests on one fact about the retain pipeline: a single
``extract_facts_from_text`` call handles exactly one retain item, so the item's
metadata is unambiguous at the point the member is chosen and nothing has to be
stored, unioned across a batch, or reconciled between operations.

These tests therefore pin three things: the config parses and validates, the
selection rules (first match wins, string comparison, list values, no match ->
primary), and that the real extraction path actually calls the selected member
with the operation's trace context intact.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import (
    LLM_STRATEGY_FAILOVER,
    LLM_STRATEGY_METADATA,
    LLM_STRATEGY_ROUND_ROBIN,
    HindsightConfig,
    LLMMetadataRoute,
    LLMStrategyConfig,
    _parse_llm_strategy,
)
from hindsight_api.engine.llm_wrapper import ConfiguredLLMProvider, LLMProvider
from hindsight_api.engine.memory_engine import validate_retain_batch_support
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.response_models import TokenUsage


def _routes(*specs: tuple[str, str, int]) -> list[LLMMetadataRoute]:
    return [LLMMetadataRoute(key=k, value=v, member=m) for k, v, m in specs]


def _chain(*specs: tuple[str, str, int], members: int = 2) -> MultiLLMProvider:
    """A chain of ``members`` distinguishable providers under metadata routing."""
    providers = [
        LLMProvider(provider="mock", api_key="sk-test", base_url=None, model=f"model-{index}")
        for index in range(members)
    ]
    return MultiLLMProvider(providers, LLMStrategyConfig(mode=LLM_STRATEGY_METADATA, routes=_routes(*specs)))


# ── config parsing ────────────────────────────────────────────────────────────


def test_parses_metadata_strategy() -> None:
    strategy = _parse_llm_strategy(
        '{"mode": "metadata", "routes": [{"key": "classification", "value": "sensitive", "member": 1}]}'
    )
    assert strategy == LLMStrategyConfig(
        mode=LLM_STRATEGY_METADATA,
        weights=None,
        routes=[LLMMetadataRoute(key="classification", value="sensitive", member=1)],
    )


def test_metadata_mode_requires_routes() -> None:
    with pytest.raises(ValueError, match="must be a non-empty list"):
        _parse_llm_strategy('{"mode": "metadata"}')
    with pytest.raises(ValueError, match="must be a non-empty list"):
        _parse_llm_strategy('{"mode": "metadata", "routes": []}')


@pytest.mark.parametrize("mode", [LLM_STRATEGY_FAILOVER, LLM_STRATEGY_ROUND_ROBIN])
def test_routes_rejected_outside_metadata_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="only valid with mode 'metadata'"):
        _parse_llm_strategy('{"mode": "%s", "routes": [{"key": "k", "value": "v", "member": 1}]}' % mode)


@pytest.mark.parametrize(
    "route,message",
    [
        ('{"key": "", "value": "v", "member": 1}', r"routes\.0\.key: String should have at least 1 character"),
        ('{"value": "v", "member": 1}', r"routes\.0\.key: Field required"),
        ('{"key": "k", "value": 3, "member": 1}', r"routes\.0\.value: Input should be a valid string"),
        ('{"key": "k", "value": "v"}', r"routes\.0\.member: Field required"),
        (
            '{"key": "k", "value": "v", "member": -1}',
            r"routes\.0\.member: Input should be greater than or equal to 0",
        ),
        # bool is an int subclass, so only strict mode rejects {"member": true},
        # which is a typo rather than a request for member 1.
        ('{"key": "k", "value": "v", "member": true}', r"routes\.0\.member: Input should be a valid integer"),
        # extra="forbid": a misspelled key must fail, not leave the route on the default member.
        ('{"key": "k", "value": "v", "member": 1, "membr": 2}', r"routes\.0\.membr: Extra inputs are not permitted"),
        ('"not-an-object"', r"routes\.0: Input should be a valid dictionary"),
    ],
)
def test_malformed_route_fails_fast(route: str, message: str) -> None:
    """Route shape is enforced by the model, so the error names the offending field."""
    with pytest.raises(ValueError, match=message):
        _parse_llm_strategy('{"mode": "metadata", "routes": [%s]}' % route)


def test_route_beyond_chain_length_fails_at_construction() -> None:
    """A typo'd index must fail at startup, not silently at the first sensitive retain."""
    with pytest.raises(ValueError, match="selects member 5, but the chain has members 0..1"):
        _chain(("classification", "sensitive", 5), members=2)


# ── member selection ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "metadata,expected_model",
    [
        (None, None),
        ({}, None),
        ({"classification": "public"}, None),
        ({"unrelated": "sensitive"}, None),
        ({"classification": "sensitive"}, "model-1"),
        # Values are compared as strings: retain metadata is free-form JSON.
        ({"classification": ["internal", "sensitive"]}, "model-1"),
        # A dict value can never equal a route's string value.
        ({"classification": {"level": "sensitive"}}, None),
    ],
)
def test_member_for_metadata(metadata: dict[str, Any] | None, expected_model: str | None) -> None:
    chain = _chain(("classification", "sensitive", 1))
    member = chain.member_for_metadata(metadata)
    if expected_model is None:
        # None means "stay where you are" — the caller is already on the primary.
        assert member is None
    else:
        assert member.model == expected_model


def test_numeric_and_bool_metadata_match_by_string_form() -> None:
    chain = _chain(("tier", "1", 1), ("archived", "true", 1))
    assert chain.member_for_metadata({"tier": 1}).model == "model-1"
    assert chain.member_for_metadata({"tier": "1"}).model == "model-1"
    assert chain.member_for_metadata({"archived": True}).model == "model-1"
    assert chain.member_for_metadata({"archived": False}) is None


def test_first_matching_route_wins() -> None:
    """Overlapping routes resolve by declaration order rather than erroring.

    One item is one prompt, so there is never a second item whose classification
    also has to be satisfied — an operator ordering their routes is enough.
    """
    chain = _chain(("classification", "sensitive", 1), ("region", "eu", 2), members=3)
    both = {"classification": "sensitive", "region": "eu"}
    assert chain.member_for_metadata(both).model == "model-1"

    reversed_chain = _chain(("region", "eu", 2), ("classification", "sensitive", 1), members=3)
    assert reversed_chain.member_for_metadata(both).model == "model-2"


def test_non_metadata_chain_never_routes() -> None:
    failover = MultiLLMProvider(
        [LLMProvider(provider="mock", api_key="sk-test", base_url=None, model=f"model-{i}") for i in range(2)],
        LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER),
    )
    assert failover.member_for_metadata({"classification": "sensitive"}) is None


def test_metadata_chain_does_not_fail_over_between_lanes() -> None:
    """A direct call has no item to route on, so it stays on the primary.

    Failing over would move a request into another member's lane, which is
    exactly the thing routing exists to prevent.
    """
    chain = _chain(("classification", "sensitive", 1))
    assert chain._member_order() == [0]


# ── re-binding the configured wrapper ─────────────────────────────────────────


def test_route_for_rebinds_member_and_keeps_trace_context() -> None:
    """A routed call must stay attributed to the same operation and trace."""
    chain = _chain(("classification", "sensitive", 1))
    trace_ctx = object()
    configured = ConfiguredLLMProvider(chain, ["safety"], trace_ctx)

    routed = configured.route_for({"classification": "sensitive"})
    assert routed is not configured
    assert routed.model == "model-1"
    assert routed.trace_context() is trace_ctx
    assert object.__getattribute__(routed, "_gemini_safety_settings") == ["safety"]

    assert configured.route_for({"classification": "public"}) is configured


def test_route_for_is_a_noop_for_a_single_provider() -> None:
    configured = ConfiguredLLMProvider(
        LLMProvider(provider="mock", api_key="sk-test", base_url=None, model="solo"), None, None
    )
    assert configured.route_for({"classification": "sensitive"}) is configured


# ── the real extraction path ──────────────────────────────────────────────────


async def _extract(metadata: dict[str, Any] | None, chain: MultiLLMProvider) -> ConfiguredLLMProvider:
    """Run real fact extraction and return the wrapper the prompt was sent with."""
    from hindsight_api.engine.retain import fact_extraction

    config = HindsightConfig.from_env()
    config.retain_extraction_mode = "facts"
    config.retain_batch_enabled = False

    seen: list[ConfiguredLLMProvider] = []

    async def _capture(*, llm_config, **kwargs):
        seen.append(llm_config)
        return [], TokenUsage()

    original = fact_extraction._extract_facts_with_auto_split
    fact_extraction._extract_facts_with_auto_split = AsyncMock(side_effect=_capture)
    try:
        await fact_extraction.extract_facts_from_text(
            text="Quarterly revenue was up.",
            event_date=None,
            llm_config=ConfiguredLLMProvider(chain, None, None),
            config=config,
            metadata=metadata,
        )
    finally:
        fact_extraction._extract_facts_with_auto_split = original
    assert seen, "extraction never reached the LLM"
    return seen[0]


async def test_extraction_sends_a_routed_item_to_its_member() -> None:
    chain = _chain(("classification", "sensitive", 1))
    assert (await _extract({"classification": "sensitive"}, chain)).model == "model-1"


async def test_extraction_sends_an_unrouted_item_to_the_primary() -> None:
    chain = _chain(("classification", "sensitive", 1))
    assert (await _extract({"classification": "public"}, chain)).model == "model-0"
    assert (await _extract(None, chain)).model == "model-0"


# ── batch retain is incompatible ──────────────────────────────────────────────


async def test_startup_rejects_batch_retain_with_metadata_routing() -> None:
    """Batch submits one job per operation, so it cannot honour per-item routes."""
    config = HindsightConfig.from_env()
    config.retain_batch_enabled = True
    with pytest.raises(RuntimeError, match="not compatible with the 'metadata' LLM strategy"):
        await validate_retain_batch_support(_chain(("classification", "sensitive", 1)), config)


async def test_metadata_routing_is_fine_when_batch_retain_is_off() -> None:
    config = HindsightConfig.from_env()
    config.retain_batch_enabled = False
    await validate_retain_batch_support(_chain(("classification", "sensitive", 1)), config)


# ── only retain can select this strategy ──────────────────────────────────────


def _build_config(**overrides: Any) -> HindsightConfig:
    from hindsight_api.config import LLMMemberConfig

    config = HindsightConfig.from_env()
    member = LLMMemberConfig(
        provider="mock",
        api_key="sk-test",
        model="member-1",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
    )
    config.llm_members = [member]
    config.llm_strategy = None
    for prefix in ("retain_", "reflect_", "consolidation_"):
        setattr(config, f"{prefix}llm_members", [])
        setattr(config, f"{prefix}llm_strategy", None)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _metadata_strategy() -> LLMStrategyConfig:
    return LLMStrategyConfig(mode=LLM_STRATEGY_METADATA, routes=_routes(("classification", "sensitive", 1)))


def _build(prefix: str, config: HindsightConfig) -> Any:
    from hindsight_api.engine.memory_engine import _build_llm, _LLMCallDefaults

    base = LLMProvider(provider="mock", api_key="sk-test", base_url=None, model="model-0")
    defaults = _LLMCallDefaults(timeout=None, max_retries=1, initial_backoff=0.1, max_backoff=1.0)
    return _build_llm(base, config, prefix, defaults)


@pytest.mark.parametrize("prefix", ["reflect_", "consolidation_"])
def test_explicit_metadata_strategy_is_rejected_for_non_retain_operations(prefix: str) -> None:
    """Accepting it would pin the chain to the primary and look like broken routes."""
    config = _build_config(**{f"{prefix}llm_strategy": _metadata_strategy()})
    with pytest.raises(ValueError, match="only supported for retain"):
        _build(prefix, config)


def test_retain_may_select_metadata_explicitly() -> None:
    config = _build_config(retain_llm_strategy=_metadata_strategy())
    assert _build("retain_", config).strategy.mode == LLM_STRATEGY_METADATA


def test_global_metadata_strategy_is_inherited_and_pins_other_operations_to_the_primary() -> None:
    """The documented setup is a single global strategy; other operations keep the primary."""
    config = _build_config(llm_strategy=_metadata_strategy())
    for prefix in ("", "retain_", "reflect_", "consolidation_"):
        chain = _build(prefix, config)
        assert chain.strategy.mode == LLM_STRATEGY_METADATA
        assert chain._member_order() == [0]
