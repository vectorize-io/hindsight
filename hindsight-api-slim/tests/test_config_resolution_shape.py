"""Shape and cost guarantees for ConfigResolver's per-request resolution.

Resolution runs on every recall and every retain sub-batch, so how it builds the
resolved config matters as much as what it resolves. These tests pin the
properties the cheap path depends on (#4209): the config is shallow-copyable,
the copy is indistinguishable from the dict round-trip it replaced, typed member
dataclasses survive it, and a caller still cannot reach into the process-global
config through a value it was handed.
"""

import copy
import dataclasses
from dataclasses import asdict, replace

import pytest

from hindsight_api.config import HindsightConfig, LLMMemberConfig, LLMStrategyConfig, _get_raw_config
from hindsight_api.config_resolver import ConfigResolver

from .test_hierarchical_config import FakeBankConfigBackend, FakeBankConfigConnection, MockTenantExtension

BANK = "test-config-shape-bank"


class _BulkCapableBackend(FakeBankConfigBackend):
    """FakeBankConfigBackend plus the ``fetch`` the bulk path needs."""

    def acquire(self):
        return _BulkCapableConnection(self)

    def transaction(self):
        return _BulkCapableConnection(self)


class _BulkCapableConnection(FakeBankConfigConnection):
    async def fetch(self, query, bank_ids):
        return [{"bank_id": bank_id, "config": self.backend.config} for bank_id in bank_ids]


def _resolver(**kwargs) -> ConfigResolver:
    return ConfigResolver(backend=FakeBankConfigBackend(), **kwargs)


def test_hindsight_config_is_safely_shallow_copyable():
    """Pin the dataclass properties ``_with_overrides`` relies on.

    ``copy.copy`` is only interchangeable with a full reconstruction while the
    config has no ``__post_init__``, no ``init=False`` fields, no ``__slots__``,
    and no instance state outside its fields. Adding any of those to
    HindsightConfig would silently change what resolution returns, so fail here
    rather than in production.
    """
    config = _get_raw_config()

    assert not hasattr(config, "__post_init__"), "a __post_init__ would be skipped by copy.copy"
    assert not hasattr(HindsightConfig, "__slots__"), "__slots__ would break vars()-based copying"
    assert [f.name for f in dataclasses.fields(config) if not f.init] == [], (
        "an init=False field cannot be round-tripped through the constructor"
    )
    assert set(vars(config)) == {f.name for f in dataclasses.fields(config)}, (
        "non-field instance attributes would not survive a reconstruction"
    )
    assert copy.copy(config) == replace(config)


@pytest.mark.asyncio
async def test_resolved_config_matches_dict_roundtrip():
    """The copy-based resolution returns exactly what the asdict() one did.

    The reference below is the pre-#4209 construction: flatten the global config
    with asdict(), splice the overrides in, rebuild, then restore the member
    dataclasses asdict() had turned into plain dicts.
    """
    tenant = MockTenantExtension({"retain_extraction_mode": "tenant-mode", "retain_chunk_size": 5000})
    resolver = ConfigResolver(backend=FakeBankConfigBackend(), tenant_extension=tenant)
    global_config = resolver._global_config
    context = _context()

    for overrides in ({}, {"retain_chunk_size": 2000, "enable_observations": False}):
        resolver._backend.config = dict(overrides)

        resolved = await resolver.resolve_full_config(BANK, context, cached=False)

        merged = asdict(global_config)
        merged.update({"retain_extraction_mode": "tenant-mode", "retain_chunk_size": 5000})
        merged.update(overrides)
        expected = replace(
            HindsightConfig(**merged),
            reranker_members=global_config.reranker_members,
            llm_members=global_config.llm_members,
            llm_strategy=global_config.llm_strategy,
            retain_llm_members=global_config.retain_llm_members,
            retain_llm_strategy=global_config.retain_llm_strategy,
            reflect_llm_members=global_config.reflect_llm_members,
            reflect_llm_strategy=global_config.reflect_llm_strategy,
            consolidation_llm_members=global_config.consolidation_llm_members,
            consolidation_llm_strategy=global_config.consolidation_llm_strategy,
        )
        assert resolved == expected
        assert resolved is not global_config, "resolution must not hand out the process-global config"


@pytest.mark.asyncio
async def test_resolution_preserves_typed_member_dataclasses():
    """Multi-LLM members stay dataclasses, not the dicts asdict() flattened them into."""
    resolver = _resolver()
    member = LLMMemberConfig(
        provider="openai",
        api_key="k",
        model="gpt-4",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
    )
    strategy = LLMStrategyConfig(mode="failover")
    resolver._global_config = replace(resolver._global_config, llm_members=[member], llm_strategy=strategy)

    resolved = await resolver.resolve_full_config(BANK, cached=False)

    assert resolved.llm_members == [member]
    assert isinstance(resolved.llm_members[0], LLMMemberConfig)
    assert isinstance(resolved.llm_strategy, LLMStrategyConfig)


@pytest.mark.asyncio
async def test_resolution_does_not_mutate_the_global_config():
    """Overrides land on the copy, never on the shared global object."""
    resolver = _resolver()
    before = replace(resolver._global_config)
    resolver._backend.config = {"retain_chunk_size": before.retain_chunk_size + 111}

    resolved = await resolver.resolve_full_config(BANK, cached=False)

    assert resolved.retain_chunk_size == before.retain_chunk_size + 111
    assert resolver._global_config == before


@pytest.mark.asyncio
async def test_bank_config_response_containers_are_detached():
    """A caller editing a returned container must not reach the global config.

    Config reads used to go through asdict(), which deep-copied every value.
    Reading fields off the object is far cheaper but would otherwise hand out the
    process-global config's own dicts and lists.
    """
    resolver = _resolver()
    resolver._global_config = replace(resolver._global_config, retain_strategies={"fast": {"retain_chunk_size": 900}})

    config = await resolver.get_bank_config(BANK, cached=False)
    config["retain_strategies"]["fast"]["retain_chunk_size"] = 1
    config["retain_strategies"]["injected"] = {}

    assert resolver._global_config.retain_strategies == {"fast": {"retain_chunk_size": 900}}
    second = await resolver.get_bank_config(BANK, cached=False)
    assert second["retain_strategies"] == {"fast": {"retain_chunk_size": 900}}


@pytest.mark.asyncio
async def test_bank_config_excludes_static_and_credential_fields():
    """The public projection is exactly configurable-minus-credential fields."""
    resolver = _resolver()

    config = await resolver.get_bank_config(BANK, cached=False)

    configurable = HindsightConfig.get_configurable_fields()
    credentials = HindsightConfig.get_credential_fields()
    assert set(config) == configurable - credentials
    assert not set(config) & credentials
    assert "database_url" not in config


@pytest.mark.asyncio
async def test_bulk_and_single_bank_config_agree():
    """get_bank_configs is the batched form of get_bank_config, including tenant overrides."""
    tenant = MockTenantExtension({"retain_chunk_size": 5000})
    resolver = ConfigResolver(backend=_BulkCapableBackend(), tenant_extension=tenant)
    resolver._backend.config = {"enable_observations": False}
    context = _context()

    single = await resolver.get_bank_config(BANK, context, cached=False)
    bulk = await resolver.get_bank_configs([BANK], context)

    assert bulk[BANK] == single
    assert single["retain_chunk_size"] == 5000
    assert single["enable_observations"] is False


def _context():
    from hindsight_api.models import RequestContext

    return RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
