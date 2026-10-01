"""Bank-local language enforcement, exercised without a database or live provider."""

import dataclasses
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import HindsightConfig, _get_raw_config, get_config
from hindsight_api.config_resolver import ConfigResolver, _coerce_stored_bank_overrides, apply_strategy
from hindsight_api.engine.language_integrity import GeneratedLanguageMismatch
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage
from hindsight_api.engine.retain.fact_extraction import ExtractionPrompt, _extract_facts_from_chunk
from hindsight_api.models import RequestContext


class BankBackend:
    def __init__(self):
        self.configs = {}

    def acquire(self):
        return self

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def fetchrow(self, query, bank_id):
        return {"config": self.configs.get(bank_id, {})}

    async def execute(self, query, updates_json, bank_id):
        self.configs.setdefault(bank_id, {}).update(json.loads(updates_json))
        return "UPDATE 1"


class Tenant:
    def __init__(self, mode):
        self.mode = mode

    async def get_tenant_config(self, context):
        return {"llm_language_integrity": self.mode}


def resolver(mode="retry", tenant=None):
    resolved = ConfigResolver(BankBackend(), tenant_extension=tenant)
    resolved._global_config = dataclasses.replace(_get_raw_config(), llm_language_integrity=mode)
    return resolved


def test_mode_is_hierarchical_but_output_language_stays_static():
    assert "llm_language_integrity" in HindsightConfig.get_configurable_fields()
    assert "llm_output_language" in HindsightConfig.get_static_fields()
    with pytest.raises(AttributeError):
        _ = get_config().llm_language_integrity


@pytest.mark.asyncio
async def test_bank_update_and_null_reset_preserve_other_banks():
    config = resolver()
    await config.update_bank_config("synthetic-enforced", {"llm_language_integrity": "reject"})
    assert (await config.get_bank_config("synthetic-enforced"))["llm_language_integrity"] == "reject"
    assert (await config.resolve_full_config("synthetic-other")).llm_language_integrity == "retry"
    await config.update_bank_config("synthetic-enforced", {"llm_language_integrity": None})
    assert (await config.resolve_full_config("synthetic-enforced")).llm_language_integrity == "retry"
    assert config._global_config.llm_language_integrity == "retry"


@pytest.mark.asyncio
async def test_bank_null_inherits_tenant_then_global():
    tenant = Tenant("observe")
    config = resolver(tenant=tenant)
    context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    await config.update_bank_config("synthetic", {"llm_language_integrity": "reject"})
    assert (await config.resolve_full_config("synthetic", context, cached=False)).llm_language_integrity == "reject"
    await config.update_bank_config("synthetic", {"llm_language_integrity": None})
    assert (await config.resolve_full_config("synthetic", context, cached=False)).llm_language_integrity == "observe"
    tenant.mode = None
    assert (await config.resolve_full_config("synthetic", context, cached=False)).llm_language_integrity == "retry"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "observe", "retry", "reject"])
async def test_all_declared_modes_are_accepted(mode):
    config = resolver()
    await config.update_bank_config("synthetic", {"HINDSIGHT_API_LLM_LANGUAGE_INTEGRITY": mode})
    assert (await config.resolve_full_config("synthetic")).llm_language_integrity == mode


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid", "REJECT", "", 1, True, {}, []])
@pytest.mark.parametrize("nested", [False, True])
async def test_invalid_modes_are_rejected_before_persistence(mode, nested):
    config = resolver()
    updates = {"llm_language_integrity": mode}
    if nested:
        updates = {"retain_strategies": {"session": updates}}
    with pytest.raises(ValueError, match="llm_language_integrity"):
        await config.update_bank_config("synthetic", updates)
    assert config._backend.configs == {}


@pytest.mark.parametrize("mode", ["invalid", {}, True])
def test_malformed_stored_mode_is_ignored_including_strategies(mode):
    config = _coerce_stored_bank_overrides(
        "synthetic",
        {"llm_language_integrity": mode, "retain_strategies": {"session": {"llm_language_integrity": mode}}},
    )
    assert "llm_language_integrity" not in config
    assert config["retain_strategies"]["session"] == {}


@pytest.mark.asyncio
async def test_invalid_tenant_mode_inherits_global():
    config = resolver(tenant=Tenant("invalid"))
    context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    assert (await config.resolve_full_config("synthetic", context, cached=False)).llm_language_integrity == "retry"


def test_strategy_null_inherits_enforcement_without_changing_other_nulls():
    config = dataclasses.replace(
        _get_raw_config(),
        llm_language_integrity="reject",
        retain_strategies={"session": {"llm_language_integrity": None, "retain_structured_chunk_size": None}},
    )
    resolved = apply_strategy(config, "session")
    assert resolved.llm_language_integrity == "reject"
    assert resolved.retain_structured_chunk_size is None


@pytest.mark.asyncio
async def test_resolved_reject_fails_after_retry_while_other_bank_accepts():
    config = resolver()
    await config.update_bank_config("synthetic-enforced", {"llm_language_integrity": "reject"})
    engine = SimpleNamespace(_config_resolver=config)
    context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    source = "The operations team completed the important review findings and the low-cost hardening work through regression tests, then ran the focused and canonical validation suites successfully."
    drift = "El equipo de operaciones completó los hallazgos importantes de la revisión y el trabajo de endurecimiento mediante pruebas de regresión, y luego ejecutó correctamente las validaciones canónicas."
    for bank, rejects in [("synthetic-enforced", True), ("synthetic-other", False)]:
        effective = dataclasses.replace(
            await MemoryEngine._resolve_retain_config(engine, bank, context, None),
            llm_output_language=None,
            retain_llm_max_retries=0,
            llm_max_retries=0,
            retain_max_completion_tokens=8192,
            retain_extraction_mode="concise",
            retain_extract_causal_links=False,
            retain_mission=None,
            llm_strict_schema_retain=False,
        )
        llm = MagicMock(spec=LLMProvider)
        llm.provider, llm.model = "mock", "mock-model"
        llm.call = AsyncMock(
            side_effect=[
                LLMCallResult(
                    content={"facts": [{"what": drift, "fact_type": "world", "fact_kind": "conversation"}]},
                    usage=TokenUsage(),
                )
                for _ in range(2)
            ]
        )
        with patch(
            "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
            return_value=ExtractionPrompt(system_prompt="system", response_schema=MagicMock()),
        ):
            call = _extract_facts_from_chunk(
                chunk=source,
                chunk_index=0,
                total_chunks=1,
                event_date=datetime(2026, 9, 4, tzinfo=timezone.utc),
                context="",
                llm_config=llm,
                config=effective,
            )
            if rejects:
                with pytest.raises(GeneratedLanguageMismatch):
                    await call
            else:
                facts, _ = await call
                assert facts[0].fact == drift
        assert llm.call.await_count == 2


@pytest.mark.asyncio
async def test_consolidation_boundary_passes_each_banks_resolved_mode():
    from hindsight_api.engine.consolidation import consolidator

    config = resolver()
    await config.update_bank_config("synthetic-consolidation-enforced", {"llm_language_integrity": "reject"})
    wrapper = MagicMock()
    engine = SimpleNamespace(_config_resolver=config, _consolidation_llm_config=wrapper)
    context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    with (
        patch.object(consolidator, "trace_context_of", return_value=None),
        patch.object(consolidator, "_run_consolidation_job", new_callable=AsyncMock) as run,
    ):
        for bank, expected in [
            ("synthetic-consolidation-enforced", "reject"),
            ("synthetic-consolidation-other", "retry"),
        ]:
            await consolidator.run_consolidation_job(engine, bank, context)
            passed_config = run.await_args.args[3]
            assert passed_config.llm_language_integrity == expected
            assert wrapper.with_config.call_args.args[0] is passed_config
            assert wrapper.with_config.call_args.kwargs == {"bank_id": bank, "operation": "consolidation"}
