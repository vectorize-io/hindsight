"""The bank-llm-routing extension, driven through its base-class interface.

What a deployment relies on: the documented env value loads it, a bank gets the
key of the first route matching it, a bank no route matches is refused (or sent
on the server key when the operator says so), and a broken config fails startup
without printing the keys it holds. That the chosen key reaches the provider on
every operation is the system-test story `test_98_bank_llm_routing.py`.
"""

import json
import os
from unittest.mock import patch

import pytest
from hindsight_api.extensions.llm_request import (
    LLMRequestExtension,
    LLMRequestRefused,
    LLMRequestTarget,
)
from hindsight_api.extensions.loader import load_extension

from hindsight_ext_bank_llm_routing import BankLLMRoutingExtension


def _ext(routes: list[dict], **config: str) -> BankLLMRoutingExtension:
    return BankLLMRoutingExtension({"routes": json.dumps(routes), **config})


def _target(bank_id: str | None, schema: str = "public", operation: str = "retain") -> LLMRequestTarget:
    return LLMRequestTarget(schema=schema, bank_id=bank_id, operation=operation, provider="openai", model="m")


def test_documented_env_value_loads_the_extension():
    env = {
        "HINDSIGHT_API_LLM_REQUEST_EXTENSION": "hindsight_ext_bank_llm_routing:BankLLMRoutingExtension",
        "HINDSIGHT_API_LLM_REQUEST_ROUTES": '[{"bank": "acme-*", "api_key": "sk-acme"}]',
        "HINDSIGHT_API_LLM_REQUEST_UNMATCHED": "server",
    }
    with patch.dict(os.environ, env, clear=False):
        extension = load_extension("LLM_REQUEST", LLMRequestExtension)

    assert isinstance(extension, BankLLMRoutingExtension)
    assert extension.unmatched == "server"


async def test_first_matching_route_wins():
    ext = _ext(
        [
            {"bank": "acme-admin", "api_key": "sk-admin"},
            {"bank": "acme-*", "api_key": "sk-acme", "headers": {"x-team": "acme"}},
            {"bank": "*", "api_key": "sk-everyone"},
        ]
    )
    assert (await ext.resolve(_target("acme-admin"))).api_key == "sk-admin"
    acme = await ext.resolve(_target("acme-42"))
    assert acme.api_key == "sk-acme"
    assert dict(acme.headers) == {"x-team": "acme"}
    assert (await ext.resolve(_target("globex-1"))).api_key == "sk-everyone"


async def test_patterns_are_case_sensitive():
    ext = _ext([{"bank": "acme-*", "api_key": "sk-acme"}], unmatched="server")
    assert await ext.resolve(_target("ACME-1")) is None


async def test_unmatched_bank_is_refused_by_default():
    ext = _ext([{"bank": "acme-*", "api_key": "sk-acme"}])
    with pytest.raises(LLMRequestRefused, match="No LLM route matches bank 'globex-1'"):
        await ext.resolve(_target("globex-1"))


async def test_a_call_outside_any_bank_never_matches_a_route():
    ext = _ext([{"bank": "*", "api_key": "sk-all"}])
    with pytest.raises(LLMRequestRefused):
        await ext.resolve(_target(None))


async def test_unmatched_server_falls_back_to_the_server_key():
    ext = _ext([{"bank": "acme-*", "api_key": "sk-acme"}], unmatched="server")
    assert await ext.resolve(_target("globex-1")) is None


async def test_schema_scoped_route_ignores_the_same_bank_id_in_another_tenant():
    """Bank ids are unique per tenant schema only: tenant B naming a bank ``acme-1``
    must not be billed to tenant A's key."""
    ext = _ext([{"bank": "acme-*", "schema": "tenant_a", "api_key": "sk-a"}])
    assert (await ext.resolve(_target("acme-1", schema="tenant_a"))).api_key == "sk-a"
    with pytest.raises(LLMRequestRefused):
        await ext.resolve(_target("acme-1", schema="tenant_b"))


@pytest.mark.parametrize(
    ("routes", "message"),
    [
        (None, "is required"),
        ("not json sk-leak", "not valid JSON"),
        ("[]", "non-empty JSON list"),
        ('[{"bank": "a-*", "api_ky": "sk-leak"}]', "api_ky"),
        ('[{"bank": "a-*"}]', "none of llm, api_key, headers"),
        ('[{"bank": "a-*", "llm": {"provider": "openai", "api_key": "sk-leak"}}]', "llm.model"),
        ('[{"bank": "a-*", "llm": {"provider": "openai", "model": "m", "key": "sk-leak"}}]', "llm.key"),
        ('[{"bank": "", "api_key": "sk-leak"}]', "bank"),
        ('[{"bank": "a-*", "api_key": "sk-leak", "headers": {"x": "a\\r\\nb: sk-leak"}}]', "control character"),
    ],
)
def test_bad_config_fails_startup_without_echoing_keys(routes, message):
    config = {} if routes is None else {"routes": routes}
    with pytest.raises(ValueError, match=message) as excinfo:
        BankLLMRoutingExtension(config)
    assert "sk-leak" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


async def test_a_route_can_send_a_bank_to_another_llm():
    ext = _ext(
        [
            {
                "bank": "acme-*",
                "llm": {"provider": "anthropic", "model": "claude-sonnet-5-5", "api_key": "sk-ant"},
                "headers": {"x-team": "acme"},
            }
        ]
    )
    overrides = await ext.resolve(_target("acme-1"))
    assert overrides.llm is not None
    assert (overrides.llm.provider, overrides.llm.model, overrides.llm.api_key) == (
        "anthropic",
        "claude-sonnet-5-5",
        "sk-ant",
    )
    assert overrides.llm.base_url is None
    assert dict(overrides.headers) == {"x-team": "acme"}


async def test_routes_can_pick_the_llm_per_operation():
    """A cheap model for retain, a strong one for reflect, on the same bank."""
    ext = _ext(
        [
            {"bank": "acme-*", "operation": "reflect", "llm": {"provider": "openai", "model": "strong"}},
            {"bank": "acme-*", "operation": "consolidation*", "llm": {"provider": "openai", "model": "mid"}},
            {"bank": "acme-*", "llm": {"provider": "openai", "model": "cheap"}},
        ]
    )
    models = {
        op: (await ext.resolve(_target("acme-1", operation=op))).llm.model
        for op in ("reflect", "consolidation_dedup", "retain")
    }
    assert models == {"reflect": "strong", "consolidation_dedup": "mid", "retain": "cheap"}


def test_startup_log_names_the_llm_but_not_its_key(caplog):
    with caplog.at_level("INFO"):
        _ext(
            [
                {
                    "bank": "acme-*",
                    "operation": "reflect",
                    "llm": {"provider": "openai", "model": "gpt-x", "api_key": "sk-secret"},
                }
            ]
        )
    assert "acme-*[reflect] -> openai/gpt-x" in caplog.text
    assert "sk-secret" not in caplog.text


def test_unknown_unmatched_policy_fails_startup():
    with pytest.raises(ValueError, match="UNMATCHED"):
        _ext([{"bank": "*", "api_key": "sk"}], unmatched="allow")


def test_startup_log_names_patterns_not_keys(caplog):
    with caplog.at_level("INFO"):
        _ext([{"bank": "acme-*", "api_key": "sk-secret"}])
    assert "acme-*" in caplog.text
    assert "sk-secret" not in caplog.text


async def test_a_route_can_be_pinned_to_the_servers_provider_and_model():
    # A multi-LLM chain asks once per member: a gateway key must not reach the fallback provider.
    extension = _ext(
        [{"bank": "acme-*", "provider": "openai", "model": "gpt-*", "api_key": "sk-gateway"}],
        unmatched="server",
    )

    def target(provider: str, model: str) -> LLMRequestTarget:
        return LLMRequestTarget(schema="public", bank_id="acme-1", operation="retain", provider=provider, model=model)

    assert (await extension.resolve(target("openai", "gpt-4o"))).api_key == "sk-gateway"
    assert await extension.resolve(target("anthropic", "gpt-4o")) is None
    assert await extension.resolve(target("openai", "o3")) is None


def test_routes_do_not_print_their_keys():
    extension = _ext(
        [{"bank": "a-*", "api_key": "sk-top", "headers": {"x-key": "sk-head"}, "llm": {"provider": "openai", "model": "m", "api_key": "sk-llm"}}]
    )
    printed = repr(extension.routes) + str(extension.routes)
    assert "sk-top" not in printed and "sk-head" not in printed and "sk-llm" not in printed
