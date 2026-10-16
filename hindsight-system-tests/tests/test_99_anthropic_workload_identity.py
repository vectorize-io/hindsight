"""A server with no Anthropic key authenticates through Workload Identity Federation.

The deployment in #5502 drops the static ``sk-ant-...`` key: the pod carries a
platform identity token, and the Anthropic SDK trades it at ``/v1/oauth/token`` for
a short-lived access token. The promise is that every model call the server makes
goes out on that exchanged token — none on a key, and none unauthenticated — so the
story retains through it and checks, at the provider, what each call carried.

The names are unusual on purpose: the stub is shared with every other server in
the session, and a straggling call from another story must not be mistaken for one
of these.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

from .conftest import WIF_FEDERATION_RULE_ID, WIF_IDENTITY_TOKEN

pytestmark = pytest.mark.asyncio

CONTENT = "Quillon Ashgrove keeps bees on a rooftop in Ljubljana."


async def test_every_model_call_carries_the_exchanged_token(wif_client, stubs, llm, bank_id):
    llm.on_step("extract_facts", contains="Quillon").returns(
        extracted(
            fact(
                "Quillon Ashgrove keeps bees on a rooftop in Ljubljana",
                where="Ljubljana",
                who="Quillon Ashgrove",
                entities=["Quillon Ashgrove", "Ljubljana"],
            )
        )
    )
    # The retain enqueues consolidation; another server's worker may claim it, so it
    # is declared rather than left to whichever provider happens to run it.
    llm.on_step("consolidate").returns(consolidation())

    await wif_client.aretain(bank_id=bank_id, content=CONTENT)
    await wait_until_settled(wif_client, bank_id)

    response = await wif_client.arecall(bank_id=bank_id, query="Who keeps bees in Ljubljana?")
    assert [result.text for result in response.results] == [
        "Quillon Ashgrove keeps bees on a rooftop in Ljubljana | Involving: Quillon Ashgrove"
    ]

    # The server presented its identity token, with the federation it was configured for.
    exchanges = [e for e in stubs.token_exchanges if e.federation_rule_id == WIF_FEDERATION_RULE_ID]
    assert exchanges, "the server never exchanged its identity token"
    for exchange in exchanges:
        assert exchange.assertion == WIF_IDENTITY_TOKEN
        assert exchange.organization_id == "org_systest"
        assert exchange.service_account_id == "svac_systest"

    # And the extraction went out on a token the exchange issued — as a bearer, with no
    # API key header beside it.
    issued = {exchange.access_token for exchange in exchanges}
    # `anthropic-version` marks a call that came in through /v1/messages, i.e. from this server.
    extraction_calls = [call for call in llm.calls if "Quillon" in call.all_text and "anthropic-version" in call.headers]
    assert len(extraction_calls) == 1
    (call,) = extraction_calls
    assert call.api_key in issued
    assert call.headers["authorization"] == f"Bearer {call.api_key}"
    assert "x-api-key" not in call.headers
