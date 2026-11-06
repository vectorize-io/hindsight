"""A server runs on an LLM provider the operator wrote, named by its import path.

An operator whose backend no built-in provider speaks — or speaks almost right, like
the OpenAI-compatible batch hosts in #6058 — sets ``HINDSIGHT_API_LLM_PROVIDER`` to
``package.module:ClassName`` instead of patching Hindsight. The promise is that the
server builds that class and every model call goes through it: here the class is the
OpenAI provider plus a header naming it (``server_extensions/marked_llm.py``), so the
story retains through it and checks, at the provider, that the extraction carried
the header.

The names are unusual on purpose: the stub is shared with every other server in
the session, and a straggling call from another story must not be mistaken for one
of these.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

CONTENT = "Thaddeus Wrenfield repairs harpsichords in Tallinn."


async def test_every_model_call_goes_through_the_custom_provider(custom_llm_client, llm, bank_id):
    llm.on_step("extract_facts", contains="Thaddeus").returns(
        extracted(
            fact(
                "Thaddeus Wrenfield repairs harpsichords in Tallinn",
                where="Tallinn",
                who="Thaddeus Wrenfield",
                entities=["Thaddeus Wrenfield", "Tallinn"],
            )
        )
    )
    # The retain enqueues consolidation; another server's worker may claim it, so it
    # is declared rather than left to whichever provider happens to run it.
    llm.on_step("consolidate").returns(consolidation())

    await custom_llm_client.aretain(bank_id=bank_id, content=CONTENT)
    await wait_until_settled(custom_llm_client, bank_id)

    response = await custom_llm_client.arecall(bank_id=bank_id, query="Who repairs harpsichords in Tallinn?")
    assert [result.text for result in response.results] == [
        "Thaddeus Wrenfield repairs harpsichords in Tallinn | Where: Tallinn | Involving: Thaddeus Wrenfield"
    ]

    # Extraction only: consolidation may be run by another server's worker, on its own provider.
    extraction_calls = [
        call for call in llm.calls if "Thaddeus" in call.all_text and anchor_for("extract_facts") in call.all_text
    ]
    assert len(extraction_calls) == 1
    (call,) = extraction_calls
    assert call.headers["x-marked-llm"] == "MarkedOpenAILLM"
    assert call.api_key == "stub-key"
