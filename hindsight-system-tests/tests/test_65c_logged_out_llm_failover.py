"""A failover chain whose primary has lost its sign-in still starts and still serves.

A ChatGPT plan login can be revoked under a running server: a token refresh is
rejected with ``invalid_grant`` and the stored session is cleared. The server
kept working, failing over on every call, until it restarted, and then it never
came back up: building the logged-out primary raised, startup died, and the
container restarted forever (#6095). The member after it was never tried, and
recall, which needs no model at all, went down with it.

The server here starts with exactly that: an empty ChatGPT credentials directory
as member 0, and the stub as member 1. If it starts at all, half the story holds;
the retain and recall show the rest of the chain serves.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CONTENT = "Odalys Brenwick repairs harpsichords in a workshop in Trieste."


async def test_a_logged_out_primary_fails_over_to_the_next_member(logged_out_primary_client, llm, bank_id):
    llm.on_step("extract_facts", contains="Odalys").returns(
        extracted(
            fact(
                "Odalys Brenwick repairs harpsichords in Trieste",
                where="Trieste",
                who="Odalys Brenwick",
                entities=["Odalys Brenwick", "Trieste"],
            )
        )
    )
    # The retain enqueues consolidation; another server's worker may claim it.
    llm.on_step("consolidate").returns(consolidation())

    await logged_out_primary_client.aretain(bank_id=bank_id, content=CONTENT)
    await wait_until_settled(logged_out_primary_client, bank_id)

    # The logged-out primary cannot reach any provider, so this fact can only have been
    # extracted by member 1, the stub.
    response = await logged_out_primary_client.arecall(bank_id=bank_id, query="Who repairs harpsichords?")
    assert [result.text for result in response.results] == [
        "Odalys Brenwick repairs harpsichords in Trieste | Where: Trieste | Involving: Odalys Brenwick"
    ]
