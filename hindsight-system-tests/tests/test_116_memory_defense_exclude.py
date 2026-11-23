"""A bank can switch off one Memory Defense pattern and keep the rest.

`credit_card` also matches numeric IDs that happen to pass the card checksum —
ad-account IDs, video IDs — and redacting them removes exactly what the memory
is searched by (#6455). A `sensitive_data` rule takes an `exclude` list of
pattern labels: those are skipped, every other pattern still redacts. This story
sets the policy through bank config and checks what a retain actually stored.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

# 1202123456789019 passes the Luhn check, so the default policy redacts it.
LUHN_ID = "1202123456789019"
TOKEN = "ghp_" + "A" * 36
CONTENT = f"Ad account {LUHN_ID} was linked using token {TOKEN}."
DOCUMENT_ID = "ad-account-note"


def _policy(**rule: object) -> dict:
    return {"updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", **rule}]}}}


async def test_excluded_pattern_is_kept_while_others_still_redact(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(extracted(fact("An ad account was linked", entities=["ad account"])))
    llm.on_step("consolidate").returns(consolidation())

    await client.banks.update_bank_config(bank_id, _policy(action="redact", exclude=["credit_card"]))
    await client.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID)
    await settled(bank_id)

    document = await client.documents.get_document(bank_id, DOCUMENT_ID)
    assert document.original_text == f"Ad account {LUHN_ID} was linked using token [REDACTED:github_token]."


async def test_an_unknown_pattern_name_is_rejected(client, bank_id):
    # A typo must not leave the bank believing a pattern is off when it isn't.
    with pytest.raises(ApiException) as error:
        await client.banks.update_bank_config(bank_id, _policy(action="redact", exclude=["credit_cards"]))
    assert error.value.status == 422
