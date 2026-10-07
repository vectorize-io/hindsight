"""Applying the same template again changes nothing and costs nothing.

Templates are applied the way infrastructure is: a GitOps sync or a
re-provisioning job posts the same manifest on every run. Each mental model in
it is a standing answer the worker writes with a full reflect loop, so an import
that treated "already there" as "changed" regenerated every model on every
apply — real LLM spend for an identical document (#5271).

The seam is between import and refresh: the import decides what to queue, the
worker spends the tokens. Only a story that lets the worker run can see the bill.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.api.bank_templates_api import BankTemplatesApi
from hindsight_client_api.models.bank_template_manifest import BankTemplateManifest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

ANSWER = "Alice lives in Berlin."


def _manifest(source_query: str = "Where does Alice live?") -> BankTemplateManifest:
    return BankTemplateManifest.from_dict(
        {
            "version": "1",
            "mental_models": [
                {
                    "id": "alice-housing",
                    "name": "Alice housing",
                    "source_query": source_query,
                    "trigger": {"mode": "delta"},
                }
            ],
            "directives": [{"name": "Precise", "content": "Answer precisely.", "priority": 2, "tags": ["style"]}],
        }
    )


async def test_reapplying_an_unchanged_template_refreshes_nothing(client, llm, bank_id, settled):
    templates = BankTemplatesApi(client.banks.api_client)
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    first = await templates.import_bank_template(bank_id, _manifest())
    assert first.mental_models_created == ["alice-housing"]
    assert first.directives_created == ["Precise"]
    await settled(bank_id)
    calls_after_first = len(llm.calls)

    again = await templates.import_bank_template(bank_id, _manifest())
    await settled(bank_id)

    assert again.mental_models_created == []
    assert again.mental_models_updated == []
    assert again.directives_created == []
    assert again.directives_updated == []
    assert again.operation_ids == []
    assert len(llm.calls) == calls_after_first, "an unchanged model was regenerated"
    model = await client.mental_models.get_mental_model(bank_id, "alice-housing", detail="full")
    assert model.content.strip() == ANSWER

    # The skip is a comparison, not a blanket no-op: a changed definition still lands and refreshes.
    changed = await templates.import_bank_template(bank_id, _manifest("Where does Alice live now?"))
    await settled(bank_id)

    assert changed.mental_models_updated == ["alice-housing"]
    assert changed.directives_updated == []
    assert len(changed.operation_ids) == 1
    assert len(llm.calls) > calls_after_first
    model = await client.mental_models.get_mental_model(bank_id, "alice-housing", detail="full")
    assert model.source_query == "Where does Alice live now?"
