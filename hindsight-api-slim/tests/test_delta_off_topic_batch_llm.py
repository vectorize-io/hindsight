"""A delta refresh handed only off-topic facts leaves the page alone (#6352).

The delta prompt used to allow ``{"operations": []}`` "only if the new facts are
already reflected in the document". A batch of new facts that are simply off-topic
for the page is neither: they are not in the document, and nothing in it should
change. A model that takes that rule literally has no legal answer, and the issue
reports models answering in prose instead, which fails the JSON parse and wedges the
refresh. The rule now names off-topic and no-op batches as an expected empty answer.

The eval behind it is test_01 in hindsight-system-evals, where every page sees the
whole corpus in waves and most of each wave is off-topic for it.
"""

from __future__ import annotations

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.reflect.delta_ops import DeltaOperationList, apply_operations, request_delta_operations
from hindsight_api.engine.reflect.prompts import STRUCTURED_DELTA_SYSTEM_PROMPT, build_structured_delta_prompt
from hindsight_api.engine.reflect.structured_doc import Block, Section, StructuredDocument, render_document
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_OFF_TOPIC = [
    "Fixed a CSS margin on the settings page in release v1.4.2.",
    "The team lunch moved from Thursday to Friday.",
    "Bumped the eslint dev dependency to the latest minor version.",
    "The staging database was restarted for routine OS patching.",
]


async def test_an_off_topic_batch_changes_nothing():
    document = StructuredDocument(
        sections=[
            Section(
                id="principles",
                heading="API design principles",
                level=2,
                blocks=[
                    Block(id="b1", text="Every list endpoint paginates with limit and offset and returns a total."),
                    Block(id="b2", text="Handlers never touch the database; all queries live in the engine layer."),
                ],
            )
        ]
    )
    user_prompt = build_structured_delta_prompt(
        current_document_json=document.model_dump_json(),
        supporting_facts=[{"id": f"f{i}", "text": text, "type": "world"} for i, text in enumerate(_OFF_TOPIC)],
        source_query="What are our API design principles?",
    )
    llm = LLMConfig.from_env().with_config(_get_raw_config())

    # Raises if the model answers in prose instead of the ops JSON: the failure #6352 reports.
    op_list = await request_delta_operations(
        llm,
        system_prompt=STRUCTURED_DELTA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        scope="test_delta_off_topic_batch",
        response_format=DeltaOperationList,
        skip_validation=True,
        document=document,
    )
    page = render_document(apply_operations(document, op_list.operations).document)

    await assert_meets_criteria(
        response=page,
        criteria=(
            "The page still states both API design principles (paginated list endpoints with a total; no "
            "database access in handlers) and contains nothing about CSS fixes, team lunches, eslint or "
            "database restarts."
        ),
        context=(
            "A page of API design principles was refreshed with a batch of facts that are all off-topic for it: "
            "a CSS fix, a lunch move, a dev-dependency bump and a routine staging restart."
        ),
    )
