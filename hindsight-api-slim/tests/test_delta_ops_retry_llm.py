"""A real model, told why its operations were refused, sends a list that parses (#4443).

``request_delta_operations`` answers a reply that misses the schema by asking once
more with the refused operations quoted back. Whether that works is a property of
the model reading the correction, which the scripted doubles in
``test_delta_operation_parse.py`` cannot show — they only prove the correction is
*sent*. So here the first reply is pinned to the exact shape #4443 reported (a
hallucinated ``block_id`` on ``append_block``) and the second comes from the real
provider, answering the real correction on top of the real delta prompt.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.llm_wrapper import ConfiguredLLMProvider
from hindsight_api.engine.reflect.delta_ops import DeltaOperationList, request_delta_operations
from hindsight_api.engine.reflect.prompts import STRUCTURED_DELTA_SYSTEM_PROMPT, build_structured_delta_prompt
from hindsight_api.engine.reflect.structured_doc import Block, Section, StructuredDocument
from hindsight_api.engine.response_models import LLMCallResult
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

#: The first reply, as #4443 reported it: right content, one invented key.
_REFUSED_REPLY = json.dumps(
    {"operations": [{"op": "append_block", "section_id": "members", "block_id": None, "text": "- Carol — SRE"}]}
)


class _FirstReplyPinned:
    """Returns the refused reply first, then hands every later call to the real model."""

    def __init__(self, real: ConfiguredLLMProvider) -> None:
        self._real = real
        self.calls = 0

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMCallResult:
        self.calls += 1
        if self.calls == 1:
            return LLMCallResult(content=_REFUSED_REPLY)
        return await self._real.call(messages, **kwargs)


async def test_the_model_repairs_a_refused_reply_from_the_correction():
    document = StructuredDocument(
        sections=[
            Section(
                id="members",
                heading="Members",
                level=2,
                blocks=[Block(id="b1a2c3d4", text="- Alice — team lead\n- Bob — backend")],
            )
        ]
    )
    user_prompt = build_structured_delta_prompt(
        current_document_json=document.model_dump_json(),
        candidate_markdown="# Team\n\nCarol joined the team as an SRE.",
        supporting_facts=[{"id": "f1", "text": "Carol joined the team as an SRE.", "type": "world", "context": None}],
        source_query="Who is on the team?",
    )
    llm = _FirstReplyPinned(LLMConfig.from_env().with_config(_get_raw_config()))

    op_list = await request_delta_operations(
        llm,  # type: ignore[arg-type]  # a pinning shim over the real provider
        system_prompt=STRUCTURED_DELTA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        scope="test_delta_retry",
        response_format=DeltaOperationList,
        skip_validation=True,
    )

    # Structural and deterministic: the helper returning at all means the
    # second reply parsed under the strict schema, and it took exactly one retry.
    assert llm.calls == 2
    assert op_list.operations, "the corrected reply dropped the edit instead of fixing it"

    # Whether the fix kept the *meaning* is the model's call.
    summary = "\n".join(
        f"- {op.op} section={getattr(op, 'section_id', '?')} text={getattr(op, 'text', getattr(op, 'blocks', ''))!r}"
        for op in op_list.operations
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "The operations add Carol, an SRE, to the 'members' section, and do not remove or overwrite Alice or Bob."
        ),
        context=(
            "A team page's Members section lists Alice (team lead) and Bob (backend). A new fact "
            "says Carol joined as an SRE. The model's first attempt was refused for an invalid "
            "field and it was asked to send the operations again."
        ),
    )
