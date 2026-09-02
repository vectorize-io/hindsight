"""The extractor attributes a fact to the picture it actually read.

The deterministic half — numbers to ids — lives in `test_attachment_attribution`.
This is the other half, and it can only be asked of a real model: given a chunk
that is mostly prose with one image in it, does the model mark the facts it could
only have got from the image, and leave the rest unmarked?

Both directions matter and fail differently. Marking nothing makes the feature
useless (no memory can show its evidence); marking everything is worse than
nothing, because a screenshot shown beside a fact it does not support is a
confident wrong citation.
"""

import io
from datetime import datetime

import pytest
from PIL import Image, ImageDraw

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.attachment_content import (
    LoadedAttachment,
    attachment_placeholder,
    compute_attachment_hash,
    short_attachment_id,
)
from hindsight_api.engine.retain.fact_extraction import _attachment_ids_for, extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


def _diagram() -> bytes:
    """A picture carrying one fact that appears nowhere in the prose."""
    image = Image.new("RGB", (640, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 60, 300, 140), outline="black", width=3)
    draw.text((40, 95), "ESCALATE TO: Tier 3 Platform", fill="black")
    draw.text((340, 95), "RESPONSE TARGET: 15 minutes", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _StubLoader:
    """Serves the bytes straight from memory; the store is not what is under test."""

    def __init__(self, attachments: dict[str, LoadedAttachment]) -> None:
        self._attachments = attachments

    async def load(self, attachment_ids) -> dict[str, LoadedAttachment]:
        return {i: self._attachments[i] for i in attachment_ids if i in self._attachments}


@pytest.mark.asyncio
async def test_only_the_facts_read_off_the_image_carry_it():
    data = _diagram()
    attachment_id = short_attachment_id(compute_attachment_hash(data))
    text = (
        "A sync is considered stuck if it has not advanced in thirty minutes.\n\n"
        "Before escalating, the sync ID must be recorded in the incident channel.\n\n"
        "The escalation path is shown below:\n\n"
        f"{attachment_placeholder(attachment_id)}\n\n"
        "Engineers must never be paged directly; follow the path above."
    )
    loader = _StubLoader({attachment_id: LoadedAttachment(media_type="image/png", data=data)})

    facts, _, _ = await extract_facts_from_text(
        text=text,
        event_date=datetime(2026, 1, 1),
        llm_config=LLMConfig.from_env(),
        agent_name=None,
        config=_get_raw_config(),
        attachment_loader=loader,
    )

    assert facts, "extraction produced no facts"
    # `extract_facts_from_text` stops at the model's numbers; resolving them
    # against the chunk is what the retain pipeline does next, so do it here too.
    ids = {id(fact): _attachment_ids_for(fact, text) for fact in facts}
    attributed = [f for f in facts if ids[id(f)]]
    unattributed = [f for f in facts if not ids[id(f)]]

    # Structural, and both directions: the whole point is that this is a
    # partition of the facts, not a flag set on all of them.
    assert attributed, "no fact was attributed to the image, so no memory can show its evidence"
    assert unattributed, "every fact claims the image, which cites it for prose it does not support"
    assert all(ids[id(f)] == [attachment_id] for f in attributed)

    # Whether the *right* facts landed in each half is a model judgement.
    summary = "\n".join(f"- [{'from image' if ids[id(f)] else 'from text'}] {f.fact}" for f in facts)
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "Facts naming the escalation target 'Tier 3 Platform' or the 15-minute response "
            "target are marked 'from image'. Facts about the thirty-minute stuck threshold, "
            "recording the sync ID in the incident channel, or not paging engineers directly "
            "are marked 'from text'."
        ),
        context=(
            "An article whose prose states the stuck-sync threshold, the sync-ID requirement "
            "and the no-direct-paging rule, plus a diagram — readable only as an image — "
            "showing 'ESCALATE TO: Tier 3 Platform' and 'RESPONSE TARGET: 15 minutes'. "
            "Each extracted fact is labelled with where the extractor said it came from."
        ),
    )
