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

import base64
import io
import uuid
from datetime import datetime

import pytest
import pytest_asyncio
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


@pytest_asyncio.fixture
async def real_llm_client(memory_real_llm):
    """An HTTP client over an engine with a real, vision-capable LLM.

    The shared `api_client` runs on MockLLM, which cannot attribute anything, so
    it can never distinguish per-fact provenance from per-chunk. Attribution is
    the one thing this test exists to check, so it needs the real extractor.
    """
    import httpx

    from hindsight_api.api import create_app

    app = create_app(memory_real_llm, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=300) as client:
        yield client


@pytest.mark.asyncio
async def test_the_api_returns_the_attachment_only_on_the_facts_that_used_it(real_llm_client):
    """The narrowing, through storage and the read path — not just the resolver.

    `test_attachment_attribution` covers the pure number-to-id mapping and the
    test above covers whether the model attributes correctly, but both stop
    before the database. Between them and a caller sit the write of
    `memory_units.attachment_ids` and the read that resolves it, and a regression
    in either — most obviously reverting to the chunk-level join this replaced —
    restores exactly the bug the feature removed while every other test passes.

    So this asserts the shape a caller actually sees: same document, same chunk,
    some memories carrying the diagram and some carrying nothing.
    """
    data = _diagram()
    attachment_id = short_attachment_id(compute_attachment_hash(data))
    bank_id = f"attrib-{uuid.uuid4().hex[:8]}"

    response = await real_llm_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "A sync is considered stuck if it has not advanced in thirty minutes.",
                        },
                        {
                            "type": "text",
                            "text": "Engineers must never be paged directly. The escalation path is shown below:",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(data).decode(),
                            },
                        },
                    ],
                    "document_id": "sync-escalation",
                }
            ],
            "async": False,
        },
    )
    assert response.status_code == 200, response.text

    listed = await real_llm_client.get(f"/v1/default/banks/{bank_id}/memories/list?limit=100")
    assert listed.status_code == 200, listed.text
    memories = listed.json()["items"]
    assert memories, "retain produced no memories"

    # Only the facts extracted from the chunk can tell the two behaviours apart.
    # Consolidation also writes observations, which have no chunk_id and never
    # carry an attachment under *either* behaviour — counting those as "memories
    # without attachments" is what made an earlier version of this test pass with
    # the per-chunk join restored, i.e. not a guard at all.
    from_chunk = [m for m in memories if m.get("chunk_id")]
    assert from_chunk, "no memory was linked to a chunk"
    assert len({m["chunk_id"] for m in from_chunk}) == 1, (
        "the document split into several chunks, so a fact without the attachment "
        "may simply be from a chunk that had none — the comparison needs one chunk"
    )

    with_attachment = [m for m in from_chunk if m.get("attachments")]
    without = [m for m in from_chunk if not m.get("attachments")]

    assert with_attachment, "no memory carried the diagram, so nothing can show its evidence"
    assert without, (
        "every fact from this one chunk carried the diagram — the per-fact edge has "
        "collapsed back to the chunk's attachments, citing the diagram for prose it "
        "does not support"
    )
    # Whatever is attached must be the attachment that was actually retained.
    assert all(a["id"] == attachment_id for m in with_attachment for a in m["attachments"]), (
        "a memory carries an attachment this document never had"
    )
