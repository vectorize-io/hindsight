"""What a vision model actually makes of an inline image.

Everything else about this feature is deterministic and asserted directly. This
is the part that is not: whether the model reads the picture, and whether it
reads it *in the context of the prose around it* rather than as a standalone
image. MockLLM cannot simulate that, and exact string matching on extracted
facts flakes across providers and runs — so this uses a real LLM plus the judge,
per the testing convention in CLAUDE.md.

The images are drawn here rather than committed as fixtures: a generated UI mock
says exactly what it is meant to say, and the test stays readable next to what it
asserts.
"""

import io
from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from hindsight_api.engine.retain.image_content import (
    LoadedImage,
    RetainImage,
    RetainText,
    canonicalize,
    compute_image_hash,
    short_image_id,
)
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


def _screenshot(label: str, *, subtitle: str = "") -> bytes:
    """Draw a screenshot-like PNG: a labelled button on a window chrome.

    Deliberately high-contrast and large: the assertion is about whether the
    model uses the image at all, so the image must not be the limiting factor.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 320), (243, 244, 246))
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.load_default(size=28)
        button_font = ImageFont.load_default(size=34)
    except TypeError:  # Pillow < 10 has no size argument
        title_font = button_font = ImageFont.load_default()

    # Window chrome, so the model reads this as a UI rather than as a poster.
    draw.rectangle([0, 0, 640, 56], fill=(31, 41, 55))
    draw.text((20, 14), "Network Settings", fill=(255, 255, 255), font=title_font)
    if subtitle:
        draw.text((40, 96), subtitle, fill=(55, 65, 81), font=title_font)

    draw.rectangle([180, 160, 460, 240], fill=(37, 99, 235))
    box = draw.textbbox((0, 0), label, font=button_font)
    draw.text(
        (320 - (box[2] - box[0]) / 2, 200 - (box[3] - box[1]) / 2),
        label,
        fill=(255, 255, 255),
        font=button_font,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _StubImageLoader:
    """Serves images from memory, standing in for the storage-backed loader.

    The storage round trip has its own deterministic tests; this one is about the
    model, so it is fed directly.
    """

    def __init__(self, images: dict[str, LoadedImage]) -> None:
        self._images = images

    async def load(self, image_ids):
        return {i: self._images[i] for i in image_ids if i in self._images}


def _content_with_image(before: str, image_bytes: bytes, after: str):
    """Canonicalize prose + image + prose the way the retain ingress does."""
    image = RetainImage(
        image_hash=compute_image_hash(image_bytes),
        media_type="image/png",
        data=image_bytes,
        block_index=1,
    )
    canonical = canonicalize([RetainText(before), image, RetainText(after)])
    loader = _StubImageLoader(
        {short_image_id(image.image_hash): LoadedImage(media_type="image/png", data=image_bytes)}
    )
    return canonical.text, loader


async def _extract(text: str, loader, context: str):
    config = _get_raw_config()
    facts, _, _ = await extract_facts_from_text(
        text=text,
        event_date=datetime(2024, 6, 1),
        llm_config=LLMConfig.from_env(),
        agent_name=None,
        config=config,
        context=context,
        image_loader=loader,
    )
    return "\n".join(f"- [{fact.fact_type}] {fact.fact}" for fact in facts)


@pytest.mark.asyncio
async def test_a_fact_is_extracted_from_what_the_image_shows() -> None:
    """The label lives only in the picture — no amount of text reading finds it."""
    text, loader = _content_with_image(
        "To reset the VPN connection, click the button shown below:",
        _screenshot("Reset VPN"),
        "After clicking it, wait ten seconds and reconnect.",
    )

    facts_summary = await _extract(text, loader, context="Support knowledge-base article about VPN troubleshooting")

    await assert_meets_criteria(
        response=facts_summary,
        criteria=(
            "At least one fact states what the button in the screenshot is labelled or says — "
            "'Reset VPN' — or otherwise conveys information that could only come from looking "
            "at the image, such as the window being titled 'Network Settings'."
        ),
        context=(
            "A knowledge-base article was retained as interleaved content: the sentence 'To reset "
            "the VPN connection, click the button shown below:', then a screenshot of a settings "
            "window with a blue button labelled 'Reset VPN', then 'After clicking it, wait ten "
            "seconds and reconnect.' The screenshot's button label appears nowhere in the text, "
            "so a text-only extractor could not produce it."
        ),
        msg="The vision model did not extract anything from the inline image",
    )


@pytest.mark.asyncio
async def test_the_image_is_read_in_the_context_of_the_prose_around_it() -> None:
    """Position is the feature: the image must be tied to the VPN instruction.

    An extractor handed the image on its own could describe a blue button. Only
    one that saw it *between* these sentences can connect that button to
    resetting the VPN.
    """
    text, loader = _content_with_image(
        "To reset the VPN connection, click the button shown below:",
        _screenshot("Reset VPN"),
        "After clicking it, wait ten seconds and reconnect.",
    )

    facts_summary = await _extract(text, loader, context="Support knowledge-base article about VPN troubleshooting")

    await assert_meets_criteria(
        response=facts_summary,
        criteria=(
            "The facts connect the button in the screenshot to the VPN reset procedure — that is, "
            "the image is described as part of resetting the VPN rather than as an unrelated "
            "picture of a user interface."
        ),
        context=(
            "The screenshot sat inline between 'To reset the VPN connection, click the button "
            "shown below:' and 'After clicking it, wait ten seconds and reconnect.' The value of "
            "inline images is that the extractor sees the image in that position."
        ),
        msg="The image was described in isolation rather than in its textual context",
    )


@pytest.mark.asyncio
async def test_a_diagram_only_detail_survives_extraction() -> None:
    """Real articles put load-bearing detail in pictures. This checks a second one."""
    text, loader = _content_with_image(
        "The escalation path for a stuck sync is documented in the diagram below.",
        _screenshot("Escalate to Tier 3", subtitle="Owner: Platform Team"),
        "Follow it before paging anyone directly.",
    )

    facts_summary = await _extract(text, loader, context="Internal runbook for on-call engineers")

    await assert_meets_criteria(
        response=facts_summary,
        criteria=(
            "At least one fact reflects detail visible only in the image — the escalation target "
            "'Tier 3', or that the owner is the 'Platform Team'."
        ),
        context=(
            "A runbook page was retained with an inline diagram. The diagram shows a button "
            "labelled 'Escalate to Tier 3' and the caption 'Owner: Platform Team'. Neither string "
            "appears in the surrounding text."
        ),
        msg="Detail present only in the image did not reach any fact",
    )
