"""Turning placeholder text back into a multimodal prompt, per provider.

All deterministic: given a chunk and some image bytes, exactly one message shape
should reach each provider. The shapes are asserted directly because getting one
wrong is silent — a data URI sent as plain text is a valid request that the model
reads as gibberish, and the retain still "succeeds".

The canonical wire vocabulary is OpenAI's content parts; each provider converts
from it. That single-canonical-shape choice is what these tests pin.
"""

import base64

from hindsight_api.engine.providers.anthropic_llm import _to_anthropic_content
from hindsight_api.engine.retain.attachment_content import (
    LoadedAttachment,
    build_prompt_parts,
    compute_attachment_hash,
    attachment_placeholder,
    short_attachment_id,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_HASH = compute_attachment_hash(PNG_BYTES)
PNG_ID = short_attachment_id(PNG_HASH)
PNG = LoadedAttachment(media_type="image/png", data=PNG_BYTES)


def test_text_without_images_stays_a_plain_string() -> None:
    """Text-only chunks must produce byte-identical requests to before."""
    assert build_prompt_parts("just prose", {}) == "just prose"


def test_the_image_lands_between_the_text_that_frames_it() -> None:
    """Position is the point: an image appended at the end loses its context."""
    text = f"click the button shown:\n\n{attachment_placeholder(PNG_HASH)}\n\nthen reconnect."

    parts = build_prompt_parts(text, {PNG_ID: PNG})

    assert [part["type"] for part in parts] == ["text", "image_url", "text"]
    assert "click the button shown" in parts[0]["text"]
    assert "then reconnect" in parts[2]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_the_encoded_payload_round_trips_to_the_original_bytes() -> None:
    parts = build_prompt_parts(attachment_placeholder(PNG_HASH), {PNG_ID: PNG})

    payload = parts[0]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(payload) == PNG_BYTES


def test_an_unresolvable_image_degrades_to_a_note_instead_of_failing() -> None:
    """Bytes can legitimately be gone; the surrounding prose is still worth extracting."""
    text = f"before {attachment_placeholder(PNG_HASH)} after"

    parts = build_prompt_parts(text, {})

    assert parts == [{"type": "text", "text": "before [attachment unavailable] after"}]


def test_repeated_images_each_get_their_own_part() -> None:
    text = f"{attachment_placeholder(PNG_HASH)} and again {attachment_placeholder(PNG_HASH)}"

    parts = build_prompt_parts(text, {PNG_ID: PNG})

    assert [part["type"] for part in parts] == ["image_url", "text", "image_url"]


def test_whitespace_only_runs_do_not_become_empty_text_parts() -> None:
    """Providers reject empty text blocks."""
    text = f"\n\n{attachment_placeholder(PNG_HASH)}\n\n"

    parts = build_prompt_parts(text, {PNG_ID: PNG})

    assert parts == [{"type": "image_url", "image_url": {"url": PNG.as_data_uri()}}]


class TestAnthropicConversion:
    def test_a_plain_string_passes_through_untouched(self) -> None:
        assert _to_anthropic_content("hello") == "hello"

    def test_an_image_part_becomes_a_native_base64_image_block(self) -> None:
        parts = build_prompt_parts(f"see:\n\n{attachment_placeholder(PNG_HASH)}", {PNG_ID: PNG})

        blocks = _to_anthropic_content(parts)

        assert blocks[-1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(PNG_BYTES).decode(),
            },
        }

    def test_text_parts_are_left_alone(self) -> None:
        blocks = _to_anthropic_content([{"type": "text", "text": "prose"}])

        assert blocks == [{"type": "text", "text": "prose"}]


class TestGeminiConversion:
    def test_a_plain_string_becomes_one_text_part(self) -> None:
        from google.genai import types as genai_types

        from hindsight_api.engine.providers.gemini_llm import _to_gemini_parts

        parts = _to_gemini_parts("hello", genai_types)

        assert len(parts) == 1
        assert parts[0].text == "hello"

    def test_an_image_part_becomes_inline_data_with_the_raw_bytes(self) -> None:
        from google.genai import types as genai_types

        from hindsight_api.engine.providers.gemini_llm import _to_gemini_parts

        prompt_parts = build_prompt_parts(f"see:\n\n{attachment_placeholder(PNG_HASH)}", {PNG_ID: PNG})

        parts = _to_gemini_parts(prompt_parts, genai_types)

        assert parts[0].text.strip() == "see:"
        assert parts[1].inline_data.mime_type == "image/png"
        # Raw bytes, not the base64 text — Gemini decodes nothing for us.
        assert parts[1].inline_data.data == PNG_BYTES
