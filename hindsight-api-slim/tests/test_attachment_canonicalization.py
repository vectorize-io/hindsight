"""Canonicalization of multimodal retain content into placeholder text.

Pure and fully deterministic, so everything here asserts directly -- no LLM.
The determinism is the point: the same blocks must always flatten to the same
body, or retain's ``content_hash`` gate would re-extract an unchanged document.
"""

from hindsight_api.engine.llm_wrapper import sanitize_text
from hindsight_api.engine.retain.attachment_content import (
    CanonicalContent,
    RetainAttachment,
    RetainText,
    canonicalize,
    compute_attachment_hash,
    contains_attachment,
    attachment_placeholder,
    iter_placeholder_ids,
    neutralize_placeholders,
    short_attachment_id,
)

PNG = b"\x89PNG\r\n\x1a\n fake bytes"
GIF = b"GIF89a other fake bytes"


def _image(data: bytes, block_index: int = 0) -> RetainAttachment:
    return RetainAttachment(
        attachment_hash=compute_attachment_hash(data),
        media_type="image/png",
        data=data,
        block_index=block_index,
    )


def test_single_text_block_matches_the_plain_string_form() -> None:
    """The string and one-text-block forms must hash identically.

    A caller migrating from `content: "..."` to the block form for an item with
    no images must not have every document re-extracted as changed.
    """
    text = "Alice mentioned she's working on a new ML model"

    result = canonicalize([RetainText(text)])

    assert result == CanonicalContent(text=text, attachments=())


def test_image_sits_alone_between_the_prose_that_frames_it() -> None:
    png = _image(PNG, block_index=1)

    result = canonicalize(
        [
            RetainText("To reset the VPN, click the button shown:"),
            png,
            RetainText("...then reconnect."),
        ]
    )

    assert result.text == (
        f"To reset the VPN, click the button shown:\n\n{attachment_placeholder(png.attachment_hash)}\n\n...then reconnect."
    )
    assert result.attachments == (png,)


def test_canonicalization_is_deterministic_across_calls() -> None:
    blocks = [RetainText("before"), _image(PNG, 1), RetainText("after")]

    assert canonicalize(blocks).text == canonicalize(blocks).text


def test_trailing_newlines_are_not_doubled_around_a_placeholder() -> None:
    png = _image(PNG, block_index=1)

    result = canonicalize([RetainText("intro:\n\n"), png])

    assert result.text == f"intro:\n\n{attachment_placeholder(png.attachment_hash)}"


def test_consecutive_images_each_get_their_own_paragraph() -> None:
    first, second = _image(PNG, 0), _image(GIF, 1)

    result = canonicalize([first, second])

    assert (
        result.text
        == f"{attachment_placeholder(first.attachment_hash)}\n\n{attachment_placeholder(second.attachment_hash)}"
    )


def test_repeated_image_is_placed_twice_but_stored_once() -> None:
    """Content-addressing means one blob, even when the document shows it twice."""
    first, again = _image(PNG, 0), _image(PNG, 2)

    result = canonicalize([first, RetainText("and again:"), again])

    short = short_attachment_id(first.attachment_hash)
    assert result.attachments == (first,)
    assert list(iter_placeholder_ids(result.text)) == [short, short]


def test_identical_bytes_hash_identically_and_different_bytes_do_not() -> None:
    assert compute_attachment_hash(PNG) == compute_attachment_hash(bytes(PNG))
    assert compute_attachment_hash(PNG) != compute_attachment_hash(GIF)


def test_caller_authored_text_cannot_forge_an_image_reference() -> None:
    """Only the canonicalizer mints placeholders.

    Otherwise a caller could hand-write the token and have extraction resolve it
    to an image the document never carried -- including another document's.
    """
    forged = attachment_placeholder("a" * 64)

    result = canonicalize([RetainText(f"see {forged} here")])

    assert result.text == "see  here"
    assert not contains_attachment(result.text)


def test_malformed_lookalikes_are_scrubbed_too() -> None:
    assert neutralize_placeholders("x ⟦hs-att:not-hex⟧ y") == "x  y"


def test_placeholder_survives_the_ingress_sanitizer_byte_for_byte() -> None:
    """`sanitize_text` runs over user content on the way to the LLM and the DB.

    If it touched the delimiters, extraction could no longer find the image the
    body refers to.
    """
    placeholder = attachment_placeholder(compute_attachment_hash(PNG))

    assert sanitize_text(placeholder) == placeholder


def test_placeholder_holds_no_chunk_separator_characters() -> None:
    """The recursive chunker must have no preferred split point inside a token.

    A placeholder split across two chunks would strand the image reference.
    """
    from hindsight_api.engine.retain.fact_extraction import _RECURSIVE_TEXT_SEPARATORS

    placeholder = attachment_placeholder(compute_attachment_hash(PNG))

    for separator in _RECURSIVE_TEXT_SEPARATORS:
        if separator:
            assert separator not in placeholder


def test_empty_block_list_yields_empty_content() -> None:
    assert canonicalize([]) == CanonicalContent(text="", attachments=())
