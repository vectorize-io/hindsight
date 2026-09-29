"""Chunking text that carries inline image placeholders.

Two properties matter, and both are deterministic:

- **Adjacency.** An image should land in the same chunk as the prose that
  introduces it. That adjacency is the entire reason to accept inline images
  rather than retain them as separate documents.
- **Idempotency.** Re-chunking any chunk must return it unchanged. The streaming
  retain pipeline pre-chunks a document and then re-chunks every piece during
  extraction; a piece that re-split would give two chunks one ``chunk_index`` and
  collide their ``chunk_id`` (issue #2301).

Text with no placeholders must also come out byte-identical to before, which the
first test pins.
"""

import pytest

from hindsight_api.engine.retain.fact_extraction import chunk_text
from hindsight_api.engine.retain.attachment_content import (
    compute_attachment_hash,
    contains_attachment,
    attachment_placeholder,
    iter_placeholder_ids,
    short_attachment_id,
)

MAX_CHARS = 3000
MAX_IMAGES = 8


def _placeholder(seed: bytes) -> str:
    return attachment_placeholder(compute_attachment_hash(seed))


def _chunk(text: str, *, max_chars: int = MAX_CHARS, max_images: int = MAX_IMAGES) -> list[str]:
    return chunk_text(text, max_chars, max_attachments_per_chunk=max_images)


def test_text_without_images_is_chunked_exactly_as_before() -> None:
    """The image budget must not perturb any document retained to date."""
    prose = ". ".join(f"Sentence number {i} carries some words" for i in range(400))

    assert _chunk(prose) == chunk_text(prose, MAX_CHARS)


def test_an_image_stays_with_the_sentence_that_introduces_it() -> None:
    image = _placeholder(b"button")
    text = f"To reset the VPN, click the button shown:\n\n{image}\n\n...then reconnect."

    chunks = _chunk(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_an_image_costs_only_the_characters_its_placeholder_occupies() -> None:
    """retain_chunk_size budgets TEXT. An image must not evict prose.

    An earlier version charged each image a large slice of the budget. That split
    an article's introduction away from the images it introduced, which is the
    one thing inline images exist to prevent.
    """
    prose = "x" * 2000
    image = _placeholder(b"diagram")

    assert len(_chunk(prose)) == 1
    assert len(_chunk(f"{prose}\n\n{image}")) == 1


def test_images_beyond_the_hard_cap_start_a_new_chunk() -> None:
    """The count cap binds even when the character budget would allow more.

    Many small images can satisfy the arithmetic and still exceed a provider's
    per-request image limit.
    """
    text = "\n\n".join(_placeholder(f"img{i}".encode()) for i in range(5))

    chunks = _chunk(text, max_images=2)

    assert len(chunks) == 3
    assert [sum(1 for _ in iter_placeholder_ids(chunk)) for chunk in chunks] == [2, 2, 1]


def test_every_placeholder_survives_chunking_intact() -> None:
    """A placeholder split across two chunks would strand its image."""
    ids = [short_attachment_id(compute_attachment_hash(f"img{i}".encode())) for i in range(6)]
    text = "\n\n".join(f"{'body text ' * 100}{attachment_placeholder(i)}" for i in ids)

    chunks = _chunk(text)

    assert [i for chunk in chunks for i in iter_placeholder_ids(chunk)] == ids


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(f"lead in:\n\n{_placeholder(b'a')}\n\ntrailer", id="single-image"),
        pytest.param("\n\n".join(_placeholder(f"m{i}".encode()) for i in range(9)), id="images-only"),
        pytest.param(f"{'prose. ' * 900}{_placeholder(b'b')}{'more prose. ' * 900}", id="long-prose-around-image"),
        pytest.param(f"{_placeholder(b'c')}{'x' * 9000}", id="oversized-run-after-image"),
    ],
)
def test_rechunking_a_chunk_returns_it_unchanged(text: str) -> None:
    """The invariant chunk_id stability depends on (#2301)."""
    for chunk in _chunk(text):
        assert _chunk(chunk) == [chunk]


def test_a_run_too_long_to_share_a_chunk_is_split_by_the_ordinary_splitter() -> None:
    """Long prose still gets sentence-aware boundaries, not arbitrary cuts."""
    image = _placeholder(b"shot")
    prose = ". ".join(f"Sentence {i} of the article body" for i in range(300))

    chunks = _chunk(f"{image}\n\n{prose}")

    assert contains_attachment(chunks[0])
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= MAX_CHARS


def test_the_chunk_sequence_reconstructs_the_document_in_order() -> None:
    """Nothing may be dropped or reordered by the image-aware path."""
    image = _placeholder(b"z")
    text = f"alpha\n\n{image}\n\nbeta\n\ngamma"

    chunks = _chunk(text)

    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_an_introduction_stays_with_all_the_images_it_introduces() -> None:
    """ "Here are the screenshots:" must not be split away from the screenshots.

    Ten images and one sentence: everything belongs in one chunk, because the
    images cost only their placeholders and the count cap allows them.
    """
    images = "\n\n".join(_placeholder(f"shot{i}".encode()) for i in range(10))

    chunks = _chunk(f"Here are the ten screenshots of the failure:\n\n{images}", max_images=16)

    assert len(chunks) == 1
    assert "ten screenshots" in chunks[0]
    assert sum(1 for _ in iter_placeholder_ids(chunks[0])) == 10


def test_images_first_then_the_text_that_explains_them() -> None:
    """The reverse ordering: pictures, then the prose about them.

    A caller writes this whenever the explanation follows the evidence. The
    packer used to flush the buffered images the moment the following prose was
    too long to fit whole, emitting a chunk of images with no text at all — and
    the explanation in a chunk with no images.
    """
    images = "\n\n".join(_placeholder(f"err{i}".encode()) for i in range(2))
    prose = ". ".join(f"These show error state {i} in the console" for i in range(120))

    chunks = _chunk(f"{images}\n\n{prose}")

    assert sum(1 for _ in iter_placeholder_ids(chunks[0])) == 2
    assert "These show error state 0" in chunks[0], "the images were stranded without their explanation"


def test_a_chunk_never_exceeds_the_image_count_cap() -> None:
    """The cap is the real bound on images, so it must hold under every ordering."""
    images = "\n\n".join(_placeholder(f"i{i}".encode()) for i in range(9))

    for text in (f"intro:\n\n{images}", f"{images}\n\ntrailing prose"):
        for chunk in _chunk(text, max_images=4):
            assert sum(1 for _ in iter_placeholder_ids(chunk)) <= 4
