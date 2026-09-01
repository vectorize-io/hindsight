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
from hindsight_api.engine.retain.image_content import (
    compute_image_hash,
    contains_image,
    image_placeholder,
    iter_placeholder_hashes,
)

COST = 1500
MAX_CHARS = 3000
MAX_IMAGES = 8


def _placeholder(seed: bytes) -> str:
    return image_placeholder(compute_image_hash(seed))


def _chunk(text: str, *, max_chars: int = MAX_CHARS, cost: int = COST, max_images: int = MAX_IMAGES) -> list[str]:
    return chunk_text(
        text,
        max_chars,
        image_cost_chars=cost,
        max_images_per_chunk=max_images,
    )


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


def test_an_image_costs_against_the_character_budget() -> None:
    """Prose that would fit alone is split once an image shares its chunk."""
    prose = "x" * 2000
    image = _placeholder(b"diagram")

    assert len(_chunk(prose)) == 1
    assert len(_chunk(f"{prose}\n\n{image}")) == 2


def test_images_beyond_the_hard_cap_start_a_new_chunk() -> None:
    """The count cap binds even when the character budget would allow more.

    Many small images can satisfy the arithmetic and still exceed a provider's
    per-request image limit.
    """
    text = "\n\n".join(_placeholder(f"img{i}".encode()) for i in range(5))

    chunks = _chunk(text, cost=10, max_images=2)

    assert len(chunks) == 3
    assert [sum(1 for _ in iter_placeholder_hashes(chunk)) for chunk in chunks] == [2, 2, 1]


def test_every_placeholder_survives_chunking_intact() -> None:
    """A placeholder split across two chunks would strand its image."""
    hashes = [compute_image_hash(f"img{i}".encode()) for i in range(6)]
    text = "\n\n".join(f"{'body text ' * 100}{image_placeholder(h)}" for h in hashes)

    chunks = _chunk(text)

    assert [h for chunk in chunks for h in iter_placeholder_hashes(chunk)] == hashes


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

    assert contains_image(chunks[0])
    assert len(chunks) > 1
    # No chunk exceeds the text budget once its images are charged for.
    for chunk in chunks:
        images = sum(1 for _ in iter_placeholder_hashes(chunk))
        assert len(chunk) + images * COST <= MAX_CHARS + len(image) * images


def test_the_chunk_sequence_reconstructs_the_document_in_order() -> None:
    """Nothing may be dropped or reordered by the image-aware path."""
    image = _placeholder(b"z")
    text = f"alpha\n\n{image}\n\nbeta\n\ngamma"

    chunks = _chunk(text)

    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_an_image_costlier_than_the_whole_budget_still_fits() -> None:
    """A small retain_chunk_size must not make images unchunkable.

    The cost and the chunk size are configured independently — a bank or retain
    strategy may lower retain_chunk_size below the image cost default for its own
    text-only reasons. Regression for a config rule that rejected exactly that
    (it broke `retain_chunk_size: 800` strategies on upgrade); the chunker clamps
    instead.
    """
    image = _placeholder(b"big")

    chunks = _chunk(f"intro\n\n{image}\n\noutro", max_chars=800, cost=5000)

    assert [h for chunk in chunks for h in iter_placeholder_hashes(chunk)] == [compute_image_hash(b"big")]
    # Still idempotent under the clamp.
    for chunk in chunks:
        assert _chunk(chunk, max_chars=800, cost=5000) == [chunk]


def test_an_over_budget_image_does_not_starve_the_surrounding_text() -> None:
    """Clamping must not drop prose — it only stops the image sharing a chunk."""
    image = _placeholder(b"big")

    chunks = _chunk(f"intro\n\n{image}\n\noutro", max_chars=800, cost=5000)

    joined = "".join(chunks)
    assert "intro" in joined
    assert "outro" in joined
