"""Unit tests for the markitdown file parser."""

import json

import pytest

from hindsight_api.engine.parsers.markitdown import MarkitdownParser


@pytest.fixture
def parser() -> MarkitdownParser:
    return MarkitdownParser()


def _utf8_json_with_ascii_prefix() -> bytes:
    """A JSON file whose first chunk is ASCII but contains multibyte UTF-8 later.

    markitdown samples only the first chunk for charset detection, so this layout
    used to be mis-detected as ASCII and crash the JSON/ipynb converter on the
    first multibyte byte (0xc3).
    """
    payload = {"messages": [{"role": "user", "text": "x" * 6400 + " café à la crème naïve über"}]}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert any(b > 127 for b in raw[6000:]), "fixture must have non-ASCII bytes past the sample window"
    return raw


async def test_convert_utf8_json_transcript(parser: MarkitdownParser):
    """A UTF-8 JSON transcript with an ASCII prefix parses without a decode error."""
    file_data = _utf8_json_with_ascii_prefix()

    content = await parser.convert(file_data, "transcript.json")

    assert "café" in content
    assert "crème" in content


async def test_convert_plain_utf8_text(parser: MarkitdownParser):
    """A plain UTF-8 text file with non-ASCII content round-trips."""
    file_data = ("über résumé\n" + "a" * 7000 + "\nfin: naïveté").encode("utf-8")

    content = await parser.convert(file_data, "notes.txt")

    assert "über" in content
    assert "naïveté" in content


def test_utf8_stream_info_for_text_extension():
    """Text files that are valid UTF-8 get an explicit UTF-8 charset hint."""
    info = MarkitdownParser._utf8_stream_info("über".encode("utf-8"), "a.json")

    assert info is not None
    assert info.charset == "utf-8"


def test_utf8_stream_info_skips_binary_extension():
    """Binary files are left to markitdown's own detection (no hint)."""
    assert MarkitdownParser._utf8_stream_info(b"%PDF-1.4 ...", "a.pdf") is None


def test_utf8_stream_info_skips_non_utf8_text():
    """Non-UTF-8 text falls back to markitdown's detection (no hint)."""
    latin1 = "café".encode("latin-1")  # 0xe9, invalid as standalone UTF-8

    assert MarkitdownParser._utf8_stream_info(latin1, "a.txt") is None
