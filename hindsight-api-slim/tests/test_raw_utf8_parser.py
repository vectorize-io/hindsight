"""Tests for the source-faithful raw UTF-8 file parser."""

import pytest

from hindsight_api.engine.parsers import FileParserRegistry, RawUtf8Parser


@pytest.mark.asyncio
async def test_raw_utf8_preserves_decoded_text_exactly() -> None:
    content = "\ufeff# Héllo\r\n\r\nline with spaces  \r\tindented\n"
    parser = RawUtf8Parser()

    assert await parser.convert(content.encode("utf-8"), "world.md") == content
    assert parser.name() == "raw_utf8"
    assert parser.contract_version() == "1"


@pytest.mark.asyncio
async def test_raw_utf8_preserves_whitespace_only_content_in_fallback_chain() -> None:
    registry = FileParserRegistry()
    registry.register(RawUtf8Parser())
    content = " \t\r\n\r\n"

    result = await registry.convert_with_fallback(
        parsers=["raw_utf8"],
        file_data=content.encode("utf-8"),
        filename="whitespace.txt",
        content_type="text/plain",
    )

    assert result.content == content
    assert result.parser_name == "raw_utf8"
    assert result.parser_contract_version == "1"
    assert result.preserves_source_text is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\xff", "valid UTF-8"),
        (b"before\x00after", "NUL"),
    ],
)
async def test_raw_utf8_rejects_unfaithful_text(content: bytes, message: str) -> None:
    parser = RawUtf8Parser()

    with pytest.raises(ValueError, match=message):
        await parser.convert(content, "world.md")
