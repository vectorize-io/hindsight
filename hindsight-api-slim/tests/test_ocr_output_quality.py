"""Tests for OCR output quality admission (#3897).

Covers the deterministic validator and its use on the parser fallback path.
OCR bodies are never asserted via logs — only reason codes / acceptance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.parsers import FileParserRegistry, MarkitdownParser
from hindsight_api.engine.parsers.base import FileParser
from hindsight_api.engine.parsers.ocr_quality import (
    assess_ocr_output_quality,
    is_ocr_quality_candidate,
)


# --- validator: hard rejects -------------------------------------------------


def test_hard_reject_explicit_refusal():
    result = assess_ocr_output_quality("I can't read this image")
    assert result.accepted is False
    assert "explicit_refusal_or_no_text" in result.reasons
    assert result.measurements["char_count"] > 0


def test_hard_reject_no_visible_text():
    result = assess_ocr_output_quality("No visible text")
    assert result.accepted is False
    assert "explicit_refusal_or_no_text" in result.reasons


def test_hard_reject_dominated_by_unclear():
    result = assess_ocr_output_quality("[unclear] [unclear] [unclear]")
    assert result.accepted is False
    assert "dominated_by_unclear" in result.reasons or "zero_meaningful_tokens" in result.reasons
    assert result.measurements["unclear_count"] >= 2


def test_hard_reject_replacement_chars():
    result = assess_ocr_output_quality("\ufffd\ufffd\ufffd\ufffd\ufffd")
    assert result.accepted is False
    assert "dominated_by_replacement_chars" in result.reasons or "zero_meaningful_tokens" in result.reasons


def test_hard_reject_zero_meaningful_tokens():
    result = assess_ocr_output_quality("... --- ***")
    assert result.accepted is False
    assert "zero_meaningful_tokens" in result.reasons


# --- validator: compound low-quality ----------------------------------------


def test_compound_reject_ui_chrome_and_repetition():
    # Dense navigation chrome — hard-reject when chrome dominates the OCR body.
    result = assess_ocr_output_quality("OK Cancel Back Home Settings Menu OK Cancel Back")
    assert result.accepted is False
    assert "dominated_by_ui_chrome" in result.reasons or len(result.reasons) >= 2


def test_compound_reject_timestamp_dense():
    result = assess_ocr_output_quality("12:34 PM 12:35 PM 12:36 PM")
    assert result.accepted is False
    assert "timestamp_dense" in result.reasons


# --- validator: legitimate sparse text must pass ----------------------------


def test_accept_verification_code():
    result = assess_ocr_output_quality("AB12-XY9")
    assert result.accepted is True
    assert result.reasons == ()


def test_accept_amount():
    result = assess_ocr_output_quality("$42.50")
    assert result.accepted is True


def test_accept_sign():
    result = assess_ocr_output_quality("STOP")
    assert result.accepted is True


def test_accept_error_code():
    result = assess_ocr_output_quality("ERR_TIMEOUT_504")
    assert result.accepted is True


def test_accept_normal_paragraph():
    text = (
        "Invoice #1042 for Acme Corp. Total due is $1,280.00 by March 15. Please remit payment to the address on file."
    )
    result = assess_ocr_output_quality(text)
    assert result.accepted is True


def test_short_length_alone_is_not_enough_to_reject():
    # A single short token with no other bad signals must pass.
    result = assess_ocr_output_quality("Q7K2")
    assert result.accepted is True


# --- filename gating --------------------------------------------------------


def test_ocr_quality_candidate_image_extensions():
    assert is_ocr_quality_candidate("shot.png")
    assert is_ocr_quality_candidate("photo.JPG")
    assert is_ocr_quality_candidate("scan.jpeg")
    assert not is_ocr_quality_candidate("doc.pdf")
    assert not is_ocr_quality_candidate("notes.txt")


# --- fallback path ----------------------------------------------------------


class _StubParser(FileParser):
    def __init__(self, name: str, content: str | None = None, error: Exception | None = None):
        self._name = name
        self._content = content
        self._error = error
        self.calls = 0

    async def convert(self, file_data: bytes, filename: str) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._content is not None
        return self._content

    def name(self) -> str:
        return self._name


@pytest.mark.asyncio
async def test_fallback_advances_after_ocr_quality_reject():
    """Rejected OCR from parser A must advance to parser B."""
    bad = _StubParser("markitdown", content="No visible text")
    good = _StubParser("iris", content="Room 12B — badge code ZX-441")

    registry = FileParserRegistry()
    registry.register(bad)
    registry.register(good)

    result = await registry.convert_with_fallback(
        parsers=["markitdown", "iris"],
        file_data=b"fake-image-bytes",
        filename="badge.png",
    )

    assert result.parser_name == "iris"
    assert "ZX-441" in result.content
    assert bad.calls == 1
    assert good.calls == 1


@pytest.mark.asyncio
async def test_terminal_failure_when_all_ocr_outputs_rejected():
    """If every parser returns inadmissible OCR, raise and create no success."""
    a = _StubParser("markitdown", content="I can't read this image")
    b = _StubParser("iris", content="[unclear] [unclear]")

    registry = FileParserRegistry()
    registry.register(a)
    registry.register(b)

    with pytest.raises(RuntimeError, match="OCR output rejected"):
        await registry.convert_with_fallback(
            parsers=["markitdown", "iris"],
            file_data=b"fake-image-bytes",
            filename="blurry.jpeg",
        )

    assert a.calls == 1
    assert b.calls == 1


@pytest.mark.asyncio
async def test_non_image_skips_ocr_quality_gate():
    """PDF/text conversions must not be gated by OCR quality heuristics."""
    # Would hard-reject if gated — but .pdf is not an OCR candidate.
    parser = _StubParser("markitdown", content="No visible text")
    registry = FileParserRegistry()
    registry.register(parser)

    result = await registry.convert_with_fallback(
        parsers=["markitdown"],
        file_data=b"%PDF-1.4",
        filename="scan.pdf",
    )
    assert result.content == "No visible text"


@pytest.mark.asyncio
async def test_markitdown_stub_convert_refusal_rejected_via_fallback():
    """Stub MarkItDown.convert() returning a refusal; fallback must reject it."""
    parser = MarkitdownParser(ocr_enabled=False)
    # Bypass the "OCR not enabled" image guard by feeding through the sync
    # convert path with a stubbed underlying convert — then run fallback gate.
    parser._ocr_enabled = True
    parser._markitdown = MagicMock()
    parser._markitdown.convert.return_value = SimpleNamespace(text_content="No visible text")

    registry = FileParserRegistry()
    registry.register(parser)

    with pytest.raises(RuntimeError, match="OCR output rejected|No visible text|reasons="):
        await registry.convert_with_fallback(
            parsers=["markitdown"],
            file_data=b"\x89PNG\r\n\x1a\n",
            filename="refuse.png",
        )


@pytest.mark.asyncio
async def test_markitdown_stub_convert_sparse_legitimate_accepted():
    """Stub MarkItDown.convert() with a verification code — must be accepted."""
    parser = MarkitdownParser(ocr_enabled=False)
    parser._ocr_enabled = True
    parser._markitdown = MagicMock()
    parser._markitdown.convert.return_value = SimpleNamespace(text_content="VERIFY-8821")

    registry = FileParserRegistry()
    registry.register(parser)

    result = await registry.convert_with_fallback(
        parsers=["markitdown"],
        file_data=b"\x89PNG\r\n\x1a\n",
        filename="code.png",
    )
    assert result.content == "VERIFY-8821"
    assert result.parser_name == "markitdown"
