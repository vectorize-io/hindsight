"""
Integration tests for the LlamaParse file parser.

Tests are skipped automatically if HINDSIGHT_API_FILE_PARSER_LLAMA_PARSE_API_KEY
is not set in the environment.
"""

import os

import pytest

from hindsight_api.config import ENV_FILE_PARSER_LLAMA_PARSE_API_KEY
from hindsight_api.engine.parsers.llama_parse import LlamaParseParser

_api_key = os.getenv(ENV_FILE_PARSER_LLAMA_PARSE_API_KEY)

pytestmark = pytest.mark.skipif(
    not _api_key,
    reason="HINDSIGHT_API_FILE_PARSER_LLAMA_PARSE_API_KEY not set",
)

# Minimal valid PDF with the text "Hello from Hindsight"
_SAMPLE_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello from Hindsight) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
369
%%EOF"""


@pytest.fixture
def llama_parse_parser() -> LlamaParseParser:
    return LlamaParseParser(api_key=_api_key)


@pytest.mark.asyncio
async def test_llama_parse_parser_converts_pdf(llama_parse_parser: LlamaParseParser):
    """LlamaParseParser should extract text from a valid PDF."""
    result = await llama_parse_parser.convert(_SAMPLE_PDF, "sample.pdf")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_llama_parse_parser_name(llama_parse_parser: LlamaParseParser):
    """LlamaParseParser.name() should return 'llama_parse'."""
    assert llama_parse_parser.name() == "llama_parse"
