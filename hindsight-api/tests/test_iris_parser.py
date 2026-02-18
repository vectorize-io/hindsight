"""
Integration tests for the Iris file parser.

Tests are skipped automatically if HINDSIGHT_API_FILE_PARSER_IRIS_TOKEN
and HINDSIGHT_API_FILE_PARSER_IRIS_ORG_ID are not set in the environment.
"""

import os

import pytest

from hindsight_api.config import ENV_FILE_PARSER_IRIS_ORG_ID, ENV_FILE_PARSER_IRIS_TOKEN
from hindsight_api.engine.parsers.iris import IrisParser

_token = os.getenv(ENV_FILE_PARSER_IRIS_TOKEN)
_org_id = os.getenv(ENV_FILE_PARSER_IRIS_ORG_ID)

pytestmark = pytest.mark.skipif(
    not (_token and _org_id),
    reason="HINDSIGHT_API_FILE_PARSER_IRIS_TOKEN and HINDSIGHT_API_FILE_PARSER_IRIS_ORG_ID not set",
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
def iris_parser() -> IrisParser:
    return IrisParser(token=_token, org_id=_org_id)


@pytest.mark.asyncio
async def test_iris_parser_converts_pdf(iris_parser: IrisParser):
    """IrisParser should extract text from a valid PDF."""
    result = await iris_parser.convert(_SAMPLE_PDF, "sample.pdf")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_iris_parser_name(iris_parser: IrisParser):
    """IrisParser.name() should return 'iris'."""
    assert iris_parser.name() == "iris"


