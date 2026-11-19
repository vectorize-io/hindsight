"""The Content-Disposition header survives any filename (#6668)."""

import re
from urllib.parse import unquote

import pytest

from hindsight_api.api.content_disposition import content_disposition


@pytest.mark.parametrize(
    ("filename", "plain_name"),
    [
        ("知识库-documents.zip", "___-documents.zip"),
        ('a "quoted" \\ name.pdf', "a__quoted____name.pdf"),
        ("../../.bashrc", ".._.._.bashrc"),
        ("plain.zip", "plain.zip"),
        ("é.md", "_.md"),
    ],
)
def test_any_filename_makes_a_latin1_header_every_client_can_read(filename, plain_name):
    header = content_disposition("attachment", filename)
    header.encode("latin-1")  # Starlette encodes headers as Latin-1; this raised before the fix.

    # The generated Python client reads only the plain filename, with this exact regex.
    plain = re.search(r'filename=[\'"]?([^\'"\s]+)[\'"]?', header)
    assert plain is not None and plain.group(1) == plain_name

    # The exact name round-trips through the RFC 5987 filename*.
    encoded = header.split("filename*=UTF-8''", 1)[1]
    assert unquote(encoded) == filename


def test_a_plain_ascii_name_is_kept_as_is():
    assert content_disposition("inline", "notes.md") == "inline; filename=\"notes.md\"; filename*=UTF-8''notes.md"
