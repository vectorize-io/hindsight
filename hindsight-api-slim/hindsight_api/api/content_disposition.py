"""The Content-Disposition header for a file response."""

import re
from typing import Literal
from urllib.parse import quote

# The plain stand-in keeps only characters every client reads safely: headers travel as Latin-1,
# simple parsers stop at a space or quote, and a `/` would let a careless client write elsewhere.
_NOT_PLAIN_ASCII = re.compile(r"[^A-Za-z0-9._-]")


def content_disposition(kind: Literal["attachment", "inline"], filename: str) -> str:
    """``filename`` for any client, encoded per RFC 6266 / RFC 5987.

    ``filename*`` carries the exact UTF-8 name; the plain ``filename`` is an ASCII stand-in for
    clients that only read that one (the generated Python client among them).
    """
    fallback = _NOT_PLAIN_ASCII.sub("_", filename)
    return f"{kind}; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"
