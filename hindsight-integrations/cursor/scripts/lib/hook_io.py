"""Cursor hook stdio helpers."""

from __future__ import annotations

import json
import sys
from typing import Any


def read_hook_input() -> dict[str, Any]:
    """Read a Cursor hook payload, including Windows' UTF-8 BOM variant."""
    stream = getattr(sys.stdin, "buffer", sys.stdin)
    raw = stream.read()
    if isinstance(raw, bytes):
        payload = raw.decode("utf-8-sig", errors="replace")
    else:
        payload = raw.lstrip("\ufeff")
    return json.loads(payload)
