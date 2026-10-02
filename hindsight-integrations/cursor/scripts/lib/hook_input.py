"""Helpers for reading the payload Cursor sends to hook scripts."""

import io
import json
import re
import sys

_DRIVE_PATH = re.compile(r"/[A-Za-z]:[/\\]")


def read_hook_input() -> dict:
    """Parse the hook's JSON stdin.

    Cursor on Windows prefixes it with a UTF-8 BOM, and Python would otherwise
    decode the pipe with the locale code page instead of UTF-8.
    """
    if isinstance(sys.stdin, io.TextIOWrapper):
        sys.stdin.reconfigure(encoding="utf-8-sig")
    return json.load(sys.stdin)


def normalize_path(path: str) -> str:
    """Drop the leading slash Cursor on Windows puts on drive paths (``/C:/Users/me``)."""
    return path[1:] if _DRIVE_PATH.match(path) else path
