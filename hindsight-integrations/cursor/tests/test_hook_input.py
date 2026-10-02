"""Tests for reading Cursor hook payloads (BOM-prefixed stdin, Windows workspace paths)."""

import io
import json

import pytest

from lib.hook_input import normalize_path, read_hook_input


def _stdin(raw: bytes) -> io.TextIOWrapper:
    # Windows pipes decode with the locale code page, not UTF-8.
    return io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252")


class TestReadHookInput:
    def test_strips_utf8_bom(self, monkeypatch):
        payload = json.dumps({"conversation_id": "c1"}).encode()
        monkeypatch.setattr("sys.stdin", _stdin(b"\xef\xbb\xbf" + payload))

        assert read_hook_input() == {"conversation_id": "c1"}

    def test_decodes_non_ascii_as_utf8(self, monkeypatch):
        payload = json.dumps({"cwd": "/home/user/café"}, ensure_ascii=False).encode("utf-8")
        monkeypatch.setattr("sys.stdin", _stdin(payload))

        assert read_hook_input() == {"cwd": "/home/user/café"}

    def test_reads_plain_text_stdin(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO('{"conversation_id": "c1"}'))

        assert read_hook_input() == {"conversation_id": "c1"}

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", _stdin(b"not json"))

        with pytest.raises(json.JSONDecodeError):
            read_hook_input()


class TestNormalizePath:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("/C:/Users/me/research", "C:/Users/me/research"),
            ("/c:\\Users\\me\\research", "c:\\Users\\me\\research"),
            ("C:\\Users\\me\\research", "C:\\Users\\me\\research"),
            ("/home/me/research", "/home/me/research"),
            ("", ""),
        ],
    )
    def test_normalize_path(self, raw, expected):
        assert normalize_path(raw) == expected
