"""Validate that JSON manifests are strict-valid JSON (no trailing commas, etc.)."""

import json
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent


def test_hooks_json_is_valid():
    path = INTEGRATION_ROOT / "hooks" / "hooks.json"
    raw = path.read_text()
    parsed = json.loads(raw)
    assert "hooks" in parsed
    assert isinstance(parsed["hooks"], dict)
