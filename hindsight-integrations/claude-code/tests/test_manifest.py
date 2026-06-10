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


def test_default_retain_tags_include_surface_attribution():
    """Shipped retainTags must keep session scoping and surface attribution.

    The surface:claude-code tag is what lets multi-surface setups sharing one
    bank attribute and filter facts per integration (recall tags / tag_groups).
    Dropping it from the shipped defaults would silently remove that filter
    axis for every user who has not overridden retainTags.
    """
    path = INTEGRATION_ROOT / "settings.json"
    parsed = json.loads(path.read_text())
    assert "{session_id}" in parsed["retainTags"]
    assert "surface:claude-code" in parsed["retainTags"]
