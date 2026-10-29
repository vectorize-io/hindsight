"""Which base URLs are OpenCode Go hosts.

``apply_opencode_session`` keys the ``x-opencode-session`` header on this (#4071), so a
look-alike domain must not match.
"""

import pytest

from hindsight_api.engine.cache_affinity import is_opencode_host


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://opencode.ai/zen/go/v1", True),
        ("https://api.opencode.ai/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://evil-opencode.ai", False),
        ("https://opencode.ai.evil.example", False),
        ("", False),
        (None, False),
    ],
)
def test_is_opencode_host(base_url, expected):
    assert is_opencode_host(base_url) is expected
