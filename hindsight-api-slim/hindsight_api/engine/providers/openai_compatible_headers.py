"""Shared HTTP headers for Hindsight's OpenAI-compatible transports."""

from hindsight_api import __version__

OPENAI_COMPATIBLE_USER_AGENT = f"hindsight/openai-compatible/{__version__}"


def with_openai_compatible_user_agent(default_headers: dict[str, str] | None) -> dict[str, str]:
    """Return client headers with Hindsight's transport identity when unset.

    Header names are case-insensitive, so inspect them accordingly before adding
    the canonical spelling. An operator-supplied value remains authoritative.
    """
    headers = dict(default_headers or {})
    if not any(name.lower() == "user-agent" for name in headers):
        headers["User-Agent"] = OPENAI_COMPATIBLE_USER_AGENT
    return headers
