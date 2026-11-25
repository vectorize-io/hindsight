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


# OpenRouter groups traffic into an app by HTTP-Referer and names it from the
# title header. Without defaults, whoever sends our URL with their own title
# can rename the app for everyone (#6888).
OPENROUTER_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/vectorize-io/hindsight",
    "X-OpenRouter-Title": "Hindsight",
}
_OPENROUTER_ATTRIBUTION_NAMES = {"http-referer", "x-title", "x-openrouter-title"}


def with_openrouter_attribution(default_headers: dict[str, str] | None) -> dict[str, str]:
    """Add Hindsight's OpenRouter app headers unless the operator set any of their own.

    All or nothing: an operator's title next to our URL would rename the
    Hindsight app, and our title next to their URL would mislabel theirs.
    """
    headers = dict(default_headers or {})
    if not any(name.lower() in _OPENROUTER_ATTRIBUTION_NAMES for name in headers):
        headers.update(OPENROUTER_ATTRIBUTION_HEADERS)
    return headers
