"""Collection of the request headers an operator forwards to extensions.

Shared by both transports (the HTTP dependency and the MCP ASGI middleware) so
they cannot disagree about which value an extension sees. Both hand over raw
ASGI header pairs — Starlette exposes them as ``request.headers.raw``, the MCP
middleware reads them straight off the ASGI scope — so one implementation covers
decoding, case-folding and duplicate handling for both.
"""

import logging
from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)


def collect_passthrough_headers(
    raw_headers: Iterable[tuple[bytes, bytes]],
    allowlist: Sequence[str],
) -> dict[str, str]:
    """Pick the allowlisted headers out of a request, keyed by lower-cased name.

    ``allowlist`` is ``HindsightConfig.extension_passthrough_headers``, already
    lower-cased at config load; empty (the default) means nothing is forwarded.

    A header sent more than once is dropped rather than resolved. These headers
    carry identity for the deployments that enable this, and there is no safe
    universal rule for picking between copies: a proxy may append its trusted
    value after a client-supplied one or before it. Dropping turns a duplicate
    into a loud failure in the extension (which sees no header) instead of a
    silent choice between a real and a spoofed value.

    Values are decoded as latin-1, matching Starlette and the HTTP/1.1 wire
    encoding, so a header carrying non-UTF-8 bytes cannot fail the request.
    """
    if not allowlist:
        return {}

    wanted = set(allowlist)
    found: dict[str, list[bytes]] = {}
    for raw_name, raw_value in raw_headers:
        name = raw_name.decode("latin-1").lower()
        if name in wanted:
            found.setdefault(name, []).append(raw_value)

    collected: dict[str, str] = {}
    for name, values in found.items():
        if len(values) > 1:
            logger.warning(
                "Header '%s' is in HINDSIGHT_API_EXTENSION_PASSTHROUGH_HEADERS but arrived %d times; "
                "not forwarding it to extensions (no safe way to choose between the copies)",
                name,
                len(values),
            )
            continue
        collected[name] = values[0].decode("latin-1")
    return collected
