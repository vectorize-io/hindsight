"""Container readiness probe. Standard library only, by rule.

``docker/standalone/start-all.sh`` polls the API's health endpoint while it
starts. That used to be ``curl -sf``, which meant shipping curl - and with it
libcurl and libssh2 - in every runtime image to make one GET request. Nothing
else in those images used it, and the three packages carried nine HIGH CVEs
with no Debian fix available.

**This module must never import the engine, the config, or any third-party
package.** It answers "is an API process up?", and pulling the application in
to ask that would put the application's startup cost - and its side effects -
on a loop that runs once per second. ``hindsight-admin`` is the cautionary
number: it takes ~5s to start in the built image because it loads the CLI and
everything behind it, against ~0.03s for this.

That rule is a test, not a convention:
``tests/test_http_probe.py::test_imports_nothing_heavy`` imports this module in
a clean subprocess and asserts that it pulled in no third-party module and none
of the engine. Living next to the code it must not touch is exactly why the
check is automated.

The contract is ``curl -sf`` *without* ``-L``, which is what this replaced:

* 2xx and 3xx succeed. curl does not follow redirects unless asked, so a 302 is
  a completed transfer, not a failure. This matters more than it looks:
  ``urllib.request.urlopen`` *does* follow redirects, so the obvious
  implementation reports a healthy service that redirects as "not ready".
* 4xx and 5xx fail, as ``-f`` does.
* Connection, DNS and timeout errors fail.
* Credentials in the URL are sent as Basic auth, as curl does.

Exit codes are not reproduced - curl's 22 and 7 both become 1. Every call site
tests zero/non-zero only.

One deliberate difference from what it replaced: curl was invoked with
``--connect-timeout``, which caps only the connection phase, and the API health
loop passed no timeout at all - so a server that accepted a connection and then
never answered hung the probe forever. The timeout here covers the whole
request.
"""

from __future__ import annotations

import base64
import http.client
import sys
from urllib.parse import unquote, urlsplit

__all__ = ["DEFAULT_TIMEOUT_SECONDS", "main", "probe"]

DEFAULT_TIMEOUT_SECONDS = 5.0


def probe(url: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Return True if ``url`` answers the way ``curl -sf`` would call success.

    Never raises: a probe that blew up on an unexpected socket error would be
    indistinguishable from a crash in the readiness loop that calls it.
    """
    parts = urlsplit(url)

    if parts.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.hostname or "", parts.port, timeout=timeout_seconds
        )
    elif parts.scheme == "http":
        connection = http.client.HTTPConnection(parts.hostname or "", parts.port, timeout=timeout_seconds)
    else:
        return False

    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"

    headers: dict[str, str] = {}
    if parts.username is not None:
        raw = f"{unquote(parts.username)}:{unquote(parts.password or '')}"
        headers["Authorization"] = "Basic " + base64.b64encode(raw.encode()).decode()

    try:
        connection.request("GET", path, headers=headers)
        status = connection.getresponse().status
    except Exception:
        return False
    finally:
        connection.close()

    return status < 400


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or len(args) > 2:
        print("usage: python -m hindsight_api.http_probe URL [TIMEOUT_SECONDS]", file=sys.stderr)
        return 2

    timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    if len(args) == 2:
        try:
            timeout_seconds = float(args[1])
        except ValueError:
            print(f"invalid timeout: {args[1]}", file=sys.stderr)
            return 2

    return 0 if probe(args[0], timeout_seconds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
