#!/usr/bin/env python3
"""Self-test for triage-labels.py's handling of an unavailable area classifier.

The script's only job is best-effort housekeeping: pick an area label so the
backlog can be filtered. When the classifier stopped answering, every
issue-opened run failed instead -- 39 consecutive red runs of the `Triage labels`
workflow whose only annotation was a bare `403`, because neither the status nor
the upstream body was ever logged (#5083). A gate that cannot do its job should
warn, not block.

The fixtures are the statuses the endpoint was measured returning:

    no credential at all      403  "Must supply an API key!"
    present but invalid key   401  "Cannot authenticate with the server..."
    unreachable               URLError

Only the first two are credential problems; a 401 means the key reached the
server and was rejected there. The cases below pin the status, the upstream body
and the reason into the run summary, because those are exactly the three things
that were missing when 39 failures in a row went unnoticed.

Against the script before this change every "degrades" case fails, because the
HTTPError propagated out of issue_label and failed the run.

Usage: python3 scripts/triage-labels-selftest.py
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(SCRIPT_DIR, "triage-labels.py")

FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILED
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}" + (f" -- {detail}" if detail else ""))
        FAILED = 1


def load_target():
    # The filename is not a module name, so load it by path.
    spec = importlib.util.spec_from_file_location("triage_labels", TARGET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def classifier(value):
    """Replace the classifier call, recording whether it was even attempted."""
    original = urllib.request.urlopen
    state = {"called": False}

    def fake(_request, timeout=None):
        state["called"] = True
        if isinstance(value, Exception):
            raise value
        return value

    urllib.request.urlopen = fake
    try:
        yield state
    finally:
        urllib.request.urlopen = original


@contextlib.contextmanager
def api_key(value: str | None):
    """TYPESAFE_API_KEY as the runner sees it: set, or absent because the secret is unset."""
    previous = os.environ.get("TYPESAFE_API_KEY")
    if value is None:
        os.environ.pop("TYPESAFE_API_KEY", None)
    else:
        os.environ["TYPESAFE_API_KEY"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TYPESAFE_API_KEY", None)
        else:
            os.environ["TYPESAFE_API_KEY"] = previous


def http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.typesafe.ai/v1/systemone", code, "Forbidden", {}, io.BytesIO(body.encode())
    )


def ok_response(label: str) -> io.BytesIO:
    return io.BytesIO(json.dumps({"answers": {"area": {"choice": label}}}).encode())


def probe(module, key: str | None) -> tuple[str | None, str, Exception | None]:
    """Call issue_label, capturing what it reported and whether it survived.

    The third element is the behaviour under test: before this change an
    HTTPError escaped from here and failed the whole run.
    """
    out = io.StringIO()
    with api_key(key), contextlib.redirect_stdout(out):
        try:
            return module.issue_label("some title", "some body"), out.getvalue(), None
        except Exception as error:
            return None, out.getvalue(), error


def expect_degrades(module, name: str, failure, key: str, *fragments: str) -> None:
    """A classifier that cannot answer must warn and return None, never raise."""
    with classifier(failure):
        label, output, error = probe(module, key)
    check(f"{name} returns None instead of raising", label is None and error is None, f"raised {error!r}")
    check(f"{name} is reported as a warning", "::warning::" in output, output.strip() or "no output")
    for fragment in fragments:
        check(f"{name} warning carries {fragment!r}", fragment in output, output.strip())


def main() -> int:
    module = load_target()

    # --- the happy path still works, and still stays quiet --------------------
    with classifier(ok_response("integration:hermes")) as state:
        label, output, error = probe(module, "a-key")
    check(
        "a reachable classifier still returns its choice",
        label == "integration:hermes" and not error,
        f"{label!r} {error!r}",
    )
    check("a reachable classifier emits no warning", "::warning::" not in output, output.strip())
    check("a reachable classifier does call the endpoint", state["called"])

    # --- 403: the credential never reached the server (#5083) -----------------
    expect_degrades(
        module,
        "HTTP 403",
        http_error(403, '{"error_type":"authentication_error","message":"Must supply an API key!"}'),
        "a-key",
        "403",
        "Must supply an API key!",
    )

    # --- 401: the credential reached the server and was rejected -------------
    expect_degrades(
        module,
        "HTTP 401",
        http_error(401, '{"error_type":"authentication_error","message":"Cannot authenticate with the server."}'),
        "a-stale-key",
        "401",
        "Cannot authenticate",
    )

    # --- unreachable ----------------------------------------------------------
    expect_degrades(
        module,
        "an unreachable classifier",
        urllib.error.URLError("connection refused"),
        "a-key",
        "connection refused",
    )

    # --- an unset or empty secret never reaches the network ------------------
    for article, name, key in (("a missing", "missing", None), ("an empty", "empty", "")):
        with classifier(ok_response("core")) as state:
            label, output, error = probe(module, key)
        check(f"{article} TYPESAFE_API_KEY returns None", label is None and error is None, f"{label!r} {error!r}")
        check(f"{article} TYPESAFE_API_KEY names itself in the warning", "TYPESAFE_API_KEY" in output, output.strip())
        check(f"{article} TYPESAFE_API_KEY skips the request entirely", not state["called"])

    # --- label_issue must survive a failure, and label nothing ----------------
    module.gh = lambda *args: json.dumps({"title": "t", "body": "b"})
    applied: list = []
    module.add_labels = lambda number, labels: applied.append((number, labels))

    with classifier(http_error(403, "Must supply an API key!")), api_key("a-key"):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                module.label_issue(4242)  # must not raise
                survived = True
            except Exception as error:
                survived = False
                print(f"      raised {error!r}")
    check("label_issue survives a failing classifier", survived)
    check("label_issue applies no label when the classifier failed", applied == [], f"applied {applied}")

    # --- ...and must still label when the classifier works -------------------
    applied.clear()
    with classifier(ok_response("core")), api_key("a-key"):
        with contextlib.redirect_stdout(io.StringIO()):
            module.label_issue(4242)
    check("label_issue still applies a label on success", applied == [(4242, {"core"})], f"applied {applied}")

    if not FAILED:
        print("all triage-labels classifier cases passed")
    return FAILED


if __name__ == "__main__":
    sys.exit(main())
