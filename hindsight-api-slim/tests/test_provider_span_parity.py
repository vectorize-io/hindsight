"""Every provider that reports a successful call to metrics must also trace it (#5381).

``LLMProvider`` records spans only for failed calls; a successful call's span is the
provider's job. Cursor and Claude Code's ``call_with_tools`` forgot it, so their
successful calls never reached the traces and no test noticed — the provider that
forgets is by definition the one nobody wrote a span test for. This guard runs over
the whole provider family instead.
"""

import ast
from pathlib import Path

import hindsight_api.engine.providers as providers_pkg

_PROVIDERS_DIR = Path(providers_pkg.__file__).parent


def _record_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "record_llm_call"
    ]


def _has_kwarg(call: ast.Call, name: str, value: object = ...) -> bool:
    for kw in call.keywords:
        if kw.arg == name:
            return value is ... or (isinstance(kw.value, ast.Constant) and kw.value.value == value)
    return False


def test_every_successful_metric_has_a_matching_span():
    checked = 0
    for path in sorted(_PROVIDERS_DIR.glob("*.py")):
        calls = _record_calls(ast.parse(path.read_text()))
        # Metrics calls carry ``success=``; span calls carry ``messages=``.
        successes = sum(1 for c in calls if _has_kwarg(c, "success", True))
        spans = sum(1 for c in calls if _has_kwarg(c, "messages"))
        assert spans >= successes, (
            f"{path.name} records {successes} successful call(s) to metrics but only {spans} span(s)"
        )
        checked += successes
    assert checked > 0, "no provider records successful calls — the scan is broken"


# A handler catching one of these and looping back is retrying a failed attempt.
# JSON-parse retries are left out: the reply is in hand there and the wrapper traces it.
_RETRIED_FAILURES = {"TimeoutError", "ClientError", "APIConnectionError", "Exception"}


def _caught_names(handler: ast.ExceptHandler) -> set[str]:
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {t.attr if isinstance(t, ast.Attribute) else getattr(t, "id", "") for t in types if t is not None}


def _calls_named(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == name for n in ast.walk(node)
    )


def test_every_retried_timeout_leaves_a_span():
    """A timed-out attempt that is retried must be traced before the retry (#6624).

    Only the final failure reaches the wrapper's error span, so an attempt a provider
    retries on its own was a gap in the trace with only a log line to explain it.
    """
    checked = 0
    for path in sorted(_PROVIDERS_DIR.glob("*.py")):
        for handler in ast.walk(ast.parse(path.read_text())):
            if not isinstance(handler, ast.ExceptHandler) or not (_caught_names(handler) & _RETRIED_FAILURES):
                continue
            if not any(isinstance(n, ast.Continue) for n in ast.walk(handler)):
                continue
            assert _calls_named(handler, "record_retried_attempt"), (
                f"{path.name}:{handler.lineno} retries a failed attempt without record_retried_attempt"
            )
            checked += 1
    assert checked > 0, "no provider retries a failed attempt — the scan is broken"
