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
