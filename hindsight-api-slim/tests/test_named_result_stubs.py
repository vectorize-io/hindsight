"""Test doubles must return the named result type the real function returns.

The refactor that replaced bare tuple returns with named types was verified by
sweeping for *callers* that still unpacked a tuple. That missed the opposite
direction, and CI caught it: five upstream tests stub one of these functions and
returned the old tuple, so production code doing ``result.results`` got an
``AttributeError`` — or, worse, hung, because the stubbed pipeline never reached
the failure the test was waiting for.

A stub is invisible to a caller-side sweep and to the type checker, so this
asserts over the whole test suite instead: any double that stands in for one of
these functions must produce the same shape the real one does.

Two ways a double is recognised, because both occur:

* ``monkeypatch.setattr(module, "<name>", replacement)`` — the target names the
  function, and the replacement is looked up by name in the same file.
* a method named after the function (``FakeGraphRetriever.retrieve``) — installed
  by handing the whole object over, so the setattr target says nothing useful.
"""

import ast
import pathlib

import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent

#: Function name -> the named type it returns. Keep in step with the dataclasses;
#: `test_retain_result_shape.py` asserts the functions themselves still return them.
NAMED_RESULTS = {
    "retain_batch": "RetainBatchResult",
    "_streaming_retain_batch": "RetainBatchResult",
    "_try_delta_retain": "RetainBatchResult",
    "_delta_metadata_only": "RetainBatchResult",
    "_retain_batch_async_internal": "RetainBatchResult",
    "_retain_batch_with_append_retry": "RetainBatchResult",
    "extract_facts_from_contents": "ExtractionResult",
    "extract_facts_from_contents_batch_api": "ExtractionResult",
    "_extract_facts_chunks": "ExtractionResult",
    "_extract_and_embed": "_EmbeddedExtraction",
    "_prepare_facts_for_entity_processing": "PreparedFactEntities",
    "build_tags_where_clause": "TagClause",
    "build_tag_groups_where_clause": "TagClause",
    "_build_group_clause": "TagClause",
    "_parse_tags_match": "TagMatchSemantics",
    "retrieve": "GraphRetrieval",
    "_fit_structured_delta_prompt_parts": "FittedDeltaPrompt",
    "_validate_operations_list": "ValidatedOperations",
}


def _test_modules():
    return sorted(p for p in TESTS_DIR.glob("test_*.py"))


def _returns_bare_tuple(fn: ast.AST) -> list[int]:
    """Line numbers where ``fn`` itself returns a tuple literal (nested defs excluded)."""
    nested = [n for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn]
    return [
        r.lineno
        for r in ast.walk(fn)
        if isinstance(r, ast.Return)
        and isinstance(r.value, ast.Tuple)
        and not any(i.lineno <= r.lineno <= i.end_lineno for i in nested)
    ]


def _doubles_in(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """(function-it-stands-in-for, the double's def) pairs found in one module."""
    by_name = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found: list[tuple[str, ast.AST]] = []

    # 1. monkeypatch.setattr(module, "<name>", replacement)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "setattr" or len(node.args) < 3:
            continue
        target, replacement = node.args[1], node.args[2]
        if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
            continue
        if target.value not in NAMED_RESULTS:
            continue
        if isinstance(replacement, ast.Name) and replacement.id in by_name:
            found.append((target.value, by_name[replacement.id]))

    # 2. a method named after the function, on any class in the module
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in NAMED_RESULTS:
                found.append((item.name, item))

    return found


def test_the_scan_finds_doubles_at_all():
    """Guard the guard: if the recognisers stop matching, every assertion below is vacuous."""
    total = 0
    for path in _test_modules():
        try:
            total += len(_doubles_in(ast.parse(path.read_text())))
        except SyntaxError:
            continue
    assert total >= 5, f"only found {total} doubles — the recognisers have probably stopped matching"


@pytest.mark.parametrize("path", _test_modules(), ids=lambda p: p.stem)
def test_doubles_return_the_named_result(path):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        pytest.skip("module does not parse")
    offenders = []
    for stands_for, fn in _doubles_in(tree):
        for lineno in _returns_bare_tuple(fn):
            offenders.append(
                f"{path.name}:{lineno} {fn.name}() stands in for "
                f"{stands_for}() and must return {NAMED_RESULTS[stands_for]}, not a tuple"
            )
    assert not offenders, "\n".join(offenders)


#: Both trees are scanned: a caller that unpacks is as broken as a stub that supplies.
_SCANNED_ROOTS = ("hindsight_api", "tests")


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _source_files():
    root = TESTS_DIR.parent
    for sub in _SCANNED_ROOTS:
        yield from sorted((root / sub).rglob("*.py"))


def test_nothing_unpacks_a_named_result_as_a_tuple():
    """The caller-side half: a named result must not be destructured positionally.

    This is what a diff-based review does catch — but only for the call sites that
    exist when the refactor lands. A rebase brings new ones, which is exactly how
    ``test_tag_resolution.py`` reached CI with ``clause, params, _ = ...``. Asserting
    it over both trees means a new caller cannot arrive un-swept.
    """
    offenders = []
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(TESTS_DIR.parent)
        for node in ast.walk(tree):
            calls: list[ast.Call] = []
            if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], (ast.Tuple, ast.List)):
                calls = [c for c in ast.walk(node.value) if isinstance(c, ast.Call)]
            elif isinstance(node, ast.For) and isinstance(node.target, (ast.Tuple, ast.List)):
                calls = [c for c in ast.walk(node.iter) if isinstance(c, ast.Call)]
            elif isinstance(node, ast.Subscript):
                inner = node.value.value if isinstance(node.value, ast.Await) else node.value
                calls = [inner] if isinstance(inner, ast.Call) else []
            for call in calls:
                name = _callee_name(call)
                if name in NAMED_RESULTS:
                    offenders.append(
                        f"{rel}:{node.lineno} treats {name}() as a tuple; "
                        f"it returns {NAMED_RESULTS[name]} — read its fields by name"
                    )
    assert not offenders, "\n".join(sorted(set(offenders)))
