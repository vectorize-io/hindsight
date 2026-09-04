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
    "call": "LLMCallResult",
    "_fit_structured_delta_prompt_parts": "FittedDeltaPrompt",
    "_validate_operations_list": "ValidatedOperations",
}


#: Shapes a double may legitimately return besides constructing the named type:
#: awaiting the real thing, or handing back a value the test already built.
def _is_named_result(node: ast.AST) -> bool:
    """Whether ``node`` plausibly evaluates to the named result type.

    Deliberately shallow — it accepts a constructor call, an await (delegating to
    the real implementation), or a conditional over those. Anything else, notably
    a bare literal or a raw payload variable, is what this guard exists to catch:
    the recurring bug was a double handing back the *payload* where production
    expects the *envelope*.
    """
    if isinstance(node, ast.Await):
        return True
    if isinstance(node, ast.IfExp):
        return _is_named_result(node.body) and _is_named_result(node.orelse)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return name in set(NAMED_RESULTS.values()) or name in {"AsyncMock", "MagicMock", "Mock"}
    return False


def _test_modules():
    """Every module in tests/, not just ``test_*.py``.

    The helpers are the dangerous ones: ``llm_judge.py`` calls ``judge.call()``
    and is imported by every LLM-behaviour test, so one stale read there fails a
    whole CI job class at once — which is exactly what happened, and why this
    glob is ``*.py``."""
    return sorted(p for p in TESTS_DIR.glob("*.py") if p.name != "__init__.py")


def _returns_bare_tuple(fn: ast.AST) -> list[int]:
    """Line numbers where ``fn`` yields a bare tuple.

    Accepts either a function (its own ``return`` statements, nested defs
    excluded) or an assignment whose value is the stubbed result directly.
    """
    if isinstance(fn, ast.Assign):
        return [] if _is_named_result(fn.value) else [fn.lineno]
    nested = [n for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn]
    return [
        r.lineno
        for r in ast.walk(fn)
        if isinstance(r, ast.Return)
        and r.value is not None
        and not _is_named_result(r.value)
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

    # 2. patch("...<module>.<Class>.<name>", new=replacement) — the target is a
    #    dotted string, so neither the setattr form nor a method name matches it.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if fname not in ("patch", "patch.object"):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        attr = first.value.rsplit(".", 1)[-1]
        if attr not in NAMED_RESULTS:
            continue
        for kw in node.keywords:
            if kw.arg == "new" and isinstance(kw.value, ast.Name) and kw.value.id in by_name:
                found.append((attr, by_name[kw.value.id]))

    # 3. `<mock>.<name>.return_value = payload` — set after construction, so the
    #    AsyncMock(return_value=...) form above never sees it.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = ast.unparse(node.targets[0])
        for fname in NAMED_RESULTS:
            if target.endswith(f".{fname}.return_value"):
                found.append((fname, node))
                break

    # 4. a method named after the function, on any class in the module
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in NAMED_RESULTS:
                found.append((item.name, item))

    # 5. `<mock>.<name> = AsyncMock(side_effect=fn)` / `(return_value=fn)` — the
    #    double is the function, not the mock, so neither the setattr target nor
    #    the mock's own return value names it.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target_name = ast.unparse(node.targets[0]).rsplit(".", 1)[-1]
        if target_name not in NAMED_RESULTS or not isinstance(node.value, ast.Call):
            continue
        for kw in node.value.keywords:
            if kw.arg in ("side_effect", "return_value") and isinstance(kw.value, ast.Name):
                if kw.value.id in by_name:
                    found.append((target_name, by_name[kw.value.id]))

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


def test_no_call_result_is_used_as_the_parsed_response():
    """``LLMInterface.call`` hands back the envelope, not the payload.

    ``call`` used to return the parsed response directly (or a ``(response,
    usage)`` tuple under a flag). It now always returns ``LLMCallResult``, so a
    site that feeds the awaited value straight into something expecting the
    parsed model, or annotates it as that model, is reading the envelope as if it
    were the payload.

    Both consolidation call sites did exactly that and no unpack-side sweep saw
    it: they assign to a single name, so there is no tuple to notice, and the
    wrong attribute only surfaces at runtime — as a caught exception that retried
    three times and skipped the batch, which is quiet enough to reach production.
    """
    root = TESTS_DIR.parent / "hindsight_api"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root.parent)

        def is_call(node):
            return (
                isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "call"
            )

        for node in ast.walk(tree):
            # annotated with something other than LLMCallResult
            if isinstance(node, ast.AnnAssign) and node.value is not None and is_call(node.value):
                ann = ast.unparse(node.annotation)
                if "LLMCallResult" not in ann:
                    offenders.append(f"{rel}:{node.lineno} annotates a call() result as {ann}")
            # passed straight into another call
            if isinstance(node, ast.Call):
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if is_call(arg) and not isinstance(node.func, ast.Attribute):
                        offenders.append(
                            f"{rel}:{node.lineno} passes a call() result into "
                            f"{ast.unparse(node.func)}() — pass .content"
                        )
            # attribute read directly off the await
            if isinstance(node, ast.Attribute) and is_call(node.value):
                if node.attr not in ("content", "usage"):
                    offenders.append(f"{rel}:{node.lineno} reads .{node.attr} off a call() result")
    assert not offenders, "\n".join(offenders)


def test_a_call_result_binding_is_only_read_through_its_fields():
    """A name bound from ``await ....call(...)`` must be used as the envelope.

    The shapes already covered are the ones where the misuse is visible at the
    expression itself — an unpack, a subscript, a wrong annotation. This covers
    the one that is not: bind the awaited value to a plain name, then treat that
    name as if it were the payload a few lines later.

    That is how ``tests/llm_judge.py`` broke every LLM-behaviour job at once. It
    did ``result = await judge.call(...)`` and then ``json.loads(str(result))``,
    which turned the envelope's repr into a JSONDecodeError — and because the
    judge is imported by every such test, one stale read failed all six provider
    acceptance jobs and 36 local tests, while looking exactly like a flaky
    provider.
    """
    #: Reading the whole envelope is legitimate here: these hand it onward
    #: unchanged rather than treating it as the payload.
    PASSTHROUGH = {"sanitize_llm_value", "isinstance_ok"}

    offenders = []
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(TESTS_DIR.parent)
        for scope in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            bound = {
                node.targets[0].id: node.lineno
                for node in ast.walk(scope)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Await)
                and isinstance(node.value.value, ast.Call)
                and isinstance(node.value.value.func, ast.Attribute)
                and node.value.value.func.attr == "call"
            }
            if not bound:
                continue
            # every Load of those names, minus the legitimate reads
            ok_nodes = set()
            for node in ast.walk(scope):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id in bound and node.attr in ("content", "usage"):
                        ok_nodes.add(id(node.value))
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                    ok_nodes.add(id(node.value))  # handed onward whole
                if isinstance(node, ast.Call):
                    fname = getattr(node.func, "id", None) or getattr(node.func, "attr", "")
                    if fname in PASSTHROUGH:
                        for a in node.args:
                            if isinstance(a, ast.Name):
                                ok_nodes.add(id(a))
            for node in ast.walk(scope):
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in bound
                    and id(node) not in ok_nodes
                    and node.lineno > bound[node.id]
                ):
                    offenders.append(
                        f"{rel}:{node.lineno} uses {node.id!r} (bound from await ....call() "
                        f"at line {bound[node.id]}) as the payload — read .content"
                    )
    assert not offenders, "\n".join(sorted(set(offenders)))
