"""Every retain pipeline path must return the named ``RetainBatchResult``.

A structural guard, not a behavioural one, because the defect it prevents is
invisible to behavioural tests. ``_streaming_retain_batch`` was annotated
``tuple[list[list[str]], TokenUsage]`` while returning three values, and
``retain_batch`` passed that straight back as its own 3-tuple. Every test still
passed — the extra element flowed through positionally — and ``ty`` cannot catch
it because ``invalid-return-type`` is disabled in ``pyproject.toml``. The
mismatch would only have surfaced as a ``ValueError`` the day someone unpacked
the annotated two names.

Mirrors ``test_migration_shape.py``: enumerate the family from the source and
assert every member satisfies the contract, so the next path added to the
pipeline cannot quietly reintroduce a bare tuple.
"""

import ast
import pathlib

import pytest

from hindsight_api.engine.retain.types import RetainBatchResult, merge_processed_content_tokens

API_ROOT = pathlib.Path(__file__).resolve().parent.parent / "hindsight_api"

#: The pipeline entry points, and the engine wrappers that forward their result.
RETAIN_RESULT_FUNCTIONS = {
    "engine/retain/orchestrator.py": [
        "retain_batch",
        "_streaming_retain_batch",
        "_try_delta_retain",
        "_delta_metadata_only",
    ],
    "engine/memory_engine.py": [
        "_retain_batch_async_internal",
        "_retain_batch_with_append_retry",
    ],
}

#: Other families where a bare tuple was the contract, and what replaced it.
#: Each entry is (module, function names, allowed return annotations).
NAMED_RESULT_FAMILIES = [
    (
        "engine/retain/fact_extraction.py",
        ["extract_facts_from_contents", "extract_facts_from_contents_batch_api", "_extract_facts_chunks"],
        {"ExtractionResult"},
    ),
    (
        "engine/search/tags.py",
        ["build_tags_where_clause", "build_tag_groups_where_clause", "_build_group_clause"],
        {"TagClause"},
    ),
    (
        "engine/search/graph_retrieval.py",
        ["retrieve"],
        {"GraphRetrieval"},
    ),
    (
        "engine/search/link_expansion_retrieval.py",
        ["retrieve"],
        {"GraphRetrieval"},
    ),
    (
        "engine/reflect/prompts.py",
        ["_fit_structured_delta_prompt_parts"],
        {"FittedDeltaPrompt"},
    ),
    (
        "engine/reflect/delta_ops.py",
        ["_validate_operations_list"],
        {"ValidatedOperations"},
    ),
    (
        "engine/retain/entity_processing.py",
        ["_prepare_facts_for_entity_processing"],
        {"PreparedFactEntities"},
    ),
]


def _function_nodes(rel_path: str, names: list[str]) -> dict[str, ast.AST]:
    tree = ast.parse((API_ROOT / rel_path).read_text())
    found = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    missing = set(names) - set(found)
    assert not missing, f"{rel_path}: expected functions vanished or were renamed: {sorted(missing)}"
    return found


def _own_returns(func: ast.AST) -> list[ast.Return]:
    """Return statements belonging to ``func`` itself, excluding nested defs."""
    nested = [n for n in ast.walk(func) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not func]
    return [
        r
        for r in ast.walk(func)
        if isinstance(r, ast.Return) and not any(inner.lineno <= r.lineno <= inner.end_lineno for inner in nested)
    ]


@pytest.mark.parametrize(
    ("rel_path", "name"),
    [(p, n) for p, names in RETAIN_RESULT_FUNCTIONS.items() for n in names],
)
def test_retain_path_is_annotated_with_the_named_result(rel_path, name):
    func = _function_nodes(rel_path, RETAIN_RESULT_FUNCTIONS[rel_path])[name]
    assert func.returns is not None, f"{name}() has no return annotation"
    annotation = ast.unparse(func.returns).replace('"', "").replace("'", "")
    assert annotation in ("RetainBatchResult", "RetainBatchResult | None"), (
        f"{name}() returns {annotation!r}; retain paths must return the named "
        f"RetainBatchResult so their arity cannot drift"
    )


@pytest.mark.parametrize(
    ("rel_path", "name"),
    [(p, n) for p, names in RETAIN_RESULT_FUNCTIONS.items() for n in names],
)
def test_retain_path_never_returns_a_bare_tuple(rel_path, name):
    func = _function_nodes(rel_path, RETAIN_RESULT_FUNCTIONS[rel_path])[name]
    offenders = [(r.lineno, ast.unparse(r.value)[:60]) for r in _own_returns(func) if isinstance(r.value, ast.Tuple)]
    assert not offenders, f"{name}() returns bare tuples at {offenders}"


def test_retain_batch_result_fields_are_ordered_and_named():
    """Positional construction is used throughout, so field order is part of the contract."""
    result = RetainBatchResult([["a"], []], usage=None, processed_content_tokens=7)
    assert result.memory_ids == [["a"], []]
    assert result.processed_content_tokens == 7


def test_merge_processed_content_tokens_treats_none_as_contagious():
    assert merge_processed_content_tokens(5, 7) == 12
    assert merge_processed_content_tokens(0, 0) == 0
    # Unknown on either side makes the total unknown — not zero, and not the
    # other side's value, both of which would under-bill the content.
    assert merge_processed_content_tokens(None, 10) is None
    assert merge_processed_content_tokens(10, None) is None


@pytest.mark.parametrize(
    ("rel_path", "name", "allowed"),
    [(p, n, a) for p, names, a in NAMED_RESULT_FAMILIES for n in names],
)
def test_named_result_families_keep_their_type(rel_path, name, allowed):
    """The other converted families must not drift back to bare tuples either.

    Each of these is a set of interchangeable implementations — three extraction
    routes that dispatch to one another, three tag-clause builders that compose
    each other's parameter offsets, an abstract graph retriever and its
    implementation. A tuple lets any one of them silently disagree with its
    siblings about arity or order.
    """
    func = _function_nodes(rel_path, [name])[name]
    assert func.returns is not None, f"{name}() has no return annotation"
    annotation = ast.unparse(func.returns).replace('"', "").replace("'", "")
    assert annotation in allowed or annotation in {f"{a} | None" for a in allowed}, (
        f"{rel_path}::{name}() returns {annotation!r}, expected one of {sorted(allowed)}"
    )
    offenders = [(r.lineno, ast.unparse(r.value)[:60]) for r in _own_returns(func) if isinstance(r.value, ast.Tuple)]
    assert not offenders, f"{rel_path}::{name}() returns bare tuples at {offenders}"
