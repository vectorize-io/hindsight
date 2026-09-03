"""Each MCP tool is registered twice; the two copies must not diverge.

Every tool is declared once with an explicit ``bank_id`` parameter (returning
JSON text) and once resolving the bank from the session (returning a dict).
FastMCP builds each tool's schema from the literal signature and docstring, so
those two declarations genuinely have to exist twice — but the *logic* does not,
and when it did, the copies drifted: the bank-id half built its error JSON by
string interpolation and emitted invalid JSON for any message containing a
double quote, while the session half returned a dict and was always correct.

These tests pin what must stay identical between the copies, and assert the
shared wrapper is actually used, so a future tool cannot quietly grow a second
implementation of the same behaviour.
"""

import ast
import pathlib

import pytest

MCP_TOOLS = pathlib.Path(__file__).resolve().parent.parent / "hindsight_api" / "mcp_tools.py"

#: Tools that legitimately keep bespoke bodies, and why. Named rather than
#: pattern-matched so adding a fifth is a deliberate act, not an accident.
BESPOKE = {
    # The two copies use different pydantic serializers -- model_dump_json() for
    # the JSON variant, model_dump() for the dict one -- which do not render
    # datetimes and enums identically. Routing both through the shared wrapper
    # would change the payload clients receive.
    "recall",
    "reflect",
    # Retain resolves its bank per content item and reports partial success, so
    # it has no single "resolve one bank, make one call" shape to share.
    "retain",
    "sync_retain",
}


def _tree():
    return ast.parse(MCP_TOOLS.read_text())


def _tool_pairs():
    pairs = {}
    for node in ast.walk(_tree()):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_register_"):
            continue
        by_name: dict[str, list] = {}
        for inner in ast.walk(node):
            if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) and inner is not node:
                by_name.setdefault(inner.name, []).append(inner)
        for name, fns in by_name.items():
            if len(fns) == 2:
                pairs[name] = sorted(fns, key=lambda f: f.lineno)
    return pairs


def _params(fn):
    return [
        (a.arg, ast.unparse(a.annotation) if a.annotation else None)
        for a in list(fn.args.args) + list(fn.args.kwonlyargs)
    ]


def test_the_tool_family_is_non_trivial():
    """Guard the guard: an empty family would make every assertion below vacuous."""
    assert len(_tool_pairs()) >= 30


@pytest.mark.parametrize("name", sorted(_tool_pairs()))
def test_both_copies_declare_the_same_parameters(name):
    """Beyond bank_id, the two copies must offer the caller the same surface.

    A parameter added to one copy only is invisible: each registration is
    exercised by whichever deployment mode uses it, so the mode nobody tested
    silently lacks the capability.
    """
    multi, single = _tool_pairs()[name]
    multi_params = [p for p in _params(multi) if p[0] != "bank_id"]
    assert multi_params == _params(single), (
        f"{name}: bank-id copy declares {multi_params}, session copy declares {_params(single)}"
    )


@pytest.mark.parametrize("name", sorted(_tool_pairs()))
def test_only_the_bank_id_copy_takes_bank_id(name):
    multi, single = _tool_pairs()[name]
    assert "bank_id" in dict(_params(multi)), f"{name}: bank-id copy is missing bank_id"
    assert "bank_id" not in dict(_params(single)), f"{name}: session copy should not take bank_id"


@pytest.mark.parametrize("name", sorted(_tool_pairs()))
def test_the_copies_share_one_implementation(name):
    """Both copies must delegate to _run_tool rather than re-implement the flow."""
    if name in BESPOKE:
        pytest.skip(f"{name} is a documented exception (see BESPOKE)")
    for fn in _tool_pairs()[name]:
        body = "\n".join(ast.unparse(s) for s in fn.body)
        assert "_run_tool" in body, (
            f"{name} (line {fn.lineno}) does not use the shared _run_tool wrapper; "
            f"if that is deliberate, add it to BESPOKE with the reason"
        )


def test_bespoke_list_has_no_stale_entries():
    """An exemption that no longer names a real tool hides a rule that stopped applying."""
    pairs = set(_tool_pairs())
    assert BESPOKE <= pairs, f"BESPOKE names tools that no longer exist: {sorted(BESPOKE - pairs)}"
