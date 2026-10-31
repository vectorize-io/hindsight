"""No provider makes native tool calls: tool calling is one path, a response schema.

``LLMProvider.call_with_tools`` makes every tool turn through the provider's ``call`` with a
response schema (``providers/tools_via_schema.py``). Native tool calling — sending the
agent's tools with a ``tool_choice`` — is how each provider used to do it, and every one
had its own quirks: a server that parsed tool calls without enforcing ``tool_choice``, an
endpoint that took only "auto", a model that rejected reasoning effort next to tools. A
provider ``call_with_tools`` is that path coming back, so it fails here.

No provider sends tools for structured output either: Anthropic uses its native
``output_config``, and a backend without a response schema gets the schema in the prompt.
Passing ``tools=[]`` to turn an SDK's own built-in tools off (Claude Code, Copilot) is the
one shape allowed.
"""

import ast
from pathlib import Path

import hindsight_api

_ROOT = Path(hindsight_api.__file__).parent

#: Where ``call_with_tools`` may be defined: the wrapper that implements it once, the
#: failover chain that forwards to its members' wrappers, and the test double.
_ALLOWED = {
    ("engine/llm_wrapper.py", "LLMProvider"),
    ("engine/llm_wrapper.py", "ConfiguredLLMProvider"),
    ("engine/multi_llm.py", "MultiLLMProvider"),
    ("engine/providers/mock_llm.py", "MockLLM"),
}


def _definitions() -> set[tuple[str, str]]:
    found = set()
    for path in _ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for item in cls.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == "call_with_tools":
                    found.add((path.relative_to(_ROOT).as_posix(), cls.name))
    return found


def test_only_the_wrapper_makes_tool_calls():
    unexpected = _definitions() - _ALLOWED
    assert not unexpected, (
        f"call_with_tools defined in {sorted(unexpected)}: tool turns go through LLMProvider.call_with_tools "
        "and the provider's call with a response schema, never a provider's own tool calling"
    )


#: Request keys and keyword arguments that hand a model tools to call.
_TOOL_FIELDS = {"tools", "tool_choice", "functions", "function_call", "tool_config"}


def _is_empty(node: ast.expr) -> bool:
    return isinstance(node, ast.List | ast.Tuple) and not node.elts


def test_no_provider_sends_tools():
    found = []
    for path in sorted((_ROOT / "engine" / "providers").glob("*.py")):
        if path.name == "mock_llm.py":
            continue  # the test double: it records the tool names it is handed, sends nothing
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Dict):
                pairs = zip(node.keys, node.values)
            elif isinstance(node, ast.Call):
                pairs = ((ast.Constant(k.arg), k.value) for k in node.keywords if k.arg)
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                pairs = [(node.slice, ast.Constant(None))]
            else:
                continue
            for key, value in pairs:
                if isinstance(key, ast.Constant) and key.value in _TOOL_FIELDS and not _is_empty(value):
                    found.append(f"{path.name}:{node.lineno} {key.value}")
    assert not found, f"tools sent to a model: {found}"


def test_the_allowlist_is_not_stale():
    assert _ALLOWED <= _definitions()
