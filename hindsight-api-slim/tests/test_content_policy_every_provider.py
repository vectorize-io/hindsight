"""Every LLM provider turns a content block into ``ProviderContentPolicyError`` (#6503).

A block is permanent, so the worker fails the operation once and HTTP answers 422.
A provider that raises anything else gets the whole retry schedule instead. That is
how the OpenAI-compatible family missed it while Gemini and Anthropic had it, so this
check runs over every provider module: a new one fails here until it handles blocks,
or it is added to the exemption list below with the reason.

It reads the source rather than importing it, because several providers need an
optional SDK that is not installed everywhere.
"""

import ast
from pathlib import Path

import hindsight_api.engine.providers as providers_pkg

PROVIDERS_DIR = Path(providers_pkg.__file__).parent

# Provider modules with no block to handle, each with its reason.
EXEMPT = {
    "mock_llm.py": "test double, never calls a model",
    "none_llm.py": "fails every call: there is no model behind it",
    "llamacpp_llm.py": "forwards every call to OpenAICompatibleLLM, which handles blocks",
}


def _classes() -> dict[str, tuple[Path, list[str]]]:
    """Every class defined in the providers package: name -> (file, base names)."""
    found: dict[str, tuple[Path, list[str]]] = {}
    for path in PROVIDERS_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases]
                found[node.name] = (path, bases)
    return found


def _files_in_chain(name: str, classes: dict[str, tuple[Path, list[str]]]) -> set[Path]:
    """The files that define ``name`` and every provider class it inherits from."""
    path, bases = classes[name]
    files = {path}
    for base in bases:
        if base in classes:
            files |= _files_in_chain(base, classes)
    return files


def test_every_provider_raises_content_policy_error_on_a_block():
    classes = _classes()
    missing = []
    for path in sorted(PROVIDERS_DIR.glob("*_llm.py")):
        if path.name in EXEMPT:
            continue
        providers = [n for n, (p, _) in classes.items() if p == path and n.endswith("LLM")]
        assert providers, f"{path.name} defines no provider class"
        chain = set().union(*(_files_in_chain(n, classes) for n in providers))
        if not any("ProviderContentPolicyError" in f.read_text() for f in chain):
            missing.append(path.name)
    assert not missing, f"providers that never raise ProviderContentPolicyError: {missing}"


def test_exemptions_name_real_modules():
    assert all((PROVIDERS_DIR / name).exists() for name in EXEMPT)
