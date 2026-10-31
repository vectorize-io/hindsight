"""The engine must not depend on the API layer (#5964).

Bank templates, the mental-model trigger and page rendering used to live in
``hindsight_api.api``, so ``MemoryEngine`` reached them through imports inside
functions to dodge an import cycle. They live in the engine now. This fails if any
engine module imports from ``hindsight_api.api`` again, at any depth.
"""

import ast
from pathlib import Path

import hindsight_api.engine

ENGINE_DIR = Path(hindsight_api.engine.__file__).parent


def _api_imports(path: Path) -> list[str]:
    package = ".".join(path.relative_to(ENGINE_DIR.parent.parent).with_suffix("").parts[:-1])
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            names = [module] + [f"{module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        if any(name == "hindsight_api.api" or name.startswith("hindsight_api.api.") for name in names):
            found.append(f"{path.relative_to(ENGINE_DIR)}:{node.lineno}")
    return found


def test_engine_does_not_import_the_api_layer():
    offenders = [hit for path in sorted(ENGINE_DIR.rglob("*.py")) for hit in _api_imports(path)]
    assert not offenders, offenders
