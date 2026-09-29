"""A provider must accept a capability's parameters only if it implements the capability.

Prompt caching is opt-in: ``supports_prompt_caching()`` defaults False and
``get_or_create_cached_prefix()`` defaults to returning None, so callers only
ever pass ``cached_prefix=`` to a provider that overrode them. Two providers had
copied the parameter out of the interface signature without any of the
behaviour -- they declared ``cached_prefix``/``cached_prefix_message_count``,
never referenced them, and could never be handed one. Dead surface that reads
like a supported feature.

This asserts over the whole provider family rather than any one provider,
because the failure mode is a member that *omits* something (or, here, keeps
something it shouldn't) and the member that forgot is by construction the one
without a test.
"""

import ast
import pathlib

import pytest

PROVIDERS_DIR = pathlib.Path(__file__).resolve().parent.parent / "hindsight_api" / "engine" / "providers"

CACHE_PARAMS = {"cached_prefix", "cached_prefix_message_count"}
#: Overriding any of these is what "implements prompt caching" means.
CACHE_METHODS = {
    "get_or_create_cached_prefix",
    "create_incremental_cache",
    "supports_prompt_caching",
    "supports_incremental_prompt_cache",
}


def _provider_modules():
    return sorted(p for p in PROVIDERS_DIR.glob("*_llm.py"))


def _classes(path):
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def test_the_provider_family_is_non_trivial():
    """Guard the guard: an empty family would make every assertion below vacuous."""
    assert len(_provider_modules()) >= 10


@pytest.mark.parametrize("path", _provider_modules(), ids=lambda p: p.stem)
def test_cache_parameters_only_where_caching_is_implemented(path):
    for cls in _classes(path):
        methods = {b.name: b for b in cls.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
        implements = bool(CACHE_METHODS & set(methods))
        for name in ("call", "call_with_tools"):
            fn = methods.get(name)
            if fn is None:
                continue
            params = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
            declared = CACHE_PARAMS & params
            if declared and not implements:
                pytest.fail(
                    f"{path.name}::{cls.name}.{name}() declares {sorted(declared)} but the class "
                    f"overrides none of {sorted(CACHE_METHODS)} — callers gate on "
                    f"get_or_create_cached_prefix() returning non-None, so it can never be passed one"
                )


@pytest.mark.parametrize("path", _provider_modules(), ids=lambda p: p.stem)
def test_declared_cache_parameters_are_actually_used(path):
    """A parameter present in the signature must be read somewhere in the body."""
    for cls in _classes(path):
        for b in cls.body:
            if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {a.arg for a in list(b.args.args) + list(b.args.kwonlyargs)}
            for param in CACHE_PARAMS & params:
                loads = [
                    n for n in ast.walk(b) if isinstance(n, ast.Name) and n.id == param and isinstance(n.ctx, ast.Load)
                ]
                assert loads, f"{path.name}::{cls.name}.{b.name}() declares {param} but never reads it"
