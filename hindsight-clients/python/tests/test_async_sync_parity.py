"""Every public convenience method must come in a sync/async pair.

The class docstring promises that ``a<name>`` exists for every convenience
method, and it is not a nicety: the sync methods route through
``_run_async`` -> ``loop.run_until_complete``, which raises
``RuntimeError: This event loop is already running`` inside FastAPI/LangGraph/
CrewAI. A sync-only method is therefore unusable from exactly the contexts the
docstring points at (#4221). Guard the whole family so a new method cannot
reintroduce the gap.
"""

import inspect

import pytest

from hindsight_client import Hindsight


def _public_methods() -> dict[str, object]:
    return {
        name: attr for name, attr in vars(Hindsight).items() if not name.startswith("_") and inspect.isfunction(attr)
    }


def _pairs() -> list[tuple[str, str]]:
    methods = _public_methods()
    return sorted((name, "a" + name) for name in methods if "a" + name in methods)


def test_every_sync_convenience_method_has_an_async_twin():
    methods = _public_methods()
    async_twins = {"a" + name for name in methods if "a" + name in methods}
    missing = sorted(name for name in methods if name not in async_twins and "a" + name not in methods)
    assert missing == [], (
        f"public convenience methods without an a-prefixed async twin: {missing}. "
        "Add `async def a<name>` next to each (the sync method should forward to it via _run_async)."
    )


def test_the_pairs_are_actually_sync_and_async():
    assert _pairs(), "expected to find sync/async method pairs on Hindsight"
    for sync_name, async_name in _pairs():
        sync_fn = getattr(Hindsight, sync_name)
        async_fn = getattr(Hindsight, async_name)
        assert not inspect.iscoroutinefunction(sync_fn), f"{sync_name} should be sync"
        assert inspect.iscoroutinefunction(async_fn), f"{async_name} should be a coroutine function"


@pytest.mark.parametrize(("sync_name", "async_name"), _pairs())
def test_pair_signatures_match(sync_name: str, async_name: str):
    """The twins must accept the same arguments, or callers silently lose one."""
    sync_params = inspect.signature(getattr(Hindsight, sync_name)).parameters
    async_params = inspect.signature(getattr(Hindsight, async_name)).parameters
    assert list(sync_params) == list(async_params), f"{sync_name} and {async_name} take different arguments"
    for name, param in sync_params.items():
        other = async_params[name]
        assert param.kind == other.kind, f"{sync_name}/{async_name}: '{name}' differs in kind"
        assert param.default == other.default, f"{sync_name}/{async_name}: '{name}' differs in default"


async def test_async_twin_is_usable_from_a_running_event_loop(monkeypatch):
    """The repro from #4221: sync-only methods are unreachable under a live loop."""
    from unittest.mock import MagicMock

    captured: dict[str, object] = {}

    async def fake_create(bank_id, request, **kwargs):
        captured["bank_id"] = bank_id
        captured["request"] = request
        return MagicMock()

    client = Hindsight(base_url="http://example.invalid")
    monkeypatch.setattr(client.mental_models, "create_mental_model", fake_create)

    await client.acreate_mental_model(bank_id="alice", name="Alice housing", source_query="Where does Alice live?")

    assert captured["bank_id"] == "alice"
