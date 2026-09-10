"""The published client keeps the promises its own documentation makes.

Not a story about the server: a story about the surface almost every consumer
actually touches. A wrapper method that exists only synchronously is unusable
from an async framework — it routes through `loop.run_until_complete`, which
raises inside a running loop — so a gap here silently puts a whole feature area
out of reach for FastAPI, LangGraph and CrewAI callers.

Written over the whole family rather than as a list, so the next method added
without a twin fails here too (the structural-guard shape from the code-review
skill, §9a).
"""

from __future__ import annotations

import inspect

import pytest
from hindsight_client import Hindsight

pytestmark = pytest.mark.asyncio


# `close`/`aclose` manage the client's own connection pool rather than calling
# the API. Both halves are excluded, not just `close` — excluding one orphans the
# other, which then reads as a method missing *its* async twin.
_NOT_API_CALLS = {"close", "aclose"}


def _convenience_methods() -> list[str]:
    """Public methods on the wrapper that are not already the async half of a pair."""
    names = [
        name
        for name, _ in inspect.getmembers(Hindsight, callable)
        if not name.startswith("_") and name not in _NOT_API_CALLS
    ]
    async_variants = {name for name in names if name.startswith("a") and name[1:] in names}
    return sorted(name for name in names if name not in async_variants)


async def test_every_convenience_method_has_an_async_variant():
    """The wrapper's own docstring says so:

        "Every convenience method has an async counterpart prefixed with `a`
        ... **Prefer the async variants** whenever you are inside an async
        context (`async def`, event loops, frameworks like
        FastAPI/LangGraph/CrewAI)."

    27 of them did not, until #4221 — which put mental models, knowledge pages,
    directives and bank config out of reach of every async caller.
    """
    missing = [name for name in _convenience_methods() if not hasattr(Hindsight, f"a{name}")]

    assert missing == [], f"{len(missing)} convenience methods have no async variant: {missing}"
