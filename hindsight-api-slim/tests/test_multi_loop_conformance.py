"""Process-wide state must work from more than one event loop.

Hindsight can be run with an event loop per thread in a single process (uvicorn on
free-threaded CPython, `--target api-only-freethreaded`). Anything shared between
those loops must therefore not be tied to one of them.

**These tests need no free-threaded build.** `asyncio.Lock`/`Semaphore`/`Future` bind
to the loop that first waits on them regardless of the GIL, so two loops in two
threads reproduce the whole failure class on 3.11 — which is the point: this file is
the cheap guard that runs in the normal suite, so the expensive free-threaded CI job
is only needed for what genuinely requires it (a C extension silently re-enabling
the GIL, and true-parallelism races).

Every case below corresponds to a bug that shipped and was found only by running the
server with eight loops. Each failed with either
``RuntimeError: ... is bound to a different event loop`` or
``RuntimeError: dictionary changed size during iteration``.
"""

import asyncio
import subprocess
import sys
import threading
from typing import Any, Callable

import pytest

LOOPS = 4
PER_LOOP = 6

# Run in a subprocess by test_query_analyzers_can_warm_up_concurrently, which needs
# both a cold interpreter and a survivable crash. See that test for why.
_WARMUP_PROBE = """
import threading
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer

QUERIES = [
    "what did I do yesterday about the deploy",
    "the meeting on 2026-06-10 with the team",
    "notes from last March about the migration",
]
errors = []


def warm_and_analyze():
    # A fresh analyzer per thread, exactly as one MemoryEngine per event loop.
    analyzer = DateparserQueryAnalyzer()
    try:
        analyzer.load()
        for query in QUERIES:
            analyzer.analyze(query)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent verbatim
        errors.append(f"{type(exc).__name__}: {exc}")


threads = [threading.Thread(target=warm_and_analyze) for _ in range(8)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

print("ERRORS " + repr(errors))
print("OK")
"""


def _across_loops(make_coros: Callable[[], list], loops: int = LOOPS) -> list[BaseException]:
    """Run `make_coros()` concurrently on `loops` independent event loops.

    Returns the exceptions raised, so a caller can assert on emptiness and print
    something useful when it is not.
    """
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run() -> None:
        async def main() -> Any:
            return await asyncio.gather(*(fn() for fn in make_coros()))

        try:
            asyncio.run(main())
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(loops)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_bank_info_cache_is_usable_from_several_loops():
    """Regression: the cache held an asyncio.Lock and coalesced behind an asyncio.Future."""
    from hindsight_api.engine.bank_stats_cache import BankStatsCache

    cache = BankStatsCache(ttl_seconds=60, max_entries=128)
    loads = 0
    guard = threading.Lock()

    async def loader() -> dict[str, Any]:
        nonlocal loads
        with guard:
            loads += 1
        await asyncio.sleep(0.01)  # widen the window so callers really do coalesce
        return {"value": 1}

    async def read() -> dict[str, Any]:
        return await cache.get_or_load("schema", "bank", loader)

    errors = _across_loops(lambda: [read] * PER_LOOP)
    assert not errors, f"cache failed across loops: {errors[:1]}"
    # The cached data is shared, so far fewer loads than callers.
    assert loads < LOOPS * PER_LOOP


def test_connection_budget_manager_is_usable_from_several_loops():
    """Regression: ConnectionBudgetManager is a process-wide singleton with a lock."""
    from hindsight_api.engine.db_budget import ConnectionBudgetManager

    manager = ConnectionBudgetManager(default_budget=2)

    async def use() -> None:
        async with manager.operation(max_connections=2):
            await asyncio.sleep(0.005)

    errors = _across_loops(lambda: [use] * PER_LOOP)
    assert not errors, f"budget manager failed across loops: {errors[:1]}"


def test_llm_concurrency_permits_are_usable_from_several_loops():
    """Regression: the LLM caps were module-level asyncio.Semaphores built at import."""
    from hindsight_api.engine.llm_wrapper import _semaphores_for_scope

    async def hold() -> None:
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            for sem in _semaphores_for_scope("retain"):
                await stack.enter_async_context(sem)
            await asyncio.sleep(0.005)

    errors = _across_loops(lambda: [hold] * PER_LOOP)
    assert not errors, f"LLM permits failed across loops: {errors[:1]}"


def test_temporal_language_detection_is_usable_from_several_threads():
    """Regression: dateparser cleans a cached locale dict in place on first use.

    Two threads reaching a locale together raised "dictionary changed size during
    iteration", which killed the lifespan of whichever loops lost the race.
    """
    from dateparser.languages.loader import LocaleDataLoader

    from hindsight_api.engine.temporal_language_detection import best_language

    locales = list(LocaleDataLoader().get_locales(languages=None, locales=None, region=None))[:40]

    # Vary the text so different character sets select different candidate locales,
    # which is what left some dictionaries cold when only English was exercised.
    texts = [
        "what happened last friday",
        "cosa e successo ieri sera",
        "was ist gestern passiert",
        "que paso el martes pasado",
    ]

    async def detect() -> None:
        for text in texts:
            best_language(text, locales)

    errors = _across_loops(lambda: [detect] * 2, loops=6)
    assert not errors, f"language detection failed across threads: {errors[:1]}"


def test_locale_dictionaries_are_never_built_by_two_threads_at_once():
    """Regression: `_char_tables` built its result *before* taking the lock.

    Only the assignment was guarded, so N threads missing the cache together each
    swept all 200+ locales concurrently. `get_wordchars_for_detection` is not a
    read: on a miss it builds the locale's dictionary under a fresh `Settings`
    (hence a fresh `registry_key`) and writes it into `Locale.dictionaries` and
    dateparser's class-level regex caches. Missing together is the normal case —
    it is what N engines warming their analyzers on the startup executor do.

    On 3.11 that surfaced as the "dictionary changed size during iteration" above;
    on 3.14t it was a SIGSEGV inside the `regex` extension, which is handed
    borrowed references into a dict another thread is resizing. This asserts the
    invariant rather than waiting for either symptom, so it fails deterministically
    on both interpreters.
    """
    from dateparser.languages.loader import LocaleDataLoader

    import hindsight_api.engine.temporal_language_detection as tld

    # A private loader, so the locales — and therefore their dictionaries — are
    # cold whatever an earlier test in this worker already parsed.
    locales = list(LocaleDataLoader().get_locales(languages=None, locales=None, region=None))[:40]
    tld._char_table_cache.clear()

    counter_lock = threading.Lock()
    inside = 0
    peak = 0
    original = type(locales[0]).get_wordchars_for_detection

    def counting_get_wordchars(self, settings=None):
        nonlocal inside, peak
        with counter_lock:
            inside += 1
            peak = max(peak, inside)
        try:
            return original(self, settings=settings)
        finally:
            with counter_lock:
                inside -= 1

    async def detect() -> None:
        for text in ("what happened last friday", "cosa e successo ieri sera"):
            tld.best_language(text, locales)

    type(locales[0]).get_wordchars_for_detection = counting_get_wordchars
    try:
        errors = _across_loops(lambda: [detect], loops=8)
    finally:
        type(locales[0]).get_wordchars_for_detection = original
        tld._char_table_cache.clear()

    assert not errors, f"language detection failed across threads: {errors[:1]}"
    assert peak == 1, f"{peak} threads built locale dictionaries concurrently; that corrupts them"


def test_query_analyzers_can_warm_up_concurrently():
    """Regression: N engines warming their analyzers at once segfaulted 3.14t.

    `MemoryEngine.initialize` warms its analyzer with `run_in_executor(None, ...)`,
    so a process with an event loop per thread warms N analyzers on the *unbounded*
    default executor simultaneously, each entering dateparser's unguarded lazy
    caches. Six loops killed the interpreter within a minute.

    A subprocess because the failure it guards is a fatal signal — there is no
    exception left to catch — and because the caches have to be cold, which they
    are not in a pytest worker that has already parsed a date.
    """
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _WARMUP_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"concurrent analyzer warm-up crashed (exit {result.returncode}; "
        f"a negative status is a fatal signal):\n{result.stderr}"
    )
    assert "OK" in result.stdout, f"probe did not finish:\n{result.stdout}\n{result.stderr}"
    assert "ERRORS []" in result.stdout, f"threads raised:\n{result.stdout}"


@pytest.mark.asyncio
async def test_asyncio_primitives_still_bind_to_a_loop():
    """Pins the premise. If this ever stops failing, the guards above can be revisited."""
    shared = asyncio.Semaphore(1)
    async with shared:
        pass

    errors = _across_loops(lambda: [lambda: _contend(shared)] * 3, loops=3)
    assert any(isinstance(e, RuntimeError) and "different event loop" in str(e) for e in errors), (
        "a module-level asyncio.Semaphore no longer breaks across loops"
    )


async def _contend(sem: asyncio.Semaphore) -> None:
    async with sem:
        await asyncio.sleep(0.02)
