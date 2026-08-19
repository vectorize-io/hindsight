"""Correctness and latency suite for recall's temporal extraction.

One file, five sections, because they are all verifying the same claim: making
``search_dates`` fast did not change what it returns.

1. **Golden corpus** — 2,538 (query x reference-date) results snapshotted from the
   pre-optimisation implementation. This is the compatibility oracle: any change
   in any answer fails here.
2. **Pre-filter soundness** — ``_query_can_score`` skips dateparser entirely when
   no span could score. Verified *against the real search*, not just against the
   word lists, on the corpus plus fuzz.
3. **Language-detection equivalence** — ``temporal_language_detection`` is a
   hand-optimised copy of a dateparser internal. Verified differentially against
   dateparser's own implementation, so a future upgrade that changes detection
   semantics fails loudly instead of silently.
4. **Off-loop execution** — the async form must return identical results and keep
   the event loop scheduling.
5. **Latency gates** — CPU-time budgets (load-insensitive, run in CI) and
   wall-clock p99 under concurrency (``slow``, skipped under xdist).

The golden values record what the old implementation did, including answers that
are arguably wrong. Optimisation must not change any of them; genuine correctness
fixes belong in a separate, reviewable change.

Regenerate the golden file (only from a known-good tree) with::

    HS_REGEN_GOLDEN=1 uv run pytest tests/test_temporal_extraction.py

No database is touched: everything here is pure CPU.
"""

import asyncio
import json
import os
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from hindsight_api.engine.query_analyzer import (
    _MONTH_WORDS,
    _PERIOD_WORDS,
    _RELATIVE_WORDS,
    _SCOREABLE_WORDS,
    _WEEKDAY_WORDS,
    DateparserQueryAnalyzer,
    _date_match_score,
    _query_can_score,
)
from hindsight_api.engine.search.temporal_extraction import (
    _get_executor,
    extract_temporal_constraint,
    extract_temporal_constraint_async,
)
from hindsight_api.engine.temporal_language_detection import best_language
from tests.query_analyzer_bench import _default_fn, measure_burst, warmup
from tests.query_analyzer_corpus import (
    REFERENCE_DATE,
    build_corpus,
    build_document_workload,
    build_perf_workload,
    build_query_workload,
)

dateparser = pytest.importorskip("dateparser")

from dateparser.search import _search_with_detection  # noqa: E402
from dateparser.search.text_detection import FullTextLanguageDetector  # noqa: E402

ALL_LOCALES = list(_search_with_detection.available_language_map.values())
GOLDEN_PATH = Path(__file__).parent / "data" / "query_analyzer_golden.json"

# Several reference dates so relative expressions exercise different weekday,
# month-start and year-boundary branches.
REFERENCE_DATES = [
    datetime(2026, 8, 12, 14, 30, 45, 123456),  # Wednesday, mid-month
    datetime(2026, 1, 1, 0, 0, 0),  # New Year's Day (Thursday)
    datetime(2026, 12, 31, 23, 59, 59),  # year end
    datetime(2024, 2, 29, 12, 0, 0),  # leap day
    datetime(2026, 8, 3, 9, 0, 0),  # a Monday
    datetime(2026, 8, 9, 18, 0, 0),  # a Sunday
]


@pytest.fixture(scope="module")
def analyzer():
    a = DateparserQueryAnalyzer()
    a.load()
    return a


# ===========================================================================
# 1. Golden corpus — behaviour must not change
# ===========================================================================


def _key(query: str, ref: datetime) -> str:
    return f"{ref.isoformat()}\x1f{query}"


def _encode(result) -> list[str] | None:
    if result is None:
        return None
    start, end = result
    return [start.isoformat(), end.isoformat()]


def _compute_all() -> dict[str, list[str] | None]:
    """Run every corpus query against every reference date."""
    analyzer = DateparserQueryAnalyzer()
    out: dict[str, list[str] | None] = {}
    for query, _category in build_corpus():
        for ref in REFERENCE_DATES:
            out[_key(query, ref)] = _encode(extract_temporal_constraint(query, reference_date=ref, analyzer=analyzer))
    return out


@pytest.mark.skipif(not os.getenv("HS_REGEN_GOLDEN"), reason="regeneration is opt-in")
def test_regenerate_golden() -> None:
    """Write the golden file. Opt-in; never runs in CI."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _compute_all()
    with GOLDEN_PATH.open("w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print(f"\nWrote {len(data)} golden cases to {GOLDEN_PATH}")


@pytest.mark.skipif(bool(os.getenv("HS_REGEN_GOLDEN")), reason="regenerating")
def test_golden_corpus_unchanged() -> None:
    """Every corpus query must produce exactly the recorded constraint."""
    if not GOLDEN_PATH.exists():
        pytest.fail(f"Golden file missing: {GOLDEN_PATH}. Regenerate with HS_REGEN_GOLDEN=1 on a known-good tree.")
    with GOLDEN_PATH.open() as fh:
        golden = json.load(fh)
    actual = _compute_all()

    missing = sorted(set(golden) - set(actual))
    added = sorted(set(actual) - set(golden))
    assert not missing, f"{len(missing)} golden cases no longer produced, e.g. {missing[:5]}"
    assert not added, f"{len(added)} new corpus cases have no golden value (regenerate deliberately), e.g. {added[:5]}"

    diffs = []
    for key in sorted(golden):
        if golden[key] != actual[key]:
            ref, query = key.split("\x1f", 1)
            diffs.append(f"  ref={ref} query={query!r}\n    golden={golden[key]}\n    actual={actual[key]}")
    assert not diffs, f"{len(diffs)} behavioural changes:\n" + "\n".join(diffs[:25])


def test_corpus_is_deduplicated_and_complete() -> None:
    queries = [q for q, _ in build_corpus()]
    assert len(queries) == len(set(queries))
    assert {c for _, c in build_corpus()} == {
        "non_temporal_en",
        "tz_abbrev_trap",
        "false_positive_trap",
        "period_en",
        "period_es",
        "period_it",
        "period_fr",
        "period_de",
        "period_ru",
        "month_year",
        "period_zh",
        "cjk_embedded",
        "explicit_date",
        "long_text",
        "edge_case",
    }


def test_analyzer_and_wrapper_agree(analyzer) -> None:
    """extract_temporal_constraint must mirror analyze() exactly, minus the guard."""
    ref = REFERENCE_DATES[0]
    for query, _category in build_corpus():
        try:
            analysis = analyzer.analyze(query, ref)
        except Exception:
            # The wrapper degrades to None where analyze() raises (#3217).
            assert extract_temporal_constraint(query, reference_date=ref, analyzer=analyzer) is None
            continue
        expected = None
        if analysis.temporal_constraint:
            expected = (
                analysis.temporal_constraint.start_date,
                analysis.temporal_constraint.end_date,
            )
        assert extract_temporal_constraint(query, reference_date=ref, analyzer=analyzer) == expected, query


def test_language_restricted_analyzer_matches_on_fastpath(analyzer) -> None:
    """Pinning languages must not change results that never reach dateparser."""
    from tests.query_analyzer_corpus import PERIOD_EN, PERIOD_RU, PERIOD_ZH

    pinned = DateparserQueryAnalyzer(languages=["en", "es", "it", "fr", "de", "ru", "zh"])
    ref = REFERENCE_DATES[0]
    for query in PERIOD_EN + PERIOD_RU + PERIOD_ZH:
        assert analyzer.analyze(query, ref) == pinned.analyze(query, ref), query


# ===========================================================================
# 2. Pre-filter soundness — never reject a query that could produce a date
# ===========================================================================


def test_scoreable_words_cover_every_scoring_set() -> None:
    """The pre-filter alternation must be the union of the scorer's word sets.

    Adding a word to one and not the other would let the pre-filter reject a
    query the scorer would have accepted.
    """
    assert _SCOREABLE_WORDS == _MONTH_WORDS | _RELATIVE_WORDS | _WEEKDAY_WORDS | _PERIOD_WORDS


@pytest.mark.parametrize("word", sorted(_MONTH_WORDS | _RELATIVE_WORDS | _WEEKDAY_WORDS | _PERIOD_WORDS))
def test_every_scoring_word_passes_the_prefilter(word: str) -> None:
    """Any word the scorer rewards must reach dateparser, alone or in a sentence."""
    assert _date_match_score(word) > 0, "corpus assumption: this word scores"
    assert _query_can_score(word)
    assert _query_can_score(f"some text {word} more text")
    assert _query_can_score(word.upper())


def test_digits_always_pass_the_prefilter() -> None:
    for digit in string.digits:
        assert _query_can_score(f"note {digit} here")


def test_prefilter_rejects_plain_prose() -> None:
    """Sanity: the common recall case really is short-circuited."""
    for query in [
        "how does the reranker work",
        "user preferences for code style",
        "who owns the billing service",
        "上海的天气",
    ]:
        assert not _query_can_score(query), query


def _scores_something(analyzer: DateparserQueryAnalyzer, query: str) -> bool:
    """Run the real dateparser search and report whether anything scored."""
    from dateparser.conf import settings as ds

    try:
        results = analyzer._find_dates(query, settings=ds)
    except Exception:
        return False
    if not results:
        return False
    from hindsight_api.engine.temporal_periods import is_embedded_cjk_dateparser_match

    return any(
        _date_match_score(text) > 0 for text, _date in results if not is_embedded_cjk_dateparser_match(query, text)
    )


def _assert_prefilter_sound(analyzer: DateparserQueryAnalyzer, query: str) -> None:
    if _query_can_score(query):
        return  # slow path taken; nothing to prove
    assert not _scores_something(analyzer, query), (
        f"pre-filter rejected {query!r} but the real search found a scoring match"
    )


def test_prefilter_is_sound_on_corpus(analyzer) -> None:
    for query, _category in build_corpus():
        _assert_prefilter_sound(analyzer, query)


@pytest.mark.parametrize("seed", range(6))
def test_prefilter_is_sound_on_random_text(analyzer, seed: int) -> None:
    """Fuzz against the real search. Fixed seeds keep failures reproducible."""
    rng = random.Random(seed)
    alphabets = [
        string.ascii_lowercase + " ",
        string.ascii_letters + string.digits + " ",
        string.printable,
        "абвгдежзийклмнопрстуфхцчшщ ",
        "一二三四五六七八九十年月日上下周昨今明 ",
        "áéíóúñàèìòùäöüß ",
    ]
    for _ in range(120):
        alphabet = rng.choice(alphabets)
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        _assert_prefilter_sound(analyzer, text)


def test_prefilter_is_sound_on_word_salad(analyzer) -> None:
    """Real words are likelier than random noise to trip the search."""
    rng = random.Random(99)
    vocabulary = [
        "the", "user", "asked", "about", "schema", "vector", "index", "recall",
        "bank", "memory", "fact", "model", "graph", "entity", "link", "token",
        "mon", "tue", "sept", "jan", "dec", "wed", "sun", "sat", "may", "march",
        "monat", "semana", "settimana", "неделя", "周", "月", "日", "ieri", "ayer",
    ]  # fmt: skip
    for _ in range(600):
        text = " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 8)))
        _assert_prefilter_sound(analyzer, text)


# ===========================================================================
# 3. Language detection — differential against dateparser itself
# ===========================================================================


def _reference_language(text: str) -> str | None:
    """dateparser's own answer. A fresh detector each call: it mutates itself."""
    return FullTextLanguageDetector(ALL_LOCALES)._best_language(text)


def _assert_same_language(text: str) -> None:
    expected = _reference_language(text)
    actual = best_language(text, ALL_LOCALES)
    assert actual == expected, f"detector disagreement on {text!r}: ours={actual!r} dateparser={expected!r}"


def test_detection_agrees_on_full_corpus() -> None:
    for query, _category in build_corpus():
        _assert_same_language(query)


@pytest.mark.parametrize(
    "text",
    [
        "", " ", "\n", "0", "2026-06-10", "12/31/1999", "(1.2.3)", "-", "a", "ab",
        "ЖЖЖ", "日本語", "한국어", "ελληνικά", "עברית", "العربية", "ไทย", "हिन्दी",
        "ᏣᎳᎩ", "🧠🧠🧠", "café naïve", "EST", "12:00 EST",
        "meeting at 3pm EST on tuesday", "GMT+5", "UTC", "x" * 2000,
    ],
)  # fmt: skip
def test_detection_agrees_on_edge_cases(text: str) -> None:
    _assert_same_language(text)


def test_detection_agrees_on_symbol_only_strings() -> None:
    """The symbol-set shortcut path (returns the first locale unconditionally)."""
    for text in ["123", "1/2/3", "(1.2)", "12:30", "2026-06-10", "1,2", "-.:", "0 0 0"]:
        _assert_same_language(text)


def test_detection_agrees_on_timezone_bearing_text() -> None:
    """The strip_timezone retry path — the branch hoisted out of the loop."""
    for text in [
        "3pm EST",
        "meeting EST",
        "connection pooling",  # 'ect' matches the tz guard mid-word
        "what happened",  # 'hat' matches the tz guard mid-word
        "expected CAT WET MET",
        "2026-06-10 12:00 UTC",
        "east of the office",
    ]:
        _assert_same_language(text)


def _random_text(rng: random.Random) -> str:
    alphabets = [
        string.ascii_lowercase,
        string.ascii_letters + string.digits,
        string.printable,
        "абвгдежзийклмнопрстуфхцчшщъыьэюя",
        "一二三四五六七八九十年月日上下周昨今明",
        "あいうえおかきくけこ日本語",
        "αβγδεζηθικλμνξοπρστυφχψω",
        "ابتثجحخدذرزسشصضطظعغ",
        "0123456789:/-. ",
        "áéíóúñàèìòùäöüßçãõ",
    ]
    alphabet = rng.choice(alphabets)
    length = rng.choice([0, 1, 2, 3, 5, 8, 13, 40, 120])
    return "".join(rng.choice(alphabet) for _ in range(length))


@pytest.mark.parametrize("seed", range(8))
def test_detection_agrees_on_random_strings(seed: int) -> None:
    """Fuzz. Fixed seeds so a failure is reproducible."""
    rng = random.Random(seed)
    for _ in range(150):
        _assert_same_language(_random_text(rng))


def test_detection_agrees_on_random_mixed_scripts() -> None:
    """Mixed-script strings exercise the unique-character short-circuit."""
    rng = random.Random(1234)
    fragments = ["hello", "上周", "вчера", "こんにちは", "2026-06-10", "ayer", "EST", "ελληνικά", "🧠", "café"]
    for _ in range(400):
        n = rng.randint(1, 5)
        _assert_same_language(" ".join(rng.choice(fragments) for _ in range(n)))


def test_char_table_cache_is_shared_and_correct() -> None:
    """The cached character tables must match a freshly computed set."""
    from dateparser.conf import settings as ds

    from hindsight_api.engine.temporal_language_detection import _char_table_cache, _char_tables

    _char_table_cache.clear()
    first = _char_tables(ALL_LOCALES, ds)
    second = _char_tables(ALL_LOCALES, ds)
    assert first is second, "second call should hit the cache"

    detector = FullTextLanguageDetector(ALL_LOCALES)
    detector.get_unique_characters(ds.replace(NORMALIZE=False))
    assert first.language_chars == detector.language_chars
    assert first.unique_chars == detector.language_unique_chars


def test_subset_locale_lists_are_cached_separately() -> None:
    """A pinned language list must not reuse the full list's tables."""
    subset = [loc for loc in ALL_LOCALES if loc.shortname in {"en", "es", "it"}]
    for text in ["ayer", "hello world", "2026-06-10", "cosa ho fatto"]:
        expected = FullTextLanguageDetector(subset)._best_language(text)
        assert best_language(text, subset) == expected, text


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires 3.11+")
def test_detection_is_thread_safe() -> None:
    """Concurrent detection must not corrupt the shared character-table cache."""
    import concurrent.futures

    from hindsight_api.engine.temporal_language_detection import _char_table_cache

    _char_table_cache.clear()
    texts = [q for q, _ in build_corpus()][:80]
    expected = {t: _reference_language(t) for t in texts}

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda t: (t, best_language(t, ALL_LOCALES)), texts * 4))
    for text, got in results:
        assert got == expected[text], text


# ===========================================================================
# 4. Off-loop execution
# ===========================================================================


async def test_async_matches_sync_on_whole_corpus(analyzer) -> None:
    """Offloading changes where the work runs, never what it returns."""
    for query, _category in build_corpus():
        expected = extract_temporal_constraint(query, reference_date=REFERENCE_DATE, analyzer=analyzer)
        actual = await extract_temporal_constraint_async(query, reference_date=REFERENCE_DATE, analyzer=analyzer)
        assert actual == expected, query


async def test_executor_is_single_worker() -> None:
    """One worker on purpose.

    The work is pure Python and holds the GIL, so extra workers add no
    parallelism — they just contend. Measured at 16 concurrent document-sized
    extractions, widening the pool cost throughput for nothing: 1 worker
    1438ms, 2 workers 2091ms, 4 workers 4751ms, unbounded 16688ms.
    """
    assert _get_executor()._max_workers == 1


async def test_concurrent_extractions_keep_the_loop_scheduling(analyzer) -> None:
    """The loop must keep getting ticks while extractions are in flight.

    Inline, the same workload gave the loop a single tick in ~1.3s. The bound
    here is deliberately loose (wall-clock on a possibly-loaded machine); it is
    checking for "the loop still runs", not a precise latency.
    """
    slow_text = build_document_workload()[2]  # ~1.8 KB, the worst case
    ticks = 0
    stop = asyncio.Event()

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.001)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    ticks = 0

    await asyncio.gather(
        *[
            extract_temporal_constraint_async(slow_text, reference_date=REFERENCE_DATE, analyzer=analyzer)
            for _ in range(8)
        ]
    )
    stop.set()
    await beat

    assert ticks > 20, f"event loop only got {ticks} ticks during 8 concurrent extractions (inline gives ~1)"


async def test_offloaded_errors_still_degrade_to_none(analyzer, monkeypatch) -> None:
    """A crash inside the worker thread must not escape as a failed recall."""

    def boom(*args, **kwargs):
        raise IndexError("list index out of range")

    monkeypatch.setattr(analyzer, "analyze", boom)
    assert (
        await extract_temporal_constraint_async("on the 3rd", reference_date=REFERENCE_DATE, analyzer=analyzer) is None
    )


async def test_module_level_patching_still_reaches_the_async_form(monkeypatch) -> None:
    """Existing tests patch the module attribute; that seam must keep working.

    ``test_recall_pipeline_toggles`` and ``test_temporal_recall_selection`` both
    monkeypatch ``temporal_extraction.extract_temporal_constraint`` to drive the
    retrieval path. Now that retrieval calls the async form, the patch only takes
    effect if the sync function is resolved at call time rather than captured.
    """
    import hindsight_api.engine.search.temporal_extraction as module

    sentinel = (datetime(2025, 1, 1), datetime(2025, 2, 1))
    monkeypatch.setattr(module, "extract_temporal_constraint", lambda *a, **k: sentinel)

    assert await extract_temporal_constraint_async("anything at all") == sentinel


async def test_reference_date_and_analyzer_are_forwarded(analyzer) -> None:
    """Arguments must survive the hop into the worker thread."""
    ref = datetime(2020, 5, 17, 8, 0, 0)
    result = await extract_temporal_constraint_async("yesterday", reference_date=ref, analyzer=analyzer)
    assert result is not None
    assert result[0].date() == datetime(2020, 5, 16).date()


# ===========================================================================
# 5. Latency gates
# ===========================================================================

# Budgets are regression tripwires, not benchmarks. They are sized against the
# behaviour this work removed, with enough headroom for the slowest CI runner —
# a shared runner measured ~5x slower than a dev machine here (0.55s vs 2.73s for
# the corpus sweep), so a budget tuned to local timings flakes in CI.
#
# What each one would have caught, before the optimisation:
#   corpus sweep        ~55s CPU (423 queries x ~130ms)   -> budget 10s
#   non-temporal query  ~60ms CPU each                    -> budget 3ms
#   period fast path    unchanged at ~0.01ms, but a regression that let it fall
#                       through to dateparser costs ~6ms/query on CI runners
#                                                          -> budget 0.5ms
CORPUS_CPU_BUDGET_S = float(os.getenv("HS_PERF_CORPUS_BUDGET", "10.0"))
NON_TEMPORAL_CPU_BUDGET_MS = float(os.getenv("HS_PERF_NON_TEMPORAL_BUDGET", "3.0"))
PERIOD_FASTPATH_BUDGET_MS = float(os.getenv("HS_PERF_FASTPATH_BUDGET", "0.5"))
BURST_P99_BUDGET_MS = float(os.getenv("HS_PERF_BURST_P99_BUDGET", "20.0"))

# Wall-clock percentiles cannot be measured while other xdist workers saturate
# the CPU: the numbers reflect the runner's load, not this code. Skip rather than
# assert something meaningless (or, worse, flake).
requires_serial = pytest.mark.skipif(
    os.getenv("PYTEST_XDIST_WORKER") is not None,
    reason="wall-clock latency is not measurable under parallel test execution; run with -n 0",
)


@pytest.fixture(scope="module")
def timed_analyze():
    fn = _default_fn()
    warmup(fn, build_perf_workload()[:60])
    return fn


def test_whole_corpus_cpu_budget(timed_analyze) -> None:
    """Total CPU to analyse every corpus query stays within budget.

    Before this work the same sweep cost ~72 s of wall time; the guard here is
    against a regression of that magnitude, not against small drift.
    """
    corpus = build_corpus()
    start = time.process_time()
    for query, _category in corpus:
        timed_analyze(query, REFERENCE_DATE)
    elapsed = time.process_time() - start
    assert elapsed < CORPUS_CPU_BUDGET_S, (
        f"corpus sweep took {elapsed:.2f}s CPU for {len(corpus)} queries (budget {CORPUS_CPU_BUDGET_S}s)"
    )


def test_non_temporal_queries_are_short_circuited(timed_analyze) -> None:
    """The common recall case must not reach dateparser at all.

    A plain question with no date word and no digit is the single most frequent
    input recall sees, and it used to be the *slowest* (~60 ms) because every one
    of 205 locales ran to completion before concluding there was no date.
    """
    from tests.query_analyzer_corpus import NON_TEMPORAL_EN, TZ_ABBREV_TRAPS

    # Only queries the pre-filter actually rejects. Some entries in these lists
    # legitimately carry a digit (e.g. a URL with "?id=42") and must still take
    # the full search path, so they are not part of this claim.
    queries = [q for q in NON_TEMPORAL_EN + TZ_ABBREV_TRAPS if q.strip() and not _query_can_score(q)]
    assert len(queries) > 20, "expected the bulk of plain prose to be short-circuited"
    for _ in range(3):  # warm
        for q in queries:
            timed_analyze(q, REFERENCE_DATE)

    worst = 0.0
    for q in queries:
        start = time.process_time()
        timed_analyze(q, REFERENCE_DATE)
        worst = max(worst, (time.process_time() - start) * 1000)
    assert worst < NON_TEMPORAL_CPU_BUDGET_MS, f"slowest non-temporal query took {worst:.3f}ms CPU"


def test_period_fastpath_never_reaches_dateparser(timed_analyze) -> None:
    """Regex-resolved period expressions must stay in the microsecond range."""
    from tests.query_analyzer_corpus import PERIOD_DE, PERIOD_EN, PERIOD_ES, PERIOD_FR, PERIOD_IT, PERIOD_RU

    queries = PERIOD_EN + PERIOD_ES + PERIOD_IT + PERIOD_FR + PERIOD_DE + PERIOD_RU
    for _ in range(3):
        for q in queries:
            timed_analyze(q, REFERENCE_DATE)

    start = time.process_time()
    for q in queries:
        timed_analyze(q, REFERENCE_DATE)
    per_query_ms = (time.process_time() - start) * 1000 / len(queries)
    assert per_query_ms < PERIOD_FASTPATH_BUDGET_MS, f"period fast path averaged {per_query_ms:.4f}ms/query"


@requires_serial
@pytest.mark.slow
@pytest.mark.parametrize("concurrency", [32, 64, 128, 256])
def test_query_burst_p99_under_concurrency(timed_analyze, concurrency: int) -> None:
    """p99 < 20 ms with N simultaneous callers on query-shaped input.

    Recall invokes temporal extraction from an ``async def``, so concurrent
    callers serialise behind each other and a caller's latency includes that
    queueing. Wall-clock, hence ``slow``: a loaded runner inflates it.
    """
    stats = asyncio.run(measure_burst(timed_analyze, build_query_workload(), concurrency))
    p99 = stats.pct(99)
    assert p99 < BURST_P99_BUDGET_MS, (
        f"burst@{concurrency} p99={p99:.2f}ms exceeds {BURST_P99_BUDGET_MS}ms "
        f"(p50={stats.pct(50):.2f} p95={stats.pct(95):.2f} max={stats.max:.2f})"
    )


@requires_serial
@pytest.mark.slow
def test_document_shaped_input_is_the_known_remaining_tail(timed_analyze) -> None:
    """Document-shaped input still exceeds the budget, and that is known.

    Language detection is O(text length x 205 locales) and every locale runs to
    completion, so cost grows linearly with the input: ~230 characters is the
    point where a single call passes 20 ms. Queries are far below that; stored
    fact text (consolidation recalls) is not.

    This test documents the boundary rather than asserting a budget we do not
    meet. Getting past it means not running every locale, which needs its own
    equivalence proof; until then this pins where the cliff is.
    """
    stats = asyncio.run(measure_burst(timed_analyze, build_document_workload(), 32))
    assert stats.pct(99) > BURST_P99_BUDGET_MS, (
        "document-shaped input now meets the budget -- delete this test and "
        "fold LONG_TEXTS back into the gated workload"
    )
