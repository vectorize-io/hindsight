"""What one knowledge-page refresh costs, measured in the LLM calls it makes.

The quality evals ask whether a refresh writes the right page. This asks what it
paid to write it: every LLM call the refresh made, with its input, cached and
output tokens, and the prompt it sent. A refresh that answers correctly off a
90k-token prompt is correct and still broken — it misses a 30s call deadline,
takes a minute, and on a metered key it is the bill (#4532, #4566, #4568).

Two things make the numbers mean something across runs:

* **The bank is frozen.** Tool payloads only grow past their budgets on a big bank
  — thousands of facts, hundreds of observations — and seeding that live would
  cost a consolidation pass per run and start every run from a different bank.
  So ``build`` makes the bank once (real consolidation, real pages), exports it,
  and the archive is committed. The eval imports it: same facts, same
  observations, same pages, every run. Regenerate it only on purpose, and say so
  in the commit, because every baseline taken before is then against a different
  bank.
* **The corpus is a project, not filler.** It is generated from a seed as the
  history of one fictional product — decisions, bugs, releases, incidents,
  benchmarks, owners, conventions — which is the shape of the coding-agent banks
  the reports came from. Every page's topic is densely present, so retrieval
  fills its budget instead of stopping early for lack of matches.

Rebuild the archive (needs a real model; takes a while — consolidation is the
point). Name the database so a failure late in the build does not throw the
consolidation away with the server::

    HINDSIGHT_EVAL_PG0_INSTANCE=refresh-cost-build uv run python -m hindsight_system_evals.refresh_cost build
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import random
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hindsight_client import Hindsight
from hindsight_client_api.api.llm_traces_api import LLMTracesApi
from pydantic import BaseModel

#: Overridable so a rebuilt archive can be measured before it replaces the committed one.
FIXTURE = Path(
    os.getenv("HINDSIGHT_EVAL_REFRESH_COST_FIXTURE")
    or Path(__file__).resolve().parents[1] / "fixtures" / "refresh-cost-bank.zip"
)

#: The seed IS the corpus. Changing it (or any template below) changes the bank,
#: which only matters when the archive is rebuilt — the eval reads the archive.
SEED = 4566
FACT_COUNT = 900

#: The pages a coding agent seeds on a project bank, with its page size.
PAGE_MAX_TOKENS = 4096
PAGES: dict[str, str] = {
    "Architecture overview": "How is the Lumen system put together: its components, what each one does, and how they talk to each other?",
    "Decisions and rationale": "Which design decisions were made on Lumen, when, by whom, and why?",
    "Known bugs and fixes": "Which bugs were found in Lumen, what caused them, and how and when were they fixed?",
    "Release history": "Which Lumen releases shipped, when, and what did each one contain?",
    "Performance characteristics": "What are Lumen's measured performance characteristics, per component, and how have they changed?",
    "Operations and incidents": "Which production incidents did Lumen have, what was the impact, and how were they mitigated?",
}

_COMPONENTS = [
    "ingestion gateway",
    "query planner",
    "storage engine",
    "billing service",
    "auth service",
    "web console",
    "CLI",
    "job scheduler",
    "search indexer",
    "notification service",
]
_PEOPLE = [
    "Priya Raman",
    "Marco Bellini",
    "Aiko Tanaka",
    "Samuel Okafor",
    "Lena Fischer",
    "Diego Alvarez",
    "Nora Lindqvist",
]
_REGIONS = ["eu-west-1", "us-east-1", "ap-southeast-2"]
_CHOICES = [
    ("batch writes in 5 MB segments", "streaming single rows", "segment writes cut fsync calls by 90%"),
    ("use Postgres advisory locks", "a Redis lock service", "it removes one moving part from the deploy"),
    ("retry with exponential backoff capped at 30 s", "fixed 1 s retries", "fixed retries amplified the March outage"),
    ("store timestamps as UTC microseconds", "local time strings", "daylight-saving shifts corrupted ordering"),
    ("paginate with opaque cursors", "offset pagination", "offsets skipped rows under concurrent inserts"),
    (
        "validate payloads with JSON Schema at the edge",
        "validating deep in the handlers",
        "bad payloads reached the database",
    ),
    ("cache plans for 10 minutes", "re-planning every query", "planning was 40% of p50 latency"),
    ("shard by tenant id", "shard by time", "time shards made per-tenant deletes scan every shard"),
    ("emit OpenTelemetry spans", "custom log lines", "traces could not be joined across services"),
    (
        "run migrations online with a shadow table",
        "locking the table",
        "a locking migration froze writes for 12 minutes",
    ),
]
_SYMPTOMS = [
    ("dropped events under backpressure", "an unbounded queue was replaced by a bounded one without a retry path"),
    ("returned stale results after a schema change", "the plan cache key ignored the schema version"),
    ("double-charged customers on retry", "the idempotency key was generated after the first attempt"),
    ("leaked file descriptors on cancelled uploads", "the cancel path skipped the cleanup handler"),
    ("rejected valid tokens near expiry", "clock skew between nodes exceeded the 5 s leeway"),
    ("rendered the dashboard blank in Safari", "a regex lookbehind unsupported by older WebKit"),
    ("hung on large exports", "the export streamed through a single 64 KB buffer"),
    ("skipped scheduled jobs at month end", "the cron parser mishandled day 31"),
    ("indexed deleted documents", "the tombstone check ran before the delete committed"),
    ("sent duplicate notifications", "two workers claimed the same outbox row"),
]
_FEATURES = [
    "tenant-scoped rate limits",
    "a new cost-based join ordering",
    "compressed column segments",
    "usage-based invoices",
    "SSO via SAML",
    "a dark-mode console",
    "shell completion",
    "priority job queues",
    "typo-tolerant search",
    "webhook retries with signing",
]
_METRICS = [
    ("p99 write latency", "ms", (8, 90)),
    ("sustained throughput", "events/s", (20_000, 400_000)),
    ("p50 query latency", "ms", (3, 60)),
    ("memory per node", "GB", (2, 48)),
    ("cold-start time", "s", (1, 25)),
    ("index build time for 10M documents", "min", (4, 70)),
]
_SETTINGS = [
    ("MAX_BATCH_BYTES", "5242880", "1048576", "larger segments cut fsync calls"),
    ("PLAN_CACHE_TTL", "600", "0", "planning was 40% of p50 latency"),
    ("TOKEN_LEEWAY_SECONDS", "30", "5", "clock skew rejected valid tokens"),
    ("WORKER_CONCURRENCY", "16", "4", "the queue backed up during nightly imports"),
    ("EXPORT_BUFFER_KB", "1024", "64", "large exports hung on the small buffer"),
    ("OUTBOX_CLAIM_BATCH", "50", "200", "big claims let two workers race for the same rows"),
]
_RULES = [
    "never log request bodies, only their size and hash",
    "wrap every external call in a timeout of at most 10 seconds",
    "add a regression test that fails on the unfixed code for every bug fix",
    "keep public API responses backwards compatible for two minor releases",
    "express every duration in seconds in configuration",
    "record every schema change as a forward-only migration",
]


def _date(rng: random.Random) -> dt.date:
    return dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(0, 260))


def _fmt(day: dt.date) -> str:
    return f"{day.day} {day.strftime('%B %Y')}"


def _version(rng: random.Random) -> str:
    return f"{rng.randint(1, 3)}.{rng.randint(0, 14)}.{rng.randint(0, 9)}"


def _fact(rng: random.Random, n: int) -> tuple[str, dt.date]:
    """One fact about the project, and the day it happened."""
    day, component, person = _date(rng), rng.choice(_COMPONENTS), rng.choice(_PEOPLE)
    kind = rng.choices(
        ["decision", "bug", "release", "perf", "setting", "incident", "owner", "rule"],
        weights=[20, 22, 12, 16, 10, 12, 4, 4],
    )[0]
    if kind == "decision":
        choice, alt, why = rng.choice(_CHOICES)
        text = f"On {_fmt(day)}, {person} decided the {component} should {choice} instead of {alt}, because {why}."
    elif kind == "bug":
        symptom, cause = rng.choice(_SYMPTOMS)
        text = (
            f"Bug LUM-{1000 + n}: the {component} {symptom}. Root cause: {cause}. "
            f"{person} fixed it in release {_version(rng)} on {_fmt(day)}."
        )
    elif kind == "release":
        text = f"Lumen release {_version(rng)} shipped on {_fmt(day)} with {rng.choice(_FEATURES)} in the {component}."
    elif kind == "perf":
        metric, unit, (low, high) = rng.choice(_METRICS)
        text = (
            f"A benchmark on {_fmt(day)} measured the {component} {metric} at {rng.randint(low, high)} {unit} "
            f"on a {rng.choice([3, 5, 8, 12])}-node cluster in {rng.choice(_REGIONS)}."
        )
    elif kind == "setting":
        name, new, old, why = rng.choice(_SETTINGS)
        text = f"{person} changed the {component} setting {name} from {old} to {new} on {_fmt(day)}, because {why}."
    elif kind == "incident":
        symptom, _ = rng.choice(_SYMPTOMS)
        choice, _, _ = rng.choice(_CHOICES)
        text = (
            f"Incident on {_fmt(day)}: the {component} {symptom} for {rng.randint(4, 180)} minutes in "
            f"{rng.choice(_REGIONS)}; {person} mitigated it by switching to {choice}."
        )
    elif kind == "owner":
        text = f"As of {_fmt(day)}, {person} owns the {component} and reviews every change to it."
    else:
        text = f"Convention adopted on {_fmt(day)}: code in the {component} must {rng.choice(_RULES)}."
    return text, day


def corpus(count: int = FACT_COUNT, seed: int = SEED) -> list[dict[str, Any]]:
    """The retain items, oldest first, deduplicated — each a fact and its timestamp."""
    rng = random.Random(seed)
    seen: dict[str, dt.date] = {}
    n = 0
    while len(seen) < count:
        text, day = _fact(rng, n)
        seen.setdefault(text, day)
        n += 1
    ordered = sorted(seen.items(), key=lambda item: item[1])
    return [{"content": text, "timestamp": f"{day.isoformat()}T12:00:00Z"} for text, day in ordered]


def new_wave(count: int = 20, seed: int = SEED + 1) -> list[dict[str, Any]]:
    """Facts that arrive after the snapshot — what a delta refresh has to fold in."""
    rng = random.Random(seed)
    items = []
    for n in range(count):
        text, _ = _fact(rng, 9000 + n)
        items.append({"content": text, "timestamp": "2026-09-20T12:00:00Z"})
    return items


# ── measuring one refresh ─────────────────────────────────────────────────────


class LLMCall(BaseModel):
    """One LLM call a refresh made, as the server traced it."""

    scope: str
    operation: str | None
    status: str
    started_at: str
    duration_ms: int | None
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    prompt: Any = None
    response: Any = None


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens, and per 1M tokens stored for an hour (explicit caches)."""

    input: float
    cached_input: float
    output: float
    cache_storage_hour: float


#: List prices, paid tier (ai.google.dev/gemini-api/docs/pricing, September 2026).
#: Explicit cache CREATION is billed at the input rate, plus storage per token-hour;
#: implicit cache hits pay only the cached rate.
PRICES: dict[str, Price] = {
    "gemini-2.5-flash-lite": Price(input=0.10, cached_input=0.01, output=0.40, cache_storage_hour=1.00),
    "gemini-2.5-flash": Price(input=0.30, cached_input=0.03, output=2.50, cache_storage_hour=1.00),
    # 3.x flash, promotional rates through 2026-12-31 (they double on 2027-01-01).
    "gemini-3.8-flash": Price(input=0.75, cached_input=0.075, output=3.75, cache_storage_hour=0.50),
    "gemini-3.7-flash": Price(input=0.75, cached_input=0.075, output=3.75, cache_storage_hour=0.50),
    "gemini-3.6-flash": Price(input=0.75, cached_input=0.075, output=3.75, cache_storage_hour=0.50),
    "gemini-3.5-flash": Price(input=1.50, cached_input=0.15, output=9.00, cache_storage_hour=1.00),
}


class CostBreakdown(BaseModel):
    """USD for one refresh (or a sum of them), by what was billed."""

    input: float = 0.0
    cached_read: float = 0.0
    output: float = 0.0
    cache_create: float = 0.0
    cache_storage: float = 0.0

    @property
    def total(self) -> float:
        return self.input + self.cached_read + self.output + self.cache_create + self.cache_storage

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(**{k: getattr(self, k) + getattr(other, k) for k in CostBreakdown.model_fields})


class ScopeUsage(BaseModel):
    """Token usage of every call sharing one trace scope."""

    calls: int = 0
    input: int = 0
    cached: int = 0
    output: int = 0

    def __add__(self, other: ScopeUsage) -> ScopeUsage:
        return ScopeUsage(**{k: getattr(self, k) + getattr(other, k) for k in ScopeUsage.model_fields})


class RefreshCost(BaseModel):
    page: str
    mode: str
    seconds: float
    #: ``completed`` or ``failed``. A failed refresh is measured, not skipped: it
    #: paid for every call it made, and a loop of them is the bill in #4532.
    status: str
    #: How the refresh ended (``content_written``, ``delta_ops_all_skipped``, ...),
    #: or the error of the failed attempt.
    outcome: str
    calls: list[LLMCall]
    #: What the refresh wrote, for a quality floor under every cost cut: a change
    #: that halves the bill by writing half a page is not a saving.
    page_chars: int = 0
    #: Distinct concrete specifics on the page (bug ids, versions, dates) — the
    #: facts a thinner prompt would drop first.
    specifics: int = 0
    #: Memories the page cites as its evidence.
    cited: int = 0

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def largest_call(self) -> LLMCall | None:
        return max(self.calls, key=lambda c: c.input_tokens, default=None)

    @property
    def final_call(self) -> LLMCall | None:
        """The synthesis call — the reflect answer the page is written from.

        The last call scoped ``reflect`` (the tool-loop turns are
        ``reflect_tool_call``; a split synthesis adds map calls before it). A delta
        refresh makes its edit call after it, but that call's prompt is the page
        plus the answer, not the evidence, and it is reported in the per-scope
        table on its own.
        """
        reflect = [c for c in self.calls if c.scope == "reflect"]
        return max(reflect, key=lambda c: c.started_at, default=None)

    def cost(self, price: Price, *, explicit_cache: bool) -> CostBreakdown:
        """USD by component.

        With explicit (Gemini ``CachedContent``) caching, every cached token was
        first written by a cache create billed at the INPUT rate, and stored from
        then until the reflect tore its caches down. Each rolling cache is read by
        one call, so a call's cached tokens approximate the cache it read: its size
        is the creation bill, and its lifetime runs from that call to the end of
        the refresh. Estimated, because creates are not LLM calls and never reach
        the trace. With implicit caching the hit is the whole story.
        """
        m = 1_000_000
        uncached = sum(c.input_tokens - c.cached_tokens for c in self.calls)
        out = CostBreakdown(
            input=uncached * price.input / m,
            cached_read=self.cached_tokens * price.cached_input / m,
            output=self.output_tokens * price.output / m,
        )
        if explicit_cache and self.calls:
            end = max(
                dt.datetime.fromisoformat(c.started_at).timestamp() + (c.duration_ms or 0) / 1000 for c in self.calls
            )
            out.cache_create = self.cached_tokens * price.input / m
            out.cache_storage = (
                sum(
                    c.cached_tokens * max(0.0, end - dt.datetime.fromisoformat(c.started_at).timestamp()) / 3600
                    for c in self.calls
                )
                * price.cache_storage_hour
                / m
            )
        return out

    def by_scope(self) -> dict[str, ScopeUsage]:
        table: dict[str, ScopeUsage] = {}
        for c in self.calls:
            one = ScopeUsage(calls=1, input=c.input_tokens, cached=c.cached_tokens, output=c.output_tokens)
            table[c.scope] = table.get(c.scope, ScopeUsage()) + one
        return table


class CostReport(BaseModel):
    """The baseline file: every refresh measured, with every prompt it sent."""

    timestamp: str
    model: str
    fixture: str
    #: Whether reflect built explicit per-step caches (HINDSIGHT_API_REFLECT_PROMPT_CACHE_ENABLED),
    #: which changes how cached tokens are billed — see ``RefreshCost.cost``.
    explicit_cache: bool = True
    refreshes: list[RefreshCost]


async def llm_calls(client: Hindsight, bank_id: str, since: str, until: str) -> list[LLMCall]:
    """Every LLM call the server traced for ``bank_id`` in ``[since, until)``, oldest first.

    Read through the public trace endpoint. The window is enough to attribute
    them because the bank is quiet: consolidation is off and the pages do not
    refresh on their own, so the only thing calling the model is the refresh the
    eval started.
    """
    # Still the published client, one layer down: the convenience wrapper has no
    # method for the trace endpoint, so the generated API class is driven with the
    # wrapper's own configured transport rather than a second HTTP client.
    api = LLMTracesApi(client._api_client)
    calls: list[LLMCall] = []
    offset = 0
    while True:
        page = await api.list_llm_requests(bank_id, start_date=since, end_date=until, limit=500, offset=offset)
        for item in page.items:
            calls.append(
                LLMCall(
                    scope=item.scope or "?",
                    operation=item.operation,
                    status=item.status,
                    started_at=item.started_at or "",
                    duration_ms=item.duration_ms,
                    input_tokens=item.input_tokens or 0,
                    cached_tokens=item.cached_tokens or 0,
                    output_tokens=item.output_tokens or 0,
                    prompt=item.input,
                    response=item.output,
                )
            )
        offset += len(page.items)
        if not page.items or offset >= page.total:
            break
    return sorted(calls, key=lambda c: c.started_at)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


@dataclass(frozen=True)
class _Attempt:
    status: str
    outcome: str


async def _first_attempt(client: Hindsight, bank_id: str, operation_id: str, timeout: float = 1800) -> _Attempt:
    """Wait for one attempt of a refresh, and stop it from retrying if it failed.

    A failed refresh is held ``pending`` for a retry. The retry would be a second
    measurement landing in the next refresh's window, so it is cancelled: this
    measures what ONE attempt costs, and a failure's retries multiply that.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        op = await client.operations.get_operation_status(bank_id, operation_id)
        if op.status == "completed":
            meta = op.result_metadata or {}
            return _Attempt("completed", str(meta.get("outcome") or meta.get("refresh_outcome") or "completed"))
        if op.status == "failed" or op.error_message:
            if op.status == "pending":
                await client.operations.cancel_operation(bank_id, operation_id)
            return _Attempt("failed", op.error_message or "failed without a message")
        await asyncio.sleep(1)
    raise TimeoutError(f"refresh {operation_id} on {bank_id} did not finish in {timeout}s")


async def measure_refresh(
    client: Hindsight, bank_id: str, mental_model_id: str, *, page: str, mode: str
) -> RefreshCost:
    """Refresh one page and collect every LLM call its first attempt made."""
    since = _now()
    started = asyncio.get_running_loop().time()
    submitted = await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)
    attempt = await _first_attempt(client, bank_id, submitted.operation_id)
    seconds = asyncio.get_running_loop().time() - started
    # Trace rows are written fire-and-forget after each call returns; give the
    # last one a moment to land before reading the window.
    await asyncio.sleep(2)
    calls = await llm_calls(client, bank_id, since, _now())
    model = await client.aget_mental_model(bank_id=bank_id, mental_model_id=mental_model_id, detail="full")
    content = getattr(model, "content", "") or ""
    based_on = (getattr(model, "reflect_response", None) or {}).get("based_on") or {}
    return RefreshCost(
        page=page,
        mode=mode,
        seconds=round(seconds, 1),
        status=attempt.status,
        outcome=attempt.outcome,
        calls=calls,
        page_chars=len(content),
        specifics=len(set(_SPECIFIC.findall(content))),
        cited=sum(len(v) for v in based_on.values() if isinstance(v, list)),
    )


#: A bug id, a version, or a date in any of the spellings a page writes them in.
_SPECIFIC = re.compile(
    r"LUM-\d+|\b\d+\.\d+\.\d+\b|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}"
)


def summary_table(refreshes: list[RefreshCost], price: Price | None = None, explicit_cache: bool = True) -> str:
    """The baseline at a glance: one row per refresh, then per-scope totals, then USD."""
    lines = [
        f"{'page':<30} {'mode':<6} {'calls':>5} {'input':>9} {'cached':>8} {'output':>7} "
        f"{'largest':>8} {'final':>8} {'synth':>5} {'secs':>6} {'chars':>6} {'spec':>5} {'cited':>5}  status"
    ]
    for r in refreshes:
        largest = r.largest_call.input_tokens if r.largest_call else 0
        final = r.final_call.input_tokens if r.final_call else 0
        # More than one synthesis call means the evidence did not fit one prompt
        # and was split into map calls plus a reduce (#4495).
        synth = sum(1 for c in r.calls if c.scope == "reflect")
        lines.append(
            f"{r.page[:30]:<30} {r.mode:<6} {len(r.calls):>5} {r.input_tokens:>9} {r.cached_tokens:>8} "
            f"{r.output_tokens:>7} {largest:>8} {final:>8} {synth:>5} {r.seconds:>6} {r.page_chars:>6} "
            f"{r.specifics:>5} {r.cited:>5}  {r.status}"
        )
    scopes: dict[str, ScopeUsage] = {}
    for r in refreshes:
        for scope, usage in r.by_scope().items():
            scopes[scope] = scopes.get(scope, ScopeUsage()) + usage
    lines.append("")
    lines.append(f"{'scope':<30} {'calls':>5} {'input':>9} {'cached':>8} {'output':>7}")
    for scope, u in sorted(scopes.items(), key=lambda kv: -kv[1].input):
        lines.append(f"{scope[:30]:<30} {u.calls:>5} {u.input:>9} {u.cached:>8} {u.output:>7}")
    if price is not None:
        totals = CostBreakdown()
        for r in refreshes:
            totals = totals + r.cost(price, explicit_cache=explicit_cache)
        lines.append("")
        lines.append(f"USD ({'explicit' if explicit_cache else 'implicit'} caching), {len(refreshes)} refreshes:")
        for key in CostBreakdown.model_fields:
            lines.append(f"  {key:<14} ${getattr(totals, key):.5f}")
        lines.append(f"  total          ${totals.total:.5f}")
        lines.append(f"  per refresh    ${totals.total / max(len(refreshes), 1):.5f}")
    return "\n".join(lines)


# ── building the frozen bank ──────────────────────────────────────────────────


def page_trigger(mode: str = "delta") -> dict[str, Any]:
    """The trigger a coding agent's page carries, minus automatic refresh.

    Sibling mental models are NOT excluded: they are the top layer of reflect's
    retrieval and the one whose payload has no token budget at all, so a cost
    measurement that hides them measures the cheap half of the loop. (The
    server's page default does exclude them; this is deliberately the broader
    scenario.) Automatic refresh is off: the eval triggers every refresh itself,
    and a refresh nobody asked for would land in the measurement window.
    """
    return {
        "mode": mode,
        "fact_types": ["world", "experience", "observation"],
        # The server's page default (exclude siblings) is the other scenario worth
        # measuring, because it is where the forced descent to raw recall has
        # nothing above it to release the loop first.
        "exclude_mental_models": os.getenv("HINDSIGHT_EVAL_REFRESH_COST_EXCLUDE_MENTAL_MODELS", "").lower() == "true",
        "refresh_after_consolidation": False,
    }


@dataclass
class _BuildResult:
    bank_id: str
    archive: bytes
    pages: list[str] = field(default_factory=list)


async def build_bank(
    client: Hindsight, settle: Callable[[str], Awaitable[None]], count: int = FACT_COUNT
) -> _BuildResult:
    """Retain the corpus with real consolidation, write the pages, and export the bank."""
    bank_id = f"refresh-cost-{uuid.uuid4().hex[:8]}"
    # `chunks` stores each fact as written: extraction would only paraphrase a
    # generated sentence, and it is the expensive half of a retain. Consolidation
    # stays ON — the observations it writes are what search_observations returns,
    # and that payload is half of what this eval exists to measure.
    await client.aupdate_bank_config(
        bank_id, retain_extraction_mode="chunks", enable_observations=True, enable_auto_consolidation=True
    )
    items = corpus(count)
    for start in range(0, len(items), 100):
        await client.aretain_batch(bank_id=bank_id, items=items[start : start + 100])
    await settle(bank_id)

    pages = []
    for name, query in PAGES.items():
        await client.knowledge_base.create_knowledge_page(
            bank_id,
            {"name": name, "source_query": query, "max_tokens": PAGE_MAX_TOKENS, "trigger": page_trigger()},
        )
        pages.append(name)
    await settle(bank_id)

    # Frozen as a quiet bank: nothing may call the model after import unless the
    # eval asked for it.
    await client.aupdate_bank_config(bank_id, enable_auto_consolidation=False)
    archive = await client.aexport_bank(bank_id, include_history=False, timeout=900)
    return _BuildResult(bank_id=bank_id, archive=archive, pages=pages)


async def _drain(client: Hindsight, bank_id: str, timeout: float = 7200) -> None:
    """Wait until nothing is queued, then fail on anything that ended failed.

    Not ``wait_until_settled``: that gives up on the first attempt that errors,
    while a build runs ~a thousand model calls and a transient provider error is
    what the worker's retries are for. Only an operation that exhausted them is
    a broken build.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        busy = [
            op
            for status in ("pending", "processing")
            for op in (await client.operations.list_operations(bank_id, status=status, limit=100)).operations
        ]
        if not busy:
            break
        await asyncio.sleep(5)
    else:
        raise TimeoutError(f"bank {bank_id} still busy after {timeout}s")
    failed = (await client.operations.list_operations(bank_id, status="failed", limit=100)).operations
    if failed:
        raise RuntimeError(f"bank {bank_id}: {', '.join(f'{op.task_type}: {op.error_message}' for op in failed)}")


async def _build(out: Path, count: int) -> None:
    from hindsight_system_evals.target import eval_target

    with eval_target() as target:
        client = Hindsight(base_url=target.url, api_key=target.api_key, timeout=900)
        try:
            result = await build_bank(client, lambda bank: _drain(client, bank), count)
        finally:
            await client.aclose()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result.archive)
    print(f"wrote {out} ({len(result.archive) / 1e6:.1f} MB) from bank {result.bank_id}: {', '.join(result.pages)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="regenerate the frozen bank archive (real model, slow)")
    build.add_argument("--out", type=Path, default=FIXTURE)
    build.add_argument(
        "--facts", type=int, default=FACT_COUNT, help="corpus size; smaller only to smoke-test the build"
    )
    args = parser.parse_args()
    if args.command == "build":
        asyncio.run(_build(args.out, args.facts))


if __name__ == "__main__":
    main()
