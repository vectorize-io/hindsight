"""``enable_text_search=false`` — pure vector recall.

The keyword arm shares one UNION query with the dense arm, so "disabled" has to mean
the BM25 half is never *built*, not that its rows are filtered away afterwards. These
tests assert the absence of work: no tokenization, no ``pg_stats`` term-selection
round trip, no BM25 arm in the SQL, and no BM25 bind parameters — while the semantic
arm is untouched.

The flag is per bank, like the other recall stage toggles, so the SQL builder takes it
as an argument rather than reading it off the global config; ``HINDSIGHT_API_ENABLE_TEXT_SEARCH``
only sets the deployment-wide default a bank inherits.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from hindsight_api.config import ENV_ENABLE_TEXT_SEARCH, HindsightConfig, _get_raw_config
from hindsight_api.engine.retain import orchestrator as retain_orchestrator
from hindsight_api.engine.search import bm25_term_selection as bm25_mod
from hindsight_api.engine.search import retrieval as retrieval_mod


class _FakeDialect:
    """Records which arms the builder asked for, and the BM25 text it prepared."""

    def __init__(self) -> None:
        self.semantic_arms = 0
        self.bm25_arms = 0
        self.prepared_text: str | None = None

    def build_semantic_arm(self, **kwargs) -> str:
        self.semantic_arms += 1
        return "SELECT 'semantic' AS source"

    def build_bm25_arm(self, **kwargs) -> str:
        self.bm25_arms += 1
        return "SELECT 'bm25' AS source"

    def prepare_bm25_text(self, tokens, query_text, **kwargs) -> str:
        self.prepared_text = " | ".join(tokens)
        return self.prepared_text


class _FakeConn:
    backend_type = "postgresql"

    def __init__(self) -> None:
        self.query: str | None = None
        self.params: tuple = ()

    async def fetch(self, query, *params):
        self.query = query
        self.params = params
        return []


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        semantic_min_similarity=0.0,
        bm25_min_score=0.0,
        text_search_extension="native",
        text_search_extension_native_language="english",
        text_search_extension_pg_search_function_schema="paradedb",
        bm25_max_query_terms=16,
        bm25_selective_terms=True,
    )


@pytest.fixture
def harness(monkeypatch):
    """Wire the SQL builder to fakes and count the pg_stats term-selection calls."""
    dialect = _FakeDialect()
    conn = _FakeConn()
    tokenized: list[str] = []
    selective_lookups: list[bool] = []

    real_tokenize = retrieval_mod.tokenize_query

    def counting_tokenize(query_text: str) -> list[str]:
        tokenized.append(query_text)
        return real_tokenize(query_text)

    async def fake_select_selective(conn_, tokens, **kwargs):
        selective_lookups.append(True)
        return tokens[:5]

    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: dialect)
    monkeypatch.setattr(retrieval_mod, "tokenize_query", counting_tokenize)
    monkeypatch.setattr(bm25_mod, "select_selective_bm25_tokens", fake_select_selective)
    monkeypatch.setattr(bm25_mod, "get_current_schema", lambda: "public")
    monkeypatch.setattr(retrieval_mod, "fq_table", lambda name: name)

    return SimpleNamespace(
        dialect=dialect,
        conn=conn,
        tokenized=tokenized,
        selective_lookups=selective_lookups,
    )


# A query long enough to trip the selective-term lookup, so a disabled arm that
# still tokenized would be caught by the pg_stats round trip as well as the SQL.
_LONG_QUERY = " ".join(f"term{i}" for i in range(20))


async def _run(harness, *, enable_text_search: bool, monkeypatch, query: str = _LONG_QUERY):
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: _config())
    return await retrieval_mod.retrieve_semantic_bm25_combined_sql(
        harness.conn,
        "[0.0]",
        query,
        "bank-1",
        ["world", "observation"],
        10,
        enable_text_search=enable_text_search,
    )


@pytest.mark.asyncio
async def test_enabled_builds_both_arms(harness, monkeypatch):
    """Baseline: the default keeps the hybrid query exactly as it was."""
    await _run(harness, enable_text_search=True, monkeypatch=monkeypatch)

    assert harness.dialect.semantic_arms == 2  # one per fact type
    assert harness.dialect.bm25_arms == 2
    assert "'bm25' AS source" in (harness.conn.query or "")
    # $3 = BM25 limit, $4 = prepared BM25 text.
    assert len(harness.conn.params) == 4


@pytest.mark.asyncio
async def test_disabled_emits_vector_only_sql(harness, monkeypatch):
    result = await _run(harness, enable_text_search=False, monkeypatch=monkeypatch)

    assert harness.dialect.semantic_arms == 2  # dense arm is untouched
    assert harness.dialect.bm25_arms == 0
    assert "'bm25' AS source" not in (harness.conn.query or "")
    assert harness.dialect.prepared_text is None
    # Only $1 (embedding) and $2 (bank_id) remain — the BM25 limit/text slots are gone.
    assert len(harness.conn.params) == 2
    assert all(not arms.bm25 for arms in result.values())


@pytest.mark.asyncio
async def test_disabled_skips_tokenization_and_term_selection(harness, monkeypatch):
    """The point of the flag is skipped work, not filtered rows.

    Tokenizing feeds the BM25 arm and nothing else, and the selective-term step is a
    real ``pg_stats`` round trip — both must be gone, not merely unused.
    """
    await _run(harness, enable_text_search=False, monkeypatch=monkeypatch)

    assert harness.tokenized == []
    assert harness.selective_lookups == []


@pytest.mark.asyncio
async def test_enabled_does_tokenize_and_select(harness, monkeypatch):
    """Guards the test above: both costs are genuinely paid when the arm is on."""
    await _run(harness, enable_text_search=True, monkeypatch=monkeypatch)

    assert harness.tokenized == [_LONG_QUERY]
    assert harness.selective_lookups == [True]


def test_config_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv(ENV_ENABLE_TEXT_SEARCH, raising=False)
    assert HindsightConfig.from_env().enable_text_search is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("false", False), ("FALSE", False), ("0", False), ("no", False), ("true", True), ("1", True)],
)
def test_config_reads_env(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_ENABLE_TEXT_SEARCH, raw)
    assert HindsightConfig.from_env().enable_text_search is expected


def test_flag_is_per_bank_like_its_siblings():
    """Hierarchical, so one bank can run lean without changing the deployment.

    It sits in the same group as the other recall stage toggles; a flag missing from
    this set is silently unreachable through the bank config API.
    """
    configurable = HindsightConfig.get_configurable_fields()
    for field in ("enable_text_search", "enable_temporal_retrieval", "enable_graph_retrieval", "enable_reranking"):
        assert field in configurable, field


def test_real_config_carries_the_field():
    """``HindsightConfig`` is a dataclass, so the field is always present."""
    config = dataclasses.replace(_get_raw_config(), enable_text_search=False)
    assert config.enable_text_search is False


@pytest.mark.parametrize("field", ["enable_text_search", "enable_graph_retrieval"])
def test_the_write_path_carries_the_flag_to_a_store_that_owns_its_index(field):
    """A store-owned retain is told what the bank currently wants.

    For a store that keeps its own index these are not only read-time settings: an arm the bank has
    switched off needs no index BUILT for it, and building one is work and bytes spent for a query
    that will not run. Postgres ignores them — its columns are maintained by the insert itself, so
    there is nothing separable to skip — which is why the interface defaults them to True.

    Asserted on the interface rather than by driving a retain: the value has to be a parameter a
    store can act on, and the orchestrator has to be the thing that reads it off the bank's config
    (so a bank that changes its mind is followed on the next write, with no out-of-band call).
    """
    import inspect

    from hindsight_api.engine.memories.base import MemoriesExtension

    sig = inspect.signature(MemoriesExtension.retain)
    assert field in sig.parameters, f"the store-owned write seam cannot see {field}"
    assert sig.parameters[field].default is True, "a store that ignores it must keep working"

    src = inspect.getsource(retain_orchestrator)
    assert f'{field}=bool(getattr(config, "{field}", True))' in src, (
        f"the orchestrator must pass the BANK's {field} on every retain — a value read once at "
        f"bank creation would not follow a bank that changes it"
    )
