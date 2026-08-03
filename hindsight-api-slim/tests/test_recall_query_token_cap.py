"""Recall queries are bounded at the engine ingress, not only at the HTTP edge (issue #3134).

`HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS` used to be enforced only in the REST handler,
so every internal caller (consolidation, reflect tools, MCP tools, the context
extension) could pass arbitrarily long text into BM25 tokenization. A degenerate
extraction — 58k words, 4 distinct — became a 54k-term OR `tsquery` whose evaluation
exceeded Postgres' stack depth (SQLSTATE 54001).
"""

from hindsight_api.engine.memory_engine import (
    _get_tiktoken_encoding,
    _truncate_query_to_token_limit,
    count_tokens,
)
from hindsight_api.engine.search.retrieval import tokenize_query


def test_short_query_is_returned_unchanged():
    query = "What did Alice say about machine learning?"
    assert _truncate_query_to_token_limit(query, 500) is query


def test_query_under_the_cap_but_over_the_char_shortcut_is_unchanged():
    """A query longer than `max` characters but under `max` tokens must survive intact.

    The character comparison is only a shortcut to skip tokenizing short queries; it
    must never truncate on its own.
    """
    query = "machine learning " * 40  # ~680 chars, ~120 tokens
    assert len(query) > 500  # past the character shortcut, so it gets tokenized
    assert count_tokens(query) < 500
    assert _truncate_query_to_token_limit(query, 500) == query


def test_long_query_is_truncated_to_the_cap():
    query = "alpha beta gamma delta " * 500
    assert count_tokens(query) > 500

    truncated = _truncate_query_to_token_limit(query, 500)

    assert count_tokens(truncated) == 500
    assert query.startswith(truncated[:100])


def test_zero_disables_the_cap():
    query = "alpha beta gamma delta " * 500
    assert _truncate_query_to_token_limit(query, 0) is query


def test_degenerate_repetition_stays_far_below_the_tsquery_stack_cliff():
    """The reported input: 58k words, 4 distinct, OR-joined into one tsquery.

    Evaluating `@@` recurses once per node, and the measured failure threshold sits
    between 30k and 50k terms. Capping the query keeps the term count ~2 orders of
    magnitude below that.
    """
    degenerate = "Ismar is mandating " + "mandatory " * 58_279

    assert len(tokenize_query(degenerate)) > 50_000  # unbounded: over the cliff

    bounded = _truncate_query_to_token_limit(degenerate, 500)

    assert len(tokenize_query(bounded)) <= 500


def test_truncation_never_raises_on_special_token_literals():
    """Internal callers must degrade, never fail — including on content that merely
    mentions a tiktoken special-token literal (issue #1883)."""
    query = "<|endoftext|> " * 1000

    bounded = _truncate_query_to_token_limit(query, 500)

    assert count_tokens(bounded) == 500


def test_truncated_query_round_trips_through_the_encoder():
    """decode(encode(x)[:n]) must yield text the downstream tokenizer accepts."""
    query = "consolidation observation recall " * 300
    bounded = _truncate_query_to_token_limit(query, 500)

    assert isinstance(bounded, str)
    assert _get_tiktoken_encoding().encode(bounded) == _get_tiktoken_encoding().encode(query)[:500]
