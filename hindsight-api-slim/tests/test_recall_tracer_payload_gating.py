"""`if tracer:` is not a trace check, and writing one is how this bug keeps coming back.

The tracer is constructed for EVERY recall so the `[phases]` accounting always has somewhere to
write, and `phases_only` drops everything expensive *inside* the tracer's own methods. So a guard
written as `if tracer:` is always true: the block runs on ordinary traffic, the caller builds the
payload, and the tracer throws it away. That has been found and fixed three separate times in
`memory_engine.py` -- at `finalize()`, at the entry-point hydration fetch, and at the four
retrieval/merge/rerank/visit payload builds, the last of which profiled at 11% of all non-idle
samples under load.

`enable_trace` is the flag that means "a caller asked for a trace". These tests pin both halves:
the tracer really does drop payloads while still recording phases, and no guard in the recall path
is written against the tracer's existence again.
"""

import ast
import inspect
from pathlib import Path

import pytest

from hindsight_api.engine import memory_engine
from hindsight_api.engine.search.tracer import SearchTracer


def _phases_only_tracer() -> SearchTracer:
    t = SearchTracer(query="anything", budget=10, max_tokens=1000)
    t.phases_only = True
    t.start()
    return t


def test_phases_are_recorded_but_payloads_are_dropped():
    """The asymmetry the call sites have to encode: timings survive `phases_only`, payloads do
    not. Anything a caller builds to hand to the payload methods is therefore wasted work."""
    t = _phases_only_tracer()

    t.add_phase_metric("parallel_retrieval", 0.4)
    t.record_query_embedding([0.1] * 384)
    t.add_entry_point("id-1", "text", 0.9, 1)
    t.add_retrieval_results(
        method_name="semantic",
        results=[("id-1", {"text": "text", "similarity": 0.9})],
        duration_seconds=0.1,
        score_field="similarity",
    )
    t.add_rrf_merged([("id-1", {"text": "text"}, {"rrf_score": 0.5})])
    t.add_reranked([{"id": "id-1", "text": "text"}], [])
    t.visit_node(
        node_id="id-1",
        text="text",
        context="",
        event_date=None,
        is_entry_point=True,
        activation=0.5,
        semantic_similarity=0.9,
        recency=0.1,
        frequency=0.0,
        final_weight=0.7,
    )

    assert [m.phase_name for m in t.phase_metrics] == ["parallel_retrieval"]
    assert t.query_embedding is None
    assert t.entry_points == []
    assert t.retrieval_results == []
    assert t.rrf_merged == []
    assert t.reranked == []
    assert t.visits == []


def test_visit_counters_are_not_read_by_anything_outside_the_visit_list():
    """`visit_node` keeps two counters under `phases_only` (`current_step`, `nodes_visited_set`),
    which is why skipping the loop entirely when no trace was asked for is safe: the summary
    counts `visits`, and nothing else reads them."""
    t = _phases_only_tracer()
    trace = t.finalize([])
    assert trace.summary.total_nodes_visited == 0


def _tracer_existence_guards() -> list[int]:
    """Line numbers of `if`/`elif` conditions in memory_engine.py that test the tracer's
    existence -- `if tracer:`, `if tracer and x:`, `if x and tracer:`."""
    source = Path(inspect.getfile(memory_engine)).read_text()
    tree = ast.parse(source)

    def _is_tracer_name(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id == "tracer"

    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        operands = test.values if isinstance(test, ast.BoolOp) else [test]
        if any(_is_tracer_name(operand) for operand in operands):
            found.append(test.lineno)
    return found


def test_no_call_site_guards_on_the_tracer_existing():
    """The structural guard. A payload block behind `if tracer:` runs on every recall; gate it on
    `enable_trace` instead, and leave `add_phase_metric` calls unguarded so timings stay always-on
    (that is the whole reason the tracer is built unconditionally)."""
    guards = _tracer_existence_guards()
    assert guards == [], (
        "memory_engine.py guards on the tracer's existence at line(s) "
        f"{guards} -- the tracer is ALWAYS constructed, so that condition is always true. "
        "Gate trace payload construction on `enable_trace`; leave phase metrics unguarded."
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "if tracer:\n    x = 1\n",
        "if tracer and include_chunks:\n    x = 1\n",
        "if include_chunks and tracer:\n    x = 1\n",
    ],
)
def test_the_structural_guard_actually_detects_the_shape(snippet, monkeypatch, tmp_path):
    """The check above passes trivially if the detector is broken, so prove it fires."""
    fake = tmp_path / "memory_engine.py"
    fake.write_text(snippet)
    monkeypatch.setattr(inspect, "getfile", lambda _module: str(fake))
    assert _tracer_existence_guards() == [1]
