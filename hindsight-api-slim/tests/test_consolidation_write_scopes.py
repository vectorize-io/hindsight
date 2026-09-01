"""Unit tests for ``_resolve_write_scopes`` / ``_resolve_obs_tags_list``.

These helpers drive the per-scope lock acquisition in the parallel consolidation
dispatcher. The lock-based safety story only works if the helpers report the
*exact* set of observation scopes a memory will write to — over-reporting is
safe (extra locks slow things down but don't lose data) but under-reporting
allows a concurrent group to race on the same observation row. So we pin the
mapping for every observation_scopes mode the dispatcher recognises.
"""

import json

import pytest

from hindsight_api.engine.consolidation.consolidator import (
    _batch_scope_signature,
    _consolidation_batch_key,
    _resolve_obs_tags_list,
    _resolve_write_scopes,
    _scope_sort_key,
)


# ---------------------------------------------------------------------------
# _resolve_write_scopes — frozenset output is what the lock dict keys on
# ---------------------------------------------------------------------------


# Production observation_scopes values come from a JSONB column. asyncpg
# returns JSONB without a codec, so the value is a JSON-encoded *string*. These
# parametrised cases cover both shapes: the raw Python form (None / list / dict
# — used in tests that build the memory dict directly) and the JSON-encoded
# string form (used in tests that mirror what asyncpg surfaces from the DB).


def _as_json_string(value):
    """Encode an observation_scopes value the way asyncpg presents it from JSONB."""
    return json.dumps(value)


class TestResolveWriteScopesCombined:
    @pytest.mark.parametrize("scopes_value", [None, _as_json_string("combined")])
    def test_default_uses_full_tag_set(self, scopes_value):
        memory = {"tags": ["alice", "session"], "observation_scopes": scopes_value}
        assert _resolve_write_scopes(memory) == [frozenset({"alice", "session"})]

    def test_empty_tags_collapses_to_untagged_scope(self):
        memory = {"tags": [], "observation_scopes": None}
        assert _resolve_write_scopes(memory) == [frozenset()]

    def test_missing_tags_collapses_to_untagged_scope(self):
        memory = {"observation_scopes": None}
        assert _resolve_write_scopes(memory) == [frozenset()]


class TestResolveWriteScopesPerTag:
    def test_emits_one_scope_per_tag(self):
        memory = {"tags": ["alice", "session", "thread"], "observation_scopes": _as_json_string("per_tag")}
        scopes = _resolve_write_scopes(memory)
        assert set(scopes) == {frozenset({"alice"}), frozenset({"session"}), frozenset({"thread"})}
        assert len(scopes) == 3  # no dedupe surprises

    def test_empty_tags_collapses_to_untagged_scope(self):
        memory = {"tags": [], "observation_scopes": _as_json_string("per_tag")}
        assert _resolve_write_scopes(memory) == [frozenset()]


class TestResolveWriteScopesAllCombinations:
    def test_two_tags_yields_three_scopes(self):
        memory = {"tags": ["alice", "session"], "observation_scopes": _as_json_string("all_combinations")}
        scopes = set(_resolve_write_scopes(memory))
        assert scopes == {
            frozenset({"alice"}),
            frozenset({"session"}),
            frozenset({"alice", "session"}),
        }

    def test_three_tags_yields_seven_scopes(self):
        memory = {"tags": ["a", "b", "c"], "observation_scopes": _as_json_string("all_combinations")}
        scopes = set(_resolve_write_scopes(memory))
        # C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7
        assert scopes == {
            frozenset({"a"}),
            frozenset({"b"}),
            frozenset({"c"}),
            frozenset({"a", "b"}),
            frozenset({"a", "c"}),
            frozenset({"b", "c"}),
            frozenset({"a", "b", "c"}),
        }

    def test_empty_tags_collapses_to_untagged_scope(self):
        memory = {"tags": [], "observation_scopes": _as_json_string("all_combinations")}
        assert _resolve_write_scopes(memory) == [frozenset()]


class TestResolveWriteScopesShared:
    def test_collapses_to_single_untagged_scope_regardless_of_tags(self):
        # "shared" ignores the memory's own tags and writes to one global scope,
        # so every memory deduplicates against the same observation.
        memory = {"tags": ["alice", "session"], "observation_scopes": _as_json_string("shared")}
        assert _resolve_write_scopes(memory) == [frozenset()]

    def test_empty_tags_also_untagged_scope(self):
        memory = {"tags": [], "observation_scopes": _as_json_string("shared")}
        assert _resolve_write_scopes(memory) == [frozenset()]


class TestResolveWriteScopesExplicitList:
    def test_uses_declared_scopes_verbatim(self):
        memory = {
            "tags": ["alice", "session"],
            "observation_scopes": _as_json_string([["alice"], ["session"], ["alice", "session"]]),
        }
        scopes = set(_resolve_write_scopes(memory))
        assert scopes == {
            frozenset({"alice"}),
            frozenset({"session"}),
            frozenset({"alice", "session"}),
        }

    def test_ignores_memory_tags(self):
        # An explicit scope list overrides per-mode logic — even if the memory's
        # own tags don't match, the declared scopes are what gets written.
        memory = {"tags": ["alice"], "observation_scopes": _as_json_string([["unrelated_scope"]])}
        assert _resolve_write_scopes(memory) == [frozenset({"unrelated_scope"})]

    def test_with_empty_inner_scope(self):
        memory = {"tags": ["alice"], "observation_scopes": _as_json_string([[], ["alice"]])}
        scopes = set(_resolve_write_scopes(memory))
        assert scopes == {frozenset(), frozenset({"alice"})}


class TestResolveWriteScopesPrepackedValues:
    """When the caller hands a memory dict with already-parsed observation_scopes
    (e.g. a unit test fixture passing a Python value directly), the helper must
    not try to JSON-decode it again. The shape gate is ``isinstance(_, str)``,
    so non-string Python values flow through untouched."""

    def test_list_passed_directly(self):
        memory = {"tags": ["alice"], "observation_scopes": [["alice"], ["other"]]}
        assert set(_resolve_write_scopes(memory)) == {frozenset({"alice"}), frozenset({"other"})}

    def test_none_passed_directly(self):
        memory = {"tags": ["alice"], "observation_scopes": None}
        assert _resolve_write_scopes(memory) == [frozenset({"alice"})]


# ---------------------------------------------------------------------------
# _resolve_obs_tags_list — drives the multi-pass dispatch
# ---------------------------------------------------------------------------


class TestResolveObsTagsList:
    """The list form is what the dispatcher passes as obs_tags_override on each
    pass; the frozenset form is what gets locked. They must agree on which
    scopes are touched."""

    def test_combined_returns_none(self):
        assert _resolve_obs_tags_list({"tags": ["a"], "observation_scopes": None}) is None
        assert _resolve_obs_tags_list({"tags": ["a"], "observation_scopes": json.dumps("combined")}) is None

    def test_per_tag_returns_one_list_per_tag(self):
        memory = {"tags": ["a", "b"], "observation_scopes": json.dumps("per_tag")}
        assert _resolve_obs_tags_list(memory) == [["a"], ["b"]]

    def test_all_combinations_returns_all_nonempty_subsets(self):
        memory = {"tags": ["a", "b"], "observation_scopes": json.dumps("all_combinations")}
        result = _resolve_obs_tags_list(memory)
        assert result is not None
        assert {tuple(sorted(r)) for r in result} == {("a",), ("b",), ("a", "b")}

    def test_per_tag_empty_tags_returns_none(self):
        # Falls back to the default single-pass behaviour; the dispatcher then
        # uses the memory's (empty) tag set, matching combined-mode behaviour.
        assert _resolve_obs_tags_list({"tags": [], "observation_scopes": json.dumps("per_tag")}) is None

    def test_explicit_list_passthrough(self):
        spec = [["a"], ["a", "b"]]
        memory = {"tags": ["a", "b"], "observation_scopes": json.dumps(spec)}
        assert _resolve_obs_tags_list(memory) == spec

    def test_shared_returns_single_empty_scope(self):
        # One pass over the empty (untagged) scope; the memory's own tags are
        # ignored so cross-tag memories consolidate into one observation.
        memory = {"tags": ["a", "b"], "observation_scopes": json.dumps("shared")}
        assert _resolve_obs_tags_list(memory) == [[]]


# ---------------------------------------------------------------------------
# Agreement between obs_tags_list (dispatch) and write_scopes (locks)
# ---------------------------------------------------------------------------


class TestDispatchLockAgreement:
    """Whatever scopes the dispatcher visits via _resolve_obs_tags_list, the
    lock layer must have a lock for. Under-locking is the race we set out to
    fix, so this is the load-bearing invariant."""

    @pytest.mark.parametrize(
        "memory",
        [
            # JSON-encoded shape (what asyncpg surfaces from JSONB without a codec).
            {"tags": ["a", "b"], "observation_scopes": None},
            {"tags": ["a", "b", "c"], "observation_scopes": json.dumps("combined")},
            {"tags": ["a", "b"], "observation_scopes": json.dumps("per_tag")},
            {"tags": ["a", "b", "c"], "observation_scopes": json.dumps("all_combinations")},
            {"tags": ["a", "b"], "observation_scopes": json.dumps("shared")},
            {"tags": ["a", "b"], "observation_scopes": json.dumps([["a"], ["b"], ["a", "b"]])},
            {"tags": ["a"], "observation_scopes": json.dumps([["a"], ["x"]])},
            {"tags": [], "observation_scopes": json.dumps("per_tag")},
            # Degenerate explicit list: resolves to no passes, so the pass loop
            # falls back to the combined single pass over the memory's own tags.
            {"tags": ["a", "b"], "observation_scopes": json.dumps([])},
            {"tags": [], "observation_scopes": json.dumps([])},
            # Pre-parsed Python shape (defensive — covers callers that hand the
            # helper a memory dict with a non-string value).
            {"tags": ["a", "b"], "observation_scopes": [["a"], ["b"], ["a", "b"]]},
            {"tags": ["a", "b"], "observation_scopes": []},
        ],
    )
    def test_every_dispatched_scope_has_a_lock(self, memory):
        dispatched = _resolve_obs_tags_list(memory)
        write_scopes = set(_resolve_write_scopes(memory))

        if not dispatched:
            # Combined-mode single pass — either the explicit ``combined``/None
            # resolution or a falsy one the pass loop treats identically: the
            # memory's own tags are the write scope.
            expected = {frozenset(memory.get("tags") or [])}
        else:
            expected = {frozenset(s) for s in dispatched}

        missing = expected - write_scopes
        assert not missing, (
            f"dispatcher will write to scopes {missing} but no lock will be acquired "
            f"for them (memory={memory!r}, write_scopes={write_scopes!r})"
        )


# ---------------------------------------------------------------------------
# _scope_sort_key — deadlock-freedom rests on a total order over frozensets
# ---------------------------------------------------------------------------


class TestScopeSortKey:
    def test_is_total_order(self):
        """Every distinct scope must have a distinct sort key so concurrent
        groups acquire shared locks in the same order."""
        scopes = [
            frozenset(),
            frozenset({"a"}),
            frozenset({"b"}),
            frozenset({"a", "b"}),
            frozenset({"a", "b", "c"}),
        ]
        keys = [_scope_sort_key(s) for s in scopes]
        assert len(set(keys)) == len(scopes), "two distinct scopes share a sort key"

    def test_order_is_stable_across_set_construction(self):
        """frozenset hash order is non-deterministic across runs, but the sort
        key should not be — otherwise two groups could try to acquire shared
        locks in opposite orders."""
        scope_a = frozenset(["x", "y", "z"])
        scope_b = frozenset({"z", "y", "x"})  # built differently, same set
        assert _scope_sort_key(scope_a) == _scope_sort_key(scope_b)


# ---------------------------------------------------------------------------
# _consolidation_batch_key — the grouping key that decides who shares an LLM call
# ---------------------------------------------------------------------------


def _effective_pass_scopes(memory) -> set[frozenset[str]]:
    """The scopes ``run_consolidation_job`` will actually write for this memory.

    Reads the production signature so the invariants below are pinned to the same
    derivation the runtime split uses; ``TestBatchScopeSignature`` is what pins
    that derivation to the pass loop.
    """
    return {frozenset(scope) for scope in _batch_scope_signature(memory)}


#: Every (tags, observation_scopes) shape the write path can produce, crossed
#: below into pairs. Kept exhaustive on purpose: the batch key is the only thing
#: standing between two memories and a shared LLM call, and the loop reads the
#: scope off ``sub_batch[0]`` for the whole batch.
_SCOPE_SPECS = [
    None,
    "combined",
    "per_tag",
    "all_combinations",
    "shared",
    [],
    [[]],
    [["a"]],
    [["b"]],
    [["a"], ["b"]],
    [["a", "b"]],
    [["x"]],
]

_TAG_SETS = [[], ["a"], ["b"], ["a", "b"], ["b", "a"], ["a", "c"]]

_ALL_MEMORIES = [
    {"tags": list(tags), "observation_scopes": json.dumps(spec) if spec is not None else None}
    for tags in _TAG_SETS
    for spec in _SCOPE_SPECS
]


class TestConsolidationBatchKey:
    """A batch's observation scope is resolved once, from ``sub_batch[0]``, and
    applied to every memory in it. So the grouping key has exactly one job: two
    memories may share a key **only if** they write the identical set of scopes.

    Break that and the failure is silent and severe — a memory tagged
    ``user:alice`` batched behind a ``shared`` memory has its observation written
    untagged (globally recallable: a cross-tag leak), and a ``shared`` memory
    batched behind a ``combined`` one has its override dropped. That is #3953,
    and it is why the invariant below is exhaustive over the spec matrix rather
    than a handful of examples.
    """

    def test_same_key_implies_same_written_scopes(self):
        """The load-bearing invariant: key collision ⇒ identical write scopes."""
        by_key: dict[tuple[str, ...], list[dict]] = {}
        for memory in _ALL_MEMORIES:
            by_key.setdefault(_consolidation_batch_key(memory), []).append(memory)

        for key, members in by_key.items():
            scope_sets = {frozenset(_effective_pass_scopes(m)) for m in members}
            assert len(scope_sets) == 1, f"batch key {key!r} pools memories that write different scopes: " + ", ".join(
                f"{m!r} -> {sorted(map(sorted, _effective_pass_scopes(m)))}" for m in members
            )

    def test_same_key_implies_same_lock_scopes(self):
        """Groups take locks per member, but a shared key must not smuggle a
        memory into a group whose lock set doesn't cover it."""
        by_key: dict[tuple[str, ...], list[dict]] = {}
        for memory in _ALL_MEMORIES:
            by_key.setdefault(_consolidation_batch_key(memory), []).append(memory)

        for key, members in by_key.items():
            lock_sets = {frozenset(_resolve_write_scopes(m)) for m in members}
            assert len(lock_sets) == 1, f"batch key {key!r} pools memories with different lock scopes: {members!r}"

    def test_key_is_stable_across_tag_order_and_encoding(self):
        """Tag order and the JSONB string/parsed shapes are incidental; a key
        that varied with them would split batches that belong together."""
        assert _consolidation_batch_key({"tags": ["a", "b"], "observation_scopes": None}) == _consolidation_batch_key(
            {"tags": ["b", "a"], "observation_scopes": json.dumps("combined")}
        )
        assert _consolidation_batch_key(
            {"tags": ["a"], "observation_scopes": json.dumps([["b"], ["a"]])}
        ) == _consolidation_batch_key({"tags": ["z"], "observation_scopes": [["a"], ["b"]]})

    def test_shared_pools_across_different_native_tags(self):
        """#3953: the whole point of ``shared`` — same target scope, so the same
        batch, whatever the source facts happen to be tagged with."""
        a = {"tags": ["suggested_tool:terraform"], "observation_scopes": json.dumps("shared")}
        b = {"tags": ["suggested_tool:gardea"], "observation_scopes": json.dumps("shared")}
        assert _consolidation_batch_key(a) == _consolidation_batch_key(b)

    @pytest.mark.parametrize(
        ("spec_a", "spec_b"),
        [
            (None, "shared"),
            (None, "per_tag"),
            (None, "all_combinations"),
            ("shared", "per_tag"),
            ("per_tag", "all_combinations"),
            (None, [["a"]]),
            ("shared", [[]]),  # equivalent target scope, reached two ways
        ],
    )
    def test_same_tags_different_modes(self, spec_a, spec_b):
        """Two memories with identical tags but different modes may share a key
        only when the scopes they write are genuinely identical."""
        mem_a = {"tags": ["a", "b"], "observation_scopes": json.dumps(spec_a) if spec_a is not None else None}
        mem_b = {"tags": ["a", "b"], "observation_scopes": json.dumps(spec_b) if spec_b is not None else None}

        same_key = _consolidation_batch_key(mem_a) == _consolidation_batch_key(mem_b)
        same_scopes = _effective_pass_scopes(mem_a) == _effective_pass_scopes(mem_b)
        assert same_key == same_scopes, (
            f"{spec_a!r} vs {spec_b!r}: keys {'match' if same_key else 'differ'} but written scopes "
            f"{'match' if same_scopes else 'differ'}"
        )

    @pytest.mark.parametrize("spec", [None, "combined", []])
    def test_falsy_resolution_keys_as_combined(self, spec):
        """``[]`` resolves to no passes and the loop falls back to combined, so
        it must key as combined — keying it as a fan-out gave every such memory
        the tag-free ``("fanout",)`` key, pooling unrelated tag sets into one
        call that then took ``memories[0]``'s tags."""
        memory = {"tags": ["a", "b"], "observation_scopes": json.dumps(spec) if spec is not None else None}
        assert _consolidation_batch_key(memory) == ("combined", "a", "b")

    def test_empty_explicit_list_does_not_pool_across_tags(self):
        """The regression the ``not resolved`` branch closes, stated directly."""
        alice = {"tags": ["user:alice"], "observation_scopes": json.dumps([])}
        bob = {"tags": ["user:bob"], "observation_scopes": json.dumps([])}
        assert _consolidation_batch_key(alice) != _consolidation_batch_key(bob)


class TestBatchScopeSignature:
    """``_batch_scope_signature`` is the runtime's last line of defence: the
    sub-batch loop splits on it, so a batch can only ever be written at one
    scope. It has to mirror the pass loop for every mode — pinned literally
    here rather than derived, so a change to either side shows up as a diff."""

    @pytest.mark.parametrize(
        ("tags", "spec", "expected"),
        [
            (["b", "a"], None, (("a", "b"),)),
            (["b", "a"], "combined", (("a", "b"),)),
            (["a", "b"], "per_tag", (("a",), ("b",))),
            (["a"], "per_tag", (("a",),)),
            ([], "per_tag", ((),)),
            (["a", "b"], "all_combinations", (("a",), ("a", "b"), ("b",))),
            (["a", "b"], "shared", ((),)),
            ([], None, ((),)),
            # Degenerate explicit list -> the combined fallback over own tags.
            (["a", "b"], [], (("a", "b"),)),
            (["x"], [["b"], ["a"]], (("a",), ("b",))),
            (["x"], [[]], ((),)),
        ],
    )
    def test_signature_per_mode(self, tags, spec, expected):
        memory = {"tags": tags, "observation_scopes": json.dumps(spec) if spec is not None else None}
        assert _batch_scope_signature(memory) == expected

    def test_signature_matches_the_pass_loop_fallback(self):
        """Cross-check: the signature and the dispatcher agree on every memory
        in the matrix, so neither can drift into agreeing with itself only."""
        for memory in _ALL_MEMORIES:
            dispatched = _resolve_obs_tags_list(memory)
            expected = {frozenset(s) for s in dispatched} if dispatched else {frozenset(memory.get("tags") or [])}
            assert _effective_pass_scopes(memory) == expected, memory

    def test_signature_is_order_insensitive(self):
        """The runtime splits a heterogeneous batch by grouping on this value,
        so equal scope sets must hash equal regardless of how they were written."""
        a = {"tags": ["x"], "observation_scopes": json.dumps([["b", "a"], ["c"]])}
        b = {"tags": ["y"], "observation_scopes": json.dumps([["c"], ["a", "b"]])}
        assert _batch_scope_signature(a) == _batch_scope_signature(b)
