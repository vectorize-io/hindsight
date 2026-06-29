"""Deterministic unit tests for the consolidation duplicate-create guard.

These exercise the dedup decision directly (no LLM, no DB), so they reliably
guard the fix in CI — unlike the real-LLM integration test, which only triggers
the path stochastically.
"""

import types
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import DEFAULT, AsyncMock, patch

from hindsight_api.engine.consolidation.consolidator import (
    _dedup_active,
    _dedup_reconcile_create,
    _dedup_reconcile_update,
    _DedupDecision,
    _duplicate_create_target,
    _norm_obs_text,
)
from hindsight_api.engine.search.types import RetrievalResult


@dataclass
class _FakeObs:
    id: str
    text: str


def _shown(*observations: _FakeObs) -> dict[str, _FakeObs]:
    return {_norm_obs_text(o.text): o for o in observations}


def test_norm_obs_text_collapses_whitespace_preserves_case() -> None:
    # Whitespace (incl. newlines) collapses; case is preserved.
    assert _norm_obs_text("  The  User  likes BASIL.\n") == "The User likes BASIL."
    assert _norm_obs_text(None) == ""


def test_create_matching_shown_observation_is_duplicate() -> None:
    shown = _shown(_FakeObs(id="11111111-aaaa", text="User waters the herbs early in the morning."))
    # Same text with only-whitespace differences still matches.
    target = _duplicate_create_target("User waters the   herbs early in the morning.", shown, set())
    assert target is not None
    assert target.startswith("shown observation 11111111")


def test_create_differing_only_in_case_is_not_duplicate() -> None:
    # Case-folding would lose information (e.g. acronyms), so a case-only difference
    # is treated as novel rather than silently dropped.
    shown = _shown(_FakeObs(id="22222222-bbbb", text="The user prefers TLS."))
    assert _duplicate_create_target("The user prefers tls.", shown, set()) is None


def test_create_matching_inresponse_update_is_duplicate() -> None:
    update_texts = {_norm_obs_text("Mint is kept in its own separate bed.")}
    target = _duplicate_create_target("Mint is kept in its own separate bed.", {}, update_texts)
    assert target == "an UPDATE in this response"


def test_novel_create_is_not_duplicate() -> None:
    shown = _shown(_FakeObs(id="22222222-bbbb", text="User waters the herbs early in the morning."))
    assert _duplicate_create_target("Rosemary is drought-tolerant.", shown, set()) is None
    assert _duplicate_create_target("", {}, set()) is None


# ── semantic dedup (_dedup_reconcile_create) ──────────────────────────────────
#
# Mocks the embedder, the obs-anchored ANN probe, and the LLM so the decision logic is
# tested without a DB or a real model.

_TWIN_ID = "33333333-3333-4333-8333-333333333333"


def _obs(text: str, sim: float, oid: str = _TWIN_ID) -> RetrievalResult:
    return RetrievalResult(id=oid, text=text, fact_type="observation", similarity=sim)


class _DedupConn:
    """Backend-shaped conn for dedup-fold tests. Enforces that the live-source filter and
    the fold UPDATE run inside the fold transaction on an acquired connection, and that the
    fold UPDATE is RETURNING-gated."""

    def __init__(self):
        self.active = 0  # >0 while a connection is acquired (set by _DedupBackend.acquire)
        self._in_txn = False
        self.fetchval_result = uuid.UUID(_TWIN_ID)  # survivor id the fold "returns"
        self.fetchrow_result = None  # update-path source snapshot
        self.live_rows = None  # override liveness rows; None -> echo all source ids as live
        # Modeled row text so the fold/snapshot text guards actually bite. None -> "match any"
        # (keeps every pre-existing test, which never sets these, behaving as before).
        self.current_twin_text = None  # survivor/twin row's current text (create + update folds)
        self.current_updated_text = None  # updated row's current text (update snapshot + fold)
        self.fetchval = AsyncMock(side_effect=self._fetchval)
        self.fetch = AsyncMock(side_effect=self._fetch)
        self.fetchrow = AsyncMock(side_effect=self._fetchrow)
        self.execute = AsyncMock()

    @asynccontextmanager
    async def transaction(self):
        assert self.active > 0, "fold transaction opened without an acquired connection"
        self._in_txn = True
        try:
            yield
        finally:
            self._in_txn = False

    async def _fetchval(self, query, *args):
        assert self._in_txn, "fold UPDATE must run inside the fold transaction"
        assert "RETURNING" in query, "fold UPDATE must be RETURNING-gated"
        # Assert the text-guard CLAUSE is present (not just that the arg is passed) so deleting the SQL
        # guard fails even if the param is left behind, then model the guarded row text so a stale-text
        # fold matches no row. ``args`` excludes the bound ``query``.
        if "u.text" in query:  # update-path fold
            assert "t.text = $4" in query and "u.text = $5" in query, "update fold must keep both text guards"
            if self.current_twin_text is not None and args[3] != self.current_twin_text:
                return None
            if self.current_updated_text is not None and args[4] != self.current_updated_text:
                return None
        else:  # create-path fold
            assert "AND text = $4" in query, "create fold must keep the twin text guard (AND text = $4)"
            if self.current_twin_text is not None and args[3] != self.current_twin_text:
                return None
        return self.fetchval_result

    async def _fetch(self, query, source_ids, bank_id):
        assert self._in_txn, "live-source filter must run inside the fold transaction"
        assert "FOR SHARE" in query, "live-source filter must hold FOR SHARE on the source rows"
        if self.live_rows is not None:
            return self.live_rows
        return [{"id": s} for s in source_ids]

    async def _fetchrow(self, query, *args):
        # Assert the text-guard CLAUSE is present (so deleting it fails even if the arg stays), then
        # model the updated row's text so a row rewritten during the LLM window snapshots as gone.
        # ``args`` excludes the bound ``query``.
        assert "AND text = $2" in query, "update snapshot must keep the updated-text guard (AND text = $2)"
        if self.current_updated_text is not None and args[1] != self.current_updated_text:
            return None
        return self.fetchrow_result


class _DedupBackend:
    """Backend-shaped stand-in matching acquire_with_retry's ``_wraps_backend`` path."""

    _wraps_backend = True

    def __init__(self, conn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        self._conn.active += 1
        try:
            yield self._conn
        finally:
            self._conn.active -= 1


def _make_dedup_llm(conn):
    """An LLM stub that asserts no pooled connection is held when it is called."""
    llm = types.SimpleNamespace(call=AsyncMock())

    def _assert_released(*a, **k):
        assert conn.active == 0, "no pooled connection may be held during the dedup LLM call"
        return DEFAULT  # fall through to the llm.call.return_value the test sets

    llm.call.side_effect = _assert_released
    return llm


def _ctx(threshold: float = 0.97):
    """Return (kwargs, conn_mock, llm_mock) for a _dedup_reconcile_create call."""
    conn = _DedupConn()
    llm = _make_dedup_llm(conn)
    kwargs = dict(
        pool=_DedupBackend(conn),
        memory_engine=types.SimpleNamespace(embeddings=object()),
        bank_id="bank1",
        config=types.SimpleNamespace(consolidation_dedup_threshold=threshold),
        dedup_llm_config=llm,
        create_text="YouTube content in Uzbek is very rich.",
        create_source_ids=[uuid.uuid4()],
        tags=["t1"],
    )
    return kwargs, conn, llm


def _patch_probe(results):
    return patch(
        "hindsight_api.engine.search.retrieval.retrieve_semantic_bm25_combined",
        AsyncMock(return_value={"observation": (results, [])}),
    )


def _patch_embed():
    return patch(
        "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
        AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
    )


async def test_dedup_no_twin_above_threshold_returns_none() -> None:
    kwargs, conn, llm = _ctx(threshold=0.97)
    with _patch_embed(), _patch_probe([_obs("something loosely related", 0.81)]):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None
    llm.call.assert_not_called()  # below threshold → no LLM call
    conn.fetchval.assert_not_called()  # no merge


async def test_dedup_llm_keep_does_not_merge() -> None:
    kwargs, conn, llm = _ctx()
    llm.call.return_value = _DedupDecision(action="keep", reason="different language")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None
    llm.call.assert_awaited_once()
    conn.fetchval.assert_not_called()  # kept distinct → no merge


async def test_dedup_llm_merge_folds_into_twin() -> None:
    kwargs, conn, llm = _ctx()
    kwargs["create_source_ids"] = [uuid.uuid4(), uuid.uuid4()]
    llm.call.return_value = _DedupDecision(action="merge", text="Uzbek content on YouTube is very rich.")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)]),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result == _TWIN_ID  # merged into the twin; caller skips the CREATE
    conn.fetchval.assert_awaited_once()
    args = conn.fetchval.await_args.args
    assert args[1] == "Uzbek content on YouTube is very rich."  # merged text persisted
    assert args[2] == kwargs["create_source_ids"]  # new source facts folded in
    assert args[3] == uuid.UUID(_TWIN_ID)  # onto the twin row


async def test_dedup_picks_highest_above_threshold_skips_below() -> None:
    # Only the >=threshold candidate is considered; a 0.95 result is ignored at threshold 0.97.
    kwargs, conn, llm = _ctx(threshold=0.97)
    llm.call.return_value = _DedupDecision(action="keep")
    with (
        _patch_embed(),
        _patch_probe([_obs("near but distinct", 0.95), _obs("the real twin", 0.98)]),
    ):
        await _dedup_reconcile_create(**kwargs)
    # the twin passed to the LLM is the >=0.97 one, not the 0.95
    sent = llm.call.await_args.kwargs["messages"][0]["content"]
    assert "the real twin" in sent
    assert "near but distinct" not in sent


# ── UPDATE-path dedup (_dedup_reconcile_update) ───────────────────────────────
#
# An UPDATE rewrites+re-embeds an observation, which can drift it into a near-twin of a
# DIFFERENT existing observation. These cover the fold-and-delete reconciliation (unlike
# CREATE, both rows already exist), the self-exclusion, and the keep/no-twin no-ops.

_UPDATED_ID = "44444444-4444-4444-8444-444444444444"


def _update_ctx(threshold: float = 0.97):
    """Return (kwargs, conn_mock, llm_mock) for a _dedup_reconcile_update call."""
    conn = _DedupConn()
    conn.fetchrow_result = {"source_memory_ids": [uuid.uuid4(), uuid.uuid4()]}
    llm = _make_dedup_llm(conn)
    kwargs = dict(
        pool=_DedupBackend(conn),
        memory_engine=types.SimpleNamespace(embeddings=object()),
        bank_id="bank1",
        config=types.SimpleNamespace(consolidation_dedup_threshold=threshold),
        dedup_llm_config=llm,
        updated_id=_UPDATED_ID,
        updated_text="Uzbek content on YouTube is very rich and growing.",
        updated_emb_str="[0.1, 0.2, 0.3]",  # already embedded by _execute_update_action
        tags=["t1"],
    )
    return kwargs, conn, llm


async def test_dedup_update_merge_folds_into_twin_and_deletes_updated() -> None:
    kwargs, conn, llm = _update_ctx()
    llm.call.return_value = _DedupDecision(action="merge", text="Uzbek YouTube content is very rich and growing.")
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_awaited_once()
    # The fold UPDATE uses fetchval (RETURNING t.id); then the updated row is DELETEd.
    conn.fetchval.assert_awaited_once()
    fold_args = conn.fetchval.await_args.args
    assert fold_args[1] == "Uzbek YouTube content is very rich and growing."  # merged text on the twin
    assert fold_args[2] == uuid.UUID(_TWIN_ID)  # survivor = the twin
    assert fold_args[3] == uuid.UUID(_UPDATED_ID)  # folded-from = the updated row
    assert fold_args[6] == conn.fetchrow_result["source_memory_ids"]
    conn.execute.assert_awaited_once()
    delete_args = conn.execute.await_args.args
    assert delete_args[1] == uuid.UUID(_UPDATED_ID)  # the updated row is deleted


async def test_dedup_update_keep_does_not_merge() -> None:
    kwargs, conn, llm = _update_ctx()
    llm.call.return_value = _DedupDecision(action="keep", reason="different growth claim")
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_awaited_once()
    conn.fetchval.assert_not_called()  # kept distinct → no fold
    conn.execute.assert_not_called()  # → no delete


async def test_dedup_update_excludes_self() -> None:
    # The probe surfaces the updated observation itself at 1.0; it must be excluded so we don't
    # "merge" a row into itself. With no other candidate, there is no twin → no LLM, no writes.
    kwargs, conn, llm = _update_ctx()
    with _patch_probe([_obs("its own current text", 1.0, oid=_UPDATED_ID)]):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_not_called()
    conn.fetchval.assert_not_called()
    conn.execute.assert_not_called()


async def test_dedup_update_no_twin_above_threshold() -> None:
    kwargs, conn, llm = _update_ctx(threshold=0.97)
    with _patch_probe([_obs("loosely related", 0.8)]):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_not_called()
    conn.fetchval.assert_not_called()
    conn.execute.assert_not_called()


async def test_dedup_create_twin_vanished_returns_none_so_caller_creates() -> None:
    # If the twin is deleted during the (connection-free) LLM window, the fold UPDATE matches
    # no row (fetchval -> None); the helper must return None so the caller still CREATEs.
    kwargs, conn, llm = _ctx()
    conn.fetchval_result = None
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)]),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None  # twin gone → don't drop the CREATE
    conn.fetchval.assert_awaited_once()


async def test_dedup_create_fold_uses_only_live_new_sources() -> None:
    kwargs, conn, llm = _ctx()
    live_source_id = uuid.uuid4()
    deleted_source_id = uuid.uuid4()
    kwargs["create_source_ids"] = [deleted_source_id, live_source_id]
    conn.live_rows = [{"id": live_source_id}]
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)]),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result == _TWIN_ID
    conn.fetchval.assert_awaited_once()
    assert conn.fetchval.await_args.args[2] == [live_source_id]


async def test_dedup_create_all_new_sources_deleted_returns_none() -> None:
    kwargs, conn, llm = _ctx()
    kwargs["create_source_ids"] = [uuid.uuid4(), uuid.uuid4()]
    conn.live_rows = []
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)]),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None
    conn.fetchval.assert_not_called()


async def test_dedup_update_twin_vanished_does_not_delete_updated() -> None:
    # If the fold matches no row (twin vanished mid-window), the updated row must NOT be deleted.
    kwargs, conn, llm = _update_ctx()
    conn.fetchval_result = None
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]):
        await _dedup_reconcile_update(**kwargs)
    conn.fetchval.assert_awaited_once()  # fold attempted
    conn.execute.assert_not_called()  # but no delete, since the fold touched nothing


async def test_dedup_update_fold_uses_only_live_updated_sources() -> None:
    kwargs, conn, llm = _update_ctx()
    live_source_id = uuid.uuid4()
    deleted_source_id = uuid.uuid4()
    conn.fetchrow_result = {"source_memory_ids": [deleted_source_id, live_source_id]}
    conn.live_rows = [{"id": live_source_id}]
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]):
        await _dedup_reconcile_update(**kwargs)
    conn.fetchval.assert_awaited_once()
    assert conn.fetchval.await_args.args[6] == [live_source_id]
    conn.execute.assert_awaited_once()


async def test_dedup_update_all_updated_sources_deleted_skips_fold_and_delete() -> None:
    kwargs, conn, llm = _update_ctx()
    conn.fetchrow_result = {"source_memory_ids": [uuid.uuid4(), uuid.uuid4()]}
    conn.live_rows = []
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]):
        await _dedup_reconcile_update(**kwargs)
    conn.fetchval.assert_not_called()
    conn.execute.assert_not_called()


# ── dedup activation gate (_dedup_active) ─────────────────────────────────────
#
# Enabled by default (threshold < 1.0), but skipped on Oracle because the merge path is
# Postgres-only — so the feature can ship on-by-default without breaking Oracle.


def _gate_cfg(threshold: float):
    return types.SimpleNamespace(consolidation_dedup_threshold=threshold)


def _patch_backend(name: str):
    return patch(
        "hindsight_api.engine.consolidation.consolidator.get_config",
        return_value=types.SimpleNamespace(database_backend=name),
    )


def test_dedup_active_enabled_on_postgres() -> None:
    with _patch_backend("postgresql"):
        assert _dedup_active(_gate_cfg(0.97)) is True


def test_dedup_active_disabled_when_threshold_is_one() -> None:
    with _patch_backend("postgresql"):
        assert _dedup_active(_gate_cfg(1.0)) is False


def test_dedup_active_skipped_on_oracle() -> None:
    # PG-only merge path → dedup is skipped on Oracle even with a sub-1.0 threshold.
    with _patch_backend("oracle"):
        assert _dedup_active(_gate_cfg(0.97)) is False


def test_dedup_active_none_config() -> None:
    assert _dedup_active(None) is False


async def test_process_batch_creates_when_dedup_target_vanished() -> None:
    # Caller contract: when _dedup_reconcile_create returns None (twin vanished mid-window),
    # _process_memory_batch must still CREATE the observation instead of dropping it.
    from hindsight_api.engine.consolidation import consolidator as C

    mem_id = str(uuid.uuid4())
    memories = [{"id": mem_id, "text": "Uzbek YouTube content is very rich.", "tags": []}]
    create = C._CreateAction(text="Uzbek YouTube content is very rich.", source_fact_ids=[mem_id])
    llm_result = C._BatchLLMResult(creates=[create])

    memory_engine = types.SimpleNamespace(
        _consolidation_llm_config=types.SimpleNamespace(with_config=lambda *a, **k: object())
    )

    with (
        patch.object(
            C,
            "_find_related_observations",
            new=AsyncMock(return_value=types.SimpleNamespace(results=[], source_facts={})),
        ),
        patch.object(C, "_consolidate_batch_with_llm", new=AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "_dedup_active", return_value=True),
        patch.object(C, "_dedup_reconcile_create", new=AsyncMock(return_value=None)),
        patch.object(C, "_execute_create_action", new=AsyncMock(return_value="created")) as create_action,
    ):
        result = await C._process_memory_batch(
            pool=object(),
            memory_engine=memory_engine,
            llm_config=object(),
            bank_id="bank1",
            memories=memories,
            request_context=object(),
            config=object(),
        )

    create_action.assert_awaited_once()
    assert create_action.await_args.kwargs["text"] == "Uzbek YouTube content is very rich."
    assert create_action.await_args.kwargs["source_memory_ids"] == [mem_id]
    assert result == ([{"action": "created"}], 0, False)


async def test_process_batch_reports_skipped_when_create_skipped() -> None:
    # _execute_create_action returns "skipped" (all sources deleted in the write txn) ->
    # _process_memory_batch must NOT mark the memory created; it falls through to skipped.
    from hindsight_api.engine.consolidation import consolidator as C

    mem_id = str(uuid.uuid4())
    memories = [{"id": mem_id, "text": "Uzbek YouTube content is very rich.", "tags": []}]
    create = C._CreateAction(text="Uzbek YouTube content is very rich.", source_fact_ids=[mem_id])
    llm_result = C._BatchLLMResult(creates=[create])
    memory_engine = types.SimpleNamespace(
        _consolidation_llm_config=types.SimpleNamespace(with_config=lambda *a, **k: object())
    )
    with (
        patch.object(
            C,
            "_find_related_observations",
            new=AsyncMock(return_value=types.SimpleNamespace(results=[], source_facts={})),
        ),
        patch.object(C, "_consolidate_batch_with_llm", new=AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "_dedup_active", return_value=True),
        patch.object(C, "_dedup_reconcile_create", new=AsyncMock(return_value=None)),
        patch.object(C, "_execute_create_action", new=AsyncMock(return_value="skipped")),
    ):
        result = await C._process_memory_batch(
            pool=object(),
            memory_engine=memory_engine,
            llm_config=object(),
            bank_id="bank1",
            memories=memories,
            request_context=object(),
            config=object(),
        )
    assert result == ([{"action": "skipped", "reason": "no_durable_knowledge"}], 0, False)


async def test_process_batch_reports_created_when_create_created() -> None:
    # _execute_create_action returns "created" -> the memory is marked created.
    from hindsight_api.engine.consolidation import consolidator as C

    mem_id = str(uuid.uuid4())
    memories = [{"id": mem_id, "text": "Uzbek YouTube content is very rich.", "tags": []}]
    create = C._CreateAction(text="Uzbek YouTube content is very rich.", source_fact_ids=[mem_id])
    llm_result = C._BatchLLMResult(creates=[create])
    memory_engine = types.SimpleNamespace(
        _consolidation_llm_config=types.SimpleNamespace(with_config=lambda *a, **k: object())
    )
    with (
        patch.object(
            C,
            "_find_related_observations",
            new=AsyncMock(return_value=types.SimpleNamespace(results=[], source_facts={})),
        ),
        patch.object(C, "_consolidate_batch_with_llm", new=AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "_dedup_active", return_value=True),
        patch.object(C, "_dedup_reconcile_create", new=AsyncMock(return_value=None)),
        patch.object(C, "_execute_create_action", new=AsyncMock(return_value="created")),
    ):
        result = await C._process_memory_batch(
            pool=object(),
            memory_engine=memory_engine,
            llm_config=object(),
            bank_id="bank1",
            memories=memories,
            request_context=object(),
            config=object(),
        )
    assert result == ([{"action": "created"}], 0, False)


async def test_dedup_create_fold_guards_on_twin_probe_text() -> None:
    # The CREATE fold guards on the twin's probe-time text (param $4) so a concurrent rewrite
    # of the survivor during the connection-free LLM window can't be clobbered by stale text.
    kwargs, conn, llm = _ctx()
    kwargs["create_source_ids"] = [uuid.uuid4(), uuid.uuid4()]
    llm.call.return_value = _DedupDecision(action="merge", text="Uzbek content on YouTube is very rich.")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)]),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result == _TWIN_ID
    conn.fetchval.assert_awaited_once()
    args = conn.fetchval.await_args.args
    assert args[4] == "Uzbek content on YouTube is described as very rich."  # twin probe-text guard


async def test_dedup_update_fold_guards_on_both_texts() -> None:
    # The UPDATE fold guards BOTH rows whose text fed the merge: the survivor twin ($4) and the
    # just-updated row ($5), so a concurrent rewrite of either aborts the fold-and-delete.
    kwargs, conn, llm = _update_ctx()
    llm.call.return_value = _DedupDecision(action="merge", text="Uzbek YouTube content is very rich and growing.")
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]):
        await _dedup_reconcile_update(**kwargs)
    conn.fetchval.assert_awaited_once()
    fold_args = conn.fetchval.await_args.args
    assert fold_args[4] == "Uzbek content on YouTube is described as very rich."  # survivor probe-text guard
    assert fold_args[5] == kwargs["updated_text"]  # updated-row text guard
    assert "FOR UPDATE" not in conn.fetchrow.await_args.args[0]  # sources-first lock order


# ── fold/snapshot text guards actually bite (modeled twin/updated row text) ────
#
# The fake above now models the survivor/updated row text, so a guard mismatch skips. This proves
# the WHERE ... text guards are load-bearing (not merely passed as params): deleting one flips a
# skip into a clobber/drop and fails the matching test below.


async def test_create_fold_skipped_when_twin_text_changed() -> None:
    # The CREATE fold is text-guarded (WHERE id = $3 AND text = $4). If the twin's text was rewritten
    # during the connection-free LLM window, the guard matches no row, so the helper returns None and
    # the caller still CREATEs (no silent drop). Deleting `AND text = $4` makes this fail.
    kwargs, conn, llm = _ctx()
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    conn.current_twin_text = "the twin was rewritten during the LLM window"
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)]),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None  # guard rejected the fold -> caller CREATEs rather than merging onto a stale twin
    conn.fetchval.assert_awaited_once()  # the RETURNING-gated fold actually ran and matched no row


async def test_update_fold_skipped_when_updated_text_changed() -> None:
    # The UPDATE path snapshots the updated row with a text guard (WHERE id = $1 AND text = $2). If the
    # updated observation's text was rewritten during the connection-free LLM window, the snapshot
    # matches no row and the reconciler bails BEFORE folding/deleting, so a concurrently-changed row is
    # never merged away. Deleting the snapshot's `AND text = $2` makes this fail.
    kwargs, conn, llm = _update_ctx()
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    conn.current_updated_text = "the updated row was rewritten during the LLM window"
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)]):
        await _dedup_reconcile_update(**kwargs)
    conn.fetchrow.assert_awaited_once()  # the text-guarded snapshot SELECT ran
    conn.fetchval.assert_not_called()  # snapshot matched no row -> no fold
    conn.execute.assert_not_called()  # -> the updated row is not deleted
