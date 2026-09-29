"""Tests for token-batched, parallel commit summarization and its dedup pass.

The LLM calls themselves are stubbed: what these pin is the batching arithmetic
and the guarantees the dedup pass makes about the model's output (order, and that
a commit_id the model invented or dropped can't corrupt the changelog).
"""

import json

from hindsight_api.engine.token_encoding import count_tokens

from hindsight_dev.generate_changelog import (
    BATCH_TOKEN_BUDGET,
    ChangelogEntry,
    Commit,
    analyze_commits_with_llm,
    batch_commits,
    deduplicate_entries,
)


def _max_commit_tokens(batch: list[Commit]) -> int:
    """A batch may exceed the budget by its last commit, which is added before the check."""
    return max(count_tokens(json.dumps({"commit_id": c.hash, "message": c.message})) for c in batch)


def _commits(n: int, message: str = "fix(api): something user-facing happened here") -> list[Commit]:
    return [Commit(hash=f"{i:09x}", message=f"{message} (#{i})") for i in range(n)]


class _StubClient:
    """Returns one entry per commit named in the prompt, and records the calls."""

    def __init__(self, dedup_result: list[ChangelogEntry] | None = None):
        self.prompts: list[str] = []
        self._dedup_result = dedup_result
        parse = self._parse
        self.beta = type(
            "_Beta",
            (),
            {"chat": type("_Chat", (), {"completions": type("_C", (), {"parse": staticmethod(parse)})()})()},
        )()

    def _parse(self, *, model, messages, response_format, max_completion_tokens):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("These changelog entries were produced independently"):
            entries = self._dedup_result if self._dedup_result is not None else []
        else:
            payload = json.loads(prompt.split("Commits:\n", 1)[1].split("\n\nFiles changed", 1)[0])
            entries = [
                ChangelogEntry(category="bugfix", summary=c["message"], commit_id=c["commit_id"]) for c in payload
            ]
        parsed = response_format(entries=entries)
        return type("_R", (), {"choices": [type("_C", (), {"message": type("_M", (), {"parsed": parsed})()})()]})()


def test_batches_stay_within_the_token_budget_and_preserve_order():
    commits = _commits(200)
    batches = batch_commits(commits)

    assert len(batches) > 1
    assert [c for batch in batches for c in batch] == commits
    for batch in batches:
        assert sum(count_tokens(json.dumps({"commit_id": c.hash, "message": c.message})) for c in batch) <= (
            BATCH_TOKEN_BUDGET + _max_commit_tokens(batch)
        )


def test_the_budget_is_read_at_call_time(monkeypatch):
    """A default bound at import would make raising the constant a silent no-op."""
    commits = _commits(200)
    assert len(batch_commits(commits)) > 1

    monkeypatch.setattr("hindsight_dev.generate_changelog.BATCH_TOKEN_BUDGET", 10**9)
    assert batch_commits(commits) == [commits]


def test_a_single_commit_over_budget_gets_its_own_batch():
    huge = Commit(hash="deadbeef1", message="x " * BATCH_TOKEN_BUDGET)
    batches = batch_commits([huge, *_commits(2)])

    assert batches[0] == [huge]
    assert len(batches) > 1


def test_a_small_release_is_one_call_with_no_dedup_pass():
    client = _StubClient()
    entries = analyze_commits_with_llm(client, "m", "0.1.0", _commits(3), file_diff="")

    assert len(client.prompts) == 1
    assert [e.commit_id for e in entries] == [c.hash for c in _commits(3)]


def test_a_large_release_fans_out_and_deduplicates():
    commits = _commits(200)
    client = _StubClient(dedup_result=[])  # model drops everything -> fall back to the union
    entries = analyze_commits_with_llm(client, "m", "0.1.0", commits, file_diff="")

    analysis_prompts = [p for p in client.prompts if p.startswith("Analyze the following")]
    assert len(analysis_prompts) == len(batch_commits(commits))
    assert sum(p.startswith("These changelog entries") for p in client.prompts) == 1
    # Every commit survives, in release order, despite the empty dedup response.
    assert [e.commit_id for e in entries] == [c.hash for c in commits]


def test_dedup_keeps_release_order_and_rejects_unknown_commit_ids():
    commits = _commits(3)
    entries = [ChangelogEntry(category="bugfix", summary=c.message, commit_id=c.hash) for c in reversed(commits)]
    kept = [
        ChangelogEntry(category="bugfix", summary="merged", commit_id=commits[2].hash),
        ChangelogEntry(category="bugfix", summary="hallucinated", commit_id="ffffffff0"),
        ChangelogEntry(category="bugfix", summary="kept", commit_id=commits[0].hash),
    ]
    client = _StubClient(dedup_result=kept)

    result = deduplicate_entries(client, "m", entries, [c.hash for c in commits])

    assert [e.commit_id for e in result] == [commits[0].hash, commits[2].hash]


def test_dedup_drops_a_repeated_commit_id_before_asking_the_model():
    commits = _commits(2)
    duplicated = [
        ChangelogEntry(category="bugfix", summary="first wording", commit_id=commits[0].hash),
        ChangelogEntry(category="bugfix", summary="second wording", commit_id=commits[0].hash),
        ChangelogEntry(category="bugfix", summary="other", commit_id=commits[1].hash),
    ]
    client = _StubClient(dedup_result=[])

    result = deduplicate_entries(client, "m", duplicated, [c.hash for c in commits])

    assert [e.summary for e in result] == ["first wording", "other"]
    sent = json.loads(client.prompts[0].split("Entries:\n", 1)[1])
    assert len(sent) == 2
