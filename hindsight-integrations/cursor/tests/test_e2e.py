"""End-to-end test for the Cursor Hindsight plugin's hook scripts.

Drives ``scripts/session_start.py`` and ``scripts/retain.py`` as Cursor itself
would — JSON on stdin, env vars for config — against a live Hindsight server.
Exercises the full stack: hook script → ``lib/client.py`` HTTP → Hindsight
server-side fact extraction → recall → rules-file workaround → ``.gitignore``
append + ``additionalContext`` stdout (forward-compat).

Run with::

    HINDSIGHT_API_URL=http://localhost:8888 uv run pytest tests/test_e2e.py -v

Environment variables:
    HINDSIGHT_API_URL   URL of a reachable Hindsight server
                        (default: http://localhost:8888)

The whole module is the real-LLM/real-service bucket
(``requires_real_llm`` marker) — excluded from the deterministic PR-CI bucket
via ``-m "not requires_real_llm"``; run on its own via ``-m requires_real_llm``.
The ``requires_hindsight`` skipif still gates runtime so a missing server
skips gracefully within the bucket.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
PLUGIN_ROOT = Path(__file__).parent.parent
SESSION_START_SCRIPT = PLUGIN_ROOT / "scripts" / "session_start.py"
RETAIN_SCRIPT = PLUGIN_ROOT / "scripts" / "retain.py"


def _hindsight_available() -> bool:
    try:
        with urllib.request.urlopen(f"{HINDSIGHT_API_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


requires_hindsight = pytest.mark.skipif(
    not _hindsight_available(),
    reason=f"Hindsight not reachable at {HINDSIGHT_API_URL}",
)
pytestmark = [requires_hindsight, pytest.mark.requires_real_llm]


@pytest.fixture
def live():
    """Yield (hindsight_client, bank_id) backed by a freshly-created test bank."""
    from hindsight_client import Hindsight

    client = Hindsight(base_url=HINDSIGHT_API_URL)
    bank_id = f"cursor-e2e-{uuid.uuid4().hex[:8]}"
    client.create_bank(bank_id, name=f"Cursor E2E {bank_id}")
    try:
        yield client, bank_id
    finally:
        try:
            client.delete_bank(bank_id)
        except Exception:
            pass
        client.close()


@pytest.fixture
def hook_env(tmp_path):
    """Env vars + isolated state/config dirs for hook subprocess invocations.

    Critically:
      * ``HOME`` is redirected to ``tmp_path`` so the hook doesn't read the
        developer's ``~/.hindsight/cursor.json`` mid-test.
      * ``CURSOR_PLUGIN_DATA`` is set so the state files (``last_recall.json``,
        ``bank_missions.json``) land in tmp and don't pollute real state.
      * ``HINDSIGHT_API_URL`` overrides the empty-string default in
        ``settings.json``, short-circuiting the daemon-start codepath.
    """
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path),
            "CURSOR_PLUGIN_DATA": str(tmp_path / "plugin-data"),
            "HINDSIGHT_API_URL": HINDSIGHT_API_URL,
            "HINDSIGHT_USE_RULES_FILE_FALLBACK": "true",
            "HINDSIGHT_APPEND_TO_GITIGNORE": "true",
            "HINDSIGHT_DEBUG": "false",
            # Force a focused bank mission so the hook's session-recall query
            # (which is bank-mission + project-name) actually matches the
            # seeded facts. The default mission in settings.json is broad
            # boilerplate; that's correct for production but too diffuse to
            # surface targeted test fixtures within a 15s wait.
            "HINDSIGHT_BANK_MISSION": (
                "Track project canary markers and user coding preferences for the development assistant."
            ),
            # retain.py batches retains every N turns (default 10). For a
            # single-shot E2E that drives one stop event, force every-turn
            # retain so the bank actually receives the transcript.
            "HINDSIGHT_RETAIN_EVERY_N_TURNS": "1",
        }
    )
    return env


def _run_hook(script: Path, hook_input: dict, env: dict) -> subprocess.CompletedProcess:
    """Invoke a hook script the way Cursor does: JSON on stdin, env vars for config."""
    return subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(hook_input),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _make_workspace(root: Path, init_git: bool = True) -> Path:
    workspace = root / "workspace"
    workspace.mkdir()
    if init_git:
        # Just a marker dir — `ensure_gitignored` checks `.git` exists, not that
        # it's a valid repository.
        (workspace / ".git").mkdir()
    (workspace / "README.md").write_text("# Test workspace\n")
    return workspace


def _wait_for_recall_to_surface(client, bank_id: str, query: str, attempts: int = 30, delay: float = 1.0) -> list:
    """Poll Hindsight until recall returns at least one result.

    Retain → server-side extraction → recall isn't instantaneous. Polling with
    a deadline (vs. a fixed sleep) keeps the test fast in the happy path and
    fails loudly if extraction never completes.
    """
    for _ in range(attempts):
        results = client.recall(bank_id=bank_id, query=query).results or []
        if results:
            return results
        time.sleep(delay)
    pytest.fail(
        f"recall({query!r}) returned no memories after {attempts * delay:.0f}s — "
        "Hindsight extraction did not surface the seeded content."
    )


class TestE2ESessionStart:
    def test_writes_rules_file_with_recalled_content_and_appends_gitignore(self, live, hook_env, tmp_path):
        client, bank_id = live
        # The fact is phrased to overlap with the bank-mission query the hook
        # builds, so it actually surfaces in the session-start recall (otherwise
        # the recall returns empty and the workaround correctly skips the
        # rules-file write — which would make the test a no-op).
        client.retain(
            bank_id,
            "The user prefers the canary marker FROBNICATE_QUUX_42 for "
            "tracking deployments. Coding preference: tabs over spaces.",
        )
        _wait_for_recall_to_surface(client, bank_id, "canary marker preference")

        workspace = _make_workspace(tmp_path)
        hook_env["HINDSIGHT_BANK_ID"] = bank_id

        result = _run_hook(
            SESSION_START_SCRIPT,
            {
                "conversation_id": "e2e-1",
                "session_id": "e2e-1",
                "hook_event_name": "sessionStart",
                "workspace_roots": [str(workspace)],
                "cursor_version": "3.6.31-test",
            },
            hook_env,
        )
        assert result.returncode == 0, (
            f"session_start exited {result.returncode}\nstdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )

        # Rules-file workaround content
        rule_path = workspace / ".cursor" / "rules" / "hindsight-session.mdc"
        assert rule_path.exists(), f"rules file missing at {rule_path}"
        content = rule_path.read_text()
        assert "alwaysApply: true" in content, "rules file missing alwaysApply: true"
        assert "<hindsight_memories>" in content, "rules file missing memory wrapper"
        assert "FROBNICATE_QUUX_42" in content, f"canary not surfaced in rules file:\n{content}"
        # In-file comment links to the upstream Cursor bug for future readers
        assert "forum.cursor.com" in content
        assert "158452" in content

        # .gitignore was appended once, with the workspace-anchored path
        gitignore = workspace / ".gitignore"
        assert gitignore.exists(), "session_start did not create .gitignore"
        gi_text = gitignore.read_text()
        assert "/.cursor/rules/hindsight-session.mdc" in gi_text

        # Forward-compat: stdout is the additionalContext JSON Cursor would consume
        # if/when the upstream bug is fixed.
        stdout_json = json.loads(result.stdout)
        assert "additionalContext" in stdout_json
        assert "FROBNICATE_QUUX_42" in stdout_json["additionalContext"]

    def test_empty_bank_writes_no_rules_file_but_still_succeeds(self, live, hook_env, tmp_path):
        _, bank_id = live  # bank exists but is empty
        workspace = _make_workspace(tmp_path)
        hook_env["HINDSIGHT_BANK_ID"] = bank_id

        result = _run_hook(
            SESSION_START_SCRIPT,
            {
                "conversation_id": "e2e-2",
                "session_id": "e2e-2",
                "hook_event_name": "sessionStart",
                "workspace_roots": [str(workspace)],
                "cursor_version": "3.6.31-test",
            },
            hook_env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr!r}"
        # No memories → no rules file (rotation cleared anything stale, write
        # was skipped because recall surfaced nothing).
        rule_path = workspace / ".cursor" / "rules" / "hindsight-session.mdc"
        assert not rule_path.exists()

    def test_useRulesFileFallback_false_disables_workaround(self, live, hook_env, tmp_path):
        client, bank_id = live
        # Use the same on-topic fixture as the first test so the hook's
        # session-recall reliably surfaces content. The point of *this* test
        # is the opt-out behaviour — that with the flag off, no workspace
        # file system mutations happen — not the stdout content.
        client.retain(
            bank_id,
            "The user prefers the canary marker FROBNICATE_QUUX_42 for "
            "tracking deployments. Coding preference: tabs over spaces.",
        )
        _wait_for_recall_to_surface(client, bank_id, "canary marker preference")

        workspace = _make_workspace(tmp_path)
        hook_env["HINDSIGHT_BANK_ID"] = bank_id
        hook_env["HINDSIGHT_USE_RULES_FILE_FALLBACK"] = "false"

        result = _run_hook(
            SESSION_START_SCRIPT,
            {
                "conversation_id": "e2e-3",
                "session_id": "e2e-3",
                "hook_event_name": "sessionStart",
                "workspace_roots": [str(workspace)],
                "cursor_version": "3.6.31-test",
            },
            hook_env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr!r}"
        # Workaround disabled → no rules-dir, no .gitignore touch. This is the
        # only assertion that matters for the opt-out contract.
        assert not (workspace / ".cursor").exists(), ".cursor was created despite useRulesFileFallback=false"
        assert not (workspace / ".gitignore").exists(), ".gitignore was created despite useRulesFileFallback=false"


class TestE2ERetain:
    def test_retain_persists_transcript_so_recall_surfaces_it(self, live, hook_env, tmp_path):
        client, bank_id = live
        hook_env["HINDSIGHT_BANK_ID"] = bank_id

        # Cursor's stop hook delivers the conversation via a JSONL transcript
        # file path, not an inline messages array. Write the test fixture in
        # that exact shape so retain.py's parsing path is the one actually
        # under test.
        transcript_path = tmp_path / "transcript.jsonl"
        with open(transcript_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "role": "user",
                        "content": "We picked Postgres 16 in us-east-1 for the new analytics service.",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "Confirmed — Postgres 16 in us-east-1 for analytics.",
                    }
                )
                + "\n"
            )

        result = _run_hook(
            RETAIN_SCRIPT,
            {
                "conversation_id": "retain-e2e-1",
                "session_id": "retain-e2e-1",
                "hook_event_name": "stop",
                "workspace_roots": [str(tmp_path / "ws")],
                "cursor_version": "3.6.31-test",
                "transcript_path": str(transcript_path),
            },
            hook_env,
        )
        # retain.py exits 0 on success or graceful no-op; don't assert on
        # output shape. What matters is that the bank ends up holding the
        # fact, verified via direct recall (not the hook's session query,
        # which is bank-mission-shaped for auto-inject).
        assert result.returncode == 0, f"retain exited {result.returncode}\nstderr: {result.stderr!r}"
        results = _wait_for_recall_to_surface(client, bank_id, "Postgres deployment region")
        joined = " | ".join(r.text for r in results).lower()
        assert "postgres" in joined or "us-east" in joined, (
            f"recall surfaced results but none referenced the retained content: {joined[:400]}"
        )
