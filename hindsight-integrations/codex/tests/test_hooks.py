"""End-to-end tests for recall.py and retain.py hook scripts.

Mocks the Codex hook runtime:
  - stdin  → io.StringIO(json.dumps(hook_input))
  - stdout → io.StringIO() captured for assertions
  - urllib.request.urlopen → fake HTTP responses
  - HOME → tmp_path (isolates ~/.hindsight/codex.json and state)
"""

import importlib
import io
import json
import os
import sys
from unittest.mock import patch

import pytest
from conftest import FakeHTTPResponse, make_hook_input, make_memory, make_transcript_file, make_user_config
from lib import state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_hook(
    module_name,
    hook_input,
    monkeypatch,
    tmp_path,
    urlopen_side_effect=None,
    user_config=None,
    api_url="http://fake:9077",
    retain_idempotency_capability=True,
    retain_serialized_upsert_capability=True,
    module_mutator=None,
):
    """Import and run a hook script's main() with mocked stdin/stdout/HTTP."""
    # Isolate HOME so ~/.hindsight/codex.json and state land in tmp_path
    monkeypatch.setenv("HOME", str(tmp_path))

    # Strip real HINDSIGHT_* env vars
    for k in list(os.environ):
        if k.startswith("HINDSIGHT_"):
            monkeypatch.delenv(k, raising=False)

    # Set required API URL via env var
    monkeypatch.setenv("HINDSIGHT_API_URL", api_url)

    # Write user config (enables retain on every turn + any overrides)
    cfg = {"retainEveryNTurns": 1, "autoRecall": True, "autoRetain": True}
    if user_config:
        cfg.update(user_config)
    make_user_config(tmp_path, cfg)

    stdin_data = io.StringIO(json.dumps(hook_input))
    stdout_capture = io.StringIO()

    # Force reimport so the module picks up patched env
    scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
    spec = importlib.util.spec_from_file_location(
        module_name + "_fresh", os.path.join(scripts_dir, f"{module_name}.py")
    )
    mod = importlib.util.module_from_spec(spec)

    default_response = FakeHTTPResponse({"results": []})
    side_effect = urlopen_side_effect or (lambda *a, **kw: default_response)

    def route_request(req, *args, **kwargs):
        if req.get_method() == "GET" and req.full_url.endswith("/version"):
            features = {}
            if retain_idempotency_capability:
                features["retain_idempotency"] = True
            if retain_serialized_upsert_capability:
                features["retain_serialized_upsert"] = True
            return FakeHTTPResponse({"api_version": "test", "features": features})
        return side_effect(req, *args, **kwargs)

    with (
        patch("sys.stdin", stdin_data),
        patch("sys.stdout", stdout_capture),
        patch("urllib.request.urlopen", side_effect=route_request),
    ):
        spec.loader.exec_module(mod)
        if module_mutator is not None:
            module_mutator(mod)
        mod.main()

    return stdout_capture.getvalue()


def _pending_retain_files(tmp_path):
    state_dir = tmp_path / ".hindsight" / "codex" / "state"
    return [path for path in state_dir.glob("retain-*.json") if not path.name.endswith(".cadence.json")]


# ---------------------------------------------------------------------------
# recall hook
# ---------------------------------------------------------------------------


class TestRecallHook:
    def test_outputs_additional_context_when_memories_found(self, monkeypatch, tmp_path):
        memory = make_memory("Paris is the capital of France", "world")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What is the capital of France?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=lambda *a, **kw: response)

        data = json.loads(output)
        context = data["hookSpecificOutput"]["additionalContext"]
        assert "Paris is the capital of France" in context
        assert "<hindsight_memories>" in context

    def test_recall_min_scores_filters_low_scoring_memories(self, monkeypatch, tmp_path):
        low_semantic = make_memory("Marginal match")
        low_semantic["scores"] = {"semantic": 0.42, "reranker": 0.8}
        low_reranker = make_memory("Junk reranker match")
        low_reranker["scores"] = {"semantic": 0.9, "reranker": 0.03}
        no_scores = make_memory("BM25-only match")
        good = make_memory("Relevant match")
        good["scores"] = {"semantic": 0.91, "reranker": 0.45}
        response = FakeHTTPResponse({"results": [low_semantic, low_reranker, no_scores, good]})

        hook_input = make_hook_input(prompt="What deployment rule applies?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
            user_config={"recallMinScores": {"semantic": 0.65, "reranker": 0.2}},
        )

        context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
        assert "Marginal match" not in context
        assert "Junk reranker match" not in context
        assert "BM25-only match" in context
        assert "Relevant match" in context

    def test_recall_min_scores_ignores_invalid_floor(self, monkeypatch, tmp_path):
        memory = make_memory("Relevant match")
        memory["scores"] = {"semantic": 0.91}
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What deployment rule applies?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
            user_config={"recallMinScores": {"semantic": None}},
        )

        context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
        assert "Relevant match" in context

    def test_no_output_when_no_memories(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hello there world")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_no_output_for_short_prompt(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hi")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_graceful_on_api_error(self, monkeypatch, tmp_path):
        def raise_error(*a, **kw):
            raise OSError("connection refused")

        hook_input = make_hook_input(prompt="What is my project about?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)
        assert output.strip() == ""

    def test_output_format_matches_codex_spec(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=lambda *a, **kw: response)

        data = json.loads(output)
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "additionalContext" in data["hookSpecificOutput"]

    def test_multi_turn_context_from_transcript(self, monkeypatch, tmp_path):
        """When recallContextTurns > 1, prior transcript is included in query."""
        messages = [
            {"role": "user", "content": "I use Python for all my scripts"},
            {"role": "assistant", "content": "Noted!"},
        ]
        transcript = make_transcript_file(tmp_path, messages)

        captured_body = {}

        def capture_and_respond(req, timeout=None):
            if "/recall" in req.full_url:
                captured_body["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"results": []})

        hook_input = make_hook_input(prompt="What language should I use?", transcript_path=transcript)
        _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture_and_respond,
            user_config={"recallContextTurns": 2},
        )

        if "body" in captured_body:
            assert "Python" in captured_body["body"].get("query", "")

    def test_recall_timeout_is_configurable(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        captured = {}

        def capture_timeout(req, timeout=None):
            captured["timeout"] = timeout
            return FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture_timeout,
            user_config={"recallTimeout": 42},
        )

        data = json.loads(output)
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert captured["timeout"] == 42

    def test_disabled_auto_recall_produces_no_output(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="What is the capital of France?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, user_config={"autoRecall": False})
        assert output.strip() == ""


# ---------------------------------------------------------------------------
# retain hook
# ---------------------------------------------------------------------------


class TestRetainHook:
    @pytest.mark.parametrize(
        ("retain_idempotency_capability", "retain_serialized_upsert_capability"),
        [(False, True), (True, False)],
    )
    def test_old_server_is_not_sent_retain_and_pending_work_is_preserved(
        self,
        monkeypatch,
        tmp_path,
        retain_idempotency_capability,
        retain_serialized_upsert_capability,
    ):
        transcript = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        )
        hook_input = make_hook_input(
            transcript_path=transcript,
            session_id="sess-old-server",
        )
        requests = []

        def capture(req, *args, **kwargs):
            requests.append((req.get_method(), req.full_url))
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            retain_idempotency_capability=retain_idempotency_capability,
            retain_serialized_upsert_capability=retain_serialized_upsert_capability,
        )

        assert requests == []
        assert _pending_retain_files(tmp_path)

    def test_posts_transcript_to_hindsight(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)

        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"status": "accepted"})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured, "retain API was not called"
        assert "hello" in captured["body"]["items"][0]["content"]

    def test_no_retain_on_empty_transcript(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(transcript_path="/nonexistent/transcript.jsonl")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)
        assert "called" not in captured

    def test_strips_memory_tags_before_retaining(self, monkeypatch, tmp_path):
        messages = [
            {"role": "user", "content": "<hindsight_memories>old memories</hindsight_memories> actual question"},
            {"role": "assistant", "content": "sure!"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            content = captured["body"]["items"][0]["content"]
            assert "old memories" not in content
            assert "actual question" in content

    def test_retain_posts_async_true(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            assert captured["body"].get("async") is True

    def test_retain_includes_codex_context_label(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            assert captured["body"]["items"][0]["context"] == "codex"

    def test_retain_skips_below_every_n_turns_threshold(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        # retainEveryNTurns=3 — first call should be skipped
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 3},
        )
        assert "called" not in captured

    def test_retain_uses_session_id_as_document_id(self, monkeypatch, tmp_path):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-doc-test")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured
        assert captured["body"]["items"][0]["document_id"] == "sess-doc-test"

    def test_graceful_on_retain_api_error(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)

        def raise_error(req, timeout=None):
            if "/memories" in req.full_url:
                raise OSError("connection refused")
            return FakeHTTPResponse({})

        # Should not raise
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)

    def test_lost_ack_retries_same_submission_before_advancing_current_chunk(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "stable"}, {"role": "assistant", "content": "answer"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-lost-ack")
        captured = []

        def lose_ack(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured.append(json.loads(req.data.decode()))
                raise TimeoutError("response lost")
            return FakeHTTPResponse({})

        # First turn advances the checkpoint but does not retain.
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *args, **kwargs: FakeHTTPResponse({}),
            user_config={"retainEveryNTurns": 2},
        )
        messages.extend([{"role": "user", "content": "second"}, {"role": "assistant", "content": "second answer"}])
        make_transcript_file(tmp_path, messages)
        # Second turn is accepted server-side but its acknowledgment is lost.
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lose_ack,
            user_config={"retainEveryNTurns": 2},
        )

        # The transcript may grow before the next Stop hook. The pending request
        # must still be replayed byte-for-byte with the same identity.
        messages.extend([{"role": "user", "content": "newer"}, {"role": "assistant", "content": "reply"}])
        make_transcript_file(tmp_path, messages)

        def acknowledge(req, timeout=None):
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "completed"})
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured.append(json.loads(req.data.decode()))
            return FakeHTTPResponse({"operation_id": "original-op"})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
            user_config={"retainEveryNTurns": 2},
        )

        assert len(captured) == 2
        assert captured[1] == captured[0]
        cadence = state.read_retain_cadence("sess-lost-ack")
        assert cadence["turn_count"] == 3
        assert not _pending_retain_files(tmp_path)

    def test_due_chunk_is_submitted_after_pending_retry(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-pending-and-current")
        captured = []

        def lose_ack(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured.append(json.loads(req.data.decode()))
                raise TimeoutError("response lost")
            return FakeHTTPResponse({})

        # Turn 1 skips. Turn 2 persists and sends the first chunk, but loses its
        # acknowledgment. Turn 3 retries and still fails, leaving it pending.
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *args, **kwargs: FakeHTTPResponse({}),
            user_config={"retainEveryNTurns": 2},
        )
        messages.extend([{"role": "user", "content": "turn-two"}, {"role": "assistant", "content": "turn-two answer"}])
        make_transcript_file(tmp_path, messages)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lose_ack,
            user_config={"retainEveryNTurns": 2},
        )
        messages.extend(
            [{"role": "user", "content": "turn-three"}, {"role": "assistant", "content": "turn-three answer"}]
        )
        make_transcript_file(tmp_path, messages)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lose_ack,
            user_config={"retainEveryNTurns": 2},
        )

        messages.extend([{"role": "user", "content": "current"}, {"role": "assistant", "content": "new answer"}])
        make_transcript_file(tmp_path, messages)

        def acknowledge(req, timeout=None):
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "completed"})
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured.append(json.loads(req.data.decode()))
            return FakeHTTPResponse({"operation_id": "ok"})

        # Turn 4 is itself due. It must replay the pending request first and
        # then submit the newly-derived current chunk with a new identity.
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
            user_config={"retainEveryNTurns": 2},
        )

        assert len(captured) == 4
        assert captured[1] == captured[0]
        assert captured[2] == captured[0]
        assert captured[3]["idempotency_key"] != captured[0]["idempotency_key"]
        assert "current" in captured[3]["items"][0]["content"]
        assert state.read_retain_cadence("sess-pending-and-current")["turn_count"] == 4

    def test_acknowledged_retain_allows_immediate_followup_for_same_session(self, monkeypatch, tmp_path):
        session_id = "sess-one-in-flight"
        transcript = make_transcript_file(
            tmp_path,
            [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}],
        )
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)
        posts = []

        def respond(req, timeout=None):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                posts.append(json.loads(req.data.decode()))
                return FakeHTTPResponse({"operation_id": f"op-{len(posts)}"})
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=respond)
        make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "new answer"},
            ],
        )
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=respond)

        assert len(posts) == 2
        assert "second" in posts[1]["items"][0]["content"]

    def test_pruned_acknowledged_operation_unblocks_queue_without_repost(self, monkeypatch, tmp_path):
        session_id = "sess-pruned-operation"
        transcript = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "already retained"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)
        posts = []

        def first_pass(req, *args, **kwargs):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                posts.append(json.loads(req.data.decode()))
                return FakeHTTPResponse({"operation_id": "op-pruned"})
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "processing"})
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=first_pass,
        )

        def pruned(req, *args, **kwargs):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                posts.append(json.loads(req.data.decode()))
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "not_found"})
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=pruned,
        )

        assert len(posts) == 1
        assert state.read_pending_retains(session_id) == []

    def test_pending_retain_is_not_replayed_to_a_different_api(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "private"}, {"role": "assistant", "content": "answer"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-api-change")
        calls = []

        def lose_ack(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                calls.append(req.full_url)
                raise TimeoutError("response lost")
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=lose_ack)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lose_ack,
            api_url="http://other:9077",
        )

        assert calls == ["http://fake:9077/v1/default/banks/codex/memories"]
        assert list((tmp_path / ".hindsight" / "codex" / "state").glob("retain-*.json"))

    def test_pending_retain_is_not_replayed_with_different_credentials(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "private"}, {"role": "assistant", "content": "answer"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-auth-change")
        calls = []

        def lose_ack(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                calls.append(req.get_header("Authorization"))
                raise TimeoutError("response lost")
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lose_ack,
            user_config={"hindsightApiToken": "token-one"},
        )
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lose_ack,
            user_config={"hindsightApiToken": "token-two"},
        )

        assert calls == ["Bearer token-one"]
        assert list((tmp_path / ".hindsight" / "codex" / "state").glob("retain-*.json"))

    def test_prolonged_failure_queues_every_due_chunk_window(self, monkeypatch, tmp_path):
        session_id = "sess-prolonged-outage"
        transcript = make_transcript_file(tmp_path, [])
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)
        captured = []

        def fail_retain(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                raise TimeoutError("still unavailable")
            return FakeHTTPResponse({})

        def acknowledge(req, timeout=None):
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "completed"})
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured.append(json.loads(req.data.decode()))
            return FakeHTTPResponse({"operation_id": "ok"})

        messages = []
        for turn in range(1, 7):
            messages.extend(
                [
                    {"role": "user", "content": f"user-turn-{turn}"},
                    {"role": "assistant", "content": f"assistant-turn-{turn}"},
                ]
            )
            make_transcript_file(tmp_path, messages)
            _run_hook(
                "retain",
                hook_input,
                monkeypatch,
                tmp_path,
                urlopen_side_effect=fail_retain,
                user_config={
                    "retainEveryNTurns": 2,
                    "retainMode": "chunked",
                    "retainOverlapTurns": 0,
                },
            )

        messages.extend(
            [
                {"role": "user", "content": "user-turn-7"},
                {"role": "assistant", "content": "assistant-turn-7"},
            ]
        )
        make_transcript_file(tmp_path, messages)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
            user_config={
                "retainEveryNTurns": 2,
                "retainMode": "chunked",
                "retainOverlapTurns": 0,
            },
        )

        assert len(captured) == 3
        retained = [body["items"][0]["content"] for body in captured]
        assert "user-turn-1" in retained[0] and "user-turn-2" in retained[0]
        assert "user-turn-3" in retained[1] and "user-turn-4" in retained[1]
        assert "user-turn-5" in retained[2] and "user-turn-6" in retained[2]
        assert not _pending_retain_files(tmp_path)

    def test_full_session_outage_coalesces_unsent_snapshots(self, monkeypatch, tmp_path):
        session_id = "sess-every-turn-outage"
        transcript = make_transcript_file(tmp_path, [])
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)
        captured = []

        def fail_retain(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                raise TimeoutError("still unavailable")
            return FakeHTTPResponse({})

        def acknowledge(req, timeout=None):
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "completed"})
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured.append(json.loads(req.data.decode()))
            return FakeHTTPResponse({"operation_id": "ok"})

        messages = []
        for turn in range(1, 4):
            messages.extend(
                [
                    {"role": "user", "content": f"user-turn-{turn}"},
                    {"role": "assistant", "content": f"assistant-turn-{turn}"},
                ]
            )
            make_transcript_file(tmp_path, messages)
            _run_hook(
                "retain",
                hook_input,
                monkeypatch,
                tmp_path,
                urlopen_side_effect=fail_retain,
            )

        messages.extend(
            [
                {"role": "user", "content": "user-turn-4"},
                {"role": "assistant", "content": "assistant-turn-4"},
            ]
        )
        make_transcript_file(tmp_path, messages)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
        )

        assert len(captured) == 2
        retained = [body["items"][0]["content"] for body in captured]
        assert "user-turn-1" in retained[0]
        assert "user-turn-4" in retained[1]
        assert not _pending_retain_files(tmp_path)

    def test_chunked_queue_limit_applies_backpressure_without_advancing_cadence(self, monkeypatch, tmp_path):
        session_id = "sess-chunked-limit"
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)

        def fail_retain(req, *args, **kwargs):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                raise TimeoutError("response lost")
            return FakeHTTPResponse({})

        def limit_to_one(module):
            module.MAX_PENDING_RETAINS = 1

        config = {"retainMode": "chunked", "retainEveryNTurns": 1}
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=fail_retain,
            user_config=config,
            module_mutator=limit_to_one,
        )

        messages.extend(
            [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        )
        make_transcript_file(tmp_path, messages)
        completed_posts = []

        def acknowledge(req, *args, **kwargs):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                completed_posts.append(json.loads(req.data.decode()))
                return FakeHTTPResponse({"operation_id": f"op-{len(completed_posts)}"})
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "completed"})
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
            user_config=config,
            module_mutator=limit_to_one,
        )

        assert len(completed_posts) == 2
        assert state.read_pending_retains(session_id) == []
        assert state.read_retain_cadence(session_id)["turn_count"] == 2

    def test_chunked_backpressure_persists_exact_window_across_outage(self, monkeypatch, tmp_path):
        session_id = "sess-chunked-overflow"
        messages = [
            {"role": "user", "content": "first-window"},
            {"role": "assistant", "content": "first-answer"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)

        def fail_retain(req, *args, **kwargs):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                raise TimeoutError("outage")
            return FakeHTTPResponse({})

        def limit_to_one(module):
            module.MAX_PENDING_RETAINS = 1

        config = {"retainMode": "chunked", "retainEveryNTurns": 1}
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=fail_retain,
            user_config=config,
            module_mutator=limit_to_one,
        )
        messages.extend(
            [
                {"role": "user", "content": "second-window"},
                {"role": "assistant", "content": "second-answer"},
            ]
        )
        make_transcript_file(tmp_path, messages)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=fail_retain,
            user_config=config,
            module_mutator=limit_to_one,
        )
        deferred = state.read_deferred_retain(session_id)
        assert deferred is not None
        assert "second-window" in deferred["request"]["content"]

        messages.extend(
            [
                {"role": "user", "content": "third-window"},
                {"role": "assistant", "content": "third-answer"},
            ]
        )
        make_transcript_file(tmp_path, messages)
        completed_posts = []

        def acknowledge(req, *args, **kwargs):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                completed_posts.append(json.loads(req.data.decode()))
                return FakeHTTPResponse({"operation_id": f"op-{len(completed_posts)}"})
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
            user_config=config,
            module_mutator=limit_to_one,
        )

        assert len(completed_posts) == 2
        assert "second-window" in completed_posts[1]["items"][0]["content"]
        assert "third-window" not in completed_posts[1]["items"][0]["content"]
        assert state.read_deferred_retain(session_id) is None

    def test_ack_state_write_failure_keeps_submission_for_idempotent_replay(self, monkeypatch, tmp_path, capsys):
        session_id = "sess-ack-write-failure"
        transcript = make_transcript_file(
            tmp_path,
            [{"role": "user", "content": "private"}, {"role": "assistant", "content": "answer"}],
        )
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)

        def fail_after_initial_persist(module):
            original_write = module.write_pending_retains
            calls = 0

            def write_then_fail_on_ack(current_session_id, submissions):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("directory fsync failed")
                return original_write(current_session_id, submissions)

            module.write_pending_retains = write_then_fail_on_ack

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda req, *args, **kwargs: (
                FakeHTTPResponse({"operation_id": "acknowledged"})
                if req.get_method() == "POST" and "/memories" in req.full_url
                else FakeHTTPResponse({})
            ),
            module_mutator=fail_after_initial_persist,
        )

        assert len(state.read_pending_retains(session_id)) == 1
        assert "idempotent replay" in capsys.readouterr().err

    def test_pending_state_write_failure_prevents_post(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "private"}, {"role": "assistant", "content": "answer"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-write-fail")
        calls = []

        monkeypatch.setenv("HOME", str(tmp_path))
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *args, **kwargs: FakeHTTPResponse({}),
            user_config={"retainEveryNTurns": 2},
        )
        messages.extend([{"role": "user", "content": "second"}, {"role": "assistant", "content": "second answer"}])
        make_transcript_file(tmp_path, messages)

        def fail_replace(source, destination):
            raise OSError("disk full")

        monkeypatch.setattr(state.os, "replace", fail_replace)

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                calls.append(req.full_url)
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 2},
        )

        assert calls == []
        assert state.read_retain_cadence("sess-write-fail")["turn_count"] == 1

    def test_corrupt_pending_state_fails_closed_without_post(self, monkeypatch, tmp_path):
        session_id = "sess-corrupt-state"
        transcript = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "private"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)
        monkeypatch.setenv("HOME", str(tmp_path))
        pending_path = state._state_file(state._retain_state_name(session_id, "json"))
        with open(pending_path, "w") as handle:
            handle.write("{corrupt")
        calls = []

        def capture(req, *args, **kwargs):
            calls.append(req.full_url)
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
        )

        assert calls == []
        with open(pending_path) as handle:
            assert handle.read() == "{corrupt"

    def test_checkpoint_failure_does_not_duplicate_durable_due_turn(self, monkeypatch, tmp_path):
        session_id = "sess-checkpoint-fail"
        messages = [{"role": "user", "content": "private"}, {"role": "assistant", "content": "answer"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)
        posts = []

        monkeypatch.setenv("HOME", str(tmp_path))
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *args, **kwargs: FakeHTTPResponse({}),
            user_config={"retainEveryNTurns": 2},
        )
        messages.extend([{"role": "user", "content": "second"}, {"role": "assistant", "content": "second answer"}])
        make_transcript_file(tmp_path, messages)
        real_write_cadence = state.write_retain_cadence

        def fail_checkpoint(*args, **kwargs):
            raise OSError("checkpoint disk full")

        monkeypatch.setattr(state, "write_retain_cadence", fail_checkpoint)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *args, **kwargs: FakeHTTPResponse({}),
            user_config={"retainEveryNTurns": 2},
        )

        assert state.read_retain_cadence(session_id)["turn_count"] == 1
        assert _pending_retain_files(tmp_path)

        monkeypatch.setattr(state, "write_retain_cadence", real_write_cadence)

        def acknowledge(req, timeout=None):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                posts.append(json.loads(req.data.decode()))
                return FakeHTTPResponse({"operation_id": "checkpoint-op"})
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "completed"})
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
            user_config={"retainEveryNTurns": 2},
        )

        assert len(posts) == 1
        assert state.read_retain_cadence(session_id)["turn_count"] == 2
        assert not _pending_retain_files(tmp_path)

    def test_api_discovery_failure_preserves_due_window_and_cadence(self, monkeypatch, tmp_path):
        session_id = "sess-api-discovery-fail"
        messages = [{"role": "user", "content": "turn-one"}, {"role": "assistant", "content": "answer-one"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            user_config={"retainEveryNTurns": 2},
        )
        messages.extend([{"role": "user", "content": "turn-two"}, {"role": "assistant", "content": "answer-two"}])
        make_transcript_file(tmp_path, messages)

        from lib import daemon

        real_get_api_url = daemon.get_api_url
        monkeypatch.setattr(
            daemon,
            "get_api_url",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("API unavailable")),
        )
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            user_config={"retainEveryNTurns": 2},
        )

        assert state.read_retain_cadence(session_id)["turn_count"] == 2
        assert _pending_retain_files(tmp_path)

        messages.extend([{"role": "user", "content": "turn-three"}, {"role": "assistant", "content": "answer-three"}])
        make_transcript_file(tmp_path, messages)
        monkeypatch.setattr(daemon, "get_api_url", real_get_api_url)
        posts = []

        def acknowledge(req, timeout=None):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                posts.append(json.loads(req.data.decode()))
                return FakeHTTPResponse({"operation_id": "recovered-op"})
            if req.get_method() == "GET" and "/operations/" in req.full_url:
                return FakeHTTPResponse({"status": "completed"})
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=acknowledge,
            user_config={"retainEveryNTurns": 2},
        )

        assert len(posts) == 1
        retained = posts[0]["items"][0]["content"]
        assert "turn-two" in retained
        assert "turn-three" not in retained
        assert state.read_retain_cadence(session_id)["turn_count"] == 3
        assert not _pending_retain_files(tmp_path)

    def test_api_discovery_failure_does_not_allow_later_endpoint_rebind(self, monkeypatch, tmp_path):
        session_id = "sess-api-discovery-rebind"
        transcript = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "private"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        hook_input = make_hook_input(transcript_path=transcript, session_id=session_id)

        from lib import daemon

        monkeypatch.setattr(
            daemon,
            "get_api_url",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("API unavailable")),
        )
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            api_url="http://first:9077",
        )

        pending = state.read_pending_retain(session_id)
        assert pending["api_url"] == "http://first:9077"

        monkeypatch.undo()
        posts = []

        def capture(req, *args, **kwargs):
            if req.get_method() == "POST" and "/memories" in req.full_url:
                posts.append(req.full_url)
            return FakeHTTPResponse({"operation_id": "unexpected"})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            api_url="http://second:9077",
        )

        assert posts == []
        assert state.read_pending_retain(session_id)["api_url"] == "http://first:9077"

    def test_disabled_auto_retain_does_not_call_api(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        captured = {}

        def capture(req, timeout=None):
            captured["called"] = True
            return FakeHTTPResponse({})

        _run_hook(
            "retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture, user_config={"autoRetain": False}
        )
        assert "called" not in captured

    def test_reads_codex_response_item_format(self, monkeypatch, tmp_path):
        """Retain should correctly parse the actual Codex on-disk transcript format."""
        messages = [
            {"role": "user", "content": "I like TypeScript"},
            {"role": "assistant", "content": "Great choice!"},
        ]
        transcript = make_transcript_file(tmp_path, messages, codex_format=True)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured, "retain API was not called"
        content = captured["body"]["items"][0]["content"]
        assert "TypeScript" in content
