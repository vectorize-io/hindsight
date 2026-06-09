"""Hook logic: recall injection, retain accumulation, graceful degradation."""

import io
import json

from conftest import base_config, make_hook, make_memory

from lib import content, hooks_impl


# ── Recall (UserPromptSubmit / TaskStart) ────────────────────────────────────


def test_recall_injects_memories_block(http):
    http.results = [make_memory("User prefers TypeScript")]
    out = hooks_impl.handle_user_prompt_submit(make_hook(prompt="how should I write this?"), base_config())
    assert "<hindsight_memories>" in out
    assert "User prefers TypeScript" in out


def test_recall_appends_prompt_to_transcript(http):
    hooks_impl.handle_user_prompt_submit(make_hook(prompt="add a login form", task_id="t9"), base_config())
    assert content.read_transcript("t9") == [{"role": "user", "content": "add a login form"}]


def test_recall_returns_empty_when_disabled(http):
    out = hooks_impl.handle_user_prompt_submit(make_hook(prompt="anything here"), base_config(autoRecall=False))
    assert out == ""


def test_recall_skips_short_prompts(http):
    assert hooks_impl.handle_user_prompt_submit(make_hook(prompt="hi"), base_config()) == ""


def test_recall_empty_when_no_memories(http):
    http.results = []
    assert hooks_impl.handle_user_prompt_submit(make_hook(prompt="a real question"), base_config()) == ""


def test_task_start_recalls_and_seeds_transcript(http):
    http.results = [make_memory("repo uses pnpm")]
    out = hooks_impl.handle_task_start(make_hook(hook_name="TaskStart", task="set up CI", task_id="tk"), base_config())
    assert "repo uses pnpm" in out
    assert content.read_transcript("tk")[0]["content"] == "set up CI"


# ── Retain (TaskComplete / TaskCancel) ───────────────────────────────────────


def test_retain_posts_accumulated_transcript(http):
    content.append_turn("done1", "user", "build the parser")
    hooks_impl.handle_retain(
        make_hook(hook_name="TaskComplete", task_id="done1", task="parser built"), base_config(), "completed"
    )
    posts = http.retain_calls()
    assert len(posts) == 1
    item = posts[0].body["items"][0]
    assert "build the parser" in item["content"]
    assert "parser built" in item["content"]  # completion summary recorded
    assert item["document_id"] == "done1"
    assert item["metadata"]["status"] == "completed"


def test_retain_clears_transcript_after_posting(http):
    content.append_turn("done2", "user", "x")
    hooks_impl.handle_retain(make_hook(task_id="done2"), base_config(), "completed")
    assert content.read_transcript("done2") == []


def test_retain_skipped_when_disabled(http):
    content.append_turn("d3", "user", "x")
    hooks_impl.handle_retain(make_hook(task_id="d3"), base_config(autoRetain=False), "completed")
    assert http.retain_calls() == []


def test_retain_skips_empty_transcript(http):
    hooks_impl.handle_retain(make_hook(task_id="empty", task=""), base_config(), "completed")
    assert http.retain_calls() == []


def test_retain_tags_render_task_id(http):
    content.append_turn("tg", "user", "x")
    hooks_impl.handle_retain(make_hook(task_id="tg"), base_config(retainTags=["{task_id}", "cline"]), "completed")
    assert http.retain_calls()[0].body["items"][0]["tags"] == ["tg", "cline"]


# ── Entrypoints (stdin → stdout JSON contract) ───────────────────────────────


def test_main_user_prompt_submit_emits_cline_json(http, monkeypatch, capsys):
    http.results = [make_memory("uses Postgres")]
    payload = json.dumps(
        {"hookName": "UserPromptSubmit", "prompt": "which database?", "taskId": "m1", "workspaceRoots": ["/x"]}
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    hooks_impl.main_user_prompt_submit()
    out = json.loads(capsys.readouterr().out)
    assert out["cancel"] is False
    assert "uses Postgres" in out["contextModification"]
    assert out["errorMessage"] == ""


def test_main_degrades_gracefully_when_server_down(http, monkeypatch, capsys):
    http.fail = True  # every request raises
    payload = json.dumps(
        {"hookName": "UserPromptSubmit", "prompt": "a real question", "taskId": "m2", "workspaceRoots": ["/x"]}
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    hooks_impl.main_user_prompt_submit()  # must not raise
    out = json.loads(capsys.readouterr().out)
    assert out["cancel"] is False
    assert out["contextModification"] == ""
