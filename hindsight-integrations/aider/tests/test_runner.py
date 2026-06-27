"""Tests for the recall -> aider -> retain orchestration."""

from __future__ import annotations

from pathlib import Path

from hindsight_aider.config import AiderConfig
from hindsight_aider.runner import (
    build_aider_command,
    do_recall,
    history_size,
    read_history_delta,
    run,
)

from .conftest import make_client


def _config(tmp_path, **kw) -> AiderConfig:
    return AiderConfig(
        bank_id="proj",
        memory_filename=str(tmp_path / "mem.md"),
        chat_history_file=str(tmp_path / "hist.md"),
        **kw,
    )


class TestBuildCommand:
    def test_injects_read_when_memory(self, tmp_path):
        cfg = _config(tmp_path)
        cmd = build_aider_command(cfg, ["-m", "hi", "src/app.py"], Path(cfg.memory_filename))
        assert cmd == ["aider", "--read", cfg.memory_filename, "-m", "hi", "src/app.py"]

    def test_no_read_when_no_memory(self, tmp_path):
        cfg = _config(tmp_path)
        cmd = build_aider_command(cfg, ["src/app.py"], None)
        assert cmd == ["aider", "src/app.py"]
        assert "--read" not in cmd

    def test_custom_aider_command(self, tmp_path):
        cfg = _config(tmp_path, aider_command="aider2")
        assert build_aider_command(cfg, [], None)[0] == "aider2"


class TestDoRecall:
    def test_writes_memory_file(self, tmp_path):
        cfg = _config(tmp_path)
        client = make_client(["prefers tabs", "deploy with make ship"])
        ok = do_recall(client, cfg, "proj", "how to deploy?", Path(cfg.memory_filename))
        assert ok is True
        text = Path(cfg.memory_filename).read_text()
        assert "prefers tabs" in text and "make ship" in text
        assert client.recall.call_args.kwargs["bank_id"] == "proj"
        assert client.recall.call_args.kwargs["query"] == "how to deploy?"

    def test_no_results_writes_nothing(self, tmp_path):
        cfg = _config(tmp_path)
        ok = do_recall(make_client([]), cfg, "proj", "q", Path(cfg.memory_filename))
        assert ok is False
        assert not Path(cfg.memory_filename).exists()

    def test_recall_error_is_swallowed(self, tmp_path):
        cfg = _config(tmp_path)
        client = make_client()
        client.recall.side_effect = RuntimeError("boom")
        assert do_recall(client, cfg, "proj", "q", Path(cfg.memory_filename)) is False


class TestHistoryDelta:
    def test_size_and_delta(self, tmp_path):
        p = tmp_path / "hist.md"
        assert history_size(p) == 0
        p.write_text("old content\n")
        prev = history_size(p)
        p.write_text("old content\nNEW SESSION\n")
        assert read_history_delta(p, prev) == "NEW SESSION\n"

    def test_rotated_returns_whole(self, tmp_path):
        p = tmp_path / "hist.md"
        p.write_text("aaaaaaaaaa")
        p.write_text("short")  # shrank
        assert read_history_delta(p, 10) == "short"


class TestRun:
    def test_recall_inject_run_retain(self, tmp_path):
        cfg = _config(tmp_path)
        client = make_client(["User prefers tabs over spaces"])
        captured = {}

        def fake_aider(cmd):
            captured["cmd"] = cmd
            # simulate aider appending to the chat history during the session
            Path(cfg.chat_history_file).write_text("# aider chat\n#### fix auth\nDone: added retry.\n")
            return 0

        code = run(["-m", "fix the auth bug"], config=cfg, client=client, run_aider=fake_aider)

        assert code == 0
        # recalled against the -m message, on the repo bank
        assert client.recall.call_args.kwargs["query"] == "fix the auth bug"
        assert client.recall.call_args.kwargs["bank_id"] == "proj"
        # memory written + aider got --read it, then the passthrough args
        assert "tabs over spaces" in Path(cfg.memory_filename).read_text()
        assert captured["cmd"][:3] == ["aider", "--read", cfg.memory_filename]
        assert captured["cmd"][-2:] == ["-m", "fix the auth bug"]
        # retained the session transcript on the same bank
        client.retain.assert_called_once()
        assert "added retry" in client.retain.call_args.kwargs["content"]
        assert client.retain.call_args.kwargs["bank_id"] == "proj"

    def test_no_memory_means_no_read_flag(self, tmp_path):
        cfg = _config(tmp_path)
        client = make_client([])  # nothing to recall
        captured = {}

        def fake_aider(cmd):
            captured["cmd"] = cmd
            return 0

        run([], config=cfg, client=client, run_aider=fake_aider)
        assert "--read" not in captured["cmd"]

    def test_retains_only_new_history(self, tmp_path):
        cfg = _config(tmp_path)
        Path(cfg.chat_history_file).write_text("PRIOR SESSION CONTENT\n")
        client = make_client(["m"])

        def fake_aider(cmd):
            with open(cfg.chat_history_file, "a") as f:
                f.write("THIS SESSION ONLY\n")
            return 0

        run([], config=cfg, client=client, run_aider=fake_aider)
        retained = client.retain.call_args.kwargs["content"]
        assert "THIS SESSION ONLY" in retained
        assert "PRIOR SESSION" not in retained

    def test_auto_recall_off(self, tmp_path):
        cfg = _config(tmp_path, auto_recall=False)
        client = make_client(["x"])
        run([], config=cfg, client=client, run_aider=lambda cmd: 0)
        client.recall.assert_not_called()

    def test_auto_retain_off(self, tmp_path):
        cfg = _config(tmp_path, auto_retain=False)
        client = make_client(["x"])

        def fake_aider(cmd):
            Path(cfg.chat_history_file).write_text("session\n")
            return 0

        run([], config=cfg, client=client, run_aider=fake_aider)
        client.retain.assert_not_called()

    def test_passes_through_exit_code(self, tmp_path):
        cfg = _config(tmp_path)
        client = make_client([])
        assert run([], config=cfg, client=client, run_aider=lambda cmd: 3) == 3

    def test_closes_client_it_created(self, tmp_path, monkeypatch):
        """A client run() resolves itself is closed on exit (avoids the aiohttp
        'Unclosed connector' warning)."""
        import hindsight_aider.runner as runner_mod

        client = make_client([])
        monkeypatch.setattr(runner_mod, "resolve_client", lambda config: client)
        cfg = _config(tmp_path, auto_recall=False, auto_retain=False)
        run([], config=cfg, run_aider=lambda cmd: 0)  # no client= -> run() owns it
        client.close.assert_called_once()

    def test_injected_client_not_closed(self, tmp_path):
        """A test/caller-injected client is left for the caller to manage."""
        client = make_client([])
        cfg = _config(tmp_path, auto_recall=False, auto_retain=False)
        run([], config=cfg, client=client, run_aider=lambda cmd: 0)
        client.close.assert_not_called()
