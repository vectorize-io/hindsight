"""Tests for the CLI passthrough."""

from hindsight_aider import cli


def test_passes_args_to_runner(monkeypatch):
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)
    rc = cli.main(["-m", "do a thing", "src/app.py"])
    assert rc == 0
    assert captured["args"] == ["-m", "do a thing", "src/app.py"]


def test_returns_runner_exit_code(monkeypatch):
    monkeypatch.setattr(cli.runner, "run", lambda args: 5)
    assert cli.main([]) == 5


def test_version_flag(capsys):
    assert cli.main(["--hindsight-version"]) == 0
    assert "hindsight-aider" in capsys.readouterr().out
