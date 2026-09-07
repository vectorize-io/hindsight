"""The llama.cpp prompt cache must stay in RAM, never on disk.

`llama_cpp.server` offers two prompt-cache backends. The `disk` one is backed
by diskcache, which pickles its entries (CVE-2025-69872): anyone able to write
to the cache directory gets code execution in the server process when an entry
is read back. diskcache has had no release since 5.6.3 in 2023 and no fixed
version exists, so choosing that backend is a permanent exposure rather than a
version to bump.

`ram` happens to be llama_cpp.server's own default, which is why this was never
a live vulnerability. That is exactly the problem: the safety rested on a
default in someone else's package, invisible at our call site. These tests
assert we pass the flag ourselves, so switching to the disk backend has to be a
deliberate edit rather than an inherited default that quietly changes.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from hindsight_api.engine.providers import llamacpp_llm
from hindsight_api.engine.providers.llamacpp_llm import LlamaCppServer


def _captured_cmd(monkeypatch, tmp_path, **kwargs) -> list[str]:
    """Start a server with the subprocess stubbed, and return the argv.

    `start()` writes a log file under MODELS_DIR before spawning, so point that
    at tmp_path; otherwise it raises before Popen and captures nothing.
    """
    captured: dict[str, list[str]] = {}

    def _fake_popen(cmd, *args, **popen_kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    monkeypatch.setattr(llamacpp_llm, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(llamacpp_llm.subprocess, "Popen", _fake_popen)
    # _wait_for_ready polls a real socket for up to 120s. The argv is captured
    # at Popen, well before that, so skip it.
    monkeypatch.setattr(LlamaCppServer, "_wait_for_ready", AsyncMock())

    server = LlamaCppServer(model_path=Path("/tmp/model.gguf"), port=18080, **kwargs)
    asyncio.run(server.start())

    assert captured.get("cmd"), "subprocess.Popen was never reached"
    return captured["cmd"]


def test_cache_type_is_passed_explicitly(monkeypatch, tmp_path):
    """Guards CVE-2025-69872: the flag must be present, not left to a default."""
    cmd = _captured_cmd(monkeypatch, tmp_path)
    assert "--cache_type" in cmd, (
        "llama.cpp server started without an explicit --cache_type. The prompt "
        "cache backend would then follow llama_cpp.server's default, which can "
        "change without notice; the `disk` backend pickles via diskcache "
        "(CVE-2025-69872, no fixed version)."
    )


def test_cache_type_is_ram_not_disk(monkeypatch, tmp_path):
    cmd = _captured_cmd(monkeypatch, tmp_path)
    value = cmd[cmd.index("--cache_type") + 1]
    assert value == "ram", f"prompt cache backend is {value!r}, must be 'ram'"
    assert "disk" not in cmd, "the diskcache-backed prompt cache must never be selected"
