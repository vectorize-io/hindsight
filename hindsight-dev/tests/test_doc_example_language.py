"""Invalid language selectors must not hang or claim a successful docs gate."""

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-doc-examples.sh"


@pytest.mark.parametrize("arguments", [["--lang"], ["--lang", "rust"], ["--lang", ""]])
def test_invalid_doc_example_language_fails_promptly(tmp_path: Path, arguments: list[str]) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    result = subprocess.run(["bash", str(scripts / SCRIPT.name), *arguments], capture_output=True, text=True, timeout=2)
    assert result.returncode != 0
    assert "All examples passed" not in result.stdout
    assert "--lang" in result.stdout + result.stderr


@pytest.mark.parametrize("language", ["python", "node", "cli", "go"])
def test_supported_doc_example_language_is_accepted(tmp_path: Path, language: str) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    examples = tmp_path / "hindsight-docs" / "examples" / "api"
    examples.mkdir(parents=True)
    for client in ("python", "go"):
        (tmp_path / "hindsight-clients" / client).mkdir(parents=True)
    # The CLI control runs a real, local shell example with no Hindsight calls.
    # A unique basename avoids sharing the script's current /tmp log path.
    (examples / f"fixture-{uuid.uuid4().hex}.sh").write_text("#!/bin/bash\necho 'CLI example ran'\n")
    result = subprocess.run(
        ["bash", str(scripts / SCRIPT.name), "--lang", language], capture_output=True, text=True, timeout=2
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Language filter: {language}" in result.stdout
    if language == "cli":
        assert "Passed: 1" in result.stdout
    assert "All examples passed" in result.stdout
