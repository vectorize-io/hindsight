"""Regression tests for issue #2961: importing Hindsight must not mutate the
host application's environment, while Hindsight's own entry points keep their
``.env`` convenience with unchanged precedence.
"""

import os
import subprocess
import sys
from pathlib import Path

API_SOURCE = str(Path(__file__).parents[1])


def _run_import(module: str, tmp_path: Path) -> str:
    """Import ``module`` from a child dir whose parent holds a hostile .env and
    return the resulting value of HOST_APP_SECRET as seen by that process."""
    (tmp_path / ".env").write_text("HOST_APP_SECRET=from-dotenv\n")
    child_dir = tmp_path / "child"
    child_dir.mkdir()

    env = os.environ.copy()
    env["HOST_APP_SECRET"] = "from-application"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (API_SOURCE, env.get("PYTHONPATH"))))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os; import {module}; print(os.environ['HOST_APP_SECRET'])",
        ],
        cwd=child_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_importing_config_does_not_modify_environment(tmp_path: Path) -> None:
    # The reporter's exact repro: importing config walked up to the parent .env
    # and overwrote the application's own value. It must not any more.
    assert _run_import("hindsight_api.config", tmp_path) == "from-application"


def test_importing_package_does_not_modify_environment(tmp_path: Path) -> None:
    # `import hindsight_api` pulls in .config transitively; still no side effect.
    assert _run_import("hindsight_api", tmp_path) == "from-application"


def test_entrypoint_dotenv_loading_is_authoritative(tmp_path: Path, monkeypatch) -> None:
    # The entry-point loader keeps override=True: a discovered .env wins over the
    # ambient process environment (unchanged precedence) AND fills missing keys.
    (tmp_path / ".env").write_text("HINDSIGHT_TEST_OVERRIDDEN=from-dotenv\nHINDSIGHT_TEST_FILLED=from-dotenv\n")
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    monkeypatch.chdir(child_dir)
    monkeypatch.setenv("HINDSIGHT_TEST_OVERRIDDEN", "from-process")
    monkeypatch.delenv("HINDSIGHT_TEST_FILLED", raising=False)

    from hindsight_api.config import load_dotenv_for_entrypoint

    load_dotenv_for_entrypoint()

    # override=True: the .env value wins over the pre-set process value.
    assert os.environ["HINDSIGHT_TEST_OVERRIDDEN"] == "from-dotenv"
    # A key absent from the process is filled from .env.
    assert os.environ["HINDSIGHT_TEST_FILLED"] == "from-dotenv"

    # Don't leak the .env-loaded key into the rest of the session; monkeypatch
    # cannot undo it because load_dotenv set it directly.
    monkeypatch.delenv("HINDSIGHT_TEST_FILLED", raising=False)
