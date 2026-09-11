"""Start a real ``hindsight-api`` against a REAL model provider.

Mirrors ``hindsight_system_tests.server`` deliberately — same scratch-directory
trick, same environment hygiene, same health wait — with one difference that is
the whole reason this package is separate: the system tests point every LLM call
at a stub, because a test wants determinism. An eval measuring answer quality
cannot do that. Stub the model and you score the stub.

The consequences follow from that and are not worked around:

* provider credentials are REQUIRED, so these evals cannot run on fork PRs, and
  the CI job has to be gated on secrets — unlike ``test-system``;
* results are rates, not equalities, because the same request can produce a
  different answer twice running.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "hindsight-api-slim"

#: Its own pg0 instance. Sharing one with the developer's server (or with the
#: system tests) means a run competes for connections and can be refused mid-way
#: with "sorry, too many clients already".
PG0_INSTANCE = "hindsight-system-evals"

SERVER_STARTUP_TIMEOUT = 180.0


@dataclass
class EvalServer:
    url: str
    log_path: Path
    _process: subprocess.Popen

    def logs(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def stop(self) -> None:
        self._process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._process.wait(timeout=30)
        if self._process.poll() is None:
            self._process.kill()


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def provider_environment() -> dict[str, str]:
    """The model settings for the server under test, from this process's env.

    Read explicitly rather than inherited: the server's environment is wiped
    below, so anything not named here does not reach it. ``HINDSIGHT_EVAL_*``
    wins over ``HINDSIGHT_API_*`` so a developer can point the evals at a model
    other than the one their own server uses.

    VertexAI authenticates with a service-account key file rather than an API
    key — that is how the perf workflow runs every quality benchmark — so its
    settings pass through as a group instead of being required individually.
    """

    def pick(suffix: str) -> str:
        return os.getenv(f"HINDSIGHT_EVAL_{suffix}") or os.getenv(f"HINDSIGHT_API_{suffix}") or ""

    provider, model = pick("LLM_PROVIDER"), pick("LLM_MODEL")
    env = {"HINDSIGHT_API_LLM_PROVIDER": provider, "HINDSIGHT_API_LLM_MODEL": model}

    if provider == "vertexai":
        for suffix in ("LLM_VERTEXAI_SERVICE_ACCOUNT_KEY", "LLM_VERTEXAI_PROJECT_ID", "LLM_VERTEXAI_REGION"):
            if value := pick(suffix):
                env[f"HINDSIGHT_API_{suffix}"] = value
        credentialed = "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY" in env
    else:
        if api_key := pick("LLM_API_KEY"):
            env["HINDSIGHT_API_LLM_API_KEY"] = api_key
        credentialed = "HINDSIGHT_API_LLM_API_KEY" in env

    if not (provider and model and credentialed):
        raise RuntimeError(
            "System evals need a real model. Set HINDSIGHT_EVAL_LLM_PROVIDER / _MODEL and either "
            "_API_KEY or, for vertexai, _VERTEXAI_SERVICE_ACCOUNT_KEY (HINDSIGHT_API_ equivalents "
            "also work). A stub cannot be used here: it would score the stub."
        )
    if base_url := pick("LLM_BASE_URL"):
        env["HINDSIGHT_API_LLM_BASE_URL"] = base_url
    return env


def start_eval_server(*, log_path: Path) -> EvalServer:
    port = free_port()

    env = os.environ.copy()
    # The repo .env is for the developer's own server. Left in place it would
    # decide which model these evals measure, and a run would mean something
    # different on every machine.
    for key in list(env):
        if key.startswith("HINDSIGHT_API_"):
            del env[key]
    env.update(provider_environment())
    env.update(
        {
            "HINDSIGHT_API_DATABASE_URL": f"pg0://{PG0_INSTANCE}",
            "HINDSIGHT_API_HOST": "127.0.0.1",
            "HINDSIGHT_API_PORT": str(port),
            "HINDSIGHT_API_LOG_LEVEL": "info",
            # The debug tools replay a captured delta-ops request verbatim, and the
            # default 50k-char cap truncates a large one mid-document. A replay of a
            # truncated prompt is a different prompt.
            "HINDSIGHT_API_LLM_TRACE_MAX_CHARS": "1000000",
        }
    )

    # `hindsight-api` loads a discovered .env with override=True (#2961), so
    # started from inside the repo it would ignore everything set above. An empty
    # .env in a scratch directory stops the upward walk. Same trick as the system
    # tests, and for the same reason.
    run_dir = log_path.parent / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".env").write_text("# intentionally empty: see start_eval_server\n")

    log_file = log_path.open("w")
    process = subprocess.Popen(
        ["uv", "run", "--project", str(API_DIR), "hindsight-api"],
        cwd=run_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    server = EvalServer(url=f"http://127.0.0.1:{port}", log_path=log_path, _process=process)
    _wait_until_healthy(server)
    return server


def _wait_until_healthy(server: EvalServer) -> None:
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server._process.poll() is not None:
            raise RuntimeError(f"hindsight-api exited during startup:\n{server.logs()}")
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(f"{server.url}/health", timeout=5).status_code == 200:
                return
        time.sleep(0.5)
    server.stop()
    raise RuntimeError(f"hindsight-api was not healthy within {SERVER_STARTUP_TIMEOUT}s:\n{server.logs()}")
