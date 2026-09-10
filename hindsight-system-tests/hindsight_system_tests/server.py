"""Boot a real Hindsight server, wired to the stub, and tear it down again.

The server runs as its own process, started the same way an operator starts it
(``hindsight-api``), against its own embedded Postgres. Nothing about the
process is special-cased for tests: the only thing the fixtures do is set
environment variables that any deployment could set.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn

from .rulebook import Stubs
from .stub_server import create_stub_app

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "hindsight-api-slim"

# A pg0 instance of its own, on a port nothing else uses. The api-slim suite and
# the dev server share the default "hindsight" instance, and pointing system
# tests at that one would both see their leftovers and block on their locks.
PG0_INSTANCE = "hindsight-systest"
PG0_PORT = 15499

SERVER_STARTUP_TIMEOUT = 180.0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class StubServer:
    """The provider stub, running in this process on a real port."""

    url: str
    stubs: Stubs
    _server: uvicorn.Server
    _thread: threading.Thread

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def start_stub_server(stubs: Stubs) -> StubServer:
    port = free_port()
    config = uvicorn.Config(
        create_stub_app(stubs),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="provider-stub")
    thread.start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("provider stub did not start")

    return StubServer(url=f"http://127.0.0.1:{port}", stubs=stubs, _server=server, _thread=thread)


@dataclass
class HindsightServer:
    """A running ``hindsight-api`` process under test."""

    url: str
    log_path: Path
    _process: subprocess.Popen

    def stop(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)

    def logs(self) -> str:
        with contextlib.suppress(OSError):
            return self.log_path.read_text()
        return ""


def stub_environment(stub_url: str) -> dict[str, str]:
    """Everything needed to point a Hindsight server at the stub.

    Deliberately expressed as plain configuration — this is the whole reason the
    suite needs no production code changes.
    """
    return {
        # LLM: the openai-compatible provider, aimed at the stub.
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_MODEL": "stub-model",
        "HINDSIGHT_API_LLM_API_KEY": "stub-key",
        "HINDSIGHT_API_LLM_BASE_URL": f"{stub_url}/v1",
        # Embeddings: same trick, and it means no torch and no model download.
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "openai",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": "stub-key",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL": f"{stub_url}/v1",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": "stub-embeddings",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS": "384",
        # Reranker: siliconflow is the plain Cohere-compatible POST {base}/rerank,
        # with no vendor SDK in the way.
        "HINDSIGHT_API_RERANKER_PROVIDER": "siliconflow",
        "HINDSIGHT_API_RERANKER_SILICONFLOW_API_KEY": "stub-key",
        "HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL": stub_url,
        "HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL": "stub-reranker",
        # In production a retry earns its keep even on a 400: a model that
        # returned malformed JSON often gets it right the second time. Against a
        # deterministic stub it cannot — the same request gets the same answer —
        # so retries here would only turn an unstubbed call into several rounds
        # of exponential backoff before the same failure. Off, the "run, paste the
        # rule, run again" loop takes milliseconds.
        "HINDSIGHT_API_LLM_MAX_RETRIES": "0",
        "HINDSIGHT_API_EMBEDDINGS_MAX_RETRIES": "0",
        "HINDSIGHT_API_RERANKER_MAX_RETRIES": "0",
        # Same reasoning one level up: the worker retries a failed operation on a
        # backoff schedule, so an unstubbed call inside background work would sit
        # in `pending` through several rounds before reaching `failed`. Against a
        # deterministic stub the retry cannot change the answer, and the wait
        # turns a fast loud-miss into a 90-second timeout.
        "HINDSIGHT_API_WORKER_MAX_RETRIES": "0",
        # Bank stats are cached for 60s by default. A test that changes the bank
        # and then asserts on a counter would be reading a value from before its
        # own action — racy at best, a minute of waiting at worst.
        "HINDSIGHT_API_BANK_STATS_CACHE_TTL_SECONDS": "0",
        # Maintenance sweeps are what pick up recovered consolidations, and the
        # default start jitter spreads them over a minute so a fleet of servers
        # does not stampede the database on boot. One server in a test has nobody
        # to stampede, and the jitter is otherwise a minute of a story waiting for
        # work it already asked for.
        "HINDSIGHT_API_MAINTENANCE_START_JITTER_SECONDS": "0",
        # Webhook destinations are SSRF-checked, and loopback is refused — the
        # right default, and story 64 asserts it. But the only receiver a
        # hermetic test can offer *is* on loopback, so the stub's host is
        # allowlisted explicitly. Nothing else is: a webhook aimed anywhere else
        # private still fails, which is what keeps the guard under test.
        "HINDSIGHT_API_WEBHOOK_ALLOWED_HOSTS": "127.0.0.1",
        "HINDSIGHT_API_CONSOLIDATION_RECONCILE_INTERVAL_SECONDS": "5",
    }


def start_hindsight_server(*, stub_url: str, log_path: Path) -> HindsightServer:
    port = free_port()

    env = os.environ.copy()
    # The repo .env is for the developer's own server; a system test must depend
    # only on what it sets here, or it passes on one machine and fails on another.
    for key in list(env):
        if key.startswith("HINDSIGHT_API_"):
            del env[key]

    env.update(stub_environment(stub_url))

    env.update(
        {
            "HINDSIGHT_API_DATABASE_URL": f"pg0://{PG0_INSTANCE}:{PG0_PORT}",
            "HINDSIGHT_API_HOST": "127.0.0.1",
            "HINDSIGHT_API_PORT": str(port),
            "HINDSIGHT_API_LOG_LEVEL": "info",
            # Everything the server does by default stays on, background work
            # included. A retain enqueues observation extraction and can trigger
            # consolidation, and those run in the worker after the request returns
            # — but this is a real server with a real worker, so they *do* finish.
            # Tests wait for them (see `waiting.py`) rather than switching them
            # off: a suite that disables the asynchronous half of the system stops
            # being a system test, and the composition bugs this suite exists to
            # catch are mostly in that half.
        }
    )

    # `hindsight-api` calls load_dotenv(find_dotenv(usecwd=True), override=True) at
    # startup, and a discovered .env deliberately wins over the ambient environment
    # (see issue #2961). Started from anywhere inside the repo it would therefore
    # pick up the developer's own .env and ignore every variable set above. Running
    # it from a scratch directory that holds an empty .env stops the upward walk at
    # a file with nothing in it, so the environment we pass is the whole config.
    run_dir = log_path.parent / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".env").write_text("# intentionally empty: see start_hindsight_server\n")

    log_file = log_path.open("w")
    process = subprocess.Popen(
        ["uv", "run", "--project", str(API_DIR), "hindsight-api"],
        cwd=run_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    url = f"http://127.0.0.1:{port}"
    server = HindsightServer(url=url, log_path=log_path, _process=process)
    _wait_until_healthy(server)
    return server


def _wait_until_healthy(server: HindsightServer) -> None:
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server._process.poll() is not None:
            raise RuntimeError(f"hindsight-api exited during startup:\n{server.logs()}")
        with contextlib.suppress(httpx.HTTPError):
            response = httpx.get(f"{server.url}/health", timeout=5)
            if response.status_code == 200:
                return
        time.sleep(0.5)

    server.stop()
    raise RuntimeError(f"hindsight-api was not healthy within {SERVER_STARTUP_TIMEOUT}s:\n{server.logs()}")
