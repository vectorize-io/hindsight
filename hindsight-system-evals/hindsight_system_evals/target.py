"""Where an eval points: a server this process starts, or one already running.

Local is the default and is what CI uses — its own `hindsight-api` on its own
pg0, configured from `HINDSIGHT_EVAL_LLM_*`, so a run means the same thing on
every machine.

Remote (`--api-url`, or `HINDSIGHT_EVAL_API_URL`) points at a deployment that is
already up: cloud dev, a colleague's box, a docker compose. Two things follow,
and neither is worked around:

* **The model is that server's.** `provider_environment()` configures a server we
  start; against a remote one it would describe a server we did not configure. So
  credentials are not required here, and the model is *read back* for the report
  instead of asserted. There is also no log to print on a failure — the message
  has to say that rather than show an empty one.
* **The banks are not disposable.** On pg0 they cost nothing and are left behind
  on purpose, for the control plane. In a shared tenant they accumulate, so a
  remote run deletes what it created unless `--keep-banks` says otherwise.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from hindsight_system_evals.server import start_eval_server

ENV_API_URL = "HINDSIGHT_EVAL_API_URL"
ENV_API_KEY = "HINDSIGHT_EVAL_API_KEY"


@dataclass(frozen=True)
class Target:
    url: str
    api_key: str | None = None
    #: Nobody started this server, so nothing here may assume it can be
    #: restarted, read a log off it, or throw its data away.
    is_remote: bool = True

    def describe(self) -> str:
        return f"{self.url} ({'remote' if self.is_remote else 'started by this run'})"


@contextlib.contextmanager
def eval_target(api_url: str | None = None, *, log_path: Path | None = None) -> Iterator[Target]:
    """Resolve the target, starting a server only when none was given."""
    url = (api_url or os.getenv(ENV_API_URL) or "").rstrip("/")
    if url:
        yield Target(url=url, api_key=os.getenv(ENV_API_KEY) or None)
        return

    if log_path is None:
        log_path = Path(tempfile.mkdtemp(prefix="hindsight-eval-server-")) / "server.log"
    server = start_eval_server(log_path=log_path)
    try:
        yield Target(url=server.url, is_remote=False)
    finally:
        server.stop()
