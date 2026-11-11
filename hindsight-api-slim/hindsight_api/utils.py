import io
import logging
import os
import sys
from urllib.parse import urlparse, urlunparse


def configure_stdio() -> None:
    """Make stdout and stderr write any text instead of crashing on it.

    Before Python 3.15, a redirected stdout on Windows (``hindsight-api > log``) uses the
    ANSI code page (cp1252, cp949), which cannot encode the banner's block characters, a
    non-Latin model name or a memory quoted in a log line. Python stops the server on the
    first such character (#6458). Use UTF-8, as Python 3.15 does by default, unless the
    operator chose an encoding with ``PYTHONIOENCODING``; a character the chosen encoding
    still cannot write is escaped instead of raising.

    Called from Hindsight's own entry points only (API server, worker, admin CLI): an
    application that imports ``hindsight_api`` keeps the streams it set up. The local MCP
    server is left out on purpose: its stdout is the protocol channel, which the MCP SDK
    already writes as UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        if not isinstance(stream, io.TextIOWrapper):
            continue
        if os.environ.get("PYTHONIOENCODING"):
            stream.reconfigure(errors="backslashreplace")
        else:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def detect_container_runtime() -> str | None:
    """Detect whether the process is running inside a container.

    Returns "kubernetes", "docker", or None. Used to warn operators that the
    default ``socket.gethostname()`` worker id is unstable across container
    recreation (the random container id changes on restart, so tasks stuck in
    'processing' under the old id are never recovered).
    """
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    # Docker (and most OCI runtimes) create this marker file in every container.
    if os.path.exists("/.dockerenv"):
        return "docker"
    # cgroup v1 fallback for runtimes that don't write /.dockerenv.
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as f:
            if any(token in f.read() for token in ("docker", "containerd", "kubepods")):
                return "docker"
    except OSError:
        pass
    return None


def warn_if_container_default_worker_id(worker_id: str | None) -> None:
    """Warn when worker id will fall back to an unstable container hostname."""
    if worker_id:
        return

    runtime = detect_container_runtime()
    if not runtime:
        return

    logging.warning(
        "\n"
        "============================================================\n"
        "  WARNING: HINDSIGHT_API_WORKER_ID is not set and Hindsight\n"
        f"  appears to be running inside {runtime}.\n"
        "\n"
        "  The worker id is defaulting to the container hostname,\n"
        "  which CHANGES every time the container is recreated.\n"
        "  When that happens, tasks left in 'processing' under the\n"
        "  old hostname are never recovered — consolidation and other\n"
        "  async operations can get stuck indefinitely.\n"
        "\n"
        "  Set HINDSIGHT_API_WORKER_ID to a STABLE value (e.g. the\n"
        "  compose service name or StatefulSet pod name) to avoid this.\n"
        "============================================================"
    )


def mask_network_location(url):
    if not url:
        return url
    parsed_url = urlparse(url)
    masked_network_location = parsed_url.hostname or ""
    if parsed_url.port:
        masked_network_location += f":{parsed_url.port}"
    if parsed_url.username or parsed_url.password:
        masked_network_location = f"***:***@{masked_network_location}"
    return urlunparse(parsed_url._replace(netloc=masked_network_location))
