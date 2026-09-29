import logging
import os
from urllib.parse import urlparse, urlunparse


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
