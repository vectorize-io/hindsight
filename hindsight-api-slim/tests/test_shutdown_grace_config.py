"""HINDSIGHT_API_SHUTDOWN_GRACE config wiring.

The worker poller's graceful-shutdown timeout was hardcoded to 30s. An
in-flight retain is an LLM call that may legitimately run for minutes
(HINDSIGHT_API_LLM_TIMEOUT defaults far above 30s); cancelling it mid-flight
on every service stop loses the operation. Deployments size this together
with their supervisor stop timeout (e.g. systemd TimeoutStopSec).
"""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    from hindsight_api.config import clear_config_cache

    keys = ["HINDSIGHT_API_LLM_PROVIDER", "HINDSIGHT_API_SHUTDOWN_GRACE"]
    original = {k: os.environ.get(k) for k in keys}
    clear_config_cache()
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    clear_config_cache()


def test_default_shutdown_grace_is_30s():
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HINDSIGHT_API_SHUTDOWN_GRACE", None)

    config = HindsightConfig.from_env()
    assert config.shutdown_grace == 30.0


def test_shutdown_grace_env_var_is_read():
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_SHUTDOWN_GRACE"] = "240"

    config = HindsightConfig.from_env()
    assert config.shutdown_grace == 240.0
