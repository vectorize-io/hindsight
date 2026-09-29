"""Test setup: keep the hook's proof-of-life log out of the real home dir."""

import os


def pytest_configure(config):  # noqa: ARG001
    # The hook writes ~/.hindsight/devin-hook.log by default; disable it in tests
    # so running the suite doesn't touch the user's home directory.
    os.environ.setdefault("HINDSIGHT_HOOK_LOG", "off")
