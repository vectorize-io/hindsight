"""Entrypoint for the Hindsight Dify plugin."""

from __future__ import annotations

from dify_plugin import DifyPluginEnv, Plugin

plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))


if __name__ == "__main__":
    plugin.run()
