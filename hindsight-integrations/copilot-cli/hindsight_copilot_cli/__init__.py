"""Hindsight long-term memory integration for GitHub Copilot CLI.

Installs Copilot CLI hooks (``sessionStart``, ``subagentStart``,
``agentStop``, ``sessionEnd``) that recall relevant memories at session/
subagent start and retain conversation transcripts to Hindsight as the
session progresses.

CLI::

    hindsight-copilot-cli install
    hindsight-copilot-cli install --api-url https://api.hindsight.vectorize.io --api-token hsk_...
    hindsight-copilot-cli uninstall
"""

__version__ = "0.1.0"
