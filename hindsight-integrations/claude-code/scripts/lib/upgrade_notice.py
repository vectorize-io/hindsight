"""Deprecation notice for this plugin, shown at every session start.

This plugin still works and is not going to stop working, but it is deprecated: development has
moved to `@vectorize-io/hindsight-coding-agents`. Someone who installed this a year ago has no way to learn
that — a changelog entry reaches nobody — so the session itself says it.

Shown every session rather than a few times: a notice that appears once is missed, and this one has
a concrete action attached. `"upgradeNotice": false` in the user config turns it off permanently,
and the message says so, which is what keeps that acceptable.

Any failure returns None rather than raising. A promotional message must never be the reason
someone's session breaks.

Deliberately duplicated in the codex plugin rather than shared: the two ship as independent
packages with no common module path.
"""

DOCS_URL = "https://hindsight.vectorize.io/sdks/integrations/coding-agents"
MIGRATION_URL = f"{DOCS_URL}#migrating-from-the-per-agent-plugins"

NOTICE = (
    "The current Hindsight Claude Code plugin is DEPRECATED, replaced by the Coding Agents "
    "plugin.\n"
    "\n"
    "One install now covers Claude Code, Codex, Cursor, Copilot, opencode, Kilo, Grok, "
    "Antigravity,\n"
    "Devin and Cline, and they all share a single memory per repository, leveraging the Hindsight "
    "Knowledge Page (v0.9.x and onwards).\n"
    "\n"
    f"See the migration guide: {MIGRATION_URL}\n"
    "Your Hindsight server and token carry over automatically; past conversations are re-imported "
    "from\n"
    'local transcripts. Silence this with "upgradeNotice": false in '
    "~/.hindsight/claude-code.json"
)


def upgrade_notice(config: dict) -> str | None:
    """Return the notice, or None when the user has turned it off."""
    try:
        return None if not config.get("upgradeNotice", True) else NOTICE
    except Exception:
        return None
