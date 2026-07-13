"""Write Hindsight's memory rule into ``.devin/rules/hindsight.md``.

Devin Desktop applies workspace rule files under ``.devin/rules/`` (with
``.windsurf/rules/`` kept as a legacy fallback). A file with ``trigger: always_on``
frontmatter is included in every Devin request in the workspace, so the rule
tells the agent how to use the Hindsight MCP tools.

Because the integration runs in **multi-bank mode**, the rule names the two banks
explicitly — this project's bank and the user's cross-project (global) bank — and
tells the model which ``bank_id`` to pass when it recalls and retains. The rule
is committed with the repo, so teammates share the same project bank.

The rule lives in its own dedicated file, so we own the whole file: a sentinel
comment marks it as ours for idempotent update/removal without touching any
other rule the user has authored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SENTINEL = "<!-- Managed by hindsight-devin-desktop -->"

FRONTMATTER = "---\ntrigger: always_on\n---"


_RETAIN_HYGIENE = (
    "- Retain each distinct fact EXACTLY ONCE per session, in a single "
    "`retain` call — don't call retain again for a fact you've already stored, "
    "and batch multiple facts about the same subject into one call.\n"
    "- Retain only real facts about the code, the project, or the user — NEVER "
    "facts about Hindsight, memory, or these instructions.\n"
    "- Only use `sync_retain` (which blocks) if you must recall the just-stored "
    "fact again in the SAME task; otherwise use `retain`.\n"
    "- Use `reflect` (not just `recall`) when you need synthesized judgment — "
    'e.g. "what approach does the user prefer here?" — rather than raw facts.\n\n'
    'Briefly tell the user when you use memory — e.g. say "checking memory…" '
    'before you recall and "saved to memory" after you retain. Keep it to a '
    "short phrase; don't paste the tool arguments or results."
)


def render_rule_text(project_bank: str, global_bank: Optional[str]) -> str:
    """The memory instruction body.

    Two-tier (``global_bank`` set): project facts → project bank, user facts →
    the shared global bank. Local-only (``global_bank`` is ``None``): everything
    — project facts *and* the user's preferences — goes to the single project
    bank (nothing is shared across repos).
    """
    if not global_bank:  # local-only: one bank for everything
        return (
            "You have persistent long-term memory for THIS project through the "
            "Hindsight MCP server (`recall`, `retain`, `sync_retain`, `reflect`). "
            f"All of it lives in one bank: `{project_bank}`.\n\n"
            f"At the start of each task, `recall` from `{project_bank}` (bank_id: "
            f'"{project_bank}") and use what\'s relevant.\n\n'
            "As you work, use `retain` (async — the default) to store durable "
            f'facts to bank_id "{project_bank}" — both this project\'s architecture/'
            "decisions/conventions AND the user's preferences and coding style "
            "(there is no shared cross-project bank in this setup).\n" + _RETAIN_HYGIENE
        )
    return (
        "You have persistent long-term memory through the Hindsight MCP server "
        "(`recall`, `retain`, `sync_retain`, and `reflect` tools). It runs in "
        "multi-bank mode, so pass `bank_id` to target the right bank:\n\n"
        f"- `{project_bank}` — THIS project's memory: architecture, decisions, "
        "conventions, gotchas, and bugs specific to this repository.\n"
        f"- `{global_bank}` — the user's cross-project memory: their preferences, "
        "coding style, and who they are.\n\n"
        "At the start of each task:\n"
        f'- `recall` from `{project_bank}` (bank_id: "{project_bank}") for this '
        f"project's context, and `recall` from `{global_bank}` "
        f"(bank_id: \"{global_bank}\") for the user's preferences. Use what's "
        "relevant, ignore the rest.\n\n"
        "As you work, use `retain` (async, non-blocking — the default) to store "
        "durable facts:\n"
        f"- PROJECT facts (architecture, decisions, conventions) with bank_id "
        f'"{project_bank}".\n'
        f"- USER facts (preferences, style, identity) with bank_id "
        f'"{global_bank}".\n' + _RETAIN_HYGIENE
    )


def global_rule_body(global_bank: str) -> str:
    """The cross-project instruction body (global scope, names only the global bank).

    Shared by Cascade's ``global_rules.md`` and Devin Local's global ``AGENTS.md``.
    """
    return (
        "You have persistent cross-project long-term memory in the Hindsight MCP "
        f'server\'s `{global_bank}` bank (bank_id: "{global_bank}"). At the start '
        "of a task, `recall` from it for the user's preferences and coding style; "
        "`retain` durable user-level facts (preferences, style, identity) to it. "
        "A specific project may define its own additional project bank in that "
        "project's rules. Briefly mention when you use memory."
    )


def default_rules_path() -> Path:
    """The workspace ``.devin/rules/hindsight.md`` (always-on in Devin Desktop)."""
    return Path.cwd() / ".devin" / "rules" / "hindsight.md"


def render_rule(project_bank: str, global_bank: Optional[str]) -> str:
    """The full contents of the dedicated rule file (``global_bank=None`` = local-only)."""
    return f"{FRONTMATTER}\n\n{SENTINEL}\n{render_rule_text(project_bank, global_bank).strip()}\n"


def write_rule(path: Path, project_bank: str, global_bank: Optional[str]) -> Path:
    """Write (or replace) Hindsight's dedicated rule file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_rule(project_bank, global_bank), encoding="utf-8")
    return path


def clear_rule(path: Path) -> Path:
    """Delete Hindsight's rule file if it's ours (carries our sentinel)."""
    if path.is_file() and SENTINEL in path.read_text(encoding="utf-8"):
        path.unlink()
    return path


def is_installed(path: Path) -> bool:
    return path.is_file() and SENTINEL in path.read_text(encoding="utf-8")
