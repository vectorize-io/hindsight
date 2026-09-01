#!/usr/bin/env python3
"""
Generate changelog entry for a new release.

This script fetches the commit diff between releases, uses an LLM to summarize,
and prepends the entry to the changelog page.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console

console = Console()

GITHUB_REPO = "vectorize-io/hindsight"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"
GITHUB_COMMIT_URL = f"https://github.com/{GITHUB_REPO}/commit"
GITHUB_PULL_URL = f"https://github.com/{GITHUB_REPO}/pull"
REPO_PATH = Path(__file__).parent.parent.parent
# Alembic migrations are enumerated deterministically from git (never via the LLM).
MIGRATIONS_DIR = "hindsight-api-slim/hindsight_api/alembic/versions"


@dataclass(frozen=True)
class VolumeTier:
    """How much data a table holds, and how that is labelled in the changelog."""

    label: str
    color: str


# Tiers in descending volume; the order doubles as the sort order in a migration line.
HIGH_VOLUME = VolumeTier("high volume", "var(--ifm-color-danger)")
MEDIUM_VOLUME = VolumeTier("medium", "var(--ifm-color-warning-darker)")
LOW_VOLUME = VolumeTier("small", "var(--ifm-color-emphasis-600)")
VOLUME_ORDER = (HIGH_VOLUME, MEDIUM_VOLUME, LOW_VOLUME)

# How much data a table holds in a real deployment, which is what decides how long a
# migration touching it runs and how wide the lock it takes is. This is a property of
# the schema, not of any one release, so it is a reviewed map rather than a per-run
# LLM judgement: the same table must never be labelled differently in two releases.
# `tests/test_generate_changelog_migrations.py` fails if a migration creates a table
# that is missing here, so new tables have to be classified deliberately.
TABLE_VOLUME: dict[str, VolumeTier] = {
    # Grows with every retained fact, link and entity; can reach millions of rows.
    "memory_units": HIGH_VOLUME,
    "memory_units_bm25": HIGH_VOLUME,
    "memory_links": HIGH_VOLUME,
    "unit_entities": HIGH_VOLUME,
    "entities": HIGH_VOLUME,
    "entity_cooccurrences": HIGH_VOLUME,
    "chunks": HIGH_VOLUME,
    "invalidated_memory_units": HIGH_VOLUME,
    "observation_history": HIGH_VOLUME,
    "llm_requests": HIGH_VOLUME,
    "audit_log": HIGH_VOLUME,
    # Grows with documents, operations and consolidated knowledge.
    "documents": MEDIUM_VOLUME,
    "mental_models": MEDIUM_VOLUME,
    "mental_model_history": MEDIUM_VOLUME,
    "mental_model_versions": MEDIUM_VOLUME,
    "observation_sources": MEDIUM_VOLUME,
    "async_operations": MEDIUM_VOLUME,
    "knowledge_pages": MEDIUM_VOLUME,
    "learnings": MEDIUM_VOLUME,
    "directives": MEDIUM_VOLUME,
    "pinned_reflections": MEDIUM_VOLUME,
    "graph_maintenance_queue": MEDIUM_VOLUME,
    "entity_maintenance_queue": MEDIUM_VOLUME,
    "file_storage": MEDIUM_VOLUME,
    # Configuration-sized: a handful of rows per bank or tenant.
    "banks": LOW_VOLUME,
    "webhooks": LOW_VOLUME,
    "bank_stats_cache": LOW_VOLUME,
}

# Table positions in Alembic ops and in raw SQL. Matches are intersected with
# TABLE_VOLUME, so prose and column names picked up by the SQL patterns drop out.
_TABLE_PATTERNS = (
    r"op\.(?:create_table|drop_table|add_column|drop_column|alter_column|rename_table)\(\s*[\"']([a-z_][a-z0-9_]*)[\"']",
    r"table_name=[\"']([a-z_][a-z0-9_]*)[\"']",
    r"(?i)\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:\{schema\}|\"[^\"]*\"\.)?\"?([a-z_][a-z0-9_]*)",
    r"(?i)\b(?:CREATE|DROP)\s+(?:MATERIALIZED\s+VIEW|TABLE)\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?(?:\{schema\}|\"[^\"]*\"\.)?\"?([a-z_][a-z0-9_]*)",
    r"(?i)\bON\s+(?:ONLY\s+)?(?:\{schema\}|\"[^\"]*\"\.)?\"?([a-z_][a-z0-9_]*)\"?\s*(?:USING|\()",
    r"(?i)\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(?:\{schema\}|\"[^\"]*\"\.)?\"?([a-z_][a-z0-9_]*)",
)

# `DROP INDEX idx_memory_units_embedding` names no table but takes ACCESS EXCLUSIVE on
# one, so the table is recovered from the index identifier (longest known name wins).
_INDEX_PATTERNS = (
    r"(?i)\b(?:CREATE|DROP)\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+(?:NOT\s+)?EXISTS\s+)?(?:\{schema\}|\"[^\"]*\"\.)?\"?([a-z_][a-z0-9_]*)",
    r"op\.(?:create_index|drop_index)\(\s*[\"']([a-z_][a-z0-9_]*)[\"']",
)
CHANGELOG_PATH = REPO_PATH / "hindsight-docs" / "src" / "pages" / "changelog" / "index.md"
INTEGRATION_CHANGELOG_DIR = REPO_PATH / "hindsight-docs" / "src" / "pages" / "changelog" / "integrations"


@dataclass(frozen=True)
class IntegrationMeta:
    package_name: str
    display_name: str | None = None  # falls back to slug


# Single source of truth for integrations. Adding a new integration here is
# enough to make this script accept it — no parallel lists to keep in sync.
INTEGRATIONS: dict[str, IntegrationMeta] = {
    "litellm": IntegrationMeta("hindsight-litellm", "LiteLLM"),
    "pydantic-ai": IntegrationMeta("hindsight-pydantic-ai", "Pydantic AI"),
    "crewai": IntegrationMeta("hindsight-crewai", "CrewAI"),
    "agent-framework": IntegrationMeta("hindsight-agent-framework", "Microsoft Agent Framework"),
    "ag2": IntegrationMeta("hindsight-ag2"),
    "ai-sdk": IntegrationMeta("@vectorize-io/hindsight-ai-sdk", "AI SDK"),
    "eliza": IntegrationMeta("@vectorize-io/hindsight-eliza", "elizaOS"),
    "chat": IntegrationMeta("@vectorize-io/hindsight-chat", "Chat SDK"),
    "openclaw": IntegrationMeta("@vectorize-io/hindsight-openclaw", "OpenClaw"),
    "langgraph": IntegrationMeta("hindsight-langgraph", "LangGraph"),
    "nemoclaw": IntegrationMeta("@vectorize-io/hindsight-nemoclaw", "NemoClaw"),
    "strands": IntegrationMeta("hindsight-strands", "Strands"),
    "claude-code": IntegrationMeta("hindsight-memory", "Claude Code"),
    # Git-distributed plugin bundle (Agent Plugins standard), not a registry
    # package — its changelog links to the source tree (see _package_url).
    "agent-plugin": IntegrationMeta("hindsight-agent-plugin", "Agent Plugins"),
    "zcode": IntegrationMeta("hindsight-zcode", "ZCode"),
    "claude-agent-sdk": IntegrationMeta("hindsight-claude-agent-sdk", "Claude Agent SDK"),
    "llamaindex": IntegrationMeta("hindsight-llamaindex", "LlamaIndex"),
    "codex": IntegrationMeta("hindsight-codex", "Codex"),
    # npm-published under the @vectorize-io scope, unlike the PyPI integrations above.
    "coding-agents": IntegrationMeta("@vectorize-io/hindsight-coding-agents", "Coding Agents"),
    "github-copilot": IntegrationMeta("hindsight-copilot", "GitHub Copilot"),
    "cline": IntegrationMeta("hindsight-cline", "Cline"),
    "cursor-cli": IntegrationMeta("hindsight-cursor-cli", "Cursor CLI"),
    "copilot-cli": IntegrationMeta("hindsight-copilot-cli", "GitHub Copilot CLI"),
    "cursor": IntegrationMeta("hindsight-cursor", "Cursor"),
    "autogen": IntegrationMeta("hindsight-autogen", "AutoGen"),
    "aider": IntegrationMeta("hindsight-aider", "Aider"),
    "paperclip": IntegrationMeta("@vectorize-io/hindsight-paperclip", "Paperclip"),
    "opencode": IntegrationMeta("@vectorize-io/opencode-hindsight", "OpenCode"),
    "eve": IntegrationMeta("@vectorize-io/hindsight-eve", "Eve"),
    "cloudflare-oauth-proxy": IntegrationMeta("hindsight-cloudflare-oauth-proxy"),
    "openai-agents": IntegrationMeta("hindsight-openai-agents"),
    "pipecat": IntegrationMeta("hindsight-pipecat", "Pipecat"),
    "agentcore": IntegrationMeta("hindsight-agentcore", "Bedrock AgentCore"),
    "smolagents": IntegrationMeta("hindsight-smolagents", "SmolAgents"),
    "n8n": IntegrationMeta("@vectorize-io/n8n-nodes-hindsight", "n8n"),
    "dify": IntegrationMeta("hindsight-dify", "Dify"),
    "vapi": IntegrationMeta("hindsight-vapi", "Vapi"),
    "gemini-spark": IntegrationMeta("hindsight-gemini-spark", "Gemini Spark"),
    "flowise": IntegrationMeta("@vectorize-io/flowise-nodes-hindsight", "Flowise"),
    "google-adk": IntegrationMeta("hindsight-google-adk", "Google ADK"),
    "superagent": IntegrationMeta("hindsight-superagent", "Superagent"),
    "obsidian": IntegrationMeta("@vectorize-io/hindsight-obsidian", "Obsidian"),
    "haystack": IntegrationMeta("hindsight-haystack", "Haystack"),
    "roo-code": IntegrationMeta("hindsight-roo-code", "Roo Code"),
    "omo": IntegrationMeta("hindsight-omo", "OMO"),
    "composio": IntegrationMeta("hindsight-composio", "Composio"),
    "continue": IntegrationMeta("hindsight-continue", "Continue"),
    "zed": IntegrationMeta("hindsight-zed", "Zed"),
    "openhands": IntegrationMeta("hindsight-openhands", "OpenHands"),
    "devin-desktop": IntegrationMeta("hindsight-devin-desktop", "Devin Desktop"),
}

VALID_INTEGRATIONS = list(INTEGRATIONS.keys())


class ChangelogEntry(BaseModel):
    """A single changelog entry."""

    category: str  # "feature", "improvement", "bugfix", "breaking", "other"
    summary: str  # Brief description of the change
    commit_id: str  # Short commit hash


class ChangelogResponse(BaseModel):
    """Structured response from LLM."""

    entries: list[ChangelogEntry]


@dataclass
class Commit:
    """Parsed commit from git log."""

    hash: str
    message: str


@dataclass(frozen=True)
class Migration:
    """An Alembic migration added in a release, enumerated from git history."""

    revision: str
    description: str
    path: str
    commit: str
    pr: int | None
    tables: tuple[str, ...] = ()


def _pr_number_from_subject(subject: str) -> int | None:
    """Extract the merge PR number from a squash-merge commit subject.

    Subjects end with the PR that merged them, e.g.
    `fix(x): ... (#3361, #3273) (#3622)` -> 3622. Earlier `(#N)` groups are
    issue references, so the *last* match is the PR.
    """
    matches = re.findall(r"\(#(\d+)\)", subject)
    return int(matches[-1]) if matches else None


def _file_at_ref(ref: str, path: str) -> str | None:
    """Read a file's contents at a git ref, or None if it doesn't exist there."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _strip_prose(source: str) -> str:
    """Drop the module docstring and `#` comments so prose can't look like a table.

    Only the *module* docstring is removed: migration SQL lives in triple-quoted
    strings further down, and stripping every triple-quoted block would throw away
    the statements this scan exists to read.
    """
    without_docstring = re.sub(r'\A\s*(?:"""|\'\'\')(?:.|\n)*?(?:"""|\'\'\')', "", source)
    return re.sub(r"(?m)#.*$", "", without_docstring)


def extract_tables(source: str) -> tuple[str, ...]:
    """Return the known tables a migration touches, ordered by volume then name.

    Intersected with TABLE_VOLUME: the SQL patterns are deliberately loose (an
    index expression or a stray identifier can match), and a reviewed table list
    is a cheaper filter than trying to parse every dialect's DDL.
    """
    body = _strip_prose(source)
    found = {match for pattern in _TABLE_PATTERNS for match in re.findall(pattern, body)}
    known = found & TABLE_VOLUME.keys()
    for pattern in _INDEX_PATTERNS:
        for index_name in re.findall(pattern, body):
            owners = [table for table in TABLE_VOLUME if table in index_name]
            if owners:
                known.add(max(owners, key=len))
    return tuple(sorted(known, key=lambda table: (VOLUME_ORDER.index(TABLE_VOLUME[table]), table)))


@dataclass(frozen=True)
class MigrationDoc:
    """The revision id and one-line description read out of a migration file."""

    revision: str
    description: str


def _parse_migration_file(source: str, path: str) -> MigrationDoc:
    """Read the revision id and description from a migration file's contents."""
    revision_match = re.search(r"^revision:\s*str\s*=\s*[\"']([^\"']+)[\"']", source, re.MULTILINE)
    revision = revision_match.group(1) if revision_match else Path(path).name.split("_", 1)[0]

    docstring_match = re.search(r'^\s*"""(.*?)$', source, re.MULTILINE)
    if docstring_match and docstring_match.group(1).strip():
        description = docstring_match.group(1).strip()
    else:
        # Fall back to the filename slug: `abc123_add_foo_index.py` -> `add foo index`
        stem = Path(path).stem.split("_", 1)
        description = stem[1].replace("_", " ") if len(stem) > 1 else Path(path).stem
    return MigrationDoc(revision=revision, description=description)


def get_new_migrations(from_ref: str | None, to_ref: str) -> list[Migration]:
    """Enumerate Alembic migrations added between two refs, oldest commit first.

    Deterministic: this reads git history directly, it never goes through the LLM.
    Migrations added and later removed within the same range are skipped (they
    don't exist at `to_ref`).
    """
    range_arg = f"{from_ref}..{to_ref}" if from_ref else to_ref
    result = subprocess.run(
        [
            "git",
            "log",
            "--diff-filter=A",
            "--no-merges",
            "--format=%x00%h|%s",
            "--name-only",
            range_arg,
            "--",
            MIGRATIONS_DIR,
        ],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        check=True,
    )

    migrations: list[Migration] = []
    seen_paths: set[str] = set()
    for block in result.stdout.split("\0"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        commit_hash, _, subject = lines[0].partition("|")
        pr = _pr_number_from_subject(subject)
        for path in sorted(lines[1:]):
            if not path.endswith(".py") or Path(path).name == "__init__.py":
                continue
            if path in seen_paths:
                continue
            source = _file_at_ref(to_ref, path)
            if source is None:
                continue
            seen_paths.add(path)
            doc = _parse_migration_file(source, path)
            migrations.append(
                Migration(
                    revision=doc.revision,
                    description=doc.description,
                    path=path,
                    commit=commit_hash,
                    pr=pr,
                    tables=extract_tables(source),
                )
            )

    # git log is newest-first; present migrations in the order they were applied.
    migrations.reverse()
    return migrations


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch)."""
    version = version.lstrip("v")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid semver: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def get_git_tags() -> list[str]:
    """Get all git tags sorted by semver (newest first)."""
    result = subprocess.run(
        ["git", "tag"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        check=True,
    )
    tags = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]

    valid_tags = []
    for tag in tags:
        try:
            parse_semver(tag)
            valid_tags.append(tag)
        except ValueError:
            continue

    valid_tags.sort(key=lambda t: parse_semver(t), reverse=True)
    return valid_tags


def get_integration_tags(integration: str) -> list[str]:
    """Get all tags for a specific integration, sorted by semver (newest first)."""
    prefix = f"integrations/{integration}/v"
    result = subprocess.run(
        ["git", "tag", "-l", f"{prefix}*"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        check=True,
    )
    tags = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]

    valid_tags = []
    for tag in tags:
        version_part = tag.removeprefix(prefix)
        try:
            parse_semver(version_part)
            valid_tags.append(tag)
        except ValueError:
            continue

    valid_tags.sort(key=lambda t: parse_semver(t.removeprefix(prefix)), reverse=True)
    return valid_tags


def find_previous_version(new_version: str, existing_tags: list[str]) -> str | None:
    """Find the previous version based on semver rules."""
    new_major, new_minor, new_patch = parse_semver(new_version)

    candidates = []
    for tag in existing_tags:
        try:
            major, minor, patch = parse_semver(tag)
        except ValueError:
            continue

        if (major, minor, patch) >= (new_major, new_minor, new_patch):
            continue

        candidates.append((tag, (major, minor, patch)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def find_previous_integration_tag(new_version: str, existing_tags: list[str], integration: str) -> str | None:
    """Find the previous integration tag based on semver rules."""
    prefix = f"integrations/{integration}/v"
    new_major, new_minor, new_patch = parse_semver(new_version)

    candidates = []
    for tag in existing_tags:
        version_part = tag.removeprefix(prefix)
        try:
            major, minor, patch = parse_semver(version_part)
        except ValueError:
            continue

        if (major, minor, patch) >= (new_major, new_minor, new_patch):
            continue

        candidates.append((tag, (major, minor, patch)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def get_commits(
    from_ref: str | None,
    to_ref: str,
    path_filter: str | None = None,
    exclude_paths: list[str] | None = None,
) -> list[Commit]:
    """Get commits between two refs as structured data."""
    if from_ref:
        cmd = ["git", "log", "--format=%h|%s", "--no-merges", f"{from_ref}..{to_ref}"]
    else:
        cmd = ["git", "log", "--format=%h|%s", "--no-merges", to_ref]

    if path_filter:
        cmd += ["--", path_filter]
    elif exclude_paths:
        cmd += ["--", ".", *[f":(exclude){p}" for p in exclude_paths]]

    result = subprocess.run(
        cmd,
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        check=True,
    )

    commits = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 1)
        if len(parts) == 2:
            commits.append(Commit(hash=parts[0], message=parts[1]))

    return commits


def get_detailed_diff(
    from_ref: str | None,
    to_ref: str,
    path_filter: str | None = None,
    exclude_paths: list[str] | None = None,
) -> str:
    """Get file change stats between two refs."""
    if from_ref:
        cmd = ["git", "diff", "--stat", f"{from_ref}..{to_ref}"]
    else:
        cmd = ["git", "diff", "--stat", f"{to_ref}^..{to_ref}"]

    if path_filter:
        cmd += ["--", path_filter]
    elif exclude_paths:
        cmd += ["--", ".", *[f":(exclude){p}" for p in exclude_paths]]

    result = subprocess.run(
        cmd,
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_commit_authors(commits: list[Commit]) -> dict[str, str]:
    """Fetch GitHub logins keyed by commit hash via the GitHub API.

    Bots (e.g. dependabot, github-actions) and missing authors are omitted.
    """
    authors: dict[str, str] = {}
    for commit in commits:
        result = subprocess.run(
            ["gh", "api", f"/repos/{GITHUB_REPO}/commits/{commit.hash}", "--jq", '.author.login // ""'],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
        )
        login = result.stdout.strip()
        if not login or login.endswith("[bot]"):
            continue
        authors[commit.hash] = login
    return authors


def _escape_mdx_text(text: str) -> str:
    """Escape MDX-significant characters in prose so MDX v3 treats it as text.

    Two hazards in LLM summaries:
    - Curly braces (`{user_id}`) parse as JSX expressions -> `ReferenceError: user_id
      is not defined` at SSG time.
    - Angle brackets (`<think>`) parse as JSX elements -> an unclosed-tag MDX
      compilation failure (mdast-util-mdx-jsx).

    Applied only to the summary text, never to the entry meta HTML, so escaping
    `<`/`>` here can't touch the rendered author/commit markup.
    """
    return text.replace("{", "\\{").replace("}", "\\}").replace("<", "&lt;").replace(">", "&gt;")


def _render_entry_meta(commit_id: str, commit_url: str, login: str | None) -> str:
    """Render inline metadata: · @author · commit-hash (GitHub-release style)."""
    sep = '<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span>'
    parts: list[str] = []
    if login:
        avatar = f"https://github.com/{login}.png?size=40"
        parts.append(sep)
        parts.append(
            f'<a href="https://github.com/{login}" target="_blank" rel="noopener noreferrer" '
            f'style={{{{color: "var(--ifm-color-primary)", textDecoration: "none", '
            f'display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}}}>'
            f'<img src="{avatar}" alt="@{login}" width="18" height="18" '
            f'style={{{{borderRadius: "50%"}}}} />@{login}</a>'
        )
    parts.append(sep)
    parts.append(
        f'<a href="{commit_url}" target="_blank" rel="noopener noreferrer" '
        f'style={{{{fontFamily: "var(--ifm-font-family-monospace, monospace)", '
        f'fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}}}>{commit_id}</a>'
    )
    return "".join(parts)


def _render_migration_meta(migration: Migration) -> str:
    """Render the trailing link for a migration: · #PR (or the commit as fallback)."""
    sep = '<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span>'
    if migration.pr is not None:
        href = f"{GITHUB_PULL_URL}/{migration.pr}"
        label = f"#{migration.pr}"
    else:
        href = f"{GITHUB_COMMIT_URL}/{migration.commit}"
        label = migration.commit
    link = (
        f'<a href="{href}" target="_blank" rel="noopener noreferrer" '
        f'style={{{{fontFamily: "var(--ifm-font-family-monospace, monospace)", '
        f'fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}}}>{label}</a>'
    )
    return sep + link


def _render_tables(tables: tuple[str, ...]) -> str:
    """Render the tables a migration touches, each tagged with its data volume."""
    if not tables:
        return ""
    sep = '<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span>'
    rendered = []
    for table in tables:
        tier = TABLE_VOLUME[table]
        rendered.append(
            f"<code>{table}</code>"
            f'<span style={{{{fontSize: "0.75em", color: "{tier.color}", marginLeft: "0.25em"}}}}>'
            f"{tier.label}</span>"
        )
    return sep + " ".join(rendered)


def render_migrations_section(migrations: list[Migration]) -> list[str]:
    """Render the deterministic "Database Migrations" section lines."""
    if not migrations:
        return []
    lines = ["**Database Migrations**", ""]
    hot = sorted({t for m in migrations for t in m.tables if TABLE_VOLUME[t] is HIGH_VOLUME})
    if hot:
        lines.append(
            f"This release alters high-volume tables ({', '.join(f'`{t}`' for t in hot)}). "
            "Migrations run on startup, so allow extra time on large deployments."
        )
        lines.append("")
    for migration in migrations:
        lines.append(
            f"- `{migration.revision}` — {_escape_mdx_text(migration.description)}"
            f"{_render_tables(migration.tables)}"
            f"{_render_migration_meta(migration)}"
        )
    lines.append("")
    return lines


def analyze_commits_with_llm(
    client: OpenAI,
    model: str,
    version: str,
    commits: list[Commit],
    file_diff: str,
    integration: str | None = None,
) -> list[ChangelogEntry]:
    """Use LLM to analyze commits and return structured changelog entries."""
    commits_json = json.dumps([{"commit_id": c.hash, "message": c.message} for c in commits], indent=2)

    subject = f"the {integration} integration for Hindsight" if integration else f"release {version} of Hindsight"

    # For the core changelog, integration/plugin commits are already filtered out
    # by path upstream; this rule is a belt-and-suspenders guard. It must NOT apply
    # when summarizing an integration's own changelog.
    skip_integrations_rule = (
        ""
        if integration
        else "\n- Skip integration and plugin changes (coding agents, framework plugins, anything shipped as "
        "its own package) — integrations are versioned and changelogged separately"
    )

    prompt = f"""Analyze the following git commits for {subject} (an AI memory system).

For each meaningful change, create a changelog entry with:
- category: one of "feature", "improvement", "bugfix", "breaking", "other"
- summary: brief one-line description of the change (user-facing, not technical)
- commit_id: the commit hash from the input

Rules:
- Group related commits into a single entry if they're part of the same change
- Skip trivial changes (typo fixes, formatting, internal refactoring)
- Skip repository-only changes: README updates, CI/GitHub Actions, release scripts, changelog updates, version bumps{skip_integrations_rule}
- Focus on user-facing changes that affect the product functionality
- Use the exact commit_id from the input (pick the most relevant one if grouping)
- If no meaningful changes remain after filtering, return an empty list

Commits:
{commits_json}

Files changed summary:
{file_diff[:4000]}"""

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=ChangelogResponse,
        max_completion_tokens=16000,
    )

    return response.choices[0].message.parsed.entries


def build_changelog_markdown(
    version: str,
    tag: str,
    entries: list[ChangelogEntry],
    integration: str | None = None,
    authors: dict[str, str] | None = None,
    migrations: list[Migration] | None = None,
) -> str:
    """Build markdown changelog from structured entries."""
    tag_url = (
        f"https://github.com/{GITHUB_REPO}/releases/tag/{tag}"
        if not integration
        else f"https://github.com/{GITHUB_REPO}/tree/{tag}"
    )

    # Group entries by category
    categories = {
        "breaking": ("Breaking Changes", []),
        "feature": ("Features", []),
        "improvement": ("Improvements", []),
        "bugfix": ("Bug Fixes", []),
        "other": ("Other", []),
    }

    for entry in entries:
        cat = entry.category.lower()
        if cat in categories:
            categories[cat][1].append(entry)
        else:
            categories["other"][1].append(entry)

    # Build markdown
    lines = [f"## [{version}]({tag_url})", ""]

    has_entries = False
    for cat_key in ["breaking", "feature", "improvement", "bugfix", "other"]:
        cat_name, cat_entries = categories[cat_key]
        if cat_entries:
            has_entries = True
            lines.append(f"**{cat_name}**")
            lines.append("")
            for entry in cat_entries:
                commit_url = f"{GITHUB_COMMIT_URL}/{entry.commit_id}"
                login = _lookup_author(entry.commit_id, authors) if authors else None
                meta = _render_entry_meta(entry.commit_id, commit_url, login)
                lines.append(f"- {_escape_mdx_text(entry.summary)}{meta}")
            lines.append("")

    migration_lines = render_migrations_section(migrations or [])
    lines.extend(migration_lines)

    if not has_entries and not migration_lines:
        lines.append("*This release contains internal maintenance and infrastructure changes only.*")
        lines.append("")

    return "\n".join(lines)


def _lookup_author(commit_id: str, authors: dict[str, str]) -> str | None:
    """Look up an author by full or short commit hash."""
    if commit_id in authors:
        return authors[commit_id]
    for full_hash, login in authors.items():
        if full_hash.startswith(commit_id) or commit_id.startswith(full_hash):
            return login
    return None


def read_existing_changelog(path: Path, default_header: str) -> tuple[str, str]:
    """Read existing changelog and split into header and content."""
    if not path.exists():
        return default_header, ""

    content = path.read_text()

    match = re.search(r"^## ", content, re.MULTILINE)
    if match:
        header = content[: match.start()].rstrip() + "\n\n"
        releases = content[match.start() :]
    else:
        header = content.rstrip() + "\n\n"
        releases = ""

    return header, releases


def write_changelog(path: Path, header: str, new_entry: str, existing_releases: str) -> None:
    """Write changelog with new entry prepended."""
    content = header + new_entry + "\n" + existing_releases
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")


def generate_changelog_entry(
    version: str,
    llm_model: str = "gpt-5.2",
) -> None:
    """Generate changelog entry for a specific version."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    tag = version if version.startswith("v") else f"v{version}"
    display_version = version.lstrip("v")

    console.print("[blue]Fetching tags from repository...[/blue]")
    existing_tags = get_git_tags()

    if tag not in existing_tags and display_version not in existing_tags:
        console.print(f"[red]Error: Tag {tag} not found in repository[/red]")
        console.print("[red]Create the tag first before generating changelog[/red]")
        sys.exit(1)

    actual_tag = tag if tag in existing_tags else display_version

    previous_tag = find_previous_version(display_version, existing_tags)

    if previous_tag:
        console.print(f"[green]Found previous version: {previous_tag}[/green]")
    else:
        console.print("[yellow]No previous version found, will include all commits[/yellow]")

    console.print("[blue]Getting commits (excluding integrations)...[/blue]")
    exclude_paths = ["hindsight-integrations"]
    commits = get_commits(previous_tag, actual_tag, exclude_paths=exclude_paths)
    # Drop any commit that *also* touched an integration/plugin. Integrations are
    # versioned and changelogged separately, but their PRs routinely touch shared
    # docs/CI/scripts as well, so a path-exclude alone still lets them leak into
    # the core changelog. Excluding every commit that touches
    # hindsight-integrations/ keeps the core changelog to core changes.
    integration_hashes = {c.hash for c in get_commits(previous_tag, actual_tag, path_filter="hindsight-integrations")}
    dropped = [c for c in commits if c.hash in integration_hashes]
    commits = [c for c in commits if c.hash not in integration_hashes]
    if dropped:
        console.print(f"[blue]Excluded {len(dropped)} integration/plugin commits from the core changelog[/blue]")
    file_diff = get_detailed_diff(previous_tag, actual_tag, exclude_paths=exclude_paths)

    if not commits:
        console.print("[red]Error: No commits found for this release[/red]")
        sys.exit(1)

    console.print(f"[blue]Found {len(commits)} commits[/blue]")

    # Log commits
    console.print("\n[bold]Commits:[/bold]")
    for c in commits:
        console.print(f"  {c.hash} {c.message}")

    console.print("\n[bold]Files changed:[/bold]")
    console.print(file_diff[:4000] if len(file_diff) > 4000 else file_diff)
    console.print("")

    console.print(f"[blue]Analyzing commits with LLM ({llm_model})...[/blue]")
    entries = analyze_commits_with_llm(client, llm_model, display_version, commits, file_diff)

    console.print(f"\n[bold]LLM identified {len(entries)} changelog entries:[/bold]")
    for entry in entries:
        console.print(f"  [{entry.category}] {entry.summary} ({entry.commit_id})")

    console.print("[blue]Fetching GitHub authors per commit...[/blue]")
    authors = get_commit_authors(commits)
    unique = sorted(set(authors.values()))
    console.print(f"[blue]Found {len(unique)} contributors: {', '.join('@' + c for c in unique)}[/blue]")

    console.print("[blue]Enumerating new database migrations...[/blue]")
    migrations = get_new_migrations(previous_tag, actual_tag)
    for migration in migrations:
        pr = f"#{migration.pr}" if migration.pr else migration.commit
        console.print(f"  {migration.revision} {migration.description} ({pr})")
    if not migrations:
        console.print("[blue]No new migrations in this release[/blue]")

    new_entry = build_changelog_markdown(display_version, tag, entries, authors=authors, migrations=migrations)

    default_header = """---
hide_table_of_contents: true
---

# Changelog

This changelog highlights user-facing changes only. Internal maintenance, CI/CD, and infrastructure updates are omitted.

For full release details, see [GitHub Releases](https://github.com/vectorize-io/hindsight/releases).

"""
    header, existing_releases = read_existing_changelog(CHANGELOG_PATH, default_header)

    if f"## [{display_version}]" in existing_releases:
        console.print(f"[red]Error: Version {display_version} already exists in changelog[/red]")
        sys.exit(1)

    write_changelog(CHANGELOG_PATH, header, new_entry, existing_releases)

    console.print(f"\n[green]Changelog updated: {CHANGELOG_PATH}[/green]")
    console.print(f"\n[bold]New entry:[/bold]\n{new_entry}")


def generate_integration_changelog_entry(
    integration: str,
    version: str,
    llm_model: str = "gpt-5.2",
) -> None:
    """Generate changelog entry for a specific integration version."""
    if integration not in VALID_INTEGRATIONS:
        console.print(f"[red]Error: Unknown integration '{integration}'. Valid: {', '.join(VALID_INTEGRATIONS)}[/red]")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    display_version = version.lstrip("v")
    path_filter = f"hindsight-integrations/{integration}/"
    changelog_path = INTEGRATION_CHANGELOG_DIR / f"{integration}.md"

    console.print(f"[blue]Fetching integration tags for {integration}...[/blue]")
    existing_tags = get_integration_tags(integration)

    previous_tag = find_previous_integration_tag(display_version, existing_tags, integration)

    if previous_tag:
        console.print(f"[green]Found previous tag: {previous_tag}[/green]")
    else:
        console.print("[yellow]No previous tag found, will include all commits touching this integration[/yellow]")

    console.print(f"[blue]Getting commits for {path_filter}...[/blue]")
    commits = get_commits(previous_tag, "HEAD", path_filter=path_filter)
    file_diff = get_detailed_diff(previous_tag, "HEAD", path_filter=path_filter)

    if not commits:
        console.print("[yellow]Warning: No commits found touching this integration path[/yellow]")
        entries = []
    else:
        console.print(f"[blue]Found {len(commits)} commits[/blue]")

        console.print("\n[bold]Commits:[/bold]")
        for c in commits:
            console.print(f"  {c.hash} {c.message}")

        console.print("\n[bold]Files changed:[/bold]")
        console.print(file_diff[:4000] if len(file_diff) > 4000 else file_diff)
        console.print("")

        console.print(f"[blue]Analyzing commits with LLM ({llm_model})...[/blue]")
        entries = analyze_commits_with_llm(
            client, llm_model, display_version, commits, file_diff, integration=integration
        )

        console.print(f"\n[bold]LLM identified {len(entries)} changelog entries:[/bold]")
        for entry in entries:
            console.print(f"  [{entry.category}] {entry.summary} ({entry.commit_id})")

    if commits:
        console.print("[blue]Fetching GitHub authors per commit...[/blue]")
        authors = get_commit_authors(commits)
        unique = sorted(set(authors.values()))
        console.print(f"[blue]Found {len(unique)} contributors: {', '.join('@' + c for c in unique)}[/blue]")
    else:
        authors = {}

    integration_tag = f"integrations/{integration}/v{display_version}"
    new_entry = build_changelog_markdown(
        display_version, integration_tag, entries, integration=integration, authors=authors
    )

    package_name = _get_package_name(integration)
    default_header = f"""---
hide_table_of_contents: true
---

# {_integration_display_name(integration)} Integration Changelog

Changelog for [`{package_name}`]({_package_url(integration, package_name)}).

For the source code, see [`hindsight-integrations/{integration}`](https://github.com/{GITHUB_REPO}/tree/main/hindsight-integrations/{integration}).

← [Back to main changelog](/changelog)

"""
    header, existing_releases = read_existing_changelog(changelog_path, default_header)

    if f"## [{display_version}]" in existing_releases:
        console.print(f"[red]Error: Version {display_version} already exists in integration changelog[/red]")
        sys.exit(1)

    write_changelog(changelog_path, header, new_entry, existing_releases)

    console.print(f"\n[green]Integration changelog updated: {changelog_path}[/green]")
    console.print(f"\n[bold]New entry:[/bold]\n{new_entry}")


def _get_package_name(integration: str) -> str:
    return INTEGRATIONS[integration].package_name


def _package_url(integration: str, package_name: str) -> str:
    # Git-distributed plugin bundles have no npm/pypi package — link to the
    # source tree instead of a registry page.
    if integration in ("claude-code", "agent-plugin"):
        return f"https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/{integration}"
    if package_name.startswith("@"):
        return f"https://www.npmjs.com/package/{package_name}"
    return f"https://pypi.org/project/{package_name}/"


def _integration_display_name(integration: str) -> str:
    return INTEGRATIONS[integration].display_name or integration


def main():
    parser = argparse.ArgumentParser(
        description="Generate changelog entry for a release",
        usage="generate-changelog VERSION [--model MODEL] [--integration NAME]",
    )
    parser.add_argument(
        "version",
        help="Version to generate changelog for (e.g., 1.0.5, v1.0.5)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--integration",
        default=None,
        help=f"Generate changelog for a specific integration. Valid: {', '.join(VALID_INTEGRATIONS)}",
    )

    args = parser.parse_args()

    if args.integration:
        generate_integration_changelog_entry(
            integration=args.integration,
            version=args.version,
            llm_model=args.model,
        )
    else:
        generate_changelog_entry(
            version=args.version,
            llm_model=args.model,
        )


if __name__ == "__main__":
    main()
