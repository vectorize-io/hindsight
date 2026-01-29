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
REPO_PATH = Path(__file__).parent.parent.parent
CHANGELOG_PATH = REPO_PATH / "hindsight-docs" / "src" / "pages" / "changelog.md"


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


def get_commits(from_ref: str | None, to_ref: str) -> list[Commit]:
    """Get commits between two refs as structured data."""
    if from_ref:
        cmd = ["git", "log", "--format=%h|%s", "--no-merges", f"{from_ref}..{to_ref}"]
    else:
        cmd = ["git", "log", "--format=%h|%s", "--no-merges", to_ref]

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


def get_detailed_diff(from_ref: str | None, to_ref: str) -> str:
    """Get file change stats between two refs."""
    if from_ref:
        cmd = ["git", "diff", "--stat", f"{from_ref}..{to_ref}"]
    else:
        cmd = ["git", "diff", "--stat", f"{to_ref}^..{to_ref}"]

    result = subprocess.run(
        cmd,
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def analyze_commits_with_llm(
    client: OpenAI,
    model: str,
    version: str,
    commits: list[Commit],
    file_diff: str,
) -> list[ChangelogEntry]:
    """Use LLM to analyze commits and return structured changelog entries."""
    commits_json = json.dumps([{"commit_id": c.hash, "message": c.message} for c in commits], indent=2)

    prompt = f"""Analyze the following git commits for release {version} of Hindsight (an AI memory system).

For each meaningful change, create a changelog entry with:
- category: one of "feature", "improvement", "bugfix", "breaking", "other"
- summary: brief one-line description of the change (user-facing, not technical)
- commit_id: the commit hash from the input

Rules:
- Group related commits into a single entry if they're part of the same change
- Skip trivial changes (typo fixes, formatting, internal refactoring)
- Skip repository-only changes: README updates, CI/GitHub Actions, release scripts, changelog updates, version bumps
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
) -> str:
    """Build markdown changelog from structured entries."""
    release_url = f"{GITHUB_RELEASES_URL}/tag/{tag}"

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
    lines = [f"## [{version}]({release_url})", ""]

    has_entries = False
    for cat_key in ["breaking", "feature", "improvement", "bugfix", "other"]:
        cat_name, cat_entries = categories[cat_key]
        if cat_entries:
            has_entries = True
            lines.append(f"**{cat_name}**")
            lines.append("")
            for entry in cat_entries:
                commit_url = f"{GITHUB_COMMIT_URL}/{entry.commit_id}"
                lines.append(f"- {entry.summary} ([`{entry.commit_id}`]({commit_url}))")
            lines.append("")

    if not has_entries:
        lines.append("*This release contains internal maintenance and infrastructure changes only.*")
        lines.append("")

    return "\n".join(lines)


def read_existing_changelog() -> tuple[str, str]:
    """Read existing changelog and split into header and content."""
    if not CHANGELOG_PATH.exists():
        header = """---
hide_table_of_contents: true
---

# Changelog

This changelog highlights user-facing changes only. Internal maintenance, CI/CD, and infrastructure updates are omitted.

For full release details, see [GitHub Releases](https://github.com/vectorize-io/hindsight/releases).
"""
        return header, ""

    content = CHANGELOG_PATH.read_text()

    match = re.search(r"^## ", content, re.MULTILINE)
    if match:
        header = content[: match.start()].rstrip() + "\n\n"
        releases = content[match.start() :]
    else:
        header = content.rstrip() + "\n\n"
        releases = ""

    return header, releases


def write_changelog(header: str, new_entry: str, existing_releases: str) -> None:
    """Write changelog with new entry prepended."""
    content = header + new_entry + "\n" + existing_releases
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGELOG_PATH.write_text(content.rstrip() + "\n")


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

    console.print("[blue]Getting commits...[/blue]")
    commits = get_commits(previous_tag, actual_tag)
    file_diff = get_detailed_diff(previous_tag, actual_tag)

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

    new_entry = build_changelog_markdown(display_version, tag, entries)

    header, existing_releases = read_existing_changelog()

    if f"## [{display_version}]" in existing_releases:
        console.print(f"[red]Error: Version {display_version} already exists in changelog[/red]")
        sys.exit(1)

    write_changelog(header, new_entry, existing_releases)

    console.print(f"\n[green]Changelog updated: {CHANGELOG_PATH}[/green]")
    console.print(f"\n[bold]New entry:[/bold]\n{new_entry}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate changelog entry for a release",
        usage="generate-changelog VERSION [--model MODEL]",
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

    args = parser.parse_args()

    generate_changelog_entry(
        version=args.version,
        llm_model=args.model,
    )


if __name__ == "__main__":
    main()
