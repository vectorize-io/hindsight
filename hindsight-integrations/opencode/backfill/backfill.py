#!/usr/bin/env python3
"""Multi-source backfill CLI for Hindsight.

Imports historical AI coding sessions into Hindsight for long-term memory.
Supports two input sources:

  opencode  - Reads OpenCode's SQLite database (~/.local/share/opencode/opencode.db)
  jsonl     - Reads JSONL transcript files (one JSON object per line)

Usage:
  python backfill.py opencode --hindsight-url http://localhost:8888 --bank-id opencode
  python backfill.py jsonl --hindsight-url http://localhost:8888 --bank-id my-agent --input ./transcripts/*.jsonl

Requirements:
  pip install hindsight-client
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from hindsight_client import Hindsight
except ImportError:
    print(
        "Error: hindsight-client not installed. Run: pip install hindsight-client",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# OpenCode source
# ---------------------------------------------------------------------------

DEFAULT_OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def get_opencode_sessions(
    conn, since=None, project=None, skip_subagent=True, min_chars=200
):
    """Query sessions from OpenCode's SQLite database."""
    query = """
        SELECT
            s.id, s.title, s.directory, s.time_created, s.time_updated,
            count(DISTINCT m.id) as msg_count,
            sum(CASE WHEN json_extract(p.data, '$.type') = 'text'
                THEN length(json_extract(p.data, '$.text')) ELSE 0 END) as text_chars
        FROM session s
        JOIN message m ON m.session_id = s.id
        JOIN part p ON p.message_id = m.id
        GROUP BY s.id
        HAVING text_chars >= ?
        ORDER BY s.time_created ASC
    """
    rows = conn.execute(query, (min_chars,)).fetchall()

    sessions = []
    for (
        sid,
        title,
        directory,
        time_created,
        time_updated,
        msg_count,
        text_chars,
    ) in rows:
        if skip_subagent and "subagent" in (title or "").lower():
            continue
        if since:
            session_date = datetime.fromtimestamp(time_created / 1000, tz=timezone.utc)
            if session_date < since:
                continue
        if project:
            dir_name = Path(directory).name if directory else ""
            if dir_name.lower() != project.lower():
                continue

        sessions.append(
            {
                "id": sid,
                "title": title,
                "directory": directory,
                "time_created": time_created,
                "msg_count": msg_count,
                "text_chars": text_chars or 0,
            }
        )

    return sessions


def reconstruct_opencode_transcript(conn, session_id, include_tools=False):
    """Reconstruct a conversation transcript from OpenCode's database."""
    query = """
        SELECT
            json_extract(m.data, '$.role') as role,
            json_extract(m.data, '$.modelID') as model,
            json_extract(p.data, '$.type') as part_type,
            json_extract(p.data, '$.text') as text,
            json_extract(p.data, '$.tool') as tool_name,
            m.time_created as msg_time
        FROM message m
        JOIN part p ON p.message_id = m.id
        WHERE m.session_id = ?
        AND json_extract(p.data, '$.type') IN ('text', 'tool')
        ORDER BY m.time_created ASC, p.time_created ASC
    """
    rows = conn.execute(query, (session_id,)).fetchall()

    lines = []
    message_count = 0
    for role, model, part_type, text, tool_name, msg_time in rows:
        ts = datetime.fromtimestamp(msg_time / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        if part_type == "text" and text:
            label = "User" if role == "user" else f"Assistant ({model or 'unknown'})"
            lines.append(f"{label} ({ts}):\n{text}")
            message_count += 1
        elif part_type == "tool" and tool_name and include_tools:
            lines.append(f"[Tool: {tool_name}]")

    return "\n\n".join(lines), message_count


def backfill_opencode(args, client):
    """Backfill from OpenCode's SQLite database."""
    db_path = args.db or DEFAULT_OPENCODE_DB
    if not os.path.exists(db_path):
        print(f"Error: OpenCode database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    sessions = get_opencode_sessions(
        conn,
        since=since,
        project=args.project,
        skip_subagent=args.skip_subagent,
        min_chars=args.min_chars,
    )

    print(f"Found {len(sessions)} sessions to backfill")
    if not sessions:
        return

    ingested = 0
    skipped = 0
    errors = 0

    for session in sessions:
        transcript, msg_count = reconstruct_opencode_transcript(
            conn,
            session["id"],
            include_tools=args.include_tools,
        )

        if not transcript or len(transcript) < args.min_chars:
            skipped += 1
            continue

        project_name = (
            Path(session["directory"]).name if session["directory"] else "unknown"
        )
        ts = datetime.fromtimestamp(
            session["time_created"] / 1000, tz=timezone.utc
        ).isoformat()

        if args.verbose or args.dry_run:
            print(
                f"  [{ingested + 1}] {session['title']} "
                f"({len(transcript):,} chars, {msg_count} msgs, {project_name})"
            )

        if args.dry_run:
            ingested += 1
            continue

        try:
            client.retain(
                bank_id=args.bank_id,
                content=transcript,
                document_id=f"opencode-session-{session['id']}",
                timestamp=ts,
                context=f"opencode coding session in {project_name}",
                metadata={
                    "source": "opencode",
                    "session_id": session["id"],
                    "project": project_name,
                    "directory": session["directory"] or "",
                    "title": session["title"] or "",
                },
                tags=[f"project:{project_name}"],
                retain_async=args.use_async,
            )
            ingested += 1
        except Exception as e:
            errors += 1
            print(f"  Error retaining session {session['id']}: {e}", file=sys.stderr)

    conn.close()

    print(f"\nDone. Ingested: {ingested}, Skipped: {skipped}, Errors: {errors}")
    if args.dry_run:
        print("(dry run, nothing was actually ingested)")


# ---------------------------------------------------------------------------
# JSONL source
# ---------------------------------------------------------------------------


def backfill_jsonl(args, client):
    """Backfill from JSONL transcript files.

    Each file is treated as one document. Lines are JSON objects with
    at minimum {role, content}. Optional fields: timestamp, model.

    Supported formats:
      {"role": "user", "content": "hello"}
      {"role": "assistant", "content": "hi there", "model": "gpt-4o"}
      {"type": "user", "message": {"role": "user", "content": "hello"}}  (Claude Code format)
    """
    input_paths = args.input
    if not input_paths:
        print("Error: --input is required for jsonl source", file=sys.stderr)
        sys.exit(1)

    # Expand globs
    files = []
    for pattern in input_paths:
        expanded = list(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        files.extend(f for f in expanded if f.is_file())

    if not files:
        print("Error: no files matched the input pattern(s)", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} transcript files to backfill")

    ingested = 0
    errors = 0

    for filepath in sorted(files):
        messages = []
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Claude Code nested format
                    if entry.get("type") in ("user", "assistant"):
                        msg = entry.get("message", {})
                        if isinstance(msg, dict) and msg.get("role"):
                            messages.append(msg)
                    # Flat format
                    elif "role" in entry and "content" in entry:
                        messages.append(entry)
        except OSError as e:
            print(f"  Error reading {filepath}: {e}", file=sys.stderr)
            errors += 1
            continue

        if not messages:
            continue

        # Reconstruct transcript
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle content blocks format
                content = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not content or not content.strip():
                continue
            model = msg.get("model", "")
            label = (
                "User"
                if role == "user"
                else f"Assistant ({model})"
                if model
                else f"Assistant"
            )
            lines.append(f"{label}:\n{content.strip()}")

        transcript = "\n\n".join(lines)
        if len(transcript) < args.min_chars:
            continue

        doc_id = f"jsonl-{filepath.stem}"
        context = args.context or f"transcript from {filepath.name}"

        if args.verbose or args.dry_run:
            print(
                f"  [{ingested + 1}] {filepath.name} ({len(transcript):,} chars, {len(messages)} msgs)"
            )

        if args.dry_run:
            ingested += 1
            continue

        try:
            tags = [f"source:{filepath.stem}"]
            if args.tags:
                tags.extend(args.tags)

            client.retain(
                bank_id=args.bank_id,
                content=transcript,
                document_id=doc_id,
                context=context,
                metadata={
                    "source": "jsonl",
                    "file": str(filepath),
                    "message_count": str(len(messages)),
                },
                tags=tags,
                retain_async=args.use_async,
            )
            ingested += 1
        except Exception as e:
            errors += 1
            print(f"  Error retaining {filepath}: {e}", file=sys.stderr)

    print(f"\nDone. Ingested: {ingested}, Errors: {errors}")
    if args.dry_run:
        print("(dry run, nothing was actually ingested)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_bank_if_needed(client, args):
    """Create the memory bank with a sensible mission if it doesn't exist."""
    try:
        client.create_bank(
            bank_id=args.bank_id,
            name=f"Backfill - {args.bank_id}",
            mission=(
                "I am a memory system for a software developer's coding sessions with AI assistants. "
                "I track architectural decisions, tool preferences, coding patterns, project context, "
                "and technical discussions. I prioritize decisions and their rationale, recurring patterns, "
                "and project-specific knowledge."
            ),
        )
        print(f"Created bank: {args.bank_id}")
    except Exception:
        pass  # Bank already exists


def main():
    parser = argparse.ArgumentParser(
        description="Backfill AI coding session history into Hindsight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill all OpenCode sessions
  python backfill.py opencode --hindsight-url http://localhost:8888 --bank-id opencode

  # Backfill only OKU_MONEY sessions since March
  python backfill.py opencode --hindsight-url http://localhost:8888 --bank-id opencode \\
    --since 2026-03-01 --project OKU_MONEY

  # Backfill from JSONL transcript files
  python backfill.py jsonl --hindsight-url http://localhost:8888 --bank-id my-agent \\
    --input "transcripts/*.jsonl"

  # Dry run (preview without ingesting)
  python backfill.py opencode --hindsight-url http://localhost:8888 --bank-id opencode --dry-run
""",
    )

    # Global arguments
    parser.add_argument(
        "source", choices=["opencode", "jsonl"], help="Input source type"
    )
    parser.add_argument("--hindsight-url", required=True, help="Hindsight API URL")
    parser.add_argument("--bank-id", required=True, help="Target memory bank ID")
    parser.add_argument("--token", default=None, help="Hindsight API token")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum transcript length to retain (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be ingested without calling retain",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print per-session details"
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async retain for faster ingestion",
    )

    # OpenCode-specific arguments
    parser.add_argument(
        "--db",
        default=None,
        help="Path to OpenCode database (default: ~/.local/share/opencode/opencode.db)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only sessions after this date (ISO 8601, e.g., 2026-03-01)",
    )
    parser.add_argument(
        "--project", default=None, help="Only sessions in this project directory name"
    )
    parser.add_argument(
        "--skip-subagent",
        action="store_true",
        default=True,
        help="Skip subagent sessions (default: true)",
    )
    parser.add_argument(
        "--no-skip-subagent",
        dest="skip_subagent",
        action="store_false",
        help="Include subagent sessions",
    )
    parser.add_argument(
        "--include-tools",
        action="store_true",
        help="Include tool call markers in transcripts",
    )

    # JSONL-specific arguments
    parser.add_argument(
        "--input", nargs="+", default=None, help="Input file paths or glob patterns"
    )
    parser.add_argument(
        "--context", default=None, help="Context label for retained memories"
    )
    parser.add_argument(
        "--tags", nargs="+", default=None, help="Additional tags for retained memories"
    )

    args = parser.parse_args()

    # Initialize client
    client = Hindsight(base_url=args.hindsight_url, token=args.token)
    if not args.dry_run:
        create_bank_if_needed(client, args)

    if args.source == "opencode":
        backfill_opencode(args, client)
    elif args.source == "jsonl":
        backfill_jsonl(args, client)


if __name__ == "__main__":
    main()
