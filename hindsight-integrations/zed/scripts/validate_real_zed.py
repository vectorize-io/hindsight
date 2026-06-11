#!/usr/bin/env python3
"""Validate the Zed threads.db reader against a *real* Zed database.

Run this after having at least one AI conversation in Zed. It opens your actual
threads.db, parses every thread with the same reader the integration uses, and
prints what it extracted — so you can eyeball whether the parsed transcript
matches the conversation you actually had.

A healthy result: thread count > 0, titles you recognize, and the last thread's
transcript reading back the messages you sent/received. If it finds 0 threads,
0 messages, or garbled text, the on-disk format differs from what the reader
expects and the reader needs adjusting.

    python scripts/validate_real_zed.py
    python scripts/validate_real_zed.py /path/to/threads.db   # explicit path
"""

import sys
from pathlib import Path

# Make the package importable when run from the integration directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hindsight_zed.threads_db import default_threads_db_path, read_threads  # noqa: E402


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else default_threads_db_path()
    print(f"threads.db: {db}")
    if not db.exists():
        print("  ✗ not found — open Zed, have one AI conversation, then re-run.")
        return 1

    threads = read_threads(db)
    print(f"  parsed {len(threads)} thread(s)\n")
    if not threads:
        print("  ✗ 0 threads parsed. Either no conversations yet, or the on-disk")
        print("    format differs from what the reader expects (a real bug to fix).")
        return 1

    threads.sort(key=lambda t: t.updated_at, reverse=True)
    for t in threads[:5]:
        print(f"- {t.title!r}  ({len(t.messages)} msgs, updated {t.updated_at})")
        if t.folder_paths:
            print(f"    project: {t.folder_paths}")

    last = threads[0]
    print(f"\n=== most recent thread transcript: {last.title!r} ===")
    if not last.messages:
        print("  ✗ thread parsed but 0 messages extracted — format mismatch in the")
        print("    messages array. Compare against a raw dump (see step B4).")
        return 1
    for m in last.messages:
        snippet = m.text.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        print(f"  [{m.role}] {snippet}")

    print("\n✓ Reader works against your real Zed database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
