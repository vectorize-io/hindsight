"""``hindsight-aider`` — a drop-in wrapper for ``aider`` with long-term memory.

Use it exactly like ``aider`` (all arguments pass straight through); it recalls
relevant project memory before the session and retains the transcript after::

    hindsight-aider                         # interactive, memory loaded
    hindsight-aider -m "fix the auth bug"   # recall uses the message as the query
    hindsight-aider src/app.py              # any aider args work

Configure via ``~/.hindsight/aider.json`` or environment variables
(``HINDSIGHT_API_TOKEN``, ``HINDSIGHT_AIDER_BANK_ID``, ...). The bank defaults to
the git repo name, so memory follows the project.
"""

from __future__ import annotations

import sys
from typing import Optional

from . import __version__, runner


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--hindsight-version",):
        print(f"hindsight-aider {__version__}")
        return 0
    return runner.run(args)


if __name__ == "__main__":
    sys.exit(main())
