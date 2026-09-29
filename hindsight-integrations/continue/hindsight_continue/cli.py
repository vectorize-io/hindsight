"""Command-line entry point: ``hindsight-continue``."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import configure
from .server import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-continue",
        description="Run the Hindsight context-provider adapter for Continue.dev.",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: env or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Listen port (default: env or 8123)")
    parser.add_argument("--bank-id", default=None, help="Default Hindsight bank id (default: env)")
    parser.add_argument("--api-url", default=None, help="Hindsight API URL (default: env or cloud)")
    parser.add_argument("--budget", choices=["low", "mid", "high"], default="mid", help="Recall budget")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = configure(
        hindsight_api_url=args.api_url,
        bank_id=args.bank_id,
        budget=args.budget,
        host=args.host,
        port=args.port,
    )
    run(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
