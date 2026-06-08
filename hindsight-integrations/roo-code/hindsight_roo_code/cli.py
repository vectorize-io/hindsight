"""Command-line interface for the Roo Code Hindsight integration.

Exposed as the ``hindsight-roo-code`` console script:

    hindsight-roo-code install
    hindsight-roo-code install --api-url http://localhost:8888
    hindsight-roo-code install --project-dir /path/to/project
    hindsight-roo-code install --global
"""

import argparse
import os
from pathlib import Path

from .install import DEFAULT_API_URL, run_install


def _run_install(args: argparse.Namespace) -> int:
    run_install(
        api_url=args.api_url,
        project_dir=Path(args.project_dir),
        global_install=args.global_install,
    )
    return 0


def _add_install_parser(subparsers: argparse._SubParsersAction) -> None:
    install = subparsers.add_parser(
        "install",
        help="Install the Hindsight MCP server and rules into a Roo Code config dir.",
    )
    install.add_argument(
        "--api-url",
        default=os.environ.get("HINDSIGHT_API_URL", DEFAULT_API_URL),
        help=f"Hindsight API base URL (default: {DEFAULT_API_URL})",
    )
    install.add_argument(
        "--project-dir",
        default=".",
        help="Project directory to install into (default: current directory)",
    )
    install.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install globally to ~/.roo/ instead of the project directory",
    )
    install.set_defaults(func=_run_install)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-roo-code",
        description="Install Hindsight long-term memory for Roo Code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_install_parser(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
