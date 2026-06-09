"""Command-line interface for the Cursor CLI Hindsight integration.

Exposed as the ``hindsight-cursor-cli`` console script:

    hindsight-cursor-cli install
    hindsight-cursor-cli install --api-url https://api.hindsight.vectorize.io --api-token hsk_...
    hindsight-cursor-cli uninstall
"""

import argparse
import os

from .install import run_install, run_uninstall

DEFAULT_API_URL = "https://api.hindsight.vectorize.io"


def _run_install(args: argparse.Namespace) -> int:
    run_install(api_url=args.api_url, api_token=args.api_token)
    return 0


def _run_uninstall(_args: argparse.Namespace) -> int:
    run_uninstall()
    return 0


def _add_install_parser(subparsers: argparse._SubParsersAction) -> None:
    install = subparsers.add_parser(
        "install",
        help="Install the Hindsight hook scripts into Cursor CLI.",
    )
    install.add_argument(
        "--api-url",
        default=os.environ.get("HINDSIGHT_API_URL"),
        help=(
            "Hindsight API base URL written to ~/.hindsight/cursor-cli.json. "
            f"For Hindsight Cloud use {DEFAULT_API_URL}. "
            "Omit to connect to a local hindsight-embed daemon."
        ),
    )
    install.add_argument(
        "--api-token",
        default=os.environ.get("HINDSIGHT_API_TOKEN"),
        help="Hindsight API token (required for Hindsight Cloud).",
    )
    install.set_defaults(func=_run_install)


def _add_uninstall_parser(subparsers: argparse._SubParsersAction) -> None:
    uninstall = subparsers.add_parser(
        "uninstall",
        help="Remove the Hindsight hook scripts from Cursor CLI.",
    )
    uninstall.set_defaults(func=_run_uninstall)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-cursor-cli",
        description="Install Hindsight long-term memory for Cursor CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_install_parser(subparsers)
    _add_uninstall_parser(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
