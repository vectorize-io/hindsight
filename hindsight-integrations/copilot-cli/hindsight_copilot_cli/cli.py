"""Command-line interface for the GitHub Copilot CLI Hindsight integration.

Exposed as the ``hindsight-copilot-cli`` console script:

    hindsight-copilot-cli install
    hindsight-copilot-cli install --api-url https://api.hindsight.vectorize.io --api-token hsk_...
    hindsight-copilot-cli install --scope repo
    hindsight-copilot-cli uninstall
    hindsight-copilot-cli uninstall --scope repo
"""

import argparse
import os

from .install import run_install, run_uninstall

DEFAULT_API_URL = "https://api.hindsight.vectorize.io"


def _run_install(args: argparse.Namespace) -> int:
    run_install(api_url=args.api_url, api_token=args.api_token, scope=args.scope)
    return 0


def _run_uninstall(args: argparse.Namespace) -> int:
    run_uninstall(scope=args.scope)
    return 0


def _add_scope_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        choices=["user", "repo"],
        default="user",
        help=(
            "Where to register hooks: 'user' (~/.copilot/hooks/, default, active for "
            "every session on this machine) or 'repo' (.github/hooks/ in the current "
            "repository, for team-shared hook config)."
        ),
    )


def _add_install_parser(subparsers: argparse._SubParsersAction) -> None:
    install = subparsers.add_parser(
        "install",
        help="Install the Hindsight hook scripts into GitHub Copilot CLI.",
    )
    install.add_argument(
        "--api-url",
        default=os.environ.get("HINDSIGHT_API_URL"),
        help=(
            "Hindsight API base URL written to ~/.hindsight/copilot-cli.json. "
            f"For Hindsight Cloud use {DEFAULT_API_URL}. "
            "Omit to connect to a local hindsight-embed daemon."
        ),
    )
    install.add_argument(
        "--api-token",
        default=os.environ.get("HINDSIGHT_API_TOKEN"),
        help="Hindsight API token (required for Hindsight Cloud).",
    )
    _add_scope_argument(install)
    install.set_defaults(func=_run_install)


def _add_uninstall_parser(subparsers: argparse._SubParsersAction) -> None:
    uninstall = subparsers.add_parser(
        "uninstall",
        help="Remove the Hindsight hook scripts from GitHub Copilot CLI.",
    )
    _add_scope_argument(uninstall)
    uninstall.set_defaults(func=_run_uninstall)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-copilot-cli",
        description="Install Hindsight long-term memory for GitHub Copilot CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_install_parser(subparsers)
    _add_uninstall_parser(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
