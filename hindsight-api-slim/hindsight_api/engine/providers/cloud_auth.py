"""Sign in with Hindsight Cloud: ``hindsight-admin cloud login | status | logout``.

The login gets a Hindsight Cloud API key for the ``hindsight`` decision-model, embeddings
and reranker providers (Hindsight Inference) without copying one by hand:

1. Listen on ``127.0.0.1:<free port>/callback`` and open
   ``<cloud>/connect/desktop?port=..&state=..&source=hindsight_cli&host=<hostname>``.
2. The user signs in (or signs up), picks an organization and approves.
3. Cloud creates a key that can call the memory and inference APIs and redirects the
   browser to ``/callback?state=..&key=hsk_..&org_id=..&org_name=..``.
4. The key is saved to ``~/.hindsight/cloud/credentials.json`` (mode 0600).

Each provider's own key (``HINDSIGHT_API_DECISION_MODEL_API_KEY``,
``HINDSIGHT_API_EMBEDDINGS_HINDSIGHT_API_KEY``, ``HINDSIGHT_API_RERANKER_HINDSIGHT_API_KEY``)
still wins over the saved key.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import html
import os
import re
import secrets
import socket
import sys
import webbrowser
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from aiohttp import web
from pydantic import BaseModel, ValidationError

from ...banner import GRADIENT_END, GRADIENT_START, LOGO
from .chatgpt_auth import _print_now, _state_matches, _write_private
from .oauth_store_lock import oauth_store_lock

CLOUD_URL = "https://ui.hindsight.vectorize.io"
HOME = Path("~/.hindsight/cloud").expanduser()
CONNECT_PATH = "/connect/desktop"
CALLBACK_PATH = "/callback"
SOURCE = "hindsight_cli"
CREDENTIAL_FILE = "credentials.json"
LOGIN_COMMAND = "hindsight-admin cloud login"
KEY_PREFIX = "hsk_"

_LOGO = Path(__file__).with_name("cloud_logo.png")
_LOGIN_TIMEOUT_SECONDS = 600.0
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_LABEL = "hindsight cloud credential store"
_PAGE_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
# The callback URL carries the key, so the page loads nothing and sends no referrer.
# It continues the Cloud approval page: same tokens, card and success state.
_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Signed in · Hindsight</title>
<style>
:root{{--bg:#F3F5F9;--card:#FFFFFF;--fg:#0D1117;--muted:#525866;--border:rgba(0,0,0,.08);--ok:#00BC7D;--ring:none}}
@media (prefers-color-scheme:dark){{:root{{--bg:#080C17;--card:#0F1724;--fg:#DCE8FF;--muted:#6E8AAF;
--border:rgba(100,160,255,.10);--ok:#00D492;--ring:0 0 0 1px rgba(100,160,255,.07),0 1px 3px rgba(0,0,0,.5)}}}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;padding:16px;background:var(--bg);color:var(--fg);
font:13px/19px Inter,ui-sans-serif,system-ui,sans-serif}}
main{{width:100%;max-width:448px;padding:21px;text-align:center;background:var(--card);border:1px solid var(--border);
border-radius:16px;box-shadow:var(--ring)}}
img{{display:block;margin:0 auto;height:30px}}
svg{{display:block;margin:20px auto 12px;color:var(--ok)}}
h1{{margin:0 0 6px;font-size:20px;font-weight:600;letter-spacing:-.01em}}
p{{margin:0;color:var(--muted)}}.small{{margin-top:12px;font-size:12px}}
</style></head><body><main>
<img src="data:image/png;base64,{logo}" alt="Hindsight">
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
<h1>{heading}</h1><p>{body}</p><p class="small">You can close this tab.</p>
</main></body></html>"""

# Terminal look: the API's startup logo and its blue-to-teal brand colors.
_BLUE = GRADIENT_START
_TEAL = GRADIENT_END
_GREEN = (34, 197, 94)
_GRAY = (153, 153, 153)
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _color_on() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _paint(text: str, rgb: tuple[int, int, int], *, bold: bool = False) -> str:
    if not _color_on():
        return text
    return f"\033[{'1;' if bold else ''}38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _color_on() else text


def _banner() -> str:
    """The startup logo with "Hindsight Cloud" and the version beside it. Nothing without colors."""
    from ... import __version__

    if not _color_on():
        return ""
    rows = LOGO.splitlines()
    width = max(len(_ANSI.sub("", row)) for row in rows)
    beside = {
        1: _paint("Hind", _BLUE, bold=True) + _paint("sight", _TEAL, bold=True) + _bold(" Cloud"),
        2: _paint(f"v{__version__}", _GRAY),
    }
    return "\n".join(
        f"  {row}{' ' * (width - len(_ANSI.sub('', row)))}   {beside.get(i, '')}" for i, row in enumerate(rows)
    )


async def _spin(text: str) -> None:
    """A Claude-style spinner on one line, until cancelled. Only on a terminal."""
    frames = "·✢✳✶✻✽✻✶✳✢"
    i = 0
    try:
        while True:
            sys.stdout.write(f"\r  {_paint(frames[i % len(frames)], _TEAL)} {text}")
            sys.stdout.flush()
            i += 1
            await asyncio.sleep(0.12)
    finally:
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


class CloudAuthError(RuntimeError):
    pass


class CloudCredential(BaseModel):
    api_key: str
    org_id: str | None = None
    org_name: str | None = None
    cloud_url: str = CLOUD_URL
    created_at: datetime | None = None


class _Callback(BaseModel):
    """What Cloud sends back on ``/callback`` (besides the state, checked before this)."""

    key: str = ""
    org_id: str | None = None
    org_name: str | None = None


def read_credential(path: Path) -> CloudCredential | None:
    """The saved login, or None when there is none (or it is unreadable)."""
    try:
        return CloudCredential.model_validate_json(path.read_text())
    except (OSError, ValidationError):
        return None


def saved_api_key(home: Path | None = None) -> str | None:
    """The key ``hindsight-admin cloud login`` saved, or None. ``home`` defaults to ~/.hindsight/cloud."""
    credential = read_credential((home or HOME) / CREDENTIAL_FILE)
    return credential.api_key if credential else None


async def _write(path: Path, credential: CloudCredential) -> None:
    async with oauth_store_lock(path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label=_LOCK_LABEL):
        _write_private(path, credential.model_dump_json(indent=2) + "\n")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def connect_url(cloud_url: str, *, port: int, state: str, host: str) -> str:
    query = urlencode({"port": port, "state": state, "source": SOURCE, "host": host})
    return f"{cloud_url}{CONNECT_PATH}?{query}"


async def _start_callback_listener(
    port: int, expected_state: str, timeout: float
) -> Callable[[], Awaitable[_Callback]]:
    """Listen on 127.0.0.1; return a waiter for the first callback carrying ``expected_state``."""
    received: asyncio.Future[_Callback] = asyncio.get_running_loop().create_future()

    async def handle(request: web.Request) -> web.Response:
        # A stray or forged request to the port cannot finish (or abort) the login.
        if not _state_matches(request.query.get("state", ""), expected_state):
            return web.Response(status=400, text="Unexpected sign-in response.", content_type="text/plain")
        callback = _Callback(
            key=request.query.get("key", ""),
            org_id=request.query.get("org_id") or None,
            org_name=request.query.get("org_name") or None,
        )
        if not received.done():
            received.set_result(callback)
        org = callback.org_name
        page = _PAGE.format(
            heading=f"Signed in to {html.escape(org)}" if org else "Signed in to Hindsight Cloud",
            body="Your API key is saved. Return to the terminal.",
            # Inline: the listener stops right after this response, before a browser could fetch it.
            logo=base64.b64encode(_LOGO.read_bytes()).decode(),
        )
        return web.Response(text=page, content_type="text/html", headers=_PAGE_HEADERS)

    app = web.Application()
    app.router.add_get(CALLBACK_PATH, handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
    except OSError as exc:
        await runner.cleanup()
        raise CloudAuthError(
            f"Cannot open 127.0.0.1:{port} for the sign-in callback ({exc.strerror}); try --port."
        ) from exc

    async def wait() -> _Callback:
        try:
            return await asyncio.wait_for(received, timeout=timeout)
        finally:
            await runner.cleanup()

    return wait


async def login(
    home: Path,
    *,
    cloud_url: str = CLOUD_URL,
    port: int = 0,
    open_browser: Callable[[str], Any] | None = webbrowser.open,
    out: Callable[[str], None] = _print_now,
    timeout: float = _LOGIN_TIMEOUT_SECONDS,
) -> CloudCredential:
    """Run the browser sign-in and save the key. ``open_browser=None`` prints the link instead."""
    port = port or _free_port()
    state = secrets.token_urlsafe(32)
    wait = await _start_callback_listener(port, state, timeout)
    url = connect_url(cloud_url, port=port, state=state, host=socket.gethostname())
    if open_browser is None or not open_browser(url):
        out(f"Open this link in a browser on this machine to sign in:\n\n{url}\n")
    else:
        out(f"  {_paint('·', _TEAL)} Opened your browser to sign in.")
        out(_paint("  Browser didn't open? Use this link:", _GRAY) + f"\n\n  {url}\n")
    spinner = asyncio.create_task(_spin("Waiting for you to approve in the browser…")) if _color_on() else None
    try:
        callback = await wait()
    except TimeoutError:
        raise CloudAuthError("Timed out waiting for the browser sign-in.") from None
    finally:
        if spinner:
            spinner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await spinner

    if not callback.key.startswith(KEY_PREFIX):
        raise CloudAuthError("Hindsight Cloud returned no API key.")
    credential = CloudCredential(
        api_key=callback.key,
        org_id=callback.org_id,
        org_name=callback.org_name,
        cloud_url=cloud_url,
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
    )
    await _write(home / CREDENTIAL_FILE, credential)
    return credential


def success_lines(credential: CloudCredential, path: Path) -> list[str]:
    org = credential.org_name or credential.org_id or "Hindsight Cloud"
    return [
        _paint(f"  ✓ Signed in to {org}", _GREEN, bold=True),
        "",
        "  Your API key:",
        "",
        f"  {_paint(credential.api_key, _BLUE, bold=True)}",
        "",
        _paint(f"  Saved to {path} (only you can read it).", _GRAY),
        _paint(f"  Keep it secret. Revoke it any time at {credential.cloud_url}/api-keys", _GRAY),
        "",
        "  Add these to your Hindsight .env, then restart Hindsight:",
        "",
        f"    {_paint('HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight', _TEAL)}",
        f"    {_paint('HINDSIGHT_API_DECISION_MODEL_API_KEY=' + credential.api_key, _TEAL)}",
        "",
        "  Optionally, embeddings and reranking on Hindsight Inference too:",
        "",
        f"    {_paint('HINDSIGHT_API_EMBEDDINGS_PROVIDER=hindsight', _TEAL)}",
        f"    {_paint('HINDSIGHT_API_EMBEDDINGS_HINDSIGHT_API_KEY=' + credential.api_key, _TEAL)}",
        f"    {_paint('HINDSIGHT_API_RERANKER_PROVIDER=hindsight', _TEAL)}",
        f"    {_paint('HINDSIGHT_API_RERANKER_HINDSIGHT_API_KEY=' + credential.api_key, _TEAL)}",
    ]


def status_lines(home: Path) -> list[str]:
    path = home / CREDENTIAL_FILE
    credential = read_credential(path)
    if credential is None:
        return [f"Not signed in (no {path}). Run `{LOGIN_COMMAND}`."]
    return [
        f"Credential file: {path}",
        f"Organization: {credential.org_name or '(unknown)'} ({credential.org_id or '?'})",
        f"Key: {KEY_PREFIX}…{credential.api_key[-4:]}",
        f"Cloud: {credential.cloud_url}",
        f"Signed in: {credential.created_at.isoformat() if credential.created_at else '(unknown)'}",
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-admin cloud",
        description="Sign in to Hindsight Cloud for the hindsight decision model provider.",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Credential directory (default: ~/.hindsight/cloud)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    login_parser = sub.add_parser(
        "login", help="Sign in in a browser, or install a login saved on another machine (--from-file)."
    )
    login_parser.add_argument("--port", type=int, default=0, help="Loopback callback port (default: a free one).")
    login_parser.add_argument("--no-browser", action="store_true", help="Print the sign-in link instead of opening it.")
    login_parser.add_argument(
        "--from-file", type=Path, default=None, help="Copy a credentials.json made on a machine with a browser."
    )
    sub.add_parser("status", help="Show the organization and the key's last 4 characters.")
    sub.add_parser("logout", help="Delete the saved key (revoke it under API Keys in Hindsight Cloud).")
    args = parser.parse_args(argv)
    home: Path = args.home.expanduser() if args.home else HOME
    path = home / CREDENTIAL_FILE

    try:
        if args.command == "login" and args.from_file:
            credential = read_credential(args.from_file.expanduser())
            if credential is None:
                raise CloudAuthError(f"{args.from_file} is not a Hindsight Cloud credentials file.")
            if args.from_file.expanduser().resolve() != path.resolve():
                asyncio.run(_write(path, credential))
            print(f"Saved {path}.")
        elif args.command == "login":
            if banner := _banner():
                print(banner + "\n")
            print(_bold("  This signs you in to Hindsight Cloud and saves an API key for"))
            print(_bold("  the hindsight decision model. It is billed to your organization's credits."))
            print()
            credential = asyncio.run(
                login(
                    home,
                    port=args.port,
                    open_browser=None if args.no_browser else webbrowser.open,
                )
            )
            print("\n".join(success_lines(credential, path)))
        elif args.command == "status":
            print("\n".join(status_lines(home)))
        elif args.command == "logout":
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            print(f"Signed out; removed {path}. The key stays valid until you revoke it under API Keys.")
    except CloudAuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
