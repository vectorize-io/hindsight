"""Configuration for the Hindsight Devin Desktop integration.

Devin Desktop is the editor formerly known as Windsurf (Codeium).

Settings layer (later wins): built-in defaults -> ``~/.hindsight/devin-desktop.json``
-> environment variables. Resolved into a typed :class:`DevinDesktopConfig`.

Memory is split into two banks (see :mod:`hindsight_devin_desktop.cli`):

* a **global bank** (default ``devin-desktop``) for the user's cross-project
  memory — preferences, coding style, identity — shared across every project;
* a **project bank**, derived per-repository at ``init`` time (see
  :mod:`hindsight_devin_desktop.project`) so each project keeps its own isolated
  architecture/decisions/conventions.

Only the global bank lives in this user-level config file. The project bank is
per-repository, so it is derived fresh in each repo and baked into that repo's
committed rule file — never persisted globally here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
# The user's cross-project memory bank. Also the historical single-bank default,
# so an existing setup keeps working — it simply becomes the global bank.
DEFAULT_GLOBAL_BANK = "devin-desktop"

USER_CONFIG_FILE = Path.home() / ".hindsight" / "devin-desktop.json"


@dataclass
class DevinDesktopConfig:
    """Resolved configuration for the Devin Desktop MCP setup."""

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    hindsight_api_token: Optional[str] = None
    # Cross-project bank (user preferences / coding style / identity).
    global_bank: str = DEFAULT_GLOBAL_BANK
    # Explicit project-bank override. When ``None`` the CLI derives it per-repo
    # from git; this is never read from / written to the global config file.
    project_bank: Optional[str] = None


# user-config-file key -> attribute. Legacy ``bankId`` maps onto the global bank
# so a pre-0.2 setup (single bank) keeps that bank as its global one.
_FILE_KEYS = {
    "hindsightApiUrl": "hindsight_api_url",
    "hindsightApiToken": "hindsight_api_token",
    "globalBank": "global_bank",
    "bankId": "global_bank",
}

_ENV_KEYS = {
    "HINDSIGHT_API_URL": "hindsight_api_url",
    "HINDSIGHT_API_TOKEN": "hindsight_api_token",
    "HINDSIGHT_DEVIN_DESKTOP_GLOBAL_BANK": "global_bank",
    # Per-run project-bank override (not persisted).
    "HINDSIGHT_DEVIN_DESKTOP_BANK_ID": "project_bank",
}


def load_config(config_file: Optional[Path] = None, env: Optional[dict] = None) -> DevinDesktopConfig:
    """Load and resolve configuration from file then environment."""
    cfg = DevinDesktopConfig()
    env = os.environ if env is None else env

    path = config_file if config_file is not None else USER_CONFIG_FILE
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        # ``globalBank`` wins over the legacy ``bankId`` alias when both are set.
        for key, attr in _FILE_KEYS.items():
            value = data.get(key)
            if value and not (key == "bankId" and data.get("globalBank")):
                setattr(cfg, attr, str(value))

    for key, attr in _ENV_KEYS.items():
        value = env.get(key)
        if value:
            setattr(cfg, attr, str(value))

    if not cfg.hindsight_api_url:
        cfg.hindsight_api_url = DEFAULT_HINDSIGHT_API_URL
    if not cfg.global_bank:
        cfg.global_bank = DEFAULT_GLOBAL_BANK

    return cfg
