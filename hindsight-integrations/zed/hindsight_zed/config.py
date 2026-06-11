"""Configuration for the Hindsight Zed daemon.

Settings layer (later wins): built-in defaults → ``~/.hindsight/zed.json`` →
environment variables. Resolved into a typed :class:`ZedConfig`.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Cross-integration cloud-default convention.
DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"

USER_CONFIG_FILE = Path.home() / ".hindsight" / "zed.json"


@dataclass
class ZedConfig:
    """Resolved daemon configuration."""

    # Connection
    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    hindsight_api_token: Optional[str] = None

    # Bank scoping (per-project by default — one bank per git repo / folder).
    bank_prefix: str = "zed"
    fixed_bank_id: Optional[str] = None  # if set, all projects share this bank
    bank_mission: str = ""

    # Recall (passive injection)
    auto_recall: bool = True
    recall_budget: str = "mid"
    recall_max_tokens: int = 1024
    recall_types: list = field(default_factory=lambda: ["world", "experience"])
    recall_max_query_chars: int = 800
    recall_preamble: str = (
        "Relevant memory from past sessions on this project (recalled automatically; "
        "use what's relevant, ignore the rest):"
    )

    # Retain (passive capture)
    auto_retain: bool = True
    retain_context: str = "zed"
    retain_tags: list = field(default_factory=list)

    # Daemon
    poll_interval: float = 5.0  # seconds between threads.db polls
    debug: bool = False


# settings.json / user-config key  ->  (attribute, caster)
_FILE_KEYS = {
    "hindsightApiUrl": ("hindsight_api_url", str),
    "hindsightApiToken": ("hindsight_api_token", str),
    "bankPrefix": ("bank_prefix", str),
    "fixedBankId": ("fixed_bank_id", str),
    "bankMission": ("bank_mission", str),
    "autoRecall": ("auto_recall", bool),
    "recallBudget": ("recall_budget", str),
    "recallMaxTokens": ("recall_max_tokens", int),
    "recallTypes": ("recall_types", list),
    "recallMaxQueryChars": ("recall_max_query_chars", int),
    "recallPreamble": ("recall_preamble", str),
    "autoRetain": ("auto_retain", bool),
    "retainContext": ("retain_context", str),
    "retainTags": ("retain_tags", list),
    "pollInterval": ("poll_interval", float),
    "debug": ("debug", bool),
}

# env var -> (attribute, caster)
_ENV_KEYS = {
    "HINDSIGHT_API_URL": ("hindsight_api_url", str),
    "HINDSIGHT_API_TOKEN": ("hindsight_api_token", str),
    "HINDSIGHT_ZED_BANK_PREFIX": ("bank_prefix", str),
    "HINDSIGHT_ZED_FIXED_BANK_ID": ("fixed_bank_id", str),
    "HINDSIGHT_ZED_BANK_MISSION": ("bank_mission", str),
    "HINDSIGHT_ZED_AUTO_RECALL": ("auto_recall", bool),
    "HINDSIGHT_ZED_AUTO_RETAIN": ("auto_retain", bool),
    "HINDSIGHT_ZED_RECALL_BUDGET": ("recall_budget", str),
    "HINDSIGHT_ZED_RECALL_MAX_TOKENS": ("recall_max_tokens", int),
    "HINDSIGHT_ZED_RETAIN_CONTEXT": ("retain_context", str),
    "HINDSIGHT_ZED_POLL_INTERVAL": ("poll_interval", float),
    "HINDSIGHT_ZED_DEBUG": ("debug", bool),
}


def _cast(value, typ):
    """Cast a value (str from env, or JSON value from file) to ``typ``."""
    try:
        if typ is bool:
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes")
        if typ is int:
            return int(value)
        if typ is float:
            return float(value)
        if typ is list:
            return list(value) if isinstance(value, (list, tuple)) else [value]
        return str(value)
    except (ValueError, TypeError):
        return None


def load_config(config_file: Optional[Path] = None, env: Optional[dict] = None) -> ZedConfig:
    """Load and resolve daemon configuration."""
    cfg = ZedConfig()
    env = os.environ if env is None else env

    # 1. user config file
    path = config_file if config_file is not None else USER_CONFIG_FILE
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        for key, (attr, typ) in _FILE_KEYS.items():
            if key in data and data[key] is not None:
                cast = _cast(data[key], typ)
                if cast is not None:
                    setattr(cfg, attr, cast)

    # 2. environment overrides
    for key, (attr, typ) in _ENV_KEYS.items():
        if key in env and env[key] != "":
            cast = _cast(env[key], typ)
            if cast is not None:
                setattr(cfg, attr, cast)

    # Empty URL falls back to the cloud default.
    if not cfg.hindsight_api_url:
        cfg.hindsight_api_url = DEFAULT_HINDSIGHT_API_URL

    return cfg
