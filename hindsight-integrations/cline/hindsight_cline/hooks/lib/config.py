"""Configuration for the Hindsight Cline integration.

Settings are a typed ``HindsightClineConfig`` (not a raw dict) built from
built-in defaults, then merged with the plugin ``settings.json``, a user
config file, and environment variable overrides. The on-disk/env format stays
camelCase (``autoRecall``, ``bankId``, …); load converts those keys to the
dataclass's snake_case fields. Lean v1 schema — no daemon or chunked-retain
keys (see the integration README for the rationale).
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class HindsightClineConfig:
    """Typed plugin configuration. Field names are the snake_case form of the
    camelCase keys used in settings.json / env vars."""

    # Recall
    auto_recall: bool = True
    recall_budget: str = "mid"
    recall_max_tokens: int = 1024
    recall_timeout: int = 10
    recall_types: list[str] = field(default_factory=lambda: ["world", "experience"])
    recall_context_turns: int = 1
    recall_max_query_chars: int = 800
    recall_prompt_preamble: str = (
        "Relevant memories from past conversations (prioritize recent when "
        "conflicting). Only use memories that are directly useful to continue "
        "this task; ignore the rest:"
    )
    # Retain
    auto_retain: bool = True
    retain_context: str = "cline"
    retain_tags: list[str] = field(default_factory=lambda: ["{task_id}"])
    # User-defined, open-ended key/values → genuinely dynamic, so a dict is correct here.
    retain_metadata: dict[str, Any] = field(default_factory=dict)
    retain_timeout: int = 15
    # Connection
    hindsight_api_url: Optional[str] = None
    hindsight_api_token: Optional[str] = None
    api_port: int = 9077
    # Bank
    bank_id: str = "cline"
    bank_id_prefix: str = ""
    dynamic_bank_id: bool = False
    dynamic_bank_granularity: list[str] = field(default_factory=lambda: ["agent", "project"])
    bank_mission: str = (
        "You are a Cline AI coding assistant. Focus on technical decisions, "
        "code changes, debugging sessions, architectural choices, and project "
        "context relevant to the user's work."
    )
    retain_mission: Optional[str] = (
        "Extract technical decisions, code patterns, debugging solutions, user "
        "preferences, project context, and architectural choices. Ignore "
        "routine greetings and transient operational details."
    )
    agent_name: str = "cline"
    # Misc
    debug: bool = False


# Map env var names to dataclass fields and their types.
ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "HINDSIGHT_API_URL": ("hindsight_api_url", str),
    "HINDSIGHT_API_TOKEN": ("hindsight_api_token", str),
    "HINDSIGHT_BANK_ID": ("bank_id", str),
    "HINDSIGHT_AGENT_NAME": ("agent_name", str),
    "HINDSIGHT_AUTO_RECALL": ("auto_recall", bool),
    "HINDSIGHT_AUTO_RETAIN": ("auto_retain", bool),
    "HINDSIGHT_RECALL_BUDGET": ("recall_budget", str),
    "HINDSIGHT_RECALL_MAX_TOKENS": ("recall_max_tokens", int),
    "HINDSIGHT_RECALL_TIMEOUT": ("recall_timeout", int),
    "HINDSIGHT_RECALL_MAX_QUERY_CHARS": ("recall_max_query_chars", int),
    "HINDSIGHT_RECALL_CONTEXT_TURNS": ("recall_context_turns", int),
    "HINDSIGHT_API_PORT": ("api_port", int),
    "HINDSIGHT_DYNAMIC_BANK_ID": ("dynamic_bank_id", bool),
    "HINDSIGHT_BANK_MISSION": ("bank_mission", str),
    "HINDSIGHT_DEBUG": ("debug", bool),
}

_VALID_FIELDS = {f for f in HindsightClineConfig.__dataclass_fields__}


def camel_to_snake(name: str) -> str:
    """Convert a camelCase settings key to a snake_case dataclass field."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def find_settings_path() -> Optional[Path]:
    """Locate the plugin's settings.json next to this lib directory.

    The repo and installed layouts differ: in the repo settings.json sits at
    the integration root (two levels above lib/), while install.py copies it
    into the hooks dir (one level above lib/). Search upward and take the
    first hit so both layouts work without special-casing.
    """
    base = Path(__file__).resolve()
    for up in (1, 2, 3):
        candidate = base.parents[up] / "settings.json"
        if candidate.exists():
            return candidate
    return None


def _cast_env(value: str, typ: type) -> Any:
    """Cast an environment-variable string to the target type; None on failure."""
    try:
        if typ is bool:
            return value.lower() in ("true", "1", "yes")
        if typ is int:
            return int(value)
        return value
    except (ValueError, AttributeError):
        return None


def _merge_settings_file(path: str, config: HindsightClineConfig) -> None:
    """Merge a camelCase settings.json into config in place. Silently skips if missing."""
    if not path or not os.path.exists(path):
        return
    try:
        with open(path) as f:
            file_config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        debug_log(config, f"Failed to load {path}: {e}")
        return
    for key, value in file_config.items():
        field_name = camel_to_snake(key)
        if value is not None and field_name in _VALID_FIELDS:
            setattr(config, field_name, value)


def load_config() -> HindsightClineConfig:
    """Load plugin configuration.

    Loading order (later entries win):
      1. Built-in defaults (the dataclass defaults)
      2. Plugin settings.json (found via find_settings_path)
      3. User config (~/.hindsight/cline.json) — stable across updates
      4. Environment variable overrides
    """
    config = HindsightClineConfig()

    settings_path = find_settings_path()
    if settings_path is not None:
        _merge_settings_file(str(settings_path), config)

    user_config_path = os.path.join(os.path.expanduser("~"), ".hindsight", "cline.json")
    _merge_settings_file(user_config_path, config)

    for env_name, (field_name, typ) in ENV_OVERRIDES.items():
        val = os.environ.get(env_name)
        if val is not None:
            cast_val = _cast_env(val, typ)
            if cast_val is not None:
                setattr(config, field_name, cast_val)

    return config


def debug_log(config: HindsightClineConfig, *args: Any) -> None:
    """Log to stderr if debug mode is enabled."""
    if config.debug:
        print("[Hindsight]", *args, file=sys.stderr)
