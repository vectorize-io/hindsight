"""Configuration management for Hindsight Cursor plugin.

Loads settings from settings.json (plugin defaults) merged with environment
variable overrides. Follows the same layering as the Claude Code integration.
"""

import json
import os
import sys

# Default URL when no configuration is supplied — points at the hosted backend.
# Users who self-host override via ``hindsightApiUrl`` or set
# ``useLocalDaemon: True`` to fall back to the auto-managed local daemon path.
DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"

DEFAULTS = {
    # Recall
    "autoRecall": True,
    "recallBudget": "mid",
    "recallMaxTokens": 1024,
    "recallTypes": ["world", "experience"],
    "recallContextTurns": 1,
    "recallMaxQueryChars": 800,
    "recallPromptPreamble": (
        "Relevant memories from past coding sessions (prioritize recent when "
        "conflicting). Only use memories that are directly useful to continue "
        "this conversation; ignore the rest:"
    ),
    # Retain
    "autoRetain": True,
    "retainMode": "full-session",
    "retainEveryNTurns": 10,
    "retainOverlapTurns": 2,
    "retainToolCalls": False,
    "retainContext": "cursor",
    "retainTags": [],
    "retainMetadata": {},
    # Connection
    "hindsightApiUrl": "",
    "hindsightApiToken": None,
    "apiPort": 9077,
    "daemonIdleTimeout": 300,
    "embedVersion": "latest",
    "embedPackagePath": None,
    # If False (default), an empty ``hindsightApiUrl`` falls back to the hosted
    # backend (``DEFAULT_HINDSIGHT_API_URL``) — matches the cross-integration
    # cloud-default convention. Self-hosters who want the plugin to auto-start
    # a local daemon (the historical behaviour) opt in by setting True.
    "useLocalDaemon": False,
    # Workaround for Cursor's broken sessionStart additionalContext path.
    # See scripts/lib/rules_file.py for the upstream bug link.
    "useRulesFileFallback": True,
    "appendToGitignore": True,
    # Bank
    "bankId": "cursor",
    "bankIdPrefix": "",
    "dynamicBankId": False,
    "dynamicBankGranularity": ["agent", "project"],
    "bankMission": "",
    "retainMission": None,
    "agentName": "cursor",
    # LLM (for daemon mode)
    "llmProvider": None,
    "llmModel": None,
    "llmApiKeyEnv": None,
    # Misc
    "debug": False,
}

# Map env var names to config keys and their types
ENV_OVERRIDES = {
    "HINDSIGHT_API_URL": ("hindsightApiUrl", str),
    "HINDSIGHT_API_TOKEN": ("hindsightApiToken", str),
    "HINDSIGHT_BANK_ID": ("bankId", str),
    "HINDSIGHT_AGENT_NAME": ("agentName", str),
    "HINDSIGHT_AUTO_RECALL": ("autoRecall", bool),
    "HINDSIGHT_AUTO_RETAIN": ("autoRetain", bool),
    "HINDSIGHT_RETAIN_MODE": ("retainMode", str),
    "HINDSIGHT_RECALL_BUDGET": ("recallBudget", str),
    "HINDSIGHT_RECALL_MAX_TOKENS": ("recallMaxTokens", int),
    "HINDSIGHT_RECALL_MAX_QUERY_CHARS": ("recallMaxQueryChars", int),
    "HINDSIGHT_RECALL_CONTEXT_TURNS": ("recallContextTurns", int),
    "HINDSIGHT_API_PORT": ("apiPort", int),
    "HINDSIGHT_DAEMON_IDLE_TIMEOUT": ("daemonIdleTimeout", int),
    "HINDSIGHT_EMBED_VERSION": ("embedVersion", str),
    "HINDSIGHT_EMBED_PACKAGE_PATH": ("embedPackagePath", str),
    "HINDSIGHT_RETAIN_EVERY_N_TURNS": ("retainEveryNTurns", int),
    "HINDSIGHT_RETAIN_CONTEXT": ("retainContext", str),
    "HINDSIGHT_DYNAMIC_BANK_ID": ("dynamicBankId", bool),
    "HINDSIGHT_BANK_MISSION": ("bankMission", str),
    "HINDSIGHT_USE_LOCAL_DAEMON": ("useLocalDaemon", bool),
    "HINDSIGHT_USE_RULES_FILE_FALLBACK": ("useRulesFileFallback", bool),
    "HINDSIGHT_APPEND_TO_GITIGNORE": ("appendToGitignore", bool),
    "HINDSIGHT_LLM_PROVIDER": ("llmProvider", str),
    "HINDSIGHT_LLM_MODEL": ("llmModel", str),
    "HINDSIGHT_DEBUG": ("debug", bool),
}


def _cast_env(value: str, typ):
    """Cast environment variable string to target type. Returns None on failure."""
    try:
        if typ is bool:
            return value.lower() in ("true", "1", "yes")
        if typ is int:
            return int(value)
        return value
    except (ValueError, AttributeError):
        return None


def _load_settings_file(path: str, config: dict) -> None:
    """Merge a settings.json file into config in-place. Silently skips if missing."""
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            file_config = json.load(f)
        config.update({k: v for k, v in file_config.items() if v is not None})
    except (json.JSONDecodeError, OSError) as e:
        debug_log(config, f"Failed to load {path}: {e}")


def load_config() -> dict:
    """Load plugin configuration from settings.json + env overrides.

    Loading order (later entries win):
      1. Built-in defaults
      2. Plugin default settings.json  (CURSOR_PLUGIN_ROOT/settings.json)
      3. User config                   (~/.hindsight/cursor.json)
      4. Environment variable overrides
    """
    config = dict(DEFAULTS)

    # 1. Plugin default settings.json
    plugin_root = os.environ.get("CURSOR_PLUGIN_ROOT", "")
    if not plugin_root:
        plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _load_settings_file(os.path.join(plugin_root, "settings.json"), config)

    # 2. User config
    user_config_path = os.path.join(os.path.expanduser("~"), ".hindsight", "cursor.json")
    _load_settings_file(user_config_path, config)

    # Apply environment variable overrides
    for env_name, (key, typ) in ENV_OVERRIDES.items():
        val = os.environ.get(env_name)
        if val is not None:
            cast_val = _cast_env(val, typ)
            if cast_val is not None:
                config[key] = cast_val

    return config


def debug_log(config: dict, *args):
    """Log to stderr if debug mode is enabled."""
    if config.get("debug"):
        print("[Hindsight]", *args, file=sys.stderr)
