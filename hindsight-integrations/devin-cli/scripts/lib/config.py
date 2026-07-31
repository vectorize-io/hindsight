"""Configuration management for the Hindsight Devin CLI plugin.

Loads settings from settings.json (plugin defaults) merged with environment
variable overrides. Port of the Claude Code plugin's config.py — same schema,
same env var names, so a shared external Hindsight server + settings can serve
both integrations if you use both.
"""

import json
import os
import sys

DEFAULTS = {
    # Recall
    "autoRecall": True,
    "recallBudget": "mid",
    "recallMaxTokens": 1024,
    "recallTypes": ["observation"],
    "recallContextTurns": 1,
    "recallMaxQueryChars": 800,
    "recallRoles": ["user", "assistant"],
    "recallTags": [],
    "recallTagsMatch": "any",
    "recallTagGroups": None,
    "recallAdditionalBankFilters": {},
    "recallMinScores": {},
    "recallPromptPreamble": (
        "Relevant memories from past conversations (prioritize recent when "
        "conflicting). Only use memories that are directly useful to continue "
        "this conversation; ignore the rest:"
    ),
    # Retain
    "autoRetain": True,
    "retainMode": "full-session",
    "retainRoles": ["user", "assistant"],
    "retainEveryNTurns": 10,
    "retainOverlapTurns": 2,
    "retainToolCalls": False,
    "retainContext": "devin-cli",
    "retainTags": [],
    "retainMetadata": {},
    "recallAdditionalBanks": [],
    # Connection
    "hindsightApiUrl": None,
    "hindsightApiToken": None,
    "apiPort": 9078,
    "daemonIdleTimeout": 0,
    "embedVersion": "latest",
    "embedPackagePath": None,
    "requestTimeoutSeconds": None,
    # Bank
    "bankId": None,
    "bankIdPrefix": "",
    "dynamicBankId": False,
    "dynamicBankGranularity": ["agent", "project"],
    "bankMission": "",
    "retainMission": None,
    "agentName": "devin-cli",
    "resolveWorktrees": True,
    "directoryBankMap": {},
    # LLM (for daemon mode)
    "llmProvider": None,
    "llmModel": None,
    "llmApiKeyEnv": None,
    # Misc
    "enableKnowledgeTools": True,
    "debug": False,
}

# Map env var names to config keys and their types. Shared with the Claude Code
# plugin intentionally: HINDSIGHT_* env vars configure whichever integration
# reads them, so a single shell profile can drive both.
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
    "HINDSIGHT_RECALL_TAGS": ("recallTags", list),
    "HINDSIGHT_RECALL_TAGS_MATCH": ("recallTagsMatch", str),
    "HINDSIGHT_RECALL_TAG_GROUPS": ("recallTagGroups", dict),
    "HINDSIGHT_RECALL_ADDITIONAL_BANK_FILTERS": ("recallAdditionalBankFilters", dict),
    "HINDSIGHT_API_PORT": ("apiPort", int),
    "HINDSIGHT_DAEMON_IDLE_TIMEOUT": ("daemonIdleTimeout", int),
    "HINDSIGHT_REQUEST_TIMEOUT_SECONDS": ("requestTimeoutSeconds", int),
    "HINDSIGHT_EMBED_VERSION": ("embedVersion", str),
    "HINDSIGHT_EMBED_PACKAGE_PATH": ("embedPackagePath", str),
    "HINDSIGHT_DYNAMIC_BANK_ID": ("dynamicBankId", bool),
    "HINDSIGHT_BANK_MISSION": ("bankMission", str),
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
        if typ is list:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return None
        if typ is dict:
            parsed = json.loads(value)
            # A dict, not "anything JSON structural". Accepting a list here
            # hands a list to a caller that will .get() on it — the exact
            # AttributeError-inside-a-hook this coercion exists to prevent.
            # The list branch above already requires a list; this is the same
            # rule, applied consistently.
            if isinstance(parsed, dict):
                return parsed
            return None
        return value
    except (ValueError, AttributeError):
        if typ is list:
            return [part.strip() for part in value.split(",") if part.strip()]
        return None


def _load_settings_file(path: str, config: dict) -> None:
    """Merge a settings.json file into config in-place. Silently skips if missing."""
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            file_config = json.load(f)
        # A settings file holding valid JSON that is not an object (a list, a bare
        # string) would raise AttributeError from .items() — not caught below, so
        # it would escape load_config() and stop every hook from running.
        if not isinstance(file_config, dict):
            debug_log(config, f"Ignoring {path}: expected a JSON object, got {type(file_config).__name__}")
            return
        config.update({k: v for k, v in file_config.items() if v is not None})
    # UnicodeDecodeError is what a settings file with invalid UTF-8 raises, and
    # it subclasses ValueError rather than OSError — so without it here a single
    # bad byte escaped load_config() and stopped every hook, which is the exact
    # opposite of the fall-back-to-defaults this function exists to provide.
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        debug_log(config, f"Failed to load {path}: {e}")


def _drop_malformed_settings(config: dict) -> None:
    """Revert any setting whose type cannot support the way it is used.

    A settings file is arbitrary user JSON, so a value can be any shape, and
    most wrong shapes are not a different behaviour but a crash: a string where
    a list belongs iterates one character at a time, a list where a dict
    belongs raises AttributeError from .get(), a string where an int belongs
    raises TypeError from arithmetic. All of that happens inside a hook, so one
    mistyped optional setting used to take recall or retain down entirely —
    a setting added to *tune* the plugin silently switching it off.

    DEFAULTS is the type table. Deriving the expected shape from it means any
    new setting is covered the moment it has a default, with no parallel list
    to drift out of sync. Defaults of None carry no type information and are
    skipped; those settings are passed through to the API rather than used
    structurally here.

    Exact types rather than isinstance: bool subclasses int, so `true` would
    otherwise satisfy an int setting and reach arithmetic as 1.
    """
    for key, default in DEFAULTS.items():
        if default is None:
            continue
        value = config.get(key)
        if value is not None and type(value) is not type(default):
            debug_log(
                config,
                f"Ignoring {key}: expected {type(default).__name__}, got {type(value).__name__}",
            )
            config[key] = default


def plugin_root() -> str:
    """Directory this plugin is installed at (wherever `devin plugins install` or
    scripts/install.py put it) — resolved from this file's own location rather
    than an env var, since Devin CLI does not set one for hook/MCP processes."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config() -> dict:
    """Load plugin configuration from settings.json + env overrides.

    Loading order (later entries win):
      1. Built-in defaults
      2. Plugin default settings.json  (<plugin_root>/settings.json)
      3. User config                   (~/.hindsight/devin-cli.json)
      4. Environment variable overrides

    ~/.hindsight/devin-cli.json is the recommended place to configure the
    plugin — stable across plugin updates, matching the Claude Code plugin's
    ~/.hindsight/claude-code.json and Openclaw's ~/.openclaw/openclaw.json.
    """
    config = dict(DEFAULTS)

    _load_settings_file(os.path.join(plugin_root(), "settings.json"), config)

    user_config_path = os.path.join(os.path.expanduser("~"), ".hindsight", "devin-cli.json")
    _load_settings_file(user_config_path, config)

    # After the files, before the env vars: the files are the only source of
    # arbitrary JSON. ENV_OVERRIDES below coerces its own values by type
    # already, so re-checking them here would reject nothing and could only
    # fight that coercion.
    _drop_malformed_settings(config)

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
