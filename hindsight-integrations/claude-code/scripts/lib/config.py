"""Configuration management for Hindsight plugin.

Loads settings from settings.json (plugin defaults) merged with environment
variable overrides. Full config schema matching Openclaw's 30+ options.
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
    "retainContext": "claude-code",
    "retainTags": [],
    "retainMetadata": {},
    "recallAdditionalBanks": [],
    # Connection
    "hindsightApiUrl": None,
    "hindsightApiToken": None,
    "apiPort": 9077,
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
    "agentName": "claude-code",
    "resolveWorktrees": True,
    "directoryBankMap": {},
    # LLM (for daemon mode)
    "llmProvider": None,
    "llmModel": None,
    "llmApiKeyEnv": None,
    # Misc
    "profile": None,
    "enableKnowledgeTools": True,
    "debug": False,
}

# Named presets that tune grouped defaults for a usage pattern. A preset
# overrides built-in defaults and the plugin's shipped settings.json
# (vendor defaults), but never a key the user set explicitly in
# ~/.hindsight/claude-code.json or via environment variable.
PROFILE_PRESETS = {
    # Coding sessions (Claude Code's primary use). Tool calls carry the
    # session's real substance, so they are retained; banks isolate per
    # project so recall from one repo doesn't surface another repo's
    # context; recall budget is lowered because coding turns are far more
    # frequent than chat turns and recall latency is paid on every prompt.
    "coding": {
        "retainToolCalls": True,
        "dynamicBankId": True,
        "dynamicBankGranularity": ["agent", "project"],
        "recallBudget": "low",
        "bankMission": (
            "You are a coding assistant with long-term memory of this "
            "project's engineering history: decisions, bug fixes, "
            "conventions, and workflows."
        ),
        "retainMission": (
            "Extract durable engineering knowledge: technical decisions and "
            "their rationale, bug root causes and their fixes, architecture "
            "and API constraints, commands that worked for building/testing/"
            "running, code style and workflow preferences, and file- or "
            "module-specific gotchas. Ignore greetings, routine tool output, "
            "and transient operational chatter."
        ),
    },
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
    "HINDSIGHT_PROFILE": ("profile", str),
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
            if isinstance(parsed, (dict, list)):
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
        config.update({k: v for k, v in file_config.items() if v is not None})
    except (json.JSONDecodeError, OSError) as e:
        debug_log(config, f"Failed to load {path}: {e}")


def load_config() -> dict:
    """Load plugin configuration from settings.json + env overrides.

    Loading order (later entries win):
      1. Built-in defaults
      2. Plugin default settings.json  (CLAUDE_PLUGIN_ROOT/settings.json)
      3. User config                   (~/.hindsight/claude-code.json)
      4. Environment variable overrides

    ~/.hindsight/claude-code.json is the recommended place to configure the
    plugin — same convention as ~/.openclaw/openclaw.json. It is stable across
    plugin updates and marketplace changes.
    """
    config = dict(DEFAULTS)

    # 1. Plugin default settings.json (ships with the plugin, version-specific path)
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if not plugin_root:
        plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _load_settings_file(os.path.join(plugin_root, "settings.json"), config)

    # 2. User config — stable, version-independent, matches openclaw convention
    user_config_path = os.path.join(os.path.expanduser("~"), ".hindsight", "claude-code.json")
    user_config = {}
    _load_settings_file(user_config_path, user_config)
    config.update(user_config)

    # 3. Environment variable overrides
    env_set_keys = set()
    for env_name, (key, typ) in ENV_OVERRIDES.items():
        val = os.environ.get(env_name)
        if val is not None:
            cast_val = _cast_env(val, typ)
            if cast_val is not None:
                config[key] = cast_val
                env_set_keys.add(key)

    # 4. Profile preset — applies profile-tuned values for keys the user did
    #    not set explicitly (user config file or env var). The plugin's
    #    shipped settings.json counts as vendor defaults, which a profile
    #    intentionally overrides.
    profile = config.get("profile")
    if profile:
        preset = PROFILE_PRESETS.get(profile)
        if preset is None:
            print(
                f"[Hindsight] Unknown profile '{profile}' — valid: {', '.join(sorted(PROFILE_PRESETS))}",
                file=sys.stderr,
            )
        else:
            explicit_keys = set(user_config) | env_set_keys
            for key, value in preset.items():
                if key not in explicit_keys:
                    config[key] = value

    return config


def debug_log(config: dict, *args):
    """Log to stderr if debug mode is enabled."""
    if config.get("debug"):
        print("[Hindsight]", *args, file=sys.stderr)
