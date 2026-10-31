#!/usr/bin/env bash
#
# Compatibility gate: hermes-agent @ main + THIS checkout's Hindsight plugin,
# installed the way a Hermes user installs it.
#
# Every step goes through Hermes' own user-facing path, so the job breaks only
# where a user would break:
#
#   1. Install   — Hermes' installer (scripts/install.sh). Its package manager
#                  builds the venv from Hermes' uv.lock with Hermes' overrides
#                  and runs its own `uv pip check`. (#5974 was this script
#                  resolving Hermes by hand and skipping those overrides.)
#   2. Plugin    — `hermes plugins install hindsight`, by catalog name: clone,
#                  security scan, dependency resolution against Hermes' pins,
#                  enable as memory.provider. Then `hermes plugins validate`, the
#                  catalog admission gate.
#   3. Wiring    — configure local_embedded and check `hermes memory status`.
#   4. Upgrade   — `hermes update` to a new Hermes commit; the plugin and its
#                  dependencies must survive the venv rebuild.
#   5. Provider  — load the provider through Hermes' registry and round-trip a
#                  turn (auto-retain, auto-recall, tools). This boots the
#                  embedded daemon the way the plugin does for users: `uvx
#                  hindsight-api@<embed version>` in its own environment.
#
# The one patch: the catalog entry for `hindsight` pins a published commit of
# this repo. We rewrite it in the Hermes checkout to point at a git snapshot of
# this working tree's plugin (file:// — the catalog normally accepts https only,
# so that check is relaxed too), and commit it before installing so `hermes
# update` keeps it. Everything else is unmodified Hermes.
#
# What this does NOT test: this checkout's hindsight-api / -embed / -client. The
# plugin declares `hindsight-client` and `hindsight-embed` from PyPI (with upper
# bounds), and Hermes users get exactly those releases, so this job does too.
#
# Previously this script built every Hindsight distribution from the working
# tree and resolved them together with an editable Hermes in one `uv pip
# install` (the #3251 co-install gate, from when the setup wizard put
# hindsight-all into Hermes' venv). That stopped matching reality twice over:
# Hermes no longer installs hindsight-all (the daemon runs under uvx in its own
# environment), and a hand-rolled resolve skips Hermes' [tool.uv] overrides, so
# an upstream pin clash its own installer handles turned every PR red (#5974).
#
# The retain/recall round-trip needs a working LLM, so it runs only when
# HERMES_COMPAT_LLM_API_KEY is set (HERMES_COMPAT_LLM_PROVIDER / _MODEL pick the
# backend). Without a key every other step still runs.
#
# Usage: scripts/test-hermes-compat.sh [--keep] [--hermes-ref BRANCH_OR_TAG]
#
#   --keep         leave the temporary work dir in place for inspection
#   --hermes-ref   Hermes branch or tag to test against (default: main)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_REF="main"
KEEP_WORKDIR=0

while [ $# -gt 0 ]; do
    case "$1" in
        --keep) KEEP_WORKDIR=1; shift ;;
        --hermes-ref) HERMES_REF="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

# pg0's initdb refuses to run as root by design, and the plugin bails out of
# local_embedded mode when it detects euid 0.
if [ "$(id -u)" = "0" ]; then
    echo "ERROR: this script boots an embedded PostgreSQL and cannot run as root." >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
export HERMES_HOME="$WORKDIR/hermes-home"
export HERMES_COMPAT_PROFILE="hermes-ci"
# Hermes' installer and updater edit shell rc files and ~/.local/bin; give them
# a throwaway HOME so a local run leaves the developer's machine alone. Every
# other command keeps the real HOME, so the daemon reuses the uv / HuggingFace
# caches and its logs land where the CI failure step looks
# (~/.hindsight/profiles/hermes-ci*). The "hermes-ci" profile keeps it clear of
# a developer's real "hermes" profile.
INSTALL_HOME="$WORKDIR/home"
mkdir -p "$INSTALL_HOME"
HERMES_SRC="$WORKDIR/hermes-src"
INSTALL_DIR="$HERMES_HOME/hermes-agent"
HERMES="$INSTALL_DIR/.hermes/bin/hermes"
GIT=(git -c user.name=hermes-compat -c user.email=hermes-compat@localhost)

# The venv Hermes is running right now. Its package manager replaces the venv
# on every dependency change (plugin install, update), so always re-read it.
hermes_python() {
    python3 - "$HERMES_HOME" <<'PYVENV'
import glob, json, sys
(facts,) = glob.glob(f"{sys.argv[1]}/installs/*/facts.json")
print(json.load(open(facts))["packages"]["venv"]["environment"] + "/bin/python")
PYVENV
}

cleanup() {
    local status=$?
    cd / || true
    # Best-effort daemon stop so a failed run doesn't strand a pg0 instance.
    if [ -x "$HERMES" ]; then
        "$(hermes_python)" - <<'PYSTOP' || echo "  (daemon stop failed; it may still be running)" >&2
import os
from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

DaemonEmbedManager().stop(os.environ.get("HERMES_COMPAT_PROFILE", "hermes-ci"))
PYSTOP
    fi
    if [ "$KEEP_WORKDIR" = "1" ]; then
        echo "Work dir kept at: $WORKDIR"
    else
        rm -rf "$WORKDIR"
    fi
    exit $status
}
trap cleanup EXIT

echo "=== Hermes ↔ Hindsight compatibility test ==="
echo "  Hermes ref:   $HERMES_REF"
echo "  Hindsight:    $REPO_ROOT (working tree)"
echo "  Work dir:     $WORKDIR"
echo ""

# ---------------------------------------------------------------------------
# 0. Sources — a Hermes checkout with the catalog patched, and a plugin snapshot
# ---------------------------------------------------------------------------
echo "--- [0/5] Preparing Hermes@$HERMES_REF and a snapshot of this plugin ---"
# `git clone` gets rate-limited on shared CI runner IPs; retry with the token.
clone_ok=0
for attempt in 1 2 3; do
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="http.https://github.com/.extraheader" \
            GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 | tr -d '\n')" \
            git clone -q --depth 1 --branch "$HERMES_REF" \
            https://github.com/NousResearch/hermes-agent.git "$HERMES_SRC" && clone_ok=1
    else
        git clone -q --depth 1 --branch "$HERMES_REF" \
            https://github.com/NousResearch/hermes-agent.git "$HERMES_SRC" && clone_ok=1
    fi
    [ "$clone_ok" = "1" ] && break
    rm -rf "$HERMES_SRC"
    sleep $((attempt * 10))
done
if [ "$clone_ok" != "1" ]; then
    echo "ERROR: could not clone hermes-agent after 3 attempts." >&2
    exit 1
fi
UPSTREAM_SHA="$(git -C "$HERMES_SRC" rev-parse --short HEAD)"
echo "    hermes-agent @ $UPSTREAM_SHA"
# The installer clones from this repo, and git refuses to clone from a shallow
# one, so restart history at a single commit. --no-verify: Hermes' own hooks
# are not ours to run. The branch must be `main`: `hermes update` follows the
# main channel for a non-official origin, whatever ref was installed.
rm -rf "$HERMES_SRC/.git"
git -C "$HERMES_SRC" init -q -b main
git -C "$HERMES_SRC" add -A
"${GIT[@]}" -C "$HERMES_SRC" commit -q --no-verify -m "hermes-agent@$UPSTREAM_SHA"

# Snapshot the working tree's plugin (tracked + new files, as they are on disk)
# into its own repo, laid out like this one so the catalog subdir still applies.
PLUGIN_REPO="$WORKDIR/plugin-repo"
mkdir -p "$PLUGIN_REPO"
(cd "$REPO_ROOT" && git ls-files -z --cached --others --exclude-standard hindsight-integrations/hermes \
    | xargs -0 tar cf -) | tar xf - -C "$PLUGIN_REPO"
git -C "$PLUGIN_REPO" init -q -b main
git -C "$PLUGIN_REPO" add -A
"${GIT[@]}" -C "$PLUGIN_REPO" commit -q -m "hindsight plugin under test"
PLUGIN_SHA="$(git -C "$PLUGIN_REPO" rev-parse HEAD)"

python3 - "$HERMES_SRC" "file://$PLUGIN_REPO" "$PLUGIN_SHA" <<'PYPATCH'
import re, sys
from pathlib import Path

src, repo, sha = sys.argv[1:]

def patch(path, pattern, repl):
    p = Path(src, path)
    text, n = re.subn(pattern, repl, p.read_text())
    # Hermes moved or rewrote the code this relies on: fail here, by name.
    assert n == 1, f"{path}: expected one match for {pattern!r}, found {n}"
    p.write_text(text)

patch("plugin-catalog/hindsight.yaml", r"(?m)^repo: .*$", f"repo: {repo}")
patch("plugin-catalog/hindsight.yaml", r"(?m)^sha: .*$", f"sha: {sha}")
patch("hermes_cli/plugin_catalog.py", r'not repo\.startswith\("https://"\)',
      'not repo.startswith(("https://", "file://"))')
PYPATCH
"${GIT[@]}" -C "$HERMES_SRC" commit -q --no-verify -am "ci: point the hindsight catalog entry at the plugin under test"
echo "✓ catalog entry 'hindsight' -> plugin snapshot ${PLUGIN_SHA:0:12}"
echo ""

# ---------------------------------------------------------------------------
# 1. Install — Hermes' own installer
# ---------------------------------------------------------------------------
echo "--- [1/5] Installing Hermes with its installer ---"
HOME="$INSTALL_HOME" HERMES_REPO_URL="file://$HERMES_SRC" \
    bash "$HERMES_SRC/scripts/install.sh" --non-interactive --branch main \
    --dir "$INSTALL_DIR" --skip-browser --skip-computer-use 2>&1 | sed 's/^/    /'
[ -x "$HERMES" ] || { echo "ERROR: installer finished without $HERMES" >&2; exit 1; }
echo "✓ Hermes installed"
echo ""

# Run from the work dir: hindsight_api force-loads a .env discovered from the
# working directory, and a developer's repo .env would hand the daemon
# credentials CI does not have.
cd "$WORKDIR"

# ---------------------------------------------------------------------------
# 2. Plugin — `hermes plugins install hindsight`, by catalog name
# ---------------------------------------------------------------------------
echo "--- [2/5] Installing the Hindsight plugin from the catalog ---"
"$HERMES" plugins install hindsight --enable --yes-deps 2>&1 | sed 's/^/    /'

PLUGIN_DIR="$HERMES_HOME/plugins/hindsight"
check_plugin() {
    "$(hermes_python)" - "$PLUGIN_DIR" "$PLUGIN_SHA" <<'PYPLUGIN'
import json, sys
from importlib.metadata import version
from pathlib import Path

from plugins.memory import find_provider_dir

plugin_dir, sha = Path(sys.argv[1]), sys.argv[2]
# The install record must name our snapshot: a live catalog that won over the
# patched in-tree one would have installed the published commit instead.
record = json.loads((plugin_dir / ".hermes-catalog.json").read_text())
assert record.get("sha") == sha, f"installed {record}, not the snapshot {sha}"
resolved = find_provider_dir("hindsight")
assert resolved and Path(resolved).resolve() == plugin_dir.resolve(), (
    f"Hermes resolves 'hindsight' to {resolved}, not {plugin_dir}"
)
print(f"    ✓ provider is the plugin under test ({sha[:12]})")
for name in ("hindsight-client", "hindsight-embed", "pillow", "cryptography"):
    print(f"    {name:18} {version(name)}")
PYPLUGIN
}
check_plugin
"$HERMES" plugins validate "$PLUGIN_DIR" 2>&1 | sed 's/^/    /'
echo "✓ Plugin installed through the catalog and passes Hermes' admission checks"
echo ""

# ---------------------------------------------------------------------------
# 3. Wiring — local_embedded, the way `hermes memory setup` leaves it
# ---------------------------------------------------------------------------
# `hermes memory setup` is an interactive wizard, so write what it writes: the
# plugin's config.json. `plugins install --enable` already set memory.provider.
# memory_enabled/user_profile_enabled off is the setup our docs recommend (only
# Hindsight remembers); it must not take the provider down with it.
# A bank per run: the profile's pg0 data survives between local runs, and a bank
# still holding the last run's turn would let step 5's recall pass on old data.
echo "--- [3/5] Configuring Hermes for local_embedded Hindsight ---"
"$HERMES" config set memory.memory_enabled false >/dev/null
"$HERMES" config set memory.user_profile_enabled false >/dev/null
mkdir -p "$HERMES_HOME/hindsight"
cat > "$HERMES_HOME/hindsight/config.json" <<EOF
{
  "mode": "local_embedded",
  "profile": "$HERMES_COMPAT_PROFILE",
  "bank_id": "hermes-compat-ci-$(date +%s)",
  "llm_provider": "${HERMES_COMPAT_LLM_PROVIDER:-openai}",
  "llm_model": "${HERMES_COMPAT_LLM_MODEL:-gpt-4o-mini}",
  "llm_api_key": "${HERMES_COMPAT_LLM_API_KEY:-}",
  "idle_timeout": 0
}
EOF

# Match the "Provider:" and "Status:" lines, not a bare "hindsight": the command
# also lists every installed plugin.
check_memory_status() {
    local out
    out="$("$HERMES" memory status 2>&1)" || { echo "$out"; echo "ERROR: 'hermes memory status' failed" >&2; exit 1; }
    echo "$out" | sed 's/^/    /'
    echo "$out" | grep -qE "^ *Provider: +hindsight *$" || {
        echo "ERROR: 'hermes memory status' does not report hindsight as the active provider" >&2; exit 1; }
    echo "$out" | grep -qE "^ *Status: +available" || {
        echo "ERROR: Hermes reports the Hindsight plugin as unavailable" >&2; exit 1; }
}
check_memory_status
echo "✓ Hermes reports Hindsight as the active, available memory provider"
echo ""

# ---------------------------------------------------------------------------
# 4. Upgrade — `hermes update`
# ---------------------------------------------------------------------------
# Give the update something to pull, so it runs the full path (fetch, fast-
# forward, venv rebuild with enabled plugins) rather than "already up to date".
echo "--- [4/5] Upgrading Hermes with 'hermes update' ---"
"${GIT[@]}" -C "$HERMES_SRC" commit -q --no-verify --allow-empty -m "ci: next hermes commit"
HOME="$INSTALL_HOME" "$HERMES" update --yes --no-backup --no-gateway-restart 2>&1 | sed 's/^/    /'
[ "$(git -C "$INSTALL_DIR" rev-parse HEAD)" = "$(git -C "$HERMES_SRC" rev-parse HEAD)" ] || {
    echo "ERROR: 'hermes update' did not move the install to the new commit" >&2; exit 1; }
check_plugin
check_memory_status
echo "✓ The plugin survives 'hermes update'"
echo ""

# ---------------------------------------------------------------------------
# 5. Provider — the plugin's own surface, driven the way Hermes drives it
# ---------------------------------------------------------------------------
# Loads the provider through Hermes' registry and calls the MemoryProvider
# contract agent_init uses. The first call that needs memory boots the embedded
# daemon through the plugin's own path.
#
# Without an LLM key: boot on a placeholder key (the daemon refuses to start
# with none; nothing below calls the LLM) and recall from the empty bank, which
# proves the daemon runs. Exit 3 tells the caller the round-trip was skipped.
#
# recall_sync makes prefetch answer on this turn instead of the next one.
echo "--- [5/5] Exercising the plugin's MemoryProvider surface ---"
(set +e; "$(hermes_python)" - <<'PYPROVIDER'
import json
import os
import sys
import time
from pathlib import Path

has_key = bool(os.environ.get("HERMES_COMPAT_LLM_API_KEY"))
config_path = Path(os.environ["HERMES_HOME"]) / "hindsight" / "config.json"
config = json.loads(config_path.read_text())
config.update(recall_sync=True, retain_async=False)
if not has_key:
    config["llm_api_key"] = "hermes-compat-ci-placeholder-not-a-real-key"
config_path.write_text(json.dumps(config))

from plugins.memory import load_memory_provider

provider = load_memory_provider("hindsight")
assert provider is not None, "Hermes registry returned no 'hindsight' provider"
print(f"    ✓ loaded {type(provider).__name__} from {type(provider).__module__}")

assert provider.is_available(), f"provider reports unavailable: {provider.unavailable_reason()}"
print("    ✓ is_available()")

provider.initialize(session_id="hermes-compat-ci")
tools = {schema.get("name") for schema in provider.get_tool_schemas()}
assert {"hindsight_retain", "hindsight_recall", "hindsight_reflect"} <= tools, f"missing tools: {tools}"
print(f"    ✓ initialize() and {len(tools)} tools registered")

try:
    if not has_key:
        recalled = provider.handle_tool_call("hindsight_recall", {"query": "anything"})
        # Tools answer {"result": ...} on success and {"error": ...} on failure.
        assert "result" in json.loads(recalled), f"hindsight_recall failed: {recalled!r}"
        print("    ✓ embedded daemon booted and answered hindsight_recall")
        print("    ⓘ skipping the retain/recall round-trip — no HERMES_COMPAT_LLM_API_KEY set")
        sys.exit(3)

    # Auto-retain: what the post-turn hook calls on every exchange.
    provider.sync_turn(
        user_content="My name is Ada Lovelace and I only drink espresso.",
        assistant_content="Noted, Ada.",
        session_id="hermes-compat-ci",
    )
    print("    ✓ sync_turn() enqueued a turn")

    # Extraction is an LLM call; poll rather than sleep a fixed amount.
    deadline, injected = time.time() + 180, ""
    while time.time() < deadline:
        time.sleep(5)
        injected = provider.prefetch("What does the user drink?", session_id="hermes-compat-ci") or ""
        if "espresso" in injected.lower():
            break
    assert "espresso" in injected.lower(), (
        f"auto-recall never surfaced the retained turn within 180s; last injection: {injected!r}"
    )
    print("    ✓ prefetch() injected the retained memory")

    status = provider.recall_status()
    assert status is not None and status.count > 0, f"recall_status() reported nothing: {status}"
    print(f"    ✓ recall_status() -> {status.count} memor(ies)")

    recalled = provider.handle_tool_call("hindsight_recall", {"query": "What is the user's name?"})
    assert "ada" in recalled.lower(), f"hindsight_recall did not find the name: {recalled!r}"
    print("    ✓ hindsight_recall tool")

    provider.on_session_end([{"role": "user", "content": "bye"}])
    print("    ✓ on_session_end()")
finally:
    provider.shutdown()
    print("    ✓ shutdown()")
PYPROVIDER
) && provider_status=0 || provider_status=$?
if [ "$provider_status" = "0" ]; then
    echo "✓ The plugin round-trips a turn through Hermes' MemoryProvider contract"
elif [ "$provider_status" = "3" ]; then
    echo "✓ The plugin boots the daemon and answers recall (round-trip skipped: no LLM key)"
else
    exit "$provider_status"
fi

echo ""
echo "=== PASS: hermes-agent@$HERMES_REF ($UPSTREAM_SHA) works with this Hindsight plugin ==="
