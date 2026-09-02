#!/usr/bin/env bash
#
# Compatibility gate: hermes-agent @ main + the Hindsight stack in THIS checkout.
#
# Why this exists (see #3251):
#
# Hermes exact-pins every direct dependency (`==X.Y.Z`) as a deliberate
# supply-chain policy — they will not loosen a pin to accommodate us. Hindsight
# is installed *into Hermes' own venv* by `hermes memory setup` in
# `local_embedded` mode (the wizard installs `hindsight-all`), so any version
# range we declare that excludes one of their pins makes the pair impossible to
# co-install. #3251 was exactly that: our `cryptography>=48.0.1` / `pillow>=12.3.0`
# floors against their `==46.0.7` / `==12.2.0`, leaving `pip check` permanently
# broken for every Hermes user on embedded memory.
#
# That class of breakage is structural and recurring — both sides bump on their
# own schedule — so it needs a build gate, not a one-off fix. This script tracks
# Hermes' *main* branch rather than a PyPI release so the conflict surfaces here
# while it is still cheap to fix, instead of in a released Hermes.
#
# What it verifies, in order of increasing depth:
#   1. Resolution   — Hermes @ main and this checkout's Hindsight co-resolve.
#   2. pip check    — the resulting environment is internally consistent.
#   3. Wiring       — `hermes memory status` sees Hindsight in local_embedded mode.
#   4. Runtime      — the modules Hermes imports for embedded memory import cleanly.
#   5. Embedded     — the daemon actually boots under Hermes' pinned dependency
#                     set and serves bank operations.
#
# Step 5 boots a real embedded daemon (pg0 + local embedding models), which is
# the only way to prove the co-installed dependency set can actually *run*
# Hindsight rather than merely import it. The daemon refuses to construct a
# MemoryEngine without an LLM key, so it boots on a placeholder one — nothing
# calls the LLM during startup or bank operations.
#
# A retain/recall round-trip does need a working LLM, so that part runs only
# when HERMES_COMPAT_LLM_API_KEY is set (with HERMES_COMPAT_LLM_PROVIDER /
# _MODEL selecting the backend). Without a key all five steps still run — only
# the round-trip is skipped, and the script says so.
#
# Usage: scripts/test-hermes-compat.sh [--keep] [--hermes-ref BRANCH_OR_TAG]
#
#   --keep         leave the temporary work dir in place for inspection
#   --hermes-ref   Hermes branch or tag to test against (default: main). A bare
#                  commit SHA will not work — the clone is --depth 1.

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

# pg0's initdb refuses to run as root by design, and Hermes' own plugin bails
# out of local_embedded mode when it detects euid 0. Fail loudly here rather
# than let step 5 die deep inside the daemon with an opaque error.
if [ "$(id -u)" = "0" ]; then
    echo "ERROR: this script boots an embedded PostgreSQL and cannot run as root." >&2
    echo "       Run it as an unprivileged user (GitHub's ubuntu runners already are)." >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
cleanup() {
    local status=$?
    # Step 5 runs from inside the work dir; step out before removing it.
    cd / || true
    # Best-effort daemon stop so a failed run doesn't strand a pg0 instance
    # holding the profile's data directory.
    if [ -n "${VENV_PY:-}" ] && [ -x "${VENV_PY:-}" ]; then
        # Failures are reported, not swallowed: a silent `except: pass` here
        # would turn a typo in the import into a teardown that never runs and
        # leaves the daemon up with nobody the wiser.
        "$VENV_PY" - <<'PYSTOP' || echo "  (daemon stop failed; it may still be running)" >&2
import os
import sys

try:
    from hindsight_embed.daemon_embed_manager import DaemonEmbedManager
except ImportError as exc:  # install step failed before the venv was usable
    sys.exit(f"cleanup: {exc}")

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

# Hermes state goes in a throwaway dir via its documented override.
#
# Hindsight state is isolated by *profile* rather than by redirecting HOME: the
# embedded daemon keys both its profile env (~/.hindsight/profiles/<profile>.env)
# and its pg0 data directory (~/.pg0/instances/hindsight-embed-<profile>) on the
# profile name, so a dedicated "hermes-ci" profile cannot collide with a
# developer's real "hermes" profile. Redirecting HOME would isolate those too,
# but it would also hide the uv and HuggingFace caches from the runner and make
# CI re-download the whole local-ml stack (torch et al.) on every run.
export HERMES_HOME="$WORKDIR/hermes-home"
export HERMES_COMPAT_PROFILE="hermes-ci"
mkdir -p "$HERMES_HOME"

VENV="$WORKDIR/venv"
VENV_PY="$VENV/bin/python"

echo "=== Hermes ↔ Hindsight compatibility test ==="
echo "  Hermes ref:   $HERMES_REF"
echo "  Hindsight:    $REPO_ROOT (working tree)"
echo "  Work dir:     $WORKDIR"
echo ""

# ---------------------------------------------------------------------------
# 1. Resolution — the #3251 gate
# ---------------------------------------------------------------------------
# Everything goes into ONE resolution on purpose. Installing Hermes first and
# Hindsight second would let the second install silently downgrade a Hermes pin
# and report success; resolved together, an unsatisfiable pin is a hard error.
#
# The Hindsight distributions are built into wheels from this checkout and
# installed from those, so the gate tests the branch under review.
#
# Wheels, NOT `file://` directory requirements: uv installs a workspace member
# given as a directory in a way that leaves `hindsight_embed.__file__` pointing
# back at the source tree. The embedded daemon manager keys dev-mode detection on
# that path (`__file__/../../../hindsight-api-slim`), so a directory install makes
# it launch the API via `uv run --project <repo>/hindsight-api-slim` — i.e. out of
# the *monorepo's* venv, with the monorepo's dependency set and .env, completely
# bypassing the Hermes venv this script exists to test. Wheels land in
# site-packages, dev mode stays off, and step 5 exercises the real thing. The
# assertion in step 5 keeps it that way.
#
# Hermes must be a *source clone + editable* install: its build backend raises
# "Building wheels or sdists for hermes-agent is not supported" on purpose, so a
# plain `git+https://...` requirement fails at the wheel-build step. Editable is
# the path their own docs point developers at. It also puts the repo's
# `plugins/` tree on sys.path, which is where the Hindsight memory plugin lives.
echo "--- [1/5] Resolving hermes-agent@$HERMES_REF + local Hindsight stack ---"
HERMES_SRC="$WORKDIR/hermes-agent"
# Use the runner token when available to avoid the low unauthenticated limit on
# shared CI runner IPs. Environment-based Git config is ephemeral: it is not
# visible in the command line and is not persisted in the cloned repository.
# stdout is silenced but stderr is not: a clone that fails on a network blip
# should say why rather than surface as an opaque failure in the next command.
clone_ok=0
for attempt in 1 2 3; do
    if [ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]; then
        token="${GITHUB_TOKEN:-${GH_TOKEN}}"
        GIT_CONFIG_COUNT=1 \
        GIT_CONFIG_KEY_0="http.https://github.com/.extraheader" \
        GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$token" | base64 | tr -d '\n')" \
        git clone --depth 1 --branch "$HERMES_REF" \
            https://github.com/NousResearch/hermes-agent.git "$HERMES_SRC" >/dev/null && clone_ok=1
    else
        git clone --depth 1 --branch "$HERMES_REF" \
            https://github.com/NousResearch/hermes-agent.git "$HERMES_SRC" >/dev/null && clone_ok=1
    fi
    [ "$clone_ok" -eq 1 ] && break
    rm -rf "$HERMES_SRC"
    if [ "$attempt" -lt 3 ]; then
        delay=$((5 * 3 ** (attempt - 1)))
        echo "    clone failed (attempt $attempt/3); retrying in ${delay}s..." >&2
        sleep "$delay"
    fi
done
if [ "$clone_ok" -ne 1 ]; then
    echo "ERROR: could not clone hermes-agent after 3 attempts." >&2
    exit 1
fi
echo "    hermes-agent @ $(git -C "$HERMES_SRC" rev-parse --short HEAD)"

# Pinned rather than taken from .python-version: Hermes caps itself at
# `>=3.11,<3.14` (their Rust-backed transitives have no cp314 wheels), so this
# venv has to sit inside both projects' windows. 3.12 does; the repo default
# would silently stop doing so the day Hindsight moves to 3.14.
uv venv --python 3.12 "$VENV" >/dev/null

DIST="$WORKDIR/dist"
for project in hindsight-api-slim hindsight-clients/python hindsight-embed hindsight-all; do
    uv build --wheel -q -o "$DIST" "$REPO_ROOT/$project"
done
echo "    built $(find "$DIST" -name '*.whl' | wc -l | tr -d ' ') wheels from the working tree"

# hindsight-all depends on hindsight-api-slim[all], so the extras come along
# without naming them here; the sibling wheels satisfy its exact-version pins.
# shellcheck disable=SC2046 # deliberate word splitting over the wheel list
uv pip install --python "$VENV_PY" \
    -e "$HERMES_SRC" \
    $(find "$DIST" -name '*.whl')

echo "✓ Resolved and installed together"
echo ""

# ---------------------------------------------------------------------------
# 2. pip check — the exact symptom users reported in #3251
# ---------------------------------------------------------------------------
echo "--- [2/5] Verifying environment consistency ---"
uv pip check --python "$VENV_PY"

# Surface the versions of the packages this has historically broken on, so a
# CI log shows *which* side won each contested pin without a re-run.
"$VENV_PY" - <<'PYVER'
from importlib.metadata import version, PackageNotFoundError

for name in ("hermes-agent", "hindsight-api-slim", "hindsight-embed", "hindsight-client",
             "cryptography", "pillow", "protobuf", "opentelemetry-sdk", "pydantic", "openai"):
    try:
        print(f"    {name:22} {version(name)}")
    except PackageNotFoundError:
        print(f"    {name:22} (not installed)")
PYVER
echo "✓ Environment is internally consistent"
echo ""

# ---------------------------------------------------------------------------
# 3. Wiring — configure embedded mode the way `hermes memory setup` does
# ---------------------------------------------------------------------------
# `hermes memory setup` is a curses wizard and cannot be driven headlessly, so
# we write the same two artifacts it writes: the provider selection in
# config.yaml and the Hindsight provider config under $HERMES_HOME. Keep this in
# sync with plugins/memory/hindsight/__init__.py::post_setup if Hermes changes
# the shape.
#
# memory_enabled/user_profile_enabled are the flags our docs tell users to set
# so Hindsight is the only memory path (they silence the built-in MEMORY.md and
# USER.md stores). They are set here so the status assertion below covers the
# configuration we actually document: turning the flat-file stores off must not
# take the Hindsight provider down with it.
echo "--- [3/5] Configuring Hermes for local_embedded Hindsight ---"
mkdir -p "$HERMES_HOME/hindsight"
cat > "$HERMES_HOME/config.yaml" <<EOF
memory:
  provider: hindsight
  memory_enabled: false
  user_profile_enabled: false
EOF
cat > "$HERMES_HOME/hindsight/config.json" <<EOF
{
  "mode": "local_embedded",
  "profile": "$HERMES_COMPAT_PROFILE",
  "bank_id": "hermes-compat-ci",
  "llm_provider": "${HERMES_COMPAT_LLM_PROVIDER:-openai}",
  "llm_model": "${HERMES_COMPAT_LLM_MODEL:-gpt-4o-mini}",
  "llm_api_key": "${HERMES_COMPAT_LLM_API_KEY:-}",
  "idle_timeout": 0
}
EOF

# Hermes' own CLI must agree that Hindsight is the active provider and that its
# plugin loaded — this catches plugin/config-shape drift that a pure import
# check would sail straight past.
#
# Match the specific "Provider:" and "Status:" lines, not a bare "hindsight":
# the command also prints every *installed* plugin, so a loose grep would pass
# even with Hindsight inactive. Note this command reports the provider, not the
# mode (it renders `memory.<provider>` from config.yaml, while the mode lives in
# the plugin's own config.json) — step 5 is what proves local_embedded is live,
# because only that mode starts a daemon.
STATUS_OUT="$("$VENV/bin/hermes" memory status 2>&1)" || {
    echo "$STATUS_OUT"
    echo "ERROR: 'hermes memory status' failed" >&2
    exit 1
}
echo "$STATUS_OUT" | sed 's/^/    /'
echo "$STATUS_OUT" | grep -qE "^ *Provider: +hindsight *$" || {
    echo "ERROR: 'hermes memory status' does not report hindsight as the active provider" >&2
    exit 1
}
echo "$STATUS_OUT" | grep -qE "^ *Status: +available" || {
    echo "ERROR: Hermes reports the Hindsight plugin as unavailable" >&2
    exit 1
}
echo "✓ Hermes reports Hindsight as the active, available memory provider"
echo ""

# ---------------------------------------------------------------------------
# 4. Runtime probe
# ---------------------------------------------------------------------------
# Mirrors plugins/memory/hindsight/__init__.py::_check_local_runtime — the gate
# Hermes itself runs before creating an embedded client. We import the same
# three modules directly rather than calling their private helper so this does
# not break when Hermes refactors internals. sentence_transformers is in the
# list because the daemon computes embeddings through it: `hindsight` and
# `hindsight_embed` can import fine while that stack is broken.
echo "--- [4/5] Probing the embedded runtime imports ---"
"$VENV_PY" - <<'PYPROBE'
import importlib

for mod in ("hindsight", "hindsight_embed.daemon_embed_manager", "sentence_transformers"):
    importlib.import_module(mod)
    print(f"    ✓ import {mod}")

from hindsight import HindsightEmbedded  # noqa: F401
print("    ✓ from hindsight import HindsightEmbedded")
PYPROBE
echo "✓ Embedded runtime imports cleanly under Hermes' pinned dependencies"
echo ""

# ---------------------------------------------------------------------------
# 5. Embedded daemon
# ---------------------------------------------------------------------------
# Constructs the client with the same kwargs Hermes passes in _get_client(),
# boots the daemon, and exercises bank operations. Retain/recall need a real
# LLM, so they are conditional; the daemon boot and bank round-trip are not,
# which keeps this step meaningful on runs without credentials (fork PRs).
#
# Run from the work dir, not the repo: hindsight_api's config force-loads a .env
# discovered from the working directory, and a developer's repo .env would
# otherwise hand the daemon credentials that CI does not have — making this pass
# locally for a reason that does not exist on a runner.
cd "$WORKDIR"
echo "--- [5/5] Booting the embedded daemon ---"
"$VENV_PY" - <<'PYEMBED'
import os
import sys
from pathlib import Path

from hindsight import HindsightEmbedded
from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

# Guard the whole point of this step: the daemon must run from the venv Hermes
# shares, not from the monorepo. Dev-mode detection silently redirects it to
# `uv run --project <repo>/hindsight-api-slim` whenever hindsight_embed resolves
# to the source tree, which would test the monorepo's dependency set instead.
api_cmd = DaemonEmbedManager()._find_api_command("0.0.0")
venv_root = str(Path(sys.prefix).resolve())
assert str(Path(api_cmd[0]).resolve()).startswith(venv_root), (
    f"daemon would launch via {api_cmd!r}, outside the test venv ({venv_root}). "
    "That runs Hindsight from the monorepo, not from Hermes' environment, so this "
    "step would prove nothing. Install Hindsight from built wheels, not directories."
)
print(f"    ✓ daemon binary resolves inside the test venv ({Path(api_cmd[0]).name})")

profile = os.environ["HERMES_COMPAT_PROFILE"]
api_key = os.environ.get("HERMES_COMPAT_LLM_API_KEY", "")
provider = os.environ.get("HERMES_COMPAT_LLM_PROVIDER", "openai")
model = os.environ.get("HERMES_COMPAT_LLM_MODEL", "gpt-4o-mini")

# MemoryEngine refuses to construct without an LLM key, so the daemon cannot
# boot at all without one. Booting is what proves the co-installed dependency
# set can run Hindsight, and that is worth having on credential-less runs, so
# feed a placeholder: nothing calls the LLM during startup or bank operations.
# The retain/recall round-trip below stays gated on a real key.
boot_key = api_key or "hermes-compat-ci-placeholder-not-a-real-key"

# Hermes maps these two onto the OpenAI wire format before handing them to the
# daemon; mirror that so the profile env we produce matches its shape.
daemon_provider = "openai" if provider in {"openai_compatible", "openrouter"} else provider

client = HindsightEmbedded(
    profile=profile,
    llm_provider=daemon_provider,
    llm_api_key=boot_key,
    llm_model=model,
    idle_timeout=0,
)

bank_id = "hermes-compat-ci"
try:
    client._ensure_started()
    print("    ✓ daemon started")

    # The pg0 data directory survives between runs (it is keyed on the profile),
    # so a previously killed run can leave this bank behind. Drop it first to
    # keep the script re-runnable rather than failing on a stale bank.
    try:
        client.delete_bank(bank_id)
    except Exception:  # noqa: BLE001 - absent bank is the normal case
        pass

    client.create_bank(bank_id)
    print(f"    ✓ created bank {bank_id!r}")

    if api_key:
        client.retain(bank_id, "The user's favourite colour is blue.")
        print("    ✓ retained a memory")
        results = client.recall(bank_id, "What is the user's favourite colour?")
        # Assert the call round-trips into a usable result rather than just
        # returning; recall *quality* is not this script's business.
        assert results is not None, "recall returned None"
        print(f"    ✓ recalled ({type(results).__name__})")
    else:
        print("    ⓘ skipping retain/recall — no HERMES_COMPAT_LLM_API_KEY set")
        print("      (daemon booted on a placeholder key; bank operations verified)")
finally:
    try:
        client.delete_bank(bank_id)
    except Exception as exc:  # noqa: BLE001 - teardown must not mask a real failure
        print(f"    ⚠ bank cleanup failed: {exc}", file=sys.stderr)
    client.close(stop_daemon=True)
    print("    ✓ daemon stopped")
PYEMBED

echo ""
echo "=== PASS: hermes-agent@$HERMES_REF is compatible with this Hindsight ==="
