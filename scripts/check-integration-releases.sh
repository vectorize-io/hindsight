#!/bin/bash
set -e

cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
DIM='\033[2m'
NC='\033[0m'

ALL_INTEGRATIONS=("litellm" "pydantic-ai" "crewai" "ag2" "ai-sdk" "chat" "openclaw" "langgraph" "llamaindex" "nemoclaw" "strands" "claude-code" "codex" "autogen" "paperclip" "opencode" "cloudflare-oauth-proxy")

usage() {
    echo "Usage: $0 [integration]"
    echo ""
    echo "  integration   Optional. Name (e.g. 'crewai') or path (e.g. 'hindsight-integrations/crewai')."
    echo "                If omitted, checks all integrations."
    exit 1
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

if [ -n "$1" ]; then
    arg="${1%/}"
    arg="${arg#hindsight-integrations/}"
    found=false
    for v in "${ALL_INTEGRATIONS[@]}"; do
        [ "$v" = "$arg" ] && found=true && break
    done
    if [ "$found" = "false" ]; then
        echo -e "${RED}Unknown integration '$arg'${NC}"
        echo "Valid: ${ALL_INTEGRATIONS[*]}"
        exit 1
    fi
    INTEGRATIONS=("$arg")
else
    INTEGRATIONS=("${ALL_INTEGRATIONS[@]}")
fi

git fetch --tags --quiet 2>/dev/null || true

check_one() {
    local name="$1"
    local dir="hindsight-integrations/$name"

    if [ ! -d "$dir" ]; then
        echo -e "${RED}[MISSING]${NC} $name (no directory)"
        return
    fi

    local last_tag
    last_tag=$(git tag --list "integrations/$name/v*" --sort=-v:refname | head -1)

    if [ -z "$last_tag" ]; then
        echo -e "${YELLOW}[NEVER RELEASED]${NC} $name"
        echo -e "  ${DIM}all commits touching $dir:${NC}"
        git log --graph --decorate --pretty=format:'%C(yellow)%h%Creset %C(cyan)(%ar)%Creset %C(green)<%an>%Creset %s%C(auto)%d%Creset' -- "$dir" | sed 's/^/    /'
        echo ""
        echo ""
        return
    fi

    local version="${last_tag#integrations/$name/v}"
    local count
    count=$(git log --oneline "$last_tag"..HEAD -- "$dir" | wc -l | tr -d ' ')

    if [ "$count" = "0" ]; then
        echo -e "${GREEN}[OK]${NC} $name v$version (up to date with $last_tag)"
        return
    fi

    echo -e "${BLUE}[UNRELEASED]${NC} $name v$version → $count commit(s) since $last_tag"
    git log --graph --decorate --pretty=format:'%C(yellow)%h%Creset %C(cyan)(%ar)%Creset %C(green)<%an>%Creset %s%C(auto)%d%Creset' "$last_tag"..HEAD -- "$dir" | sed 's/^/    /'
    echo ""
    echo ""
}

for name in "${INTEGRATIONS[@]}"; do
    check_one "$name"
done
