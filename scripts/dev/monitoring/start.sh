#!/bin/bash
set -e

# CollabMind Dev Observability Stack
# Start engines that feed the Central API: traces, LLM observability, AI evaluation.
# The Operator Panel consumes these via the Central API — no direct operator access.
#
# Profiles:
#   core      — Grafana LGTM (metrics, logs, traces)       [default]
#   full      — core + Jaeger + Langfuse V2                 [recommended]
#   llm-obs   — full + Phoenix (AI evaluation)              [full AI stack]
#   all       — everything including heavy services          [resource heavy]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
API_PORT="${API_PORT:-8888}"

PROFILE="${1:-core}"

cd "$SCRIPT_DIR"

echo ""
echo "═══ CollabMind Dev Observability Stack ═══"
echo ""

case "$PROFILE" in
  core)
    echo "Profile: core — Grafana LGTM only"
    echo "  Ports: :3000 (Grafana), :4317 (OTLP gRPC), :4318 (OTLP HTTP)"
    echo ""
    echo "  Add engines:  start.sh full"
    echo "  Full AI obs:  start.sh llm-obs"
    echo ""
    docker compose up
    ;;
  full)
    echo "Profile: full — core + Jaeger + Langfuse V2"
    echo "  :3000  Grafana LGTM  (metrics, logs, traces)"
    echo "  :3002  CollabMind LLM Observability"
    echo "  :4317  OTLP gRPC"
    echo "  :4318  OTLP HTTP"
    echo "  :16686 Trace Explorer"
    echo "  :14317 OTLP gRPC (traces)"
    echo "  :14318 OTLP HTTP (traces)"
    echo ""
    echo "  Add Phoenix:  start.sh llm-obs"
    echo ""
    docker compose --profile full up
    ;;
  llm-obs)
    echo "Profile: llm-obs — full + Phoenix AI Evaluation Lab"
    echo "  :3000  Grafana LGTM"
    echo "  :3002  CollabMind LLM Observability"
    echo "  :6006  AI Evaluation Lab"
    echo "  :4317  OTLP gRPC"
    echo "  :4318  OTLP HTTP"
    echo "  :16686 Trace Explorer"
    echo "  :14317 OTLP gRPC (traces)"
    echo "  :14318 OTLP HTTP (traces)"
    echo ""
    echo "  Everything:  start.sh all"
    echo ""
    docker compose --profile llm-obs up
    ;;
  all)
    echo "Profile: all — full observability stack (may be resource heavy)"
    docker compose --profile all up
    ;;
  *)
    echo "Usage: $0 [profile]"
    echo ""
    echo "Profiles:"
    echo "  core       Grafana LGTM only (default)"
    echo "  full       core + Jaeger + Langfuse V2"
    echo "  llm-obs    full + Phoenix"
    echo "  all        everything (heavy)"
    echo ""
    echo "Examples:"
    echo "  $0                # core only"
    echo "  $0 full           # recommended dev stack"
    echo "  $0 llm-obs       # full AI observability"
    echo ""
    exit 1
    ;;
esac
