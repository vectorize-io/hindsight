# Deepwork: System Stabilization

## Current Understanding
Complete service topology audit done (22 services across 4 environments). Key issues identified:
- P0-1: Worker startup broken (pipx binary outdated) - FIXED (start-worker.sh now uses .venv)
- P0-2: Workers stopped (died from LM Studio timeout floods)
- P0-3: Duplicate CP instances fixed (embed intentional, dev running)
- LM Studio now responsive (tested, 4s response)
- 1204 pending consolidations on opencode bank
- 64 failed operations (APIConnectionError, APITimeoutError from LM Studio)

## Config State
- LLM: LM Studio (192.168.1.144:1234, gemma-4-e4b, timeout=300s)
- Embeddings: litellm-sdk -> Ollama :11434 (nomic-embed-text)
- Retain concurrency: 1
- API on :8888 (main), :8899 (brew-installed, used by embed CP)

## Phase 0: Start Workers & Verify Processing
- Start 1 worker to re-process opencode consolidation
- Verify no LM Studio saturation
- Then scale to 2 workers

## Phase 1: Cockpit Panel (P1-1 to P1-5)
- Live Trace Stream
- Engine Load Visualization
- Service Health Table
- Fusion Strategy Mix Bar
- Metric Tiles with Sparklines

## Phase 2: Govern Panel (P2-1 to P2-4)
- Fusion Strategy Panel
- Adapter Registry Table
- Scope->Engine Routing Map
- Guardrails & Policy Toggles

## Phase 3: Memories Panel (P3-1 to P3-4)
- Unified Semantic Search
- Sector Distribution Panel
- Memory Records Table
- Record Detail Modal

## Phase 4+: Lower Priority Fixes
- P4-1: OTLP trace export 404
- P4-2: Langfuse read-only auth
- P4-3: Duplicate React keys in CP
- P4-4: FreeLLM OpenRouter 404
- P5-1: Memlord 401 from external clients
- P5-2: TEI batch limit awareness
- P5-3: Dark terminal theme
- P5-4: Build missing API routes
