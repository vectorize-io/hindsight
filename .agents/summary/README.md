# Hindsight Codebase Summary — Documentation Index

**Generated**: 2026-06-27  
**Analysis Scope**: Full codebase (3,708 files, 465K LOC)  
**Status**: Complete — Raw analysis without modifications

---

## 📚 Documentation Files

### 1. **codebase_info.md**
- **Purpose**: Codebase metrics & subsystem overview
- **Content**:
  - File/LOC metrics
  - Language support matrix
  - 10 core subsystems (API, workers, UI, clients, integrations)
  - Architecture patterns & deployment options
  - File organization
  - Recent development notes

### 2. **AGENTS_GENERATED.md** (Root level)
- **Purpose**: AI agent navigation guide for Hindsight codebase
- **Content**:
  - 30-second overview
  - Directory map & component navigation
  - Key interfaces & data flows (retain/recall/reflect)
  - Configuration & environment variables
  - Operational patterns & conventions
  - Common development tasks
  - **Custom Instructions section** (preserved for future agent runs)
- **Audience**: AI coding assistants, developers, integrators

### 3. **DELTA_ANALYSIS.md**
- **Purpose**: Compare generated analysis vs. existing Memlord memories
- **Content**:
  - Alignment analysis (what's confirmed)
  - 15 new insights not in existing AGENTS.md
  - Discrepancies & clarifications
  - Documentation gaps summary
  - Recommendations for operational continuity

---

## 🔍 Key Findings

### ✅ Existing Operational Memory Confirmed
- Split Ollama configuration (11434/11435)
- Critical LLM config error (HINDSIGHT_API_LLM_BASE_URL)
- Docker Desktop 4.79.0 Electron bug
- MCP dual architecture (Memlord + Hindsight)
- Worker slot management
- Control plane as optional viewer

### 🆕 New Technical Insights
1. Fact extraction pipeline (2,675 LOC, includes Chinese temporal parsing)
2. Consolidation deduplication strategy (atomic hashing + delta reconciliation)
3. 4-parallel search strategies with RRF merging
4. 10+ cross-encoder reranking providers
5. 10+ embedding providers with prefix semantics
6. 12+ LLM provider ecosystem
7. PostgreSQL + Oracle database abstraction
8. Mental model refresh triggers (scheduled, post-consolidation, on-demand)
9. Directive priority ranking system
10. Observation scopes with per-scope limits & budgets
11. Memory defense blocking policy system
12. Multi-tenant extension architecture
13. Rust CLI implementation (2,122 LOC)
14. TypeScript client features
15. Comprehensive integration ecosystem (50+)

### ⚠️ Clarifications
- Control plane port: 9998 (dev) vs. embedded/standalone
- Production deployment: Uses Cloudflare tunnel, HMR cross-origin workaround
- Worker scaling: 40 slots optimal for M1 Max; configurable per operation type

---

## 🎯 How to Use This Documentation

### For AI Agents (Claude, Cursor, Kiro, etc.)
1. **Start**: Read `AGENTS_GENERATED.md` (Quick Start section)
2. **Navigate**: Use "Directory Map" to find relevant subsystem
3. **Deep Dive**: Consult `codebase_info.md` for metrics & architecture patterns
4. **Compare**: Check `DELTA_ANALYSIS.md` for gaps vs. operational memory

### For Human Developers
1. **Overview**: Start with `codebase_info.md` for system architecture
2. **Navigation**: Use `AGENTS_GENERATED.md` directory map
3. **Implementation**: Follow "Common Development Tasks" section
4. **Context**: Cross-reference with original `AGENTS.md` (in repo root) for operational procedures

### For Integration Work
1. Read integration overview in `AGENTS_GENERATED.md` (50+ plugins listed)
2. Check `codebase_info.md` "Integrations" section
3. Consult `hindsight-integrations/` directory for specific framework

---

## 📋 Metadata

| Aspect | Details |
|--------|---------|
| **Codebase Path** | `/Users/oliververmeulen/hindsight` |
| **Total Files** | 3,708 |
| **Prioritized Files** | 1,845 |
| **Lines of Code** | 465,166 |
| **Languages** | Python (core), TypeScript (UI/clients), Rust (CLI), Go (SDK) |
| **Core Package** | `hindsight-api-slim` |
| **API Port** | 8888 (REST) + 8888/mcp (Model Context Protocol) |
| **Control Plane** | 9998 (dev), embedded in standalone |
| **Worker Pool** | Configurable, slot-based capacity |
| **Databases** | PostgreSQL (default), Oracle (enterprise) |
| **Integrations** | 50+ (IDEs, frameworks, LLM wrappers, no-code) |

---

## 🔗 Related Documentation

**In Repository Root**:
- `AGENTS.md` — Existing operational memory & MCP setup details
- `CLAUDE.md` — Coding conventions & project structure
- `ARCHITECTURE_WHY.md` — Design decisions & performance rationale
- `QUICK_START.md` — End-user getting-started guide
- `HINDSIGHT_ENV_VARS_COMPLETE.md` — Full environment variable reference
- `OLLAMA_SPLIT_SETUP.md` — Split Ollama configuration guide
- `MONITORING_GUIDE.md` — Grafana dashboards, OTEL tracing, logging

**In Scripts**:
- `scripts/dev/README.md` — Development scripts overview
- `scripts/dev/start-all.sh` — Unified startup (API, workers, monitoring)
- `scripts/dev/status.sh` — Health check & monitoring

---

## 📝 Generation Notes

- **Analysis Method**: Raw codebase scanning (no runtime execution)
- **Format**: Markdown with tables, code blocks, Mermaid diagrams
- **Consolidation**: Analyzed across 10 major subsystems
- **Accuracy**: Metrics from file counting; interfaces from code inspection
- **Coverage**: Full polyglot stack (Python, TypeScript, Rust, Go)

---

## ✨ Next Steps

1. **Preserve Original AGENTS.md** — Contains valuable session context
2. **Link to Generated Docs** — Cross-reference technical insights
3. **Update Custom Instructions** — Add new findings to preserved section
4. **Run Codebase-Summary Periodically** — Catch new subsystems, integrations, providers

---

Generated for AI agents and development teams working on Hindsight.

For questions about this analysis, consult the original repository documentation or run the codebase-summary SOP again.
