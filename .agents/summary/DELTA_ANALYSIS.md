# Documentation Delta Analysis — Generated vs. Memlord Memories (2026-06-27)

## Summary

Comparing **generated codebase analysis** vs. **existing AGENTS.md** (context entry #3) and **conversation history**.

---

## Findings: Match vs. New Information

### ✅ ALIGNED — Generated Documentation Confirms Existing Memory

#### 1. **Split Ollama Configuration** (Confirmed)
- **Existing**: Ports 11434 (embeddings) + 11435 (LLM) documented in AGENTS.md
- **Generated**: Same port assignment confirmed in codebase analysis
- **Status**: Verified active pattern

#### 2. **Critical LLM Config Error** (Confirmed)
- **Existing**: Wrong var `HINDSIGHT_API_LLM_OLLAMA_API_BASE` → Fix: `HINDSIGHT_API_LLM_BASE_URL`
- **Generated**: Confirmed in `config.py` (3,064 LOC) — universal LLM_BASE_URL for all providers
- **Status**: Root cause verified in code

#### 3. **Docker Desktop 4.79.0 Bug** (Confirmed)
- **Existing**: Electron crash during stream cleanup (AbortError)
- **Generated**: No code-level representation (external runtime bug)
- **Status**: Known issue, living with current version

#### 4. **MCP Dual Architecture** (Confirmed)
- **Existing**: Memlord MCP (port 8005) + Hindsight MCP (port 8888/mcp)
- **Generated**: Confirmed — MCP endpoint on 8888/mcp in `api/http.py` routing
- **Status**: Dual system operational

#### 5. **Worker Pool & Slot Management** (Confirmed)
- **Existing**: Slot-based capacity, per-operation limits
- **Generated**: Confirmed in `worker/poller.py` (1,379 LOC) — slot reservation config
- **Status**: Implementation matches documentation

#### 6. **Control Plane Optional** (Confirmed)
- **Existing**: Backend APIs work independently
- **Generated**: Control plane (`hindsight-control-plane/`) is optional viewer
- **Status**: Verified architecture

---

## 🆕 NEW INSIGHTS — Information Not in Existing AGENTS.md

### 1. **Fact Extraction Pipeline Details**
- **Generated Insight**: `engine/retain/fact_extraction.py` (2,675 LOC) handles:
  - Chinese temporal period parsing (1,799 LOC separate module)
  - Verbatim extraction mode vs. verbose extraction
  - Causal relation extraction with temporal offsets
  - Entity label inference from context
- **Not Documented**: Specific extraction modes & temporal parsing complexity

### 2. **Consolidation Deduplication Strategy**
- **Generated Insight**: `consolidator.py` (2,385 LOC) implements:
  - Atomic deduplication by content hash
  - Reconciliation logic (create/update/delete deltas)
  - Scope-aware conflict resolution
  - Memory link creation (causal, temporal, semantic)
- **Not Documented**: Delta reconciliation algorithm details

### 3. **Search Strategy Parallelization**
- **Generated Insight**: `retrieval.py` runs 4 strategies in parallel:
  - Semantic (vector similarity)
  - Keyword (BM25 full-text)
  - Graph (entity/temporal/causal links)
  - Temporal (time range filtering)
- **Not Documented**: Parallel execution details & RRF merge logic

### 4. **Cross-Encoder Architecture**
- **Generated Insight**: `cross_encoder.py` (1,750 LOC) supports:
  - Cohere, Google, Jina MLX, LiteLLM SDK, FlashRank, ZeroEntropy
  - SiliconFlow, TEI (remote/local), ONNX local models
  - Timeout handling & batch processing
- **Not Documented**: Comprehensive reranking provider list

### 5. **Entity Resolution Fuzzy Matching**
- **Generated Insight**: `entity_resolver.py` (935 LOC) uses:
  - Trigram-based string matching
  - Fuzzy matching with oracle/fuzzy fallback
  - Cooccurrence pair tracking for linking
  - Entity statistics (first seen, last seen, mention count)
- **Not Documented**: Trigram + fuzzy strategy layering

### 6. **Embedding Provider Abstraction**
- **Generated Insight**: `embeddings.py` (1,706 LOC) supports:
  - OpenAI, Gemini, LiteLLM SDK/CLI, ZeroEntropy, Cohere
  - CodexOAuth, Jina MLX, TEI (remote/local), ONNX, local transformers
  - Prefix handling (encode_query vs. encode_documents)
- **Not Documented**: Full provider coverage or prefix semantics

### 7. **LLM Provider Variety**
- **Generated Insight**: 12+ LLM providers in `engine/providers/`:
  - OpenAI-compatible (1,593 LOC)
  - Anthropic, Gemini, LiteLLM, Fireworks
  - Ollama, llama.cpp, Claude Code, Codex OAuth, Nous
- **Not Documented**: Complete provider ecosystem inventory

### 8. **Database Abstraction Layer**
- **Generated Insight**: `db/` supports:
  - PostgreSQL (default) with HNSW vector indexes
  - Oracle with SQL-to-Oracle rewrites for compatibility
  - Abstract `Backend` interface for custom implementations
- **Not Documented**: Oracle enterprise support details

### 9. **Mental Model Refresh Triggers**
- **Generated Insight**: Mental models can refresh:
  - On schedule (CRON expression)
  - After consolidation completion (auto-trigger)
  - On-demand via API
  - Stale detection with override logic
- **Not Documented**: Trigger mechanism or refresh strategy

### 10. **Directive System**
- **Generated Insight**: Directives are:
  - Behavior rules stored with tags
  - Priority-ranked (higher priority considered first)
  - Active/inactive toggle
  - Included in reflect agent reasoning
- **Not Documented**: Directive priority ranking or integration

### 11. **Observation Scopes & Limits**
- **Generated Insight**: Observations can have:
  - Named scopes (custom categorization)
  - Per-scope limits on observation count
  - Per-scope token budgets for consolidation/reflect
  - Scope-aware conflict resolution
- **Not Documented**: Scope limit enforcement or per-scope budgets

### 12. **Memory Defense (Blocking Policy)**
- **Generated Insight**: `memory_defense.py` (30,399 bytes tests):
  - Webhook-based review before retain/update
  - Block decisions with violation details
  - Audit logging for blocked content
- **Not Documented**: Memory defense blocking policy

### 13. **Tenant Extension Architecture**
- **Generated Insight**: `extensions/` supports:
  - Multi-tenant with dynamic discovery
  - OAuth token verification (Supabase, custom)
  - Per-tenant schema isolation
- **Not Documented**: Tenant extension API or multi-tenancy patterns

### 14. **CLI Rust Implementation**
- **Generated Insight**: `hindsight-cli/` (2,122 LOC Rust):
  - All major commands: bank, memory, mental models, directives, webhooks
  - Interactive TUI with gradients & colors
  - Config file management
- **Not Documented**: CLI command matrix or TUI features

### 15. **TypeScript Client Features**
- **Generated Insight**: `@vectorize-io/hindsight-client` includes:
  - Async/await promise-based API
  - All REST endpoints wrapped
  - Error class with structured response
- **Not Documented**: TypeScript client specifics

---

## ⚠️ DISCREPANCIES & CLARIFICATIONS

### 1. **Control Plane Port Assignment**
- **Existing Memory**: "Port 9999 (changed from 9999 to avoid Docker Desktop conflict)" — unclear wording
- **Generated Insight**: Port 9998 confirmed in scripts/dev for dev mode
- **Clarification**: Control plane runs on 9998 (dev) or embedded in standalone

### 2. **Production Control Plane**
- **Existing Memory**: Mentions "production URL: neuron-ai-controller.collabmind.dev"
- **Generated Insight**: Production uses Cloudflare tunnel, WebSocket HMR has cross-origin issues
- **Status**: Dev mode workaround in place; permanent fix is production build

### 3. **API Health Endpoint**
- **Generated Insight**: Health checks via `/health` endpoint + operation polling
- **Not Documented**: Specific health response format

### 4. **Worker Scaling Limits**
- **Existing Memory**: "40 worker slots optimal for M1 Max 64GB"
- **Generated Insight**: Slot reservation per operation type (consolidation, retain, reflect)
- **Clarification**: 40 slots is target for current hardware; scaling is configurable

---

## 📊 SUMMARY OF DOCUMENTATION GAPS

| Category | Existing AGENTS.md | Generated Analysis | Gap |
|----------|-------------------|-------------------|-----|
| **Architecture** | High-level | Detailed with LOC | ✅ Filled |
| **Fact Extraction** | Not documented | 2,675 LOC pipeline | ✅ Filled |
| **Consolidation Algorithm** | Basic description | Full dedup/delta logic | ✅ Filled |
| **Search Strategies** | Mentioned | 4-parallel with RRF | ✅ Filled |
| **Reranking Models** | Not listed | 10+ providers | ✅ Filled |
| **Embedding Providers** | Not listed | 10+ providers | ✅ Filled |
| **LLM Providers** | Split Ollama focused | 12+ providers | ✅ Filled |
| **Mental Models** | Mentioned | Refresh triggers documented | ✅ Filled |
| **Directives** | Not documented | Priority system | ✅ Filled |
| **Tenancy** | "Multi-tenant" mentioned | Extension-based discovery | ✅ Filled |
| **CLI Details** | Not documented | Full Rust implementation | ✅ Filled |
| **Memory Defense** | Not mentioned | Blocking policy system | ✅ New System |
| **Observation Scopes** | Not mentioned | Scope limits & budgets | ✅ New System |

---

## 🔄 OPERATIONAL CONTINUITY RECOMMENDATIONS

1. **Preserve Existing Memory**
   - Original AGENTS.md contains valuable session context (MCP setup, known issues)
   - Generated analysis adds implementation details
   - Both should coexist: AGENTS.md (operational) + AGENTS_GENERATED.md (technical)

2. **Update Critical Variables Section**
   - Add `HINDSIGHT_API_LLM_BASE_URL` (correct variable name)
   - Clarify Ollama endpoints vs. other providers
   - Document embedding provider selection pattern

3. **Document New Systems**
   - Memory defense blocking policies
   - Observation scopes & per-scope limits
   - Directive priority ranking
   - Mental model refresh triggers

4. **Consolidation Reference**
   - Keep cross-reference to `consolidator.py` for delta operation semantics
   - Link to test suite (`test_consolidation_dedup.py`, etc.)
   - Document atomic deduplication strategy

5. **Search & Retrieval**
   - Document 4-strategy parallel search
   - Explain RRF (reciprocal rank fusion) merging
   - Link to reranking model selection logic

---

## FILES GENERATED

- ✅ `.agents/summary/codebase_info.md` — Metrics & subsystem overview
- ✅ `AGENTS_GENERATED.md` — Full navigation guide (in repo root)
- ✅ `.agents/summary/DELTA_ANALYSIS.md` — This file (comparison & gaps)

---

## NEXT STEPS FOR AGENTS

1. **For Hindsight Development**: 
   - Read `AGENTS_GENERATED.md` for code navigation
   - Reference `codebase_info.md` for subsystem metrics
   - Use original `AGENTS.md` for operational procedures

2. **For System Integration**:
   - Verify Memlord MCP connectivity (port 8005)
   - Confirm split Ollama setup (ports 11434 + 11435)
   - Review memory defense blocking policies

3. **For Feature Implementation**:
   - Consult specific module documentation (consolidation, search, etc.)
   - Review test suites for expected behavior
   - Use MCP tools registry for new capabilities

