# White Paper: Hindsight Clients, Integrations, and Inference Topology
**A Comprehensive Evaluation of In-Process Runtimes, Caching Strategy, Reranking Engines, and Administrative Synergy**

---

## 1. Executive Summary

Hindsight is a biomimetic agent memory engine designed to provide artificial intelligence agents with cross-session recall, experience consolidation, and disposition-aware reflection. This document serves as the absolute canonical evaluation of:
1.  **`hindsight-clients`**: Standardized programmatic SDKs (Python, TypeScript, Go, Rust).
2.  **`hindsight-integrations`**: 50+ first-party and community integrations (Claude Code, OpenCode, Continue, Obsidian, Vapi, Pipecat).
3.  **In-Process Inference Options**: Direct local loop SDK integrations (LiteLLM SDK, llama.cpp).
4.  **The Docusaurus Registry Deep-Dive**: Analysis of `llmProviders.json` and `integrations.json` from the documentation core.
5.  **Caching Strategy ("Cash")**: Auto-managed prompt caching.
6.  **Reranking Architecture ("Reno")**: Multi-provider cross-encoders for high-precision semantic pruning.

---

## 2. Technical Topology: Clients vs. Integrations

### 2.1 The Client Landscape (`hindsight-clients/`)
The clients form the programmatic foundation of Hindsight. They provide a type-safe, direct interface over the REST HTTP/JSON endpoints exposed by the Hindsight API server (`port 8888`).

*   **Python Client**: Hand-crafted, ergonomic wrapper featuring parallel async methods (`aretain`, `arecall`, `areflect`) and synchronous wrappers for scripts/REPLs.
*   **TypeScript Client**: Lightweight, high-performance module optimized for Node.js, Next.js, and serverless edge runtimes.
*   **Go & Rust Clients**: Strongly typed, compiled connectors built for high-throughput enterprise architectures.

### 2.2 The Integration Ecosystem (`hindsight-integrations/`)
Integrations provide vertical bindings that eliminate the need to write custom memory logic.

1.  **Claude Code Hook-Based Plugin (`claude-code/`)**: Hooks directly into Claude Code's native lifecycle. Uses the `SessionStart` hook to verify daemon health, the `UserPromptSubmit` hook to run auto-recall and inject memories as invisible `additionalContext`, and the `Stop` hook to asynchronously retain conversation transcripts. Features an auto-managed local daemon lifecycle that spawns and terminates `hindsight-embed` dynamically.
2.  **OpenCode Coding Agent Plugin (`opencode/`)**: Integrates auto-retain during the `session.idle` phase to minimize token consumption and features a **Compaction Hook** that injects memories directly during context window compaction so that critical experiences survive context trimming.
3.  **Continue.dev HTTP Context Provider (`continue/`)**: Implements an HTTP adapter server on `127.0.0.1:8123` to intercept `@hindsight` queries at chat query-time and inject relevant context dynamically.
4.  **Obsidian Note Sync (`obsidian/`)**: Features incremental vault sync using local content hash indices (`note path` -> `content hash + mtime`) to avoid redundant API transfers. Auto-tags documents with implicit multidimensional scoping (vault, folder tree, created/updated dates).
5.  **Cloudflare OAuth Proxy (`cloudflare-oauth-proxy/`)**: Handles the full OAuth 2.1 authorization server flow, dynamic client registration (RFC 7591), PKCE verification, and password gating. Bridges cloud-based agents (like `claude.ai` web app or Perplexity) to your private, self-hosted Hindsight instance via a Cloudflare Tunnel origin.
6.  **Voice Pipelines (Vapi & Pipecat)**: 
    *   *Vapi*: Pre-injects call-start memories as `assistantOverrides` and retains transcripts on call-end reports.
    *   *Pipecat*: Functions as a real-time `FrameProcessor` that intercepts `OpenAILLMContextFrame` streams, injects memories as system frames, and routes them to downstream LLMs.

---

## 3. Caching Strategy ("Cash")

To achieve low-latency recall and minimize operational costs, Hindsight implements **Explicit and Implicit Prompt Caching**:

```
[System/Mission Prefix (Stable & Shared)]  ===>  CACHED (Billed at ~10% input rate)
[Bank-Specific Context & Memories]        ===>  DYNAMIC (Appended to cached prefix)
```

### 3.1 Explicit Prompt Caching (Gemini & Vertex AI)
Hindsight leverages Gemini and Vertex AI's `CachedContent` API automatically:
*   **Bank-Agnostic Structure**: System instructions and schema definitions are kept stable and bank-agnostic, allowing a single cached prompt to be shared across all memory banks instead of creating separate cache entries per user.
*   **Soft-Fail Safety**: Cache creation is non-blocking. If cache building fails, it falls back to a normal uncached call with zero impact on the request lifecycle.
*   **Control**: Managed globally via `HINDSIGHT_API_LLM_PROMPT_CACHE_ENABLED=true` (on by default).

### 3.2 Implicit Prompt Caching (OpenAI & Anthropic)
*   **OpenAI**: Benefits from automatic, zero-config server-side caching of leading prompt blocks.
*   **Anthropic**: Implements standard `cache_control` breakpoints, which are automatically injected at stable segments of the prompt payload.

---

## 4. Reranking Architecture ("Reno")

Vector databases are prone to returning noise when querying large datasets. To ensure retrieved memories are highly relevant, Hindsight employs a **Cross-Encoder Reranking Layer ("Reno")**:

```
               [Raw Retrieved Memories (e.g., Top 50 Vector Matches)]
                                         |
                                         v
               [Cross-Encoder Reranker (Scores Query-Document Pairs)]
                                         |
                                         v
                [Pruned Relevant Memories (e.g., Top 10 High-Precision)]
```

### 4.1 Local Reranking (In-Process CPU/GPU)
*   **SentenceTransformers (`LocalSTCrossEncoder`)**: Loads models like `cross-encoder/ms-marco-MiniLM-L-6-v2` locally (~80ms for 100 pairs on CPU). Uses a dedicated thread pool to limit concurrent CPU-bound work and avoid thrashing.
*   **FlashRank**: An ultra-lightweight ONNX-powered local reranking engine that performs scoring in pure numpy, bypassing the PyTorch memory overhead.
*   **Linux Page Releasing (`malloc_trim`)**: Because local cross-encoders allocate large transient buffers, Hindsight automatically invokes `malloc_trim(0)` under Linux after each batch to release freed heap pages back to the OS and prevent memory bloat.

### 4.2 Cloud Reranking (In-Process SDK)
*   **LiteLLM SDK Reranker (`LiteLLMSDKCrossEncoder`)**: Performs direct, in-process API calls to high-power cloud reranking models like `cohere/rerank-english-v3.0`, `jina-reranker-v2-base-multilingual`, or `deepinfra/Qwen3-reranker-8B` without requiring a proxy server.

---

## 5. The Documentation Docusaurus Deep-Dive

An audit of the Docusaurus documentation workspace reveals the exact underlying schemas that drive model and integration registration:

### 5.1 Model Specifications (`llmProviders.json`)
Specifies the out-of-the-box parameters for verified model families:
*   **Built-In GGUF Server (`llamacpp`)**: Uses the `gemma-4-e2b-it` model which is automatically downloaded and executed inside the main Python process—no Ollama or external binaries required.
*   **Batch APIs**: Explicitly registers which models (OpenAI, Gemini, Groq, Fireworks) support asynchronous Batch APIs, enabling 50% cost savings for background memory consolidation.

### 5.2 Integration Specifications (`integrations.json`)
The canonical registry mapping out all 50+ integrations, categorizing their implementations (native, official, community), link structures, and icons to build dynamic visual components inside Docusaurus.

---

## 6. Actionable Roadmap & Recommendations

To capitalize on these architectural assets, the following steps are recommended:

1.  **Promote In-Process Caching**: Ensure `HINDSIGHT_API_LLM_PROMPT_CACHE_ENABLED=true` is standard in all developer templates to slash prompt ingestion costs in half.
2.  **Utilize Direct Local Reranking**: For self-hosted instances, enforce `HINDSIGHT_API_RERANKER_PROVIDER=local` using the lightweight ms-marco MiniLM.
3.  **Integrate Client Observability**: Wire up client `User-Agent` extraction to group Control Plane telemetry by the active client integration (e.g., displaying relative activity of Claude Code vs. Cursor).

---
*Document compiled and updated by Crush - AI System Architect.*
