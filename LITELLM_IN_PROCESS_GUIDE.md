# Guide: Utilizing In-Process LiteLLM SDK inside Hindsight
**Architectural Blueprint for Local, Zero-Proxy, High-Resiliency Inference Loops**

---

## 1. Executive Summary

A common pitfall when building multi-model agent systems is relying heavily on secondary proxy microservices (e.g., spinning up an independent LiteLLM proxy server in Docker, orchestrating keys, managing extra ports, and suffering extra network serialization latency). 

This document details how **Hindsight completely avoids this overhead by embedding the LiteLLM SDK directly inside its engine**. We map out how you can configure a direct, local loop that supports 100+ LLMs, embeddings, and reranker engines locally or from cloud providers—completely in-process.

---

## 2. In-Process SDK vs. External Proxy Server

Understanding this distinction is critical for local performance:

```
METHOD A: LiteLLM Proxy Server (Network Hop Overhead)
[Hindsight Engine] ---HTTP/JSON (port 4000)---> [LiteLLM Proxy Server] ---> [Cloud/Local APIs]

METHOD B: LiteLLM SDK (In-Process "Local Loop" - OPTIMAL)
[Hindsight Engine (LiteLLM SDK Inside)] -----------------------------------> [Cloud/Local APIs]
```

### 2.1 Why the In-Process SDK is Superior:
1.  **Zero Network Hops**: Eliminates the latency of sending HTTP JSON payloads through a secondary local server.
2.  **No Port/Process Wrangling**: You do not have to manage and keep port `4000` (default LiteLLM proxy port) alive.
3.  **Unified Configuration**: Everything runs natively inside your primary Hindsight configuration tree (`.env`).
4.  **Local Memory Access**: Thread pooling, backoff parameters, and retry strategies are handled in python memory without risking broken socket connections between servers.

---

## 3. Configuration & Parameter Matrix

Hindsight provides native SDK configurations across **all three core inference layers**: LLMs, Embeddings, and Cross-Encoder Rerankers.

### 3.1 Layer 1: Core LLM (Text Generation & Reflection)
To route text completion and structured extraction through the in-process LiteLLM SDK, configure:

```bash
# 1. Primary LLM Configuration
HINDSIGHT_API_LLM_PROVIDER=litellm
HINDSIGHT_API_LLM_MODEL=together_ai/meta-llama/Llama-3-70b-chat-hf
HINDSIGHT_API_LLM_API_KEY=your_together_ai_api_key_here

# Optional Fallbacks / Router Strategy
HINDSIGHT_API_LLM_PROVIDER=litellmrouter
HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG='{
  "model_list": [
    {
      "model_name": "primary-model",
      "litellm_params": {
        "model": "together_ai/meta-llama/Llama-3-70b-chat-hf",
        "api_key": "sk-111"
      }
    },
    {
      "model_name": "backup-model",
      "litellm_params": {
        "model": "openai/gpt-4o-mini",
        "api_key": "sk-222"
      }
    }
  ]
}'
```

### 3.2 Layer 2: Embeddings
By utilizing the `litellm-sdk` embeddings provider, Hindsight invokes `litellm.aembedding()` natively:

```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm-sdk
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL=cohere/embed-english-v3.0
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=your_cohere_api_key_here
HINDSIGHT_API_EMBEDDINGS_DIMENSION=1024  # Sized automatically or declared manually
```

### 3.3 Layer 3: Cross-Encoder Reranker
By choosing `litellm-sdk` for your reranker provider, Hindsight routes semantic cross-comparison natively inside the core engine:

```bash
HINDSIGHT_API_RERANKER_PROVIDER=litellm-sdk
HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL=cohere/rerank-english-v3.0
HINDSIGHT_API_RERANKER_COHERE_API_KEY=your_cohere_api_key_here
```

---

## 4. Comparable Runtimes & In-Process Alternatives

If your goal is to minimize external dependencies and build a self-contained local deployment, there are other highly optimized in-process providers inside Hindsight you should leverage:

### 4.1 Local Inference (The Local Loop)
Hindsight includes local CPU and GPU-accelerated libraries. You don't even need cloud APIs:

1.  **Local Embeddings (`SentenceTransformers`)**:
    ```bash
    HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
    HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5
    # Runs inside PyTorch/ONNX completely locally. Auto-allocates CUDA/MPS.
    ```
2.  **Local ONNX Embeddings (`ONNX Runtime`)**:
    ```bash
    HINDSIGHT_API_EMBEDDINGS_PROVIDER=onnx
    # Extremely fast, low memory footprint, no PyTorch overhead.
    ```
3.  **Local Rerankers (`FlashRank`)**:
    ```bash
    HINDSIGHT_API_RERANKER_PROVIDER=local
    HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
    # Heavy-duty semantic sorting on CPU, typically sub-100ms.
    ```

### 4.2 Local Llama.cpp Integration
For local LLM inference without Ollama, Hindsight has a built-in `llama.cpp` provider:
```bash
HINDSIGHT_API_LLM_PROVIDER=llama-cpp
# Direct C++ bindings. Auto-downloads Gemma 4 E2B or similar GGUFs and loads them into memory.
```

---

## 5. Summary Matrix of Local Benefits

| Layer | Provider Option | In-Process Mode | External Process Needed? | Performance Profile |
|---|---|---|---|---|
| **LLM** | `litellm` | YES (LiteLLM SDK) | No | Fast, native Python API calls |
| **LLM** | `llama-cpp` | YES (Local GGUF) | No | Heavy CPU/GPU memory, offline |
| **Embeddings**| `litellm-sdk` | YES (LiteLLM SDK) | No | Cloud API speeds |
| **Embeddings**| `local` / `onnx`| YES (HF/ONNX Run) | No | Sub-50ms local vectors, offline |
| **Reranker**  | `litellm-sdk` | YES (LiteLLM SDK) | No | High-accuracy Cohere rerank |
| **Reranker**  | `local` (Flash) | YES (ONNX Rerank) | No | Fast, local passage ranking |

---
*Document compiled by Crush - local execution specialist.*
