# Hindsight Embeddings: All Tested Options

## 📋 Summary Table

| Option | Dimensions | Speed | Cost | Privacy | Status | Notes |
|--------|------------|-------|------|---------|--------|-------|
| **Ollama + nomic-embed-text** ⭐ | 768 | Fast (GPU) | Free | ✅ Local | ✅ **WORKS** | **RECOMMENDED** |
| **FreeLLM + text-embedding-3-small** | 1536 | Medium | Free | ❌ Cloud | ⚠️ Blocked | Curl works, app blocked |
| **ONNX + e5-small** | 384 | Slow (CPU) | Free | ✅ Local | ✅ **WORKS** | Fallback option |

---

## Option 1: Ollama + nomic-embed-text ⭐ **RECOMMENDED**

### Configuration
```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm-sdk
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE=http://localhost:11434
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL=ollama/nomic-embed-text
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_ENCODING_FORMAT=
```

### Test Results
- ✅ **Initialization**: SUCCESS
- ✅ **Dimensions**: 768 (auto-detected)
- ✅ **Performance**: Fast (GPU-accelerated)
- ✅ **Generated**: 3 test embeddings
- ✅ **Sample**: `[0.0364, -0.0007, -0.1791, -0.0601, 0.0601]`

### Pros
- 🚀 Fast (GPU-accelerated if available)
- 🔒 Complete privacy (never leaves your machine)
- 💰 Free forever (no API costs)
- 📶 No rate limits
- 🎯 High-quality 768-dimensional embeddings
- ⚡ Works offline after model download

### Cons
- Requires Ollama installation
- ~300MB model download on first use

### Why This Is Best
This is the **goldilocks option**: not too small (ONNX's 384), not too large (3-small's 1536), and runs locally with GPU acceleration. Perfect for agent memory workloads.

---

## Option 2: FreeLLM API + text-embedding-3-small

### Configuration
```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm-sdk
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_KEY=freellmapi-b921acca21a32a48fa44544a8a5f7b04ae7801343ca0cd24
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE=https://freellmapi.collabmind.dev/v1
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL=text-embedding-3-small
```

### Test Results
- ⚠️ **Curl Test**: ✅ SUCCESS (returns 1536-dim vectors)
- ❌ **App Test**: BLOCKED ("Your request was blocked")
- ✅ **Dimensions**: 1536 (confirmed via curl)
- ✅ **Provider**: GitHub (via FreeLLM routing)

### Curl Confirmation
```bash
curl -X POST https://freellmapi.collabmind.dev/v1/embeddings \
  -H "Authorization: Bearer freellmapi-b921acca21a32a48fa44544a8a5f7b04ae7801343ca0cd24" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": ["test"]}'
  
# Returns: 1536-dimensional embedding (first 5: [-0.00986, 0.00156, 0.01567, -0.05481, -0.00641])
```

### Pros
- 🎯 Highest quality (1536 dimensions)
- ☁️ No local installation needed
- 🆓 Free API (FreeLLM credits)

### Cons
- ❌ **Currently blocked** in Hindsight app (rate limiting or auth issue)
- 📡 Requires internet connection
- 🔓 Data sent to external service
- ⏱️ Potential rate limits
- 🌐 API availability dependency

### Status
**PARTIALLY WORKING** - The API endpoint works (confirmed via curl), but Hindsight gets blocked. This could be:
1. Rate limiting (too many requests in test)
2. Auth header formatting issue
3. User-Agent blocking
4. API-side security policy

**Recommendation**: Use Ollama instead. FreeLLM works great for LLM inference but embeddings are currently unreliable.

---

## Option 3: ONNX + multilingual-e5-small (Fallback)

### Configuration
```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=onnx
HINDSIGHT_API_EMBEDDINGS_ONNX_MODEL_ID=intfloat/multilingual-e5-small
HINDSIGHT_API_EMBEDDINGS_ONNX_FILE=onnx/model.onnx
HINDSIGHT_API_EMBEDDINGS_ONNX_DIMENSIONS=384
```

### Test Results
- ✅ **Initialization**: SUCCESS (auto-downloads from HuggingFace)
- ✅ **Dimensions**: 384
- ✅ **Generated**: 3 test embeddings
- ⚠️ **Performance**: Slower (CPU-only)

### Pros
- 📦 Pure Python (no external services)
- 🌍 Multilingual support
- 💾 Small model size
- 🔧 Works anywhere (no GPU needed)

### Cons
- 🐌 Slower (CPU-only, no GPU acceleration)
- 📉 Lower quality (384 vs 768 vs 1536 dimensions)
- ⏳ First run downloads model from HuggingFace

### When to Use
- You don't have Ollama installed
- Running on CPU-only systems
- Need multilingual embeddings
- Disk space is limited

---

## 🏆 Final Recommendation

### For Production: **Ollama + nomic-embed-text**

**Your current setup** (already configured in `.env`):
```bash
# LLM
HINDSIGHT_API_LLM_PROVIDER=litellm
HINDSIGHT_API_LLM_MODEL=z-ai/glm-5.1
HINDSIGHT_API_LLM_BASE_URL=https://freellmapi.collabmind.dev/v1

# Embeddings
HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm-sdk
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE=http://localhost:11434
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL=ollama/nomic-embed-text
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_ENCODING_FORMAT=
```

This gives you:
- ✅ **Free LLM** via FreeLLM API (z-ai/glm-5.1)
- ✅ **Local embeddings** via Ollama (768-dim, GPU-accelerated)
- ✅ **No rate limits** on embeddings
- ✅ **Privacy** for your memory data
- ✅ **Fast performance** with GPU acceleration

---

## 📊 Dimension Comparison

| Model | Dimensions | Use Case |
|-------|------------|----------|
| **e5-small** | 384 | Basic semantic search, CPU-only systems |
| **nomic-embed-text** ⭐ | 768 | **Agent memory (RECOMMENDED)** |
| **text-embedding-3-small** | 1536 | Highest quality (if FreeLLM unblocks) |
| **text-embedding-3-large** | 3072 | Overkill for agent memory |

**Why 768 is the sweet spot for agent memory:**
- Better than 384 (more nuanced understanding)
- Smaller than 1536 (faster retrieval, less storage)
- Optimized for semantic similarity (Hindsight's primary use)
- Fits within pgvector's HNSW index limits

---

## 🛠️ Quick Start

### Test Your Current Setup
```bash
cd /Users/oliververmeulen/hindsight
python3 test_embeddings.py
```

### Check Database Status
```bash
python3 check_database.py
```

### Start Hindsight
```bash
cd hindsight-api-slim
python -m hindsight_api
```

---

## 📁 Configuration Files

Created during testing:
- `.env` - **Active config** (Ollama 768-dim)
- `.env.test-ollama` - Ollama config (same as above)
- `.env.test-onnx` - ONNX fallback config
- `.env.test-freellm-3-small` - FreeLLM config (blocked)

Test scripts:
- `test_embeddings.py` - Comprehensive embedding test
- `check_database.py` - Database status checker

---

## ❓ FAQ

**Q: Why not use FreeLLM for embeddings?**  
A: FreeLLM blocks embeddings requests in the app (though curl works). Ollama is more reliable for this use case.

**Q: Can I use OpenAI's API directly?**  
A: Yes! Just set:
```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=sk-your-key
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small
```
But you'll pay per request (~$0.02/1M tokens).

**Q: What if I don't have GPU?**  
A: Ollama works on CPU too (just slower). Or use ONNX fallback.

**Q: Can I change dimensions later?**  
A: Yes, but you must clear existing embeddings first (run `clear_embeddings.py`).

---

**Status**: ✅ **ALL SYSTEMS GO** with Ollama + nomic-embed-text!
