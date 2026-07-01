# Hindsight Startup Analysis & MCP OAuth Issue (2026-06-27)

## Issue: OAuth Discovery Failed

**Error**: "OAuth discovery failed: the server does not advertise OAuth endpoints"

**Root Cause Analysis**: 
- MCP is trying to authenticate but Hindsight MCP endpoint isn't advertising OAuth endpoints
- This likely means MCP service is **down** or **not responding**
- MCP endpoint should be at: `http://localhost:8888/mcp`

---

## Startup Plan (No Code Changes)

### Step 1: Check Current Status

```bash
cd /Users/oliververmeulen/hindsight

# Check if anything is running
./scripts/dev/status.sh

# Or quick health checks:
curl http://localhost:8888/health        # Hindsight API
curl http://localhost:8888/mcp           # MCP endpoint (should not error)
curl http://localhost:8005/health        # Memlord MCP
```

### Step 2: Clean Start (Safest Approach)

If services are running but broken, stop them first:

```bash
# Option A: Kill processes cleanly
pkill -f "hindsight-api"
pkill -f "hindsight-worker"
pkill -f "next dev"  # Control Plane

# Wait 3 seconds
sleep 3
```

### Step 3: Start Everything

**Recommended (Simplest)**:
```bash
cd /Users/oliververmeulen/hindsight/scripts/dev
./start-services.sh
```

This runs in order:
1. **Ollama split** (embeddings port 11434, LLM port 11435) — ⏱️ ~2-3 min
2. **Hindsight API** (port 8888) — ⏱️ ~15-30 sec, health check waits
3. **Control Plane** (port 9998) — ⏱️ ~30-60 sec
4. **Workers** (configurable, default 2) — ⏱️ ~10 sec

**Alternative (With Monitoring)**:
```bash
./start-all.sh --monitoring --workers 4
```

### Step 4: Verify MCP is Working

Once API is up (should see "✓ API is ready"):

```bash
# Check MCP endpoint responds
curl -v http://localhost:8888/mcp

# Check if OAuth endpoints advertised:
curl http://localhost:8888/.well-known/oauth-authorization-server 2>/dev/null | jq .
```

### Step 5: Monitor Status

```bash
# Watch real-time status
./status.sh --watch

# Or use interactive dashboard
./dashboard.sh
```

---

## What Each Script Does (No Code Changes)

| Script | Purpose | Command |
|--------|---------|---------|
| `start-services.sh` | **Recommended**: Full startup | `./start-services.sh` |
| `start-all.sh` | Full startup with options | `./start-all.sh --monitoring --workers 8` |
| `status.sh` | Check health (read-only) | `./status.sh --watch` |
| `dashboard.sh` | Interactive management | `./dashboard.sh` |
| `start-ollama-split.sh` | Just Ollama lanes | `./start-ollama-split.sh` |
| `start-api.sh` | Just API | `./start-api.sh --port 8888` |
| `start-control-plane.sh` | Just Control Plane | `./start-control-plane.sh` |
| `start-worker.sh` | Just one worker | `./start-worker.sh` |

---

## Expected Timeline

```
start-services.sh execution:

[1/4] Starting Ollama lanes...
      ↓ (~2-3 min, models download if not cached)
      ✓ Ollama ready

[2/4] Starting API...
      ↓ (~15 sec startup + health check loop ~15 sec)
      ✓ API ready (http://localhost:8888)

[3/4] Starting Control Plane...
      ↓ (~30-60 sec, Next.js dev mode)
      ✓ Control Plane ready (http://localhost:9998)

[4/4] Starting Workers...
      ↓ (~10 sec)
      ✓ N workers running

Total: ~4-5 minutes from start to fully operational
```

---

## Log File Locations (for Debugging)

```bash
# Real-time monitoring
tail -f /tmp/hindsight-api.log              # API startup + requests
tail -f /tmp/hindsight-control-plane.log    # Control Plane errors
tail -f /Users/oliververmeulen/hindsight/logs/worker-*.log  # Worker logs

# Or check what failed:
cat /tmp/hindsight-api.log | grep -i error
```

---

## If Ollama Needs Restart

Sometimes Ollama ports get stuck. Recovery:

```bash
# Kill stuck Ollama processes
pkill ollama
sleep 2

# Restart just Ollama
./start-ollama-split.sh

# Wait for both ports ready:
curl http://localhost:11434  # Should respond
curl http://localhost:11435/v1/models  # Should list models
```

---

## Environment Check

Before starting, verify `.env` has critical vars:

```bash
# Quick check (no code change, just reading):
grep "HINDSIGHT_API_LLM_BASE_URL" .env
grep "HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE" .env
grep "HINDSIGHT_DB_URL" .env
```

Expected output:
```
HINDSIGHT_API_LLM_BASE_URL=http://localhost:11435/v1
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434
HINDSIGHT_DB_URL=postgresql://...
```

---

## Next Steps to Fix OAuth Issue

1. **Run**: `./start-services.sh` (wait for "✓ All services started")
2. **Verify**: `curl http://localhost:8888/health` returns success
3. **Test MCP**: `curl http://localhost:8888/mcp` 
4. **Check Logs**: `tail -f /tmp/hindsight-api.log` for startup errors
5. **If still failing**: Share error from logs + output of `./status.sh`

---

**No code changes needed — just orchestration of existing startup scripts.**
