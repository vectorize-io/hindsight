# Hindsight Integration Tests

E2E and integration tests for Hindsight API that require a running server.

## Test Types

### 1. Tests with External Server
Tests like `test_mcp_e2e.py` expect a server to already be running.

**Running:**
```bash
# Start the API server
./scripts/dev/start-api.sh

# Run tests
cd hindsight-integration-tests
HINDSIGHT_API_URL=http://localhost:8888 uv run pytest tests/test_mcp_e2e.py -v
```

### 2. Self-Contained Tests
Tests like `test_base_path_deployment.py` manage their own server lifecycle and use docker-compose.

**Running:**
```bash
cd hindsight-integration-tests

# Run with pytest
uv run pytest tests/test_base_path_deployment.py -v

# Or run directly for nice output
uv run python tests/test_base_path_deployment.py
```

**Requirements:**
- Docker and docker-compose installed (for reverse proxy test)
- No nginx required on host!

**What it tests:**
- ✅ API with base path (direct server)
- ✅ Full reverse proxy via docker-compose + Nginx
- ✅ Regression: API without base path
- ✅ Full retain/recall workflow

These tests:
- Start their own API servers on dedicated ports (18888-18891)
- Use docker-compose to test actual deployment scenarios
- Run in parallel with other tests (no port conflicts)
- Clean up automatically

### 3. Hermes Agent Smoke Tests

`test_hermes_embedded_smoke.py` drives the Hermes Agent ↔ Hindsight integration
end-to-end through the `HindsightMemoryProvider` plugin in `local_embedded`
mode. Run on demand — not part of CI.

The test reuses the installed Hermes venv (which already has every dep), so
it doesn't need `uv` or this package's lockfile.

**Prerequisites:**
```bash
hermes update            # ensure plugin code at ~/.hermes/hermes-agent is current
```

**Running:**
```bash
HINDSIGHT_LLM_API_KEY=sk-... \
    ~/.hermes/hermes-agent/venv/bin/python -m pytest \
    hindsight-integration-tests/tests/test_hermes_embedded_smoke.py \
    -v -s -o addopts=""
```

(`-o addopts=""` overrides the `--timeout` flag from this package's pyproject;
`pytest-timeout` is not installed in the hermes venv and isn't needed here.)

Skipped automatically if `HINDSIGHT_LLM_API_KEY` (or `OPENAI_API_KEY`) is not set.

## Running All Tests

```bash
cd hindsight-integration-tests
uv run pytest tests/ -v
```

This runs both types. Self-contained tests won't conflict with the external server.

## Environment Variables

- `HINDSIGHT_API_URL` — Base URL for external-server tests (default: `http://localhost:8888`)
- `HINDSIGHT_LLM_API_KEY` / `OPENAI_API_KEY` — required by the Hermes embedded smoke test
- `HINDSIGHT_LLM_PROVIDER`, `HINDSIGHT_LLM_MODEL` — override defaults (`openai` / `gpt-4o-mini`)
