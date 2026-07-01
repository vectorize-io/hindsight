---
name: run-central-api
description: Run, start, launch, test, smoke, screenshot, or verify the CollabMind Central API (FastAPI server on port 8000). Use this skill whenever you need to start the central-api service, hit its endpoints, or validate a code change.
---

CollabMind Central API is a Python/FastAPI server (the memory control plane). It is driven with `curl` via `smoke.sh`. No GUI — a browser can reach `/docs` for the interactive OpenAPI explorer, but agent interaction uses curl.

## Prerequisites

macOS with Homebrew Python ≥ 3.11.  No extra apt packages needed.

## Build

```bash
cd central-api
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -q
```

The venv is gitignored.  Re-run `pip install -e` whenever `pyproject.toml` changes.

## Run (agent path)

Launch the server in the background, run the smoke script, then kill it:

```bash
cd central-api
.venv/bin/uvicorn app.main:app --port 8000 &
SERVER_PID=$!
sleep 2
bash .claude/skills/run-central-api/smoke.sh http://localhost:8000
kill $SERVER_PID
```

`smoke.sh` checks:
- `GET /` → `"service": "central-api"`
- `GET /health` → `"status": "ok"`
- `GET /api/health/engines` → lists all four engine backends
- `POST /api/memory/search` → returns `count` field
- `POST /api/context/build` → returns `audit_trace_id`

Use a different port to avoid conflicts: `--port 8100` and `smoke.sh http://localhost:8100`.

To poke a specific endpoint manually:

```bash
curl -s -X POST http://localhost:8000/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query":"hello","engine":"memlord","top_k":5}' | python3 -m json.tool
```

OpenAPI docs live at `http://localhost:8000/docs` while the server is running.

## Run (human path)

```bash
cd central-api
.venv/bin/uvicorn app.main:app --reload --port 8000
# open http://localhost:8000/docs in browser, Ctrl-C to stop
```

## Test

```bash
cd central-api
.venv/bin/pytest -q
# 28 passed in ~0.3 s (as of 2026-06-09)
```

Tests use FastAPI's `TestClient` — no live server needed.

## Lint

```bash
cd central-api
.venv/bin/ruff check .
```

## Gotchas

- `python3.12` is not available on the dev machine (Homebrew ships 3.14). `pyproject.toml` says `requires-python = ">=3.12"` — any 3.12+ works; just use `python3`.
- All engine adapters are stubs/mocks. `POST /api/memory/search` returns `count: 0` — that is correct behavior until real engines are wired.
- The `memlord` adapter (`app/adapters/memlord.py`) is the active development surface on branch `feat/central-api-memlord-adapter`. Its smoke check hits `GET /api/health/engines` which includes the memlord backend entry.
- `httpx` + `starlette.testclient` emits a `StarletteDeprecationWarning` about `httpx2` — harmless, all 28 tests pass.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `command not found: python3.12` | Use `python3` — Homebrew ships 3.14, which satisfies `>=3.12` |
| Port already in use | Use `--port 8100` and update the smoke.sh call accordingly |
| `ModuleNotFoundError: app` | Run `pip install -e .` inside the venv; or make sure `PYTHONPATH=.` |
| Smoke FAIL on `/api/memory/search` | Server may not be ready — increase `sleep 2` to `sleep 4` |
