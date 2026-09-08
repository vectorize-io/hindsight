# Hindsight system tests

Blackbox tests over a real `hindsight-api` process, driven **only** through the
published Python client. No engine imports, no SQL, no internals.

## Why this package exists

`hindsight-api-slim/tests/` holds ~500 files, each covering one mechanism. That
catches mechanism bugs. It does not catch *composition* bugs — consolidation
wiping facts, a delta refresh missing a backdated window, a transfer dropping
evidence — because no single-mechanism test spans the steps where those live.
These tests do.

## How determinism works

A stub server implements the OpenAI chat-completions and embeddings APIs plus a
Cohere-compatible rerank endpoint. The server under test is pointed at it with
ordinary environment variables, so **no production code changes are needed**, and
the real provider transport (JSON repair, retries, structured output) is
exercised for real.

- **Rules match on content, not request bodies** — the requested JSON-schema name
  and prompt substrings — so editing a prompt does not break every test.
- **An unmatched call fails the test** with the rule to paste in. Nothing is ever
  answered by a plausible-looking default.
- **Requests are validated strictly.** If Hindsight sends something real OpenAI
  would 400, so does the stub.
- **Embeddings are lexical and deterministic**, so the suite needs no torch and no
  model download.

## Running

```bash
# once: the server needs pg0, and nothing else beyond its base dependencies
(cd ../hindsight-api-slim && uv sync --frozen --extra embedded-db)

cd hindsight-system-tests
uv run pytest tests -v
```

Note the missing extra: with embeddings *and* reranking pointed at the stub,
nothing loads sentence-transformers, so the suite runs with **no torch and no
model download**. CI (`test-system`) needs no provider secrets either, which
means it runs on fork PRs — unlike every `test-api` job.

The fixtures start their own embedded Postgres (`pg0://hindsight-systest:15499`),
separate from the dev database and from the api-slim suite's, and run the server
from a scratch directory holding an empty `.env` so your own `.env` cannot leak
into the test configuration.

If your shell exports `PYTEST_ADDOPTS=-n ...`, clear it for this suite —
`pytest-xdist` is not installed here, and the session-scoped server makes it
pointless anyway.

**Background work is off by default.** Observation extraction and auto
consolidation run in the worker after a retain returns, so their LLM calls would
land at a moment no test controls — after the assertions, sometimes after the
next test has started. Both are per-bank settings, so a story about either
switches it on for its own bank and waits for the operation to finish.

## Conventions

- One story per file, named `test_NN_<story>.py`. The number is **reading order for
  a human, not execution order** — every test must pass when run alone.
- Assert through the client's responses only.
