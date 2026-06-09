# Observation deduplication

Finds near-duplicate **observations** in a Hindsight bank.

The Hindsight API has no bulk export and does not expose embedding vectors, so
the tool:

1. Pages through `GET /v1/{tenant}/banks/{bank}/memories/list?type=observation`
   to read every observation.
2. Re-embeds the text locally with the same default model Hindsight uses
   (`BAAI/bge-small-en-v1.5`).
3. Runs a block-wise cosine-similarity scan and merges pairs above a threshold
   into transitive clusters (union-find).

Cosine similarity is a cheap first pass. The pipeline in `dedup.py` is split so
an agentic verifier can later confirm candidate clusters before they're
reported (see the module docstring).

## Usage

```bash
# Against a running API (default URL http://localhost:8888)
uv run find-duplicate-observations --bank-id hermes --threshold 0.92

# Tighter threshold isolates near-verbatim duplicates; write a JSON report
uv run find-duplicate-observations --bank-id hermes --threshold 0.97 \
    --json-out report.json
```

Key flags: `--api-url`, `--api-key`, `--tenant`, `--threshold` (cosine cutoff,
default 0.92), `--min-cluster-size`, `--embedding-model`, `--json-out`.

### Picking a threshold

- **~0.97+** — near-verbatim copies (same event re-consolidated).
- **~0.92** — looser; transitive merging can chain topically-related
  observations into large clusters. Useful for spotting redundancy hot-spots,
  noisier for "true duplicate" decisions.
