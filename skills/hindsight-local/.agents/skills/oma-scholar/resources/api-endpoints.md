# knows.academy API Endpoints

Base URL: `https://knows.academy`

All endpoints are public proxies. **No authentication required**, so do not invent or send `Authorization` headers.

## Search

```
GET /api/proxy/search?q={query}
GET /api/proxy/search?q={query}&discipline={field}
```

```bash
curl -s "https://knows.academy/api/proxy/search?q=diffusion+super+resolution"
```

Response shape (verified):
```json
{
  "results": [
    {
      "record_id": "knows:generated/{slug}/{version}",
      "profile": "paper@1",
      "title": "...",
      "summary": "...",
      "venue": "...",
      "year": 2026,
      "discipline": null,
      "keywords": [],
      "coverage_statements": "exhaustive",
      "coverage_evidence": "key_evidence_only",
      "provenance_origin": "machine",
      "provenance_actor_name": "knows-gen",
      "version_record": "1.0.0",
      "lint_passed": true,
      "download_count": 0,
      "created_at": "...",
      "stats": {...}
    }
  ]
}
```

`record_id` format: `knows:generated/{slug}/{version}` includes colons and slashes that **must be URL-encoded** when used as a path/query param.

## Fetch Full Sidecar

```
GET /api/proxy/sidecars/{record_id}
```

The full v0.9.0 record (~22 KB / ~4.5K tokens, 55% smaller than the source PDF).

```bash
RID=$(python3 -c "import urllib.parse; print(urllib.parse.quote('knows:generated/{slug}/1.0.0', safe=''))")
curl -s "https://knows.academy/api/proxy/sidecars/${RID}"
```

## Partial Fetch (Token-Saving)

```
GET /api/proxy/partial?record_id={id}&section={one_of}
```

Note: parameter is **`section` (singular)**, value must match `^(statements|evidence|relations|artifacts|citation)$`.

Available sections (verified):
- `statements` (typically ~700 tokens, 93% reduction vs full PDF)
- `evidence`
- `relations`
- `artifacts`
- `citation`

**Not available** via partial fetch: `methods` (statement_type), `provenance`. Fetch the full sidecar instead.

```bash
RID=$(python3 -c "import urllib.parse; print(urllib.parse.quote('knows:generated/{slug}/1.0.0', safe=''))")

# Just the claims
curl -s "https://knows.academy/api/proxy/partial?record_id=${RID}&section=statements"

# Evidence only
curl -s "https://knows.academy/api/proxy/partial?record_id=${RID}&section=evidence"
```

Response shape:
```json
{
  "record_id": "...",
  "items": [...]
}
```

**When to use partial vs full:**
- "What claims does this paper make?" -> `section=statements` only
- "What experiments?" -> `section=evidence`
- "What does it cite?" -> `section=citation` (or `section=artifacts` filtered by `role: cited`)
- Multi-section analysis -> fetch full sidecar (cheaper than 3 partial calls)

## Platform Stats

```
GET /api/proxy/jobs/stats
```

```bash
curl -s "https://knows.academy/api/proxy/jobs/stats"
```

Returns processing queue health (verified):
```json
{
  "pending": 10735,
  "running": 288,
  "completed": 37897,
  "failed": 1243,
  "skipped": 433,
  "total": 50596
}
```

Use this for:
- Health check before bulk operations
- Estimating coverage ("how many papers are in the index?")
- Diagnosing slow responses

## Skill Self-Description

```
GET /api/proxy/skill/knows.md
```

The canonical generation guide (natural language, slightly different from the JSON Schema). Refresh `upstream-spec-cache.md` from here.

## JSON Schema (Reference)

Schema id used by sidecars: `https://knows.dev/schema/record-0.9.json`

The host `knows.dev` is currently unreachable (HTTPS times out as of 2026-04-25), so schema cannot be fetched directly. The `oma scholar lint` subcommand implements rules empirically derived from production records.

## Rate Limiting

No hard limits documented. Be polite:
- Avoid bursting more than ~10 req/sec
- Cache responses locally for repeated queries within a session
- Prefer partial fetch over full when possible

## Error Handling

| Status | Meaning | Recovery |
|--------|---------|----------|
| 404 | record_id not found | Verify ID via `/search`; the record may not be sidecared yet |
| 422 | Validation error (bad params) | Check `section` value against the allowed enum |
| 5xx | Platform issue | Retry once; check `/jobs/stats` |
| Timeout | Network / slow response | Retry with longer timeout; fall back to local sidecar if cached |

## URL Encoding

`record_id` contains `:` and `/`, so always URL-encode when placing in a path or query parameter. Examples:

```bash
# Bash with python3
RID=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$RAW_RID', safe=''))")

# Or with jq
echo -n "$RAW_RID" | jq -sRr @uri
```

## Notes

- Endpoints starting with `/api/proxy/` are stable public routes
- Direct (non-proxy) endpoints may exist but are not contract-stable; do not rely on them
- This skill **only consumes** the API. Submission/upload is out of scope
