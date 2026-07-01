# Fallback Provider Cascade

`oma-scholar` queries `knows.academy` first, then falls back to OpenAlex for
records the platform doesn't have. The fallback is what makes the skill useful
beyond the (currently 2026-only) knows.academy index.

## Coverage Matrix

| Source | Coverage | Gives | Doesn't give |
|--------|----------|-------|--------------|
| **knows.academy** | ~50K papers, **2026 only** (verified empirically) | Full v0.9 sidecar (claims, evidence, relations, methods) | Pre-2026 papers, full text |
| **OpenAlex** | ~240M works, all years | Title, authors, year, venue, DOI, abstract (reconstructed), OA PDF URL, citation count, references | Structured claims/evidence (no sidecar) |

When a paper is in **both**, prefer knows.academy (richer structure). When a
paper is **only in OpenAlex** (most pre-2026 work), use OpenAlex metadata as
the source for **Mode 1 Generate**, where the host LLM produces a local sidecar from
the abstract.

## Cascade Logic

```
search "query"
   │
   ├─ knows.academy ── any hits? ──► return immediately (sidecars available)
   │     │
   │     no hits
   │     ▼
   └─ openalex ─────────────────► return metadata + abstract (no sidecar)
                                  ↓
                             user wants deeper? → Mode 1 Generate locally
```

## CLI Reference (`oma scholar`)

### Search both sources

```bash
oma scholar search "vision language action"
```

Returns JSON with:
- `primary`: which source returned hits
- `fallback`: which source was used as fallback (if any)
- `results`: unified list with `source: "knows.academy" | "openalex"` per item

### Get a specific record

```bash
# knows.academy sidecar
oma scholar get "knows:generated/reconvla/1.0.0"

# Partial fetch (saves up to 93% tokens)
oma scholar get --section statements "knows:generated/reconvla/1.0.0"

# OpenAlex by DOI
oma scholar get "10.48550/arXiv.1706.03762"

# OpenAlex by W-id
oma scholar get "W2147144213"
```

When `oma scholar get` is asked for a `knows:...` id and the platform is
unreachable, the command extracts the slug from the record_id and searches
OpenAlex for the same paper, returning metadata with a `fallback: "openalex"`
marker. The user can then run **Mode 1 Generate** from the abstract.

### Resolve a title across both

```bash
oma scholar resolve "Attention Is All You Need"
```

Returns the best match from each source side-by-side, plus a recommendation.

## When Fallback is Needed

Fallback to OpenAlex is the right move when:

- **Pre-2026 papers**: knows.academy doesn't index them
- **Cross-archive coverage**: non-arXiv venues, journals, books
- **Citation counts**: knows.academy doesn't track citations; OpenAlex does
- **Reference resolution**: when a sidecar's `cites` predicate points to a paper not in knows.academy

Fallback is NOT needed for:
- Recent (2026) arXiv papers, since knows.academy has them with rich sidecars
- Full-text reading, since neither source provides full text; fetch the OA PDF and use `oma-pdf`

## Local Generate from Fallback Result

When OpenAlex returns metadata + abstract, but you want a structured sidecar:

```
1. oma scholar get <doi>        → get title, authors, year, venue, abstract
2. (optional) download OA PDF   → oma-pdf to extract full text
3. Mode 1 Generate              → host LLM produces .knows.yaml using sidecar-spec.md
4. oma scholar lint             → validate locally
```

The locally-generated sidecar is **not registered** with knows.academy; it's
yours. If you want it shared, that's a separate publishing flow (out of scope
for this skill).

## Authentication

OpenAlex anonymous use is free up to ~$1/day. For higher limits or polite
pool access, see `setup-openalex.md`. The skill works without any key.

`oma scholar` reads:
- `OPENALEX_API_KEY`: passed as `?api_key=` (recommended)
- `OPENALEX_EMAIL`: passed as `?mailto=` (polite pool, no signup)

Neither is required.

## Trust Considerations

- **knows.academy sidecars are AI-generated**: all have `provenance.origin: machine`. Verified `lint_passed: true` is platform-internal; our local `oma scholar lint` finds dangling refs in ~47% (use `--lenient` when consuming)
- **OpenAlex metadata is curated**: generally reliable for title/authors/DOI/year, but venue and abstract can be missing for older works
- **Reconstructed abstracts** from OpenAlex's inverted index are exact (no paraphrasing), but punctuation/formatting may be lossy

## Limitations

- arXiv-only papers without DOI may show `doi: None` in OpenAlex
- Some 2026 papers exist in both sources with slightly different titles; `resolve` does case-insensitive contain matching as a heuristic
- knows.academy proxy can timeout under load; fallback to OpenAlex is automatic
- OpenAlex rate-limits: respect ~10 req/sec courtesy limit
