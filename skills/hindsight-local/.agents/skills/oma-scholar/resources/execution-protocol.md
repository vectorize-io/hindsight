# Scholar - Execution Protocol

Choose the mode based on user intent, then follow its steps.

## Mode 1: Generate

Create a `.knows.yaml` sidecar from a paper, draft, or research notes.

### Step 0: Gather Source

1. Identify input: PDF path, LaTeX file, plain text, or pasted content
2. If PDF and only path is given -> chain via `oma-pdf` first to extract markdown
3. Determine output path: `{base}.knows.yaml` next to input (or as configured)

### Step 1: Read Source Thoroughly

- Read the entire source: abstract, methods, results, discussion, limitations
- Identify: claims, methods, evidence (tables/figures), datasets, code/repo links, cited works, assumptions
- Note paper metadata: title, authors, venue, year, DOI (only if visible)

### Step 2: Draft Sidecar Structure

Use the v0.9.0 spec rules in `sidecar-spec.md`. Top-level structure:

- `knows_version: "0.9.0"`, `profile: "paper@1"`, `subject_ref: art:paper`
- **Top-level metadata**: `title`, `authors`, `venue`, `year` (no `metadata` wrapper); omit `doi`/`venue`/`year` if not visible
- `summary`: one-paragraph overview
- `coverage`: object with `statements` and `evidence` enums (NOT a single value)
- `provenance`: `origin`, single `actor` object (`type: tool|person|org`), `generated_at`, `method`
- `version`: `{spec, record, source}`
- `freshness`: `{as_of, update_policy}`
- `artifacts`: list (`art:` prefix, `artifact_type`, `role: subject|supporting|cited`); include `representations` for the source
- `statements`: list (`stmt:` prefix, `statement_type`, `modality`, `status`, `confidence` object, `source_anchors`)
- `evidence`: list (`ev:` prefix, `evidence_type`)
- `relations`: list (`rel:` prefix, `predicate` in present tense, `subject_ref`, `object_ref`)
- `actions`: usually empty for paper profile

### Step 3: Wire Relations

For every statement, ensure at least one `supported_by` relation pointing to evidence.
Aim for **average ≥1.5 relations per statement**. Common patterns:

| Subject | Predicate | Object |
|---------|-----------|--------|
| Claim | `supported_by` | Evidence |
| Claim | `depends_on` | Assumption |
| Method | `evaluates_on` | Dataset |
| Method | `implements` | Code repo |
| Paper | `cites` | Other paper |
| Result | `contradicts` | Prior work |

### Step 4: Optional Metadata Enrichment

If `OPENALEX_API_KEY` is set and DOI/venue/year are missing:

```bash
# Sketch — actual enrichment is best done via host LLM with a curl call
curl -s "https://api.openalex.org/works?search={title}&api_key=$OPENALEX_API_KEY" \
  | jq '.results[0] | {doi, host_venue, publication_year}'
```

If the key is not set, skip enrichment and tell the user how to set it (point to `setup-openalex.md`).

### Step 5: Lint

Always validate before reporting done:

```bash
oma scholar lint {output}.knows.yaml
```

If lint fails, fix the reported issues and re-run until clean.

### Step 6: Report

Tell the user:
- Output path
- Statement count, evidence count, relations/statement ratio
- Any fields omitted due to anti-fabrication (e.g., "DOI not extracted; visible in source? If yes, paste it.")
- Lint status

## Mode 2: Validate

Lint an existing `.knows.yaml`.

### Steps

1. Confirm input file exists and ends with `.knows.yaml`, `.yaml`, or `.json`
2. Decide strictness based on origin:
   - **Own Generate output** → strict (default)
   - **Third-party / remote sidecar** → `--lenient` (dangling refs become warnings)
3. Run lint:
   ```bash
   oma scholar lint {input}
   # OR for fetched sidecars:
   oma scholar lint --lenient {input}
   ```
4. Read output; group findings by severity (error/warning)
5. Report file:line for each issue with the rule violated and the fix
6. Offer to apply fixes if the user agrees

## Mode 3: Review

Generate a peer-review sidecar covering what claims need stronger evidence, what assumptions are unstated, and what limitations are missing.

### Steps

1. Read the source paper (or its existing sidecar)
2. For each statement in the paper, evaluate:
   - Is `supported_by` evidence sufficient?
   - Is `depends_on` assumption explicit?
   - Are limitations acknowledged?
3. Produce review sidecar at `{base}.review.knows.yaml` with:
   - top-level `coverage`: `{statements: key_claims_and_limitations, evidence: partial}`
   - `statements` capturing reviewer assertions, each with `statement_type: review_comment` (or `limitation`)
   - `relations` linking review comments to original paper statements via `predicate: critiques` or `extends`
4. Lint and report (same as Generate)

## Mode 4: Analyze / Query

Answer natural-language questions over an existing sidecar.

### Steps

1. Locate sidecar: local path or `oma scholar get "{record_id}"` from knows.academy
2. For token efficiency, prefer **partial fetch** when only a subset is needed.
   The query param is **`section` (singular)**; allowed values are `statements|evidence|relations|artifacts|citation`:
   - "What claims does this paper make?" -> `?section=statements`
   - "What are the experimental results?" -> `?section=evidence`
   - "Who do they cite?" -> `?section=citation` (or `?section=artifacts` filtered by `role: cited`)
3. Parse YAML, traverse structure to answer
4. Cite IDs (`stmt:...`, `ev:...`) so the user can verify

## Mode 5: Compare

Structural diff between two sidecars.

### Steps

1. Load both sidecars (local or remote)
2. Build comparison table:
   - **Claims overlap**: shared `statement_type` themes
   - **Method differences**: same dataset? different metrics?
   - **Evidence quality**: relations-per-statement ratio for each
   - **Citation overlap**: shared `artifacts` with `role: cited`
3. Output a markdown table; cite specific IDs from each paper
4. Optional: surface contradicting claims (search for `predicate: contradicts` patterns)

## Mode 6: Remote (with OpenAlex fallback)

Search and fetch from knows.academy first, then fall back to OpenAlex when the
paper isn't in the (2026-only) knows.academy index.

### Recommended path: `oma scholar` (handles cascade automatically)

```bash
# Hybrid search
oma scholar search "<query>"

# Cross-source resolve (decides which source has the right paper)
oma scholar resolve "<title>"

# Get specific record (knows id, OpenAlex W-id, or DOI)
oma scholar get [--section <one>] "<id>"
```

See `resources/fallback-providers.md` for full cascade design.

### Manual cascade (raw curl)

1. **knows.academy search**: `curl -s "{base}/api/proxy/search?q={query}"`
2. If hits → fetch sidecar:
   - Full: `/api/proxy/sidecars/{record_id}`
   - Partial: `/api/proxy/partial?record_id={id}&section=statements|evidence|relations|artifacts|citation`
3. If no hits OR clearly wrong paper (cross-source title similarity < 0.7) →
   **OpenAlex fallback**:
   - `curl -s "https://api.openalex.org/works?search={query}"` (anonymous OK)
   - Or by DOI: `https://api.openalex.org/works/doi:{doi}`
   - Reconstruct abstract from `abstract_inverted_index`
4. If user wants a sidecar from the OpenAlex result → transition to **Mode 1 Generate**
   using the abstract as the source text. The local sidecar will not be on
   knows.academy but is structurally identical.
5. **Stats** (health check): `curl -s "{base}/api/proxy/jobs/stats"`
6. If user asks for analysis after fetch, transition to **Mode 4** (Analyze)

## Error Recovery

| Error | Recovery |
|-------|----------|
| `oma scholar` not found | Run `oma install` to install / update the CLI |
| Lint script reports many errors | Fix top error first; re-run; cascading errors often resolve together |
| Remote API timeout | Retry once; if still failing, check `/api/proxy/jobs/stats` |
| Empty search results | Broaden query; remove quotes; try keyword-only |
| YAML parse error after Generate | Re-emit with stricter quoting on string values containing `:` `#` `&` `*` |
| Source PDF has no text layer | Chain via `oma-pdf` with hybrid OCR mode first |
| OpenAlex 403/429 | Tell user to set `OPENALEX_API_KEY`; see `setup-openalex.md` |
