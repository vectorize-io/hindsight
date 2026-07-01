---
name: oma-scholar
description: >
  Scholarly research companion using Knows sidecar spec (.knows.yaml). Generates,
  validates, reviews, queries, and compares structured research-paper sidecars,
  and fetches them from knows.academy. Use for academic literature search, survey
  synthesis, paper authoring assistance, and peer review with token-efficient
  claim/evidence/relation access.
---

# Scholar - Research Paper Sidecar Companion

## Scheduling

### Goal
Search, fetch, generate, validate, analyze, review, and compare scholarly paper sidecars using the Knows `.knows.yaml` spec for token-efficient research workflows.

### Intent signature
- User asks for academic literature search, sidecar generation, sidecar validation, paper claims/evidence summary, structural paper comparison, or peer review as sidecar.
- User references Knows, `.knows.yaml`, knows.academy, OpenAlex, claims, evidence, relations, or paper sidecars.

### When to use

- Reading research papers token-efficiently via Knows sidecars (~700 tokens for claims-only vs ~10K for full PDF)
- Generating `.knows.yaml` sidecars from your own paper drafts, LaTeX, or research notes
- Validating sidecar structure (rule-based) before sharing
- Producing peer reviews as sidecars
- Querying or summarizing existing sidecars
- Structurally comparing two papers (claims, methods, evidence)
- Searching/fetching sidecars from `knows.academy` (~50K papers indexed)

### When NOT to use

- General web search or non-academic content -> use `oma-search`
- Translating papers -> use `oma-translator`
- PDF parsing only (no sidecar) -> use `oma-pdf`
- Submitting sidecars back to knows.academy -> out of scope (host LLM only consumes/produces locally)
- Full peer-review workflow with editor system -> out of scope

### Expected inputs
- Paper, abstract, draft, LaTeX, research notes, sidecar file, DOI, OpenAlex ID, Knows record ID, or search query
- Desired mode: generate, validate, review, analyze, compare, or remote fetch
- Optional strictness, section filter, or CI behavior

### Expected outputs
- `.knows.yaml` sidecar, review sidecar, lint report, search/fetch result, natural-language analysis, or structural comparison
- Sidecars conforming to v0.9.0 / `paper@1` profile
- Validation status and warnings before sharing generated sidecars

### Dependencies
- `oma scholar` CLI subcommands
- knows.academy public API and OpenAlex fallback
- `resources/sidecar-spec.md`, API endpoints, OpenAlex setup, upstream cache, checklist, and execution protocol

### Control-flow features
- Branches by mode, source availability, Knows/OpenAlex coverage, strict vs lenient validation, and fetched section
- Reads/writes YAML sidecars and may call public APIs
- Avoids fabrication when source evidence is missing

## Structural Flow

### Entry
1. Identify mode and source artifact/query.
2. Resolve paper identity through Knows or OpenAlex when needed.
3. Load sidecar spec and mode-specific protocol.

### Scenes
1. **PREPARE**: Select mode and gather source or remote identifiers.
2. **ACQUIRE**: Fetch paper metadata, sidecar sections, or local source text.
3. **REASON**: Extract claims, evidence, relations, provenance, or comparison structure.
4. **ACT**: Generate, lint, review, analyze, compare, or fetch sidecar data.
5. **VERIFY**: Validate schema, enums, IDs, relations, and provenance.
6. **FINALIZE**: Return sidecar, report, summary, or comparison with caveats.

### Transitions
- If knows.academy lacks the paper, fall back to OpenAlex metadata/abstract.
- If generating a sidecar, run lint before sharing.
- If consuming third-party sidecars with dangling references, use lenient mode when appropriate.
- If source evidence is absent, omit fields instead of guessing.

### Failure and recovery
- If remote API times out, retry or use OpenAlex fallback.
- If YAML fails parsing, fix indentation and scalar types.
- If relation density or orphan statements warn, add supported-by relations when source evidence supports them.

### Exit
- Success: requested sidecar operation completes with validation status.
- Partial success: missing metadata, fallback source, or validation warnings are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Select mode | `SELECT` | Generate/Validate/Review/Analyze/Compare/Remote |
| Read paper or sidecar | `READ` | Source files or YAML |
| Request remote data | `REQUEST` | Knows/OpenAlex APIs |
| Infer claims/evidence/relations | `INFER` | Sidecar generation/analysis |
| Write sidecar | `WRITE` | `.knows.yaml` outputs |
| Validate sidecar | `VALIDATE` | `oma scholar lint` |
| Report result | `NOTIFY` | Summary or lint report |

### Tools and instruments
- `oma scholar search|resolve|get|lint`
- Knows public API, OpenAlex fallback, sidecar spec, checklist

### Canonical command path
```bash
oma scholar search "<query>"
oma scholar resolve "<title-or-doi>"
oma scholar get "<record-id-or-doi>"
oma scholar lint "<paper.knows.yaml>"
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Paper drafts, sidecar YAML, review sidecars |
| `NETWORK` | knows.academy and OpenAlex APIs |
| `PROCESS` | `oma scholar` CLI and lint |
| `USER_DATA` | User-provided paper content and research notes |

### Preconditions
- Mode and source are identifiable.
- Spec rules are available for generation or validation.

### Effects and side effects
- May create local sidecar or review sidecar files.
- May query public scholarly APIs.
- Does not submit sidecars back to knows.academy.

### Guardrails

1. **Target spec is v0.9.0 / `paper@1` profile**: verified against production sidecars from knows.academy; see `resources/sidecar-spec.md`
2. **Host LLM generates sidecars**: never shell out to `anthropic` SDK or external LLM CLI; this skill runs inside an agent
3. **Anti-fabrication**: if DOI/venue/year is not visible in source, **omit the key entirely**; never write `doi: TODO` or guess
4. **Top-level metadata**: `title`, `authors`, `venue`, `year` live at the top level (no `metadata` wrapper)
5. **Field names are exact**: `statement_type`, `evidence_type`, `predicate`, `artifact_type` (not `type`/`claim`)
6. **Provenance has SINGLE actor**: `provenance.actor` is one object, NOT a `provenance.actors` array
7. **Confidence is an object**: `{claim_strength: ..., extraction_fidelity: ...}`, both from `high|medium|low`
8. **Coverage is an object**: `coverage.statements` (4-value enum) + `coverage.evidence` (3-value enum)
9. **Closed enums**: actor `tool|person|org` (never `ai`/`llm`/`model`); artifact role `subject|supporting|cited`; predicates in present tense
10. **Numbers unquoted**: `value: 22`, never `value: '22'`
11. **Relation density**: average ≥1.5 relations per statement; every claim needs `supported_by` evidence (lint warns when ratio is below; orphan statements warned per-id)
12. **ID format**: descriptive kebab-case with prefix: `stmt:privacy-budget-tradeoff`, `ev:cifar10-accuracy-table`, `art:paper`
13. **Validate before sharing**: run `oma scholar lint` after Generate
14. **Remote API has no auth**: `https://knows.academy/api/proxy/*` is public; do not invent auth headers
15. **Partial fetch param is `section` (singular)**: fixed enum `statements|evidence|relations|artifacts|citation`
16. **OpenAlex key is optional**: metadata enrichment only; gracefully degrade when missing
17. **Sidecar content stays English**: schema fields, IDs, statement text follow upstream convention; user-facing responses follow `oma-config.yaml` `language`
18. **Spec drift awareness**: our local rules track v0.9.0 production behavior, which differs from the upstream `knows.md` natural-language description; refresh `resources/upstream-spec-cache.md` periodically

### Modes

| Mode | Trigger | Output |
|------|---------|--------|
| **Generate** | "create sidecar from this paper / abstract / draft", "generate `.knows.yaml`" | `{paper}.knows.yaml` (host LLM emits, then `oma scholar lint` validates) |
| **Validate** | "lint this sidecar", "validate `.knows.yaml`" | Pass/fail report with file:line issues |
| **Review** | "peer review this paper as sidecar" | `{paper}.review.knows.yaml` |
| **Analyze** | "summarize this sidecar", "what claims does it make?" | Natural-language answer |
| **Compare** | "compare paper A and paper B structurally" | Diff table (claims/methods/evidence) |
| **Remote** | "find papers on X", "fetch sidecar :id", "get claims only for :id" | Search results / sidecar payload |

### Provider Fallback (knows.academy → OpenAlex)

`knows.academy` currently indexes **only 2026 papers** (~50K, mostly arXiv). For
older or non-2026 papers (Transformer 2017, BERT 2018, classics, journals),
the skill automatically falls back to **OpenAlex** for metadata and abstract.

Use the `oma scholar` CLI subcommands:

```bash
# Hybrid search: knows first, OpenAlex fallback
oma scholar search "vision language action"

# Cross-source resolve: figures out which source has the right paper
oma scholar resolve "Attention Is All You Need"

# Get by id (knows record_id, OpenAlex W-id, or DOI)
oma scholar get "10.48550/arXiv.1706.03762"
```

When OpenAlex returns the answer (knows.academy lacks the paper), use the
returned abstract as input to **Mode 1 Generate** to produce a local sidecar.

### How to Execute

Follow `resources/execution-protocol.md` step by step for the selected mode.

### Quick Reference

### Search (knows + auto OpenAlex fallback)
```bash
oma scholar search "diffusion super resolution"
oma scholar search --year-min 2024 "vision language action"
```

### Find one specific paper
```bash
oma scholar resolve "Attention Is All You Need"
# returns top hit from each source + recommendation
```

### Fetch a sidecar or work
```bash
# knows.academy full sidecar
oma scholar get "knows:generated/reconvla/1.0.0"

# Partial fetch (claims only, ~700 tokens, 93% reduction vs PDF)
oma scholar get --section statements "knows:generated/reconvla/1.0.0"

# By DOI or OpenAlex W-id (works regardless of knows.academy availability)
oma scholar get "10.48550/arXiv.1706.03762"
```

When knows.academy is unreachable, `get knows:...` automatically falls back
to OpenAlex by extracting the slug from the record_id. The result is marked
with `fallback: "openalex"` and contains metadata + abstract, useful for
running Mode 1 Generate locally.

### Validate
```bash
# Strict mode for own Generate output (default)
oma scholar lint paper.knows.yaml

# Lenient mode for third-party / fetched sidecars
oma scholar lint --lenient remote.knows.yaml

# Treat warnings as failures (CI mode)
oma scholar lint --fail-on-warning paper.knows.yaml
```

About **47% of knows.academy-served sidecars contain at least one dangling
cross-reference** (typo in `subject_ref`/`object_ref`, measured across 15
production samples). Use `--lenient` when consuming third-party records so
these surface as warnings rather than blocking errors.

### Raw API (when CLI is unavailable)
```bash
curl -s "https://knows.academy/api/proxy/search?q=..."
curl -s "https://knows.academy/api/proxy/sidecars/<encoded-id>"
curl -s "https://knows.academy/api/proxy/partial?record_id=<id>&section=statements"
curl -s "https://knows.academy/api/proxy/jobs/stats"   # platform health
```

### Configuration

Project-specific settings: `config/scholar-config.yaml`

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `[ERROR] *.value: numeric value '22' is quoted` | Remove quotes: `value: '22'` -> `value: 22` |
| `[ERROR] provenance.actor.type: 'ai' is not allowed` | Change to `tool`, `person`, or `org` |
| `[ERROR] *.type: use \`statement_type\` instead of \`type\`` | Rename `type` -> `statement_type` (or `evidence_type`/`predicate`/`artifact_type`) |
| `[ERROR] provenance.actors: v0.9 spec uses singular \`actor\`` | Replace `actors: [{...}]` array with `actor: {...}` object |
| `[ERROR] *.object_ref: reference 'X' does not match any defined id` | Fix the `subject_ref`/`object_ref` to point to a real id, OR use `--lenient` if consuming third-party data |
| `[WARN] relations: avg relations/statement is N.NN (target ≥ 1.5)` | Add more `supported_by`/`depends_on` relations |
| `[WARN] statements: only N statements; most papers warrant ≥ 8` | Expected when generating from abstract only; full-paper Generate should hit 15+ |
| `[WARN] *.predicate: past-tense '...' is suspicious` | Switch to present tense (`evaluated_on` -> `evaluates_on`) |
| Remote API returns empty results | Try broader query; check `/api/proxy/jobs/stats`; CLI auto-falls-back to OpenAlex |
| `knows.academy search failed: fetch failed` (stderr) | Platform timeout; fallback to OpenAlex is automatic; retry later for sidecars |
| OpenAlex 403/429 | Set `OPENALEX_API_KEY` (see `resources/setup-openalex.md`) |
| YAML won't parse | Check indentation; numbers/booleans must be unquoted; strings with `:` need quotes |

## References

- Execution steps: `resources/execution-protocol.md`
- Sidecar spec rules: `resources/sidecar-spec.md`
- API endpoints: `resources/api-endpoints.md`
- OpenAlex setup: `resources/setup-openalex.md`
- Upstream spec snapshot: `resources/upstream-spec-cache.md`
- Post-generation checklist: `resources/checklist.md`
- CLI subcommands: `oma scholar search|resolve|get|lint` (implementation under `cli/commands/scholar/`)
- Context loading: `../_shared/core/context-loading.md`
- Quality principles: `../_shared/core/quality-principles.md`
- i18n rules: `../../rules/i18n-guide.md`
