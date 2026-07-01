# Upstream Spec Cache

This file is a **snapshot** of the canonical Knows skill description.
Source: `https://knows.academy/api/proxy/skill/knows.md`

## How to refresh

```bash
curl -s https://knows.academy/api/proxy/skill/knows.md \
  > .agents/skills/oma-scholar/resources/upstream-spec-cache.md.new
diff .agents/skills/oma-scholar/resources/upstream-spec-cache.md \
     .agents/skills/oma-scholar/resources/upstream-spec-cache.md.new
```

If the diff shows meaningful changes, update:
1. This file with the new content
2. `sidecar-spec.md` if rules changed
3. `SKILL.md` if mode descriptions changed

Recommend refreshing every 1-2 weeks until the upstream stabilizes.

---

## Snapshot (captured 2026-04-25)

```markdown
# Knows Sidecar Skill — Complete Reference

## Overview

Knows is a structured YAML companion specification for research papers that enables LLM agents to access claims, evidence, and relations directly. A KnowsRecord is a schema-validated sidecar file that sits alongside PDFs.

## Core Modes

**Generate**: Create `.knows.yaml` from paper text, LaTeX, or research ideas
**Validate**: Run structural checks via `scripts/lint.py`
**Review**: Generate peer reviews as sidecars
**Analyze/Query**: Summarize or answer questions from existing sidecars
**Compare**: Diff two papers structurally
**Remote**: Search/download sidecars from knows.academy platform

## Critical Rules for Generation

### Field Naming (Copy Exactly)
- Statements: `statement_type` (not `type` or `claim`)
- Evidence: `evidence_type` (not `type`)
- Relations: `predicate` (not `type`)
- Artifacts: `artifact_type` (not `type`)
- Actors: `type: tool|person|org` (never `ai`, `llm`, `model`)

### Value Constraints
- Numbers unquoted: `value: 22` not `value: '22'`
- Actor origin: `origin: machine` for AI-generated, `author` for human curation
- Confidence: `high|medium|low` only
- Coverage statements: `exhaustive|main_claims_only|key_claims_and_limitations|partial`
- Artifact roles: `subject|supporting|cited` only

### Anti-Fabrication
"If the exact DOI is not visible in the PDF text, omit the `doi` key entirely. Do NOT write `doi: TODO`." Same principle applies to venue and year — omit unknown fields rather than placeholder them.

### Relation Wiring
Every statement needs ≥1.5 relations per statement on average. Minimum patterns:
- Claims: `supported_by` evidence, optionally `depends_on` assumptions
- Evidence: must be `object_ref` of at least one relation
- Methods: `evaluates_on` dataset, `implements` repo, or `documents` paper

### ID Format
Use descriptive kebab-case with prefix: `stmt:privacy-budget-tradeoff`, `ev:cifar10-accuracy-table`, `rel:ablation-supports-claim` — never numbered IDs like `stmt:c1`.

## Post-Generation Checklist

1. Verify statement count (complex papers need 15+)
2. Wire relations systematically per required patterns
3. Run sanitize if YAML won't parse: `python3 scripts/sanitize.py`
4. Lint validation: `python3 scripts/lint.py` until 0 errors
5. Verify metadata: `python3 scripts/verify_metadata.py` (with `--auto-enrich` if DOI missing)

## Common Mistakes

| Error | Wrong | Correct |
|---|---|---|
| Actor type | `type: ai` | `type: tool` |
| Observation value | `'22'` | `22` |
| Artifact field | `type: paper` | `artifact_type: paper` |
| Wrong tense | `evaluated_on` | `evaluates_on` |
| Fabricated DOI | `doi: "TODO"` | Omit entirely |
| Missing metric | `qualitative_value: "..."` | Add `metric: "name"` |

## Dependencies

- **Always available**: YAML template, JSON Schema, generation prompt
- **For lint**: `pip install pyyaml jsonschema`
- **For CLI**: `pip install knows-sidecar`
- **For LLM generation**: `pip install anthropic`
- **For verify**: Free OpenAlex/CrossRef API (set `OPENALEX_API_KEY` in `~/.claude/.env`)
```

---

## Local Adaptations

Where our local skill diverges from upstream (recorded for spec drift tracking):

1. **Generation engine**: upstream suggests `pip install anthropic`; we use the host LLM directly (no subprocess SDK call). Cost and key savings.
2. **Validation tooling**: upstream references `pip install knows-sidecar`; that package is **not yet on PyPI** (verified 2026-04-25). We ship rule-based validation as the `oma scholar lint` CLI subcommand (TypeScript, no Python dependency).
3. **JSON Schema host unreachable**: production sidecars reference `https://knows.dev/schema/record-0.9.json` but the host's HTTPS port times out. We cannot redistribute or fetch the schema, so lint encodes empirically-derived rules instead.
4. **Submission**: not implemented. This skill is read/generate-only, not a publisher.

## v0.9.0 Production Spec: Differences From Upstream `knows.md`

The upstream `knows.md` document above describes a simplified shape; production
sidecars served by `knows.academy/api/proxy/sidecars/*` follow the v0.9.0 schema
which differs significantly. **Our `sidecar-spec.md` reflects v0.9.0 production**.
Key differences:

| Field | Upstream `knows.md` | v0.9.0 Production |
|-------|---------------------|-------------------|
| Title/authors location | `metadata.title`, `metadata.authors` | top-level `title`, `authors`, `venue`, `year` (no `metadata` wrapper) |
| Provenance attribution | `provenance.actors[]` array | `provenance.actor` (single object) |
| Confidence shape | string `high\|medium\|low` | object `{claim_strength, extraction_fidelity}` |
| Coverage shape | single string at `provenance.coverage` | top-level `coverage` object with `statements` + `evidence` keys |
| Coverage evidence enum | (not specified) | `exhaustive` \| `key_evidence_only` \| `partial` |
| Partial fetch param | (not specified) | **`section=`** (singular), enum `statements\|evidence\|relations\|artifacts\|citation` |

## v0.9.0 Production-Only Fields

Not mentioned in upstream `knows.md` but present in every production sidecar:

- `$schema`: URL of the JSON Schema (currently unreachable but referenced)
- `knows_version`: e.g., `"0.9.0"`
- `record_id`: `knows:generated/{slug}/{version}` for published records
- `profile`: e.g., `"paper@1"`
- `subject_ref`: points to the artifact representing the paper itself
- `summary`: one-paragraph overview
- `license`: e.g., `"CC-BY-4.0"`
- `actions`: list (typically empty for paper profile)
- `version`: `{spec, record, source}`
- `freshness`: `{as_of, update_policy}`

## Statement Extra Fields (v0.9.0)

Not mentioned in upstream `knows.md`:

- `modality`: `empirical` \| `theoretical` \| `descriptive`
- `about_ref`: what the statement is about (usually the subject artifact)
- `status`: observed value `asserted`
- `source_anchors`: list of `{representation_ref, locator_type, locator}` pointers
- `provenance`: per-statement provenance block (same shape as top-level)

## Artifact Extra Fields (v0.9.0)

- `identifiers`: `{url, doi, ...}` (omit unknown keys)
- `representations`: list of `{id: rep:..., media_type, locator: {type, value}}`

## Predicate Vocabulary (Verified in Production)

Across 330 relations in 15 production sidecars, the following predicates
appear (descending frequency):

```
supported_by, depends_on, evaluates_on, limited_by, documents,
uses, challenged_by, cites, implements, used_by, defines
```

Other predicates from upstream natural-language doc, valid per the spec but
not yet observed in our sample: `extends`, `contradicts`, `critiques`,
`generalizes`, `specializes`, `introduces`, `refutes`, `replicates`.

## Vocabulary Survey (15 production sidecars / 2026-04-25)

Field enums verified by frequency, all production AI-generated (`origin: machine`):

| Field | Verified values |
|-------|-----------------|
| `knows_version` | `0.9.0` (only one observed) |
| `profile` | `paper@1` (only one observed) |
| `provenance.method` | `extraction` (only one observed) |
| `coverage.statements` | `exhaustive`, `main_claims_only` |
| `coverage.evidence` | `key_evidence_only` |
| `statement_type` | `claim`, `method`, `limitation`, `assumption`, `definition`, `question` |
| `statement.modality` | `descriptive`, `empirical`, `theoretical` |
| `statement.status` | `asserted` (only value across 252 statements) |
| `evidence_type` | `table_result`, `figure`, `proof`, `observation`, `experiment_run`, `case_study`, `citation_backed` |
| `artifact_type` | `paper`, `dataset`, `benchmark`, `repository`, `model`, `software`, `other` |
| `artifact.role` | `subject`, `supporting`, `cited` |
