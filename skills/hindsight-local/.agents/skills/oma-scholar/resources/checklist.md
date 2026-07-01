# Post-Generation Checklist (v0.9.0)

Run this after Mode 1 (Generate) or Mode 3 (Review) before reporting done.

## Top-Level Structure

- [ ] `title` is set at the **top level** (not `metadata.title`)
- [ ] `authors` is a top-level list of strings
- [ ] `venue`, `year`, `doi` keys present **only** if visible in source (no TODO/TBD)
- [ ] `knows_version` set (e.g., `"0.9.0"`)
- [ ] `profile` set (e.g., `"paper@1"`)
- [ ] `subject_ref` points to an `art:` artifact id
- [ ] `coverage` is an object with `statements` and `evidence` keys (each from its own enum)
- [ ] `provenance` is present with single `actor` object (not `actors` array)
- [ ] `version` block has `spec`, `record`, `source`
- [ ] `freshness` block has `as_of`, `update_policy`

## Provenance

- [ ] `provenance.origin` is `machine` or `author`
- [ ] `provenance.actor.type` is `tool`, `person`, or `org` (never `ai`/`llm`/`model`)
- [ ] `provenance.actor.name` is set
- [ ] `provenance.method` describes how the sidecar was produced (e.g., `extraction`)
- [ ] `provenance.generated_at` is a valid ISO timestamp

## IDs

- [ ] All IDs use descriptive kebab-case (no `stmt:c1`, `ev:001`)
- [ ] Type prefixes correct: `stmt:`, `ev:`, `rel:`, `art:`, `rep:`
- [ ] No duplicate IDs across the document

## Field Names

- [ ] Statements use `statement_type` (not `type` or `claim`)
- [ ] Evidence uses `evidence_type`
- [ ] Relations use `predicate`
- [ ] Artifacts use `artifact_type`

## Statement Internals

- [ ] Each statement has `statement_type` from: `claim`, `method`, `limitation`, `assumption`, `definition`, `question`
- [ ] Each statement has `modality` from: `descriptive`, `empirical`, `theoretical`
- [ ] Each statement has `status` (commonly `asserted`)
- [ ] `confidence` is an object: `{claim_strength: ..., extraction_fidelity: ...}`, both from `high|medium|low`
- [ ] `source_anchors` reference a valid `representation_ref` (e.g., `rep:paper-pdf`)

## Values

- [ ] Numbers unquoted (`value: 22`, not `value: '22'`)
- [ ] `coverage.statements` from: `exhaustive`, `main_claims_only`, `key_claims_and_limitations`, `partial`
- [ ] `coverage.evidence` from: `exhaustive`, `key_evidence_only`, `partial`
- [ ] `artifacts[].role` from: `subject`, `supporting`, `cited`
- [ ] Predicates use present tense (`evaluates_on`, not `evaluated_on`)

## Relations

- [ ] Every statement has at least one relation (incoming or outgoing)
- [ ] Every claim has a `supported_by` relation pointing to evidence
- [ ] Average relations-per-statement ratio ≥ 1.5
- [ ] Methods have at least one of: `implements`, `uses`, `evaluates_on`, `documents`
- [ ] No dangling references; every `subject_ref` and `object_ref` points to an existing id

## Density

- [ ] Statement count appropriate for paper length (complex papers ≥ 15)
- [ ] Limitations and discussion mined for additional statements
- [ ] No important section ignored (abstract, methods, results, discussion, limits)

## Lint

- [ ] `oma scholar lint` returns 0 errors
- [ ] Warnings reviewed (recommended-key warnings are usually acceptable for local drafts)

## Anti-Fabrication

- [ ] No fabricated DOIs, ORCIDs, or URLs
- [ ] No "TODO", "TBD", "N/A" placeholder values
- [ ] All quoted statement text is paraphrased or quoted accurately from source
- [ ] No invented author names or affiliations

## Final Report to User

Include:
- Output file path
- Counts: `statements`, `evidence`, `relations`, `artifacts`
- Ratio: relations/statements (target ≥ 1.5)
- Lint status: clean / N warnings / N errors
- Fields explicitly omitted due to anti-fabrication (e.g., "DOI not visible; please paste if you have it")
