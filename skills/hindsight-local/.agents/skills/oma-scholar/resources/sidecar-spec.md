# Knows Sidecar Spec: Generation Rules (v0.9.0)

This is the v0.9.0 record shape verified against `knows.academy` production sidecars
(JSON Schema id: `https://knows.dev/schema/record-0.9.json`, profile `paper@1`).
For the upstream natural-language description see `upstream-spec-cache.md`.

## Top-level Structure (paper@1 profile)

```yaml
$schema: "https://knows.dev/schema/record-0.9.json"
knows_version: "0.9.0"
record_id: "knows:generated/{slug}/1.0.0"   # only for published records; omit for local drafts
profile: "paper@1"
subject_ref: "art:paper"                      # points to the artifact representing the paper itself

# Top-level metadata (NOT inside a `metadata` block)
title: "..."
authors: ["..."]
venue: "..."          # only if visible in source
year: 2026            # only if visible in source
summary: "..."

# Coverage is an object, not a single value
coverage:
  statements: exhaustive | main_claims_only | key_claims_and_limitations | partial
  evidence: exhaustive | key_evidence_only | partial

license: "CC-BY-4.0"   # if known

artifacts: [...]
statements: [...]
evidence: [...]
relations: [...]
actions: []            # usually empty for paper profile

# Provenance has SINGLE actor (object), not actors (array)
provenance:
  origin: machine | author
  actor:
    name: "knows-gen"
    type: tool         # tool | person | org   (NEVER ai/llm/model)
    version: "0.9.0"
  generated_at: "2026-04-25T00:00:00Z"
  method: extraction

version:
  spec: "0.9.0"
  record: "1.0.0"
  source: original

freshness:
  as_of: "2026-04-25T00:00:00Z"
  update_policy: versioned
```

## Statement Shape

```yaml
- id: stmt:descriptive-kebab-case
  statement_type: claim    # claim | method | limitation | assumption | definition | question
  modality: empirical      # descriptive | empirical | theoretical
  text: "..."
  about_ref: art:paper     # what the statement is about (usually the subject artifact)
  status: asserted
  source_anchors:
    - representation_ref: rep:paper-pdf
      locator_type: section
      locator: "Section 5"
  confidence:
    claim_strength: high       # high | medium | low
    extraction_fidelity: high  # high | medium | low
  provenance:
    origin: machine
    actor:
      name: "knows-gen"
      type: tool
    generated_at: "..."
```

### Statement Type Frequencies (252 statements / 15 papers)

| Type | Frequency | Use |
|------|-----------|-----|
| `claim` | 117 (46%) | Headline assertion of the paper |
| `method` | 58 (23%) | Procedure, technique, or pipeline |
| `limitation` | 31 (12%) | Acknowledged limit or threat to validity |
| `assumption` | 18 (7%) | Precondition for the claim/method |
| `definition` | 18 (7%) | Term or concept introduction |
| `question` | 10 (4%) | Open research question / RQ |

## Evidence Shape

```yaml
- id: ev:descriptive-kebab-case
  evidence_type: table_result   # see frequencies below
  summary: "..."
  source_anchors: [...]
  provenance: {...}
```

### Evidence Type Frequencies (138 evidence / 15 papers)

| Type | Frequency | Use |
|------|-----------|-----|
| `table_result` | 64 (46%) | Numeric results in a table |
| `figure` | 34 (25%) | Graphical/diagram-based result |
| `proof` | 15 (11%) | Theoretical/mathematical proof |
| `observation` | 11 (8%) | Qualitative observation |
| `experiment_run` | 6 (4%) | Single experimental run/trial |
| `case_study` | 4 (3%) | Detailed walkthrough of a case |
| `citation_backed` | 4 (3%) | Cited from another paper |

## Artifact Shape

```yaml
- id: art:paper
  artifact_type: paper       # see frequencies below
  role: subject              # subject | supporting | cited
  title: "..."
  identifiers:
    url: "https://arxiv.org/abs/..."
    doi: "..."     # OMIT entirely if not visible in source
  representations:
    - id: rep:paper-pdf
      media_type: application/pdf
      locator:
        type: path | url
        value: "..."
```

### Artifact Type Frequencies (50 artifacts / 15 papers)

| Type | Frequency | Use |
|------|-----------|-----|
| `paper` | 15 (30%) | The subject paper or a cited paper |
| `dataset` | 12 (24%) | Training/eval dataset |
| `benchmark` | 10 (20%) | Standard benchmark suite |
| `repository` | 6 (12%) | Code repository (GitHub etc.) |
| `model` | 4 (8%) | Pre-trained or released model |
| `software` | 2 (4%) | Library/toolkit |
| `other` | 1 (2%) | Catch-all |

### Artifact Role Frequencies (50 / 15 papers)

| Role | Frequency | Use |
|------|-----------|-----|
| `cited` | 18 (36%) | Referenced as prior work |
| `supporting` | 17 (34%) | Used by the paper (dataset, code, model) |
| `subject` | 15 (30%) | The paper itself (always 1 per record) |

## Relation Shape

```yaml
- id: rel:descriptive-kebab-case
  predicate: supported_by  # see "Predicates" section below
  subject_ref: stmt:...
  object_ref: ev:... | stmt:... | art:...
```

## Field Naming (copy exactly)

| Concept | Correct field | Wrong |
|---------|---------------|-------|
| Statement category | `statement_type` | `type`, `claim` |
| Evidence category | `evidence_type` | `type` |
| Relation verb | `predicate` | `type`, `relation_type` |
| Artifact category | `artifact_type` | `type` |
| Actor category | `type` (with closed enum) | (none) |
| Top-level metadata | `title`, `authors`, `venue`, `year` (no wrapper) | `metadata.title` etc. |
| Provenance attribution | `provenance.actor` (single object) | `provenance.actors` (array) |
| Confidence | object with `claim_strength` + `extraction_fidelity` | bare string |

## Value Constraints

| Field | Allowed values |
|-------|----------------|
| `actor.type` | `tool` \| `person` \| `org` (never `ai`/`llm`/`model`) |
| `provenance.origin` | `machine` (AI-generated) \| `author` (human curation) |
| `confidence.claim_strength` | `high` \| `medium` \| `low` |
| `confidence.extraction_fidelity` | `high` \| `medium` \| `low` |
| `coverage.statements` | `exhaustive` \| `main_claims_only` \| `key_claims_and_limitations` \| `partial` |
| `coverage.evidence` | `exhaustive` \| `key_evidence_only` \| `partial` |
| `artifacts[].role` | `subject` \| `supporting` \| `cited` |
| `statement.status` | `asserted` (the only value observed across 252 production statements) |
| `statement.modality` | `descriptive` \| `empirical` \| `theoretical` |
| `statement_type` | `claim` \| `method` \| `limitation` \| `assumption` \| `definition` \| `question` |
| `evidence_type` | `table_result` \| `figure` \| `proof` \| `observation` \| `experiment_run` \| `case_study` \| `citation_backed` |
| `artifact_type` | `paper` \| `dataset` \| `benchmark` \| `repository` \| `model` \| `software` \| `other` |

## Numeric Values

```yaml
# Correct
- value: 22
- accuracy: 0.945

# Wrong
- value: '22'        # never quote numbers
- accuracy: "0.945"
```

## Anti-Fabrication

If a field is not visible in source, **omit the key entirely**.

```yaml
# Wrong
doi: TODO
venue: TBD

# Correct (just omit the keys)
title: "Paper Title"
authors: ["A. Author"]
```

Applies to: `doi`, `venue`, `year`, ORCIDs, GitHub URLs, dataset URLs.

## ID Format

Descriptive kebab-case with type prefix. **Never** use opaque IDs.

```yaml
# Correct
- id: stmt:standard-transformer-fails-unseen-tokens
- id: ev:lemma-41-contraction-proof
- id: rel:collapse-causes-failure
- id: art:paper
- id: rep:paper-pdf

# Wrong
- id: stmt:c1
- id: ev:001
- id: rel:r-23
```

## Predicates

Verified frequencies across 330 relations in 15 production sidecars:

| Predicate | Direction | Use |
|-----------|-----------|-----|
| `supported_by` | claim → evidence | most common; primary evidence wiring |
| `depends_on` | claim/method → assumption | preconditions |
| `evaluates_on` | method → dataset/benchmark | empirical evaluation target |
| `limited_by` | claim → limitation | acknowledged limit |
| `documents` | statement → artifact (paper) | references the source paper |
| `uses` | method → artifact (model/software) | active usage |
| `used_by` | artifact → method | reverse direction of `uses` |
| `challenged_by` | claim → counter-claim | opposed/debated |
| `cites` | artifact → cited paper | citation graph |
| `implements` | method → repository | code implementation |
| `defines` | statement → definition | term introduction |

Less common predicates valid per upstream natural-language doc (not yet observed):

```
extends, contradicts, critiques, generalizes,
specializes, introduces, refutes, replicates
```

```yaml
# Wrong tense (warning)
predicate: evaluated_on

# Correct
predicate: evaluates_on

# Passive forms accepted in production
predicate: supported_by
predicate: used_by
predicate: challenged_by
```

## Relation Wiring (CRITICAL)

Average **≥1.5 relations per statement**. Minimum patterns:

| Subject | Predicate | Object |
|---------|-----------|--------|
| Claim (statement) | `supported_by` | Evidence |
| Claim | `depends_on` | Assumption (statement) |
| Method (statement_type=method) | `implements` | Repository (artifact) |
| Method | `uses` | Model/Dataset (artifact) |
| Limitation | `limited_by` | Specific cause (statement/evidence) |

## Common Mistakes Cheatsheet

| Wrong | Correct |
|-------|---------|
| `metadata: {title: ...}` | top-level `title:` |
| `provenance.actors: [...]` | `provenance.actor: {...}` (single) |
| `confidence: high` | `confidence: {claim_strength: high, extraction_fidelity: high}` |
| `coverage: exhaustive` | `coverage: {statements: exhaustive, evidence: ...}` |
| `type: ai` | `type: tool` |
| `value: '22'` | `value: 22` |
| `type: paper` (on artifact) | `artifact_type: paper` |
| `type: claim` (on statement) | `statement_type: claim` |
| `evaluated_on` | `evaluates_on` |
| `doi: TODO` | omit `doi` key entirely |
| `id: stmt:c1` | `id: stmt:descriptive-name` |

## Statement Density Guidance

- Short methods paper: 8-12 statements
- Standard ML paper: 12-18 statements
- Complex/long paper: **15+ required**
- Survey/review paper: 20+ recommended

If under-extracted, re-read:
- Limitations sections
- Discussion/conclusions
- Footnotes and ablation studies
