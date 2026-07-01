---
name: oma-market
description: "Market research skill for pain-point extraction, trend detection, competitor positioning, and discovery across community sources (Reddit, HN, Bluesky, Mastodon, GitHub Issues, web). Routes via oma-search transport, deterministic CLI compute, intent-auto SWOT/Porter's 5F/PESTEL frameworks. Use for market research, pain point analysis, trend detection, competitor research, user complaints, voice-of-customer, 시장조사, 사용자 페인, 트렌드, 경쟁구도."
---

# Market Research Agent - Community Signal Intelligence

## Scheduling

### Goal
Classify user intent into pain / trend / competitor / discovery, fan-out to community sources via `oma search fetch`, score and cluster findings with deterministic CLI compute, auto-apply strategic frameworks, and emit a single LAW-compliant markdown brief.

### Intent signature
- User asks about pain points, user complaints, or voice-of-customer signals for a product or category.
- User asks what is trending, growing, or declining in a space this week or month.
- User asks how one product compares to another in community sentiment or positioning.
- User asks for discovery or exploratory market research on a topic.

### When to use
- Extracting real user pain points from community posts (Reddit, HN, GitHub Issues, Bluesky, Mastodon)
- Detecting trends in a product category over a time window (7d / 30d / 90d / 180d)
- Competitor sentiment analysis and SWOT positioning
- Open-ended discovery research across multiple sources

### When NOT to use
- General web research without market framing -> use oma-search directly
- Single-source queries only -> use `oma search fetch` standalone
- Delta tracking or trend velocity over time (v2 feature) -> defer
- Live dashboards or scheduled monitoring -> out of scope (v1 one-shot only)

### Expected inputs
- Topic string and optional `--intent pain|trend|competitor|discovery`
- Optional `--window 7d|30d|90d|180d` (default: `30d`)
- Optional `--sources <list>` to override defaults
- Optional `--vs <entity>` for competitor COMPARISON mode
- Optional `--frameworks auto|none|swot,5f,pestel`

### Expected outputs
- Single markdown brief at `.agents/results/market/{topic-slug}-{YYYYMMDD}.md`
- Badge first-line, `What we learned:` body opener (or COMPARISON title), engine footer
- No raw evidence dump; no Sources block; no em-dash; no `##` in body (framework/COMPARISON sections excepted)

### Dependencies
- `oma-search` for all fetches (`oma search fetch --only api`); never fetches directly
- Serena `trust-registry-cache` (read-only); Trust Registry labels inherited from oma-search
- `resources/intent-rules.md`, `resources/operator-packs/`, `resources/output-laws.md`

### Control-flow features
- Branches by classified intent, window, source availability, and env key presence
- detect-trap gate before harvest (exit 2 on broad/ambiguous topic, exit 4 on invalid)
- Paid sources (X, TikTok, Instagram, YouTube, Perplexity) auto-skip when env key absent
- Framework auto-toggle by intent (see Routes table)

## Structural Flow

### Entry
1. Run `oma market detect-trap "<topic>"` to preflight the query.
2. Classify or confirm intent from user prompt or `--intent` flag.
3. Select operator pack and framework set for the intent.

### Scenes
1. **PREPARE**: Parse topic and flags; run detect-trap; resolve intent, operator pack, window.
2. **ACT**: Build per-source `oma search fetch` URLs with operator pack query expansion.
3. **ACQUIRE**: Fan-out harvest via `oma market harvest` (parallel, per-source-limit 12, cache TTL 15m).
4. **VERIFY**: Score, fuse, and cluster candidates; validate JSON at each pipe stage.
5. **FINALIZE**: Render LAW-compliant markdown brief; run self-check; write to output path.

### Transitions
- If detect-trap exits 2 (REFUSE), surface reframe suggestion and halt.
- If all sources blocked, exit 2 with per-source diagnostics.
- If partial harvest failure, proceed; render annotates "coverage: N/M sources".
- If zero clusters, emit preview message and suggest wider window.
- If `--vs <entity>` flag is present, switch to COMPARISON template.

### Failure and recovery
- detect-trap exit 2: surface REFUSE reason and suggested reframe; do not proceed to harvest.
- Network timeout (exit 6): report and suggest `--window` reduction or `--no-cache`.
- Invalid JSON from any pipe stage: exit 4 with offending line in stderr.
- Render LAW self-check violation: strip or regenerate; exit 1 only if regeneration also fails.
- FS permission denied at write: exit 5.

### Exit
- Success: brief file written; first 50 lines previewed; engine footer present.
- Partial success: source failures and framework skips are explicit in footer and stderr.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Run detect-trap preflight | `VALIDATE` | Topic arg, trap pattern rules |
| Classify intent | `SELECT` | Intent rules, user flags |
| Select operator pack | `SELECT` | `resources/operator-packs/` |
| Fan-out harvest | `CALL_TOOL` | `oma market harvest` -> `oma search fetch` |
| Score candidates | `INFER` | Engagement weights, freshness, intent blends |
| Fuse and deduplicate | `INFER` | URL canonicalize, RRF k=60, author cap |
| Cluster by entity overlap | `INFER` | Overlap coefficient >= 0.4, MMR lambda=0.75 |
| Select frameworks | `SELECT` | Intent-to-framework toggle table |
| Render and self-check | `WRITE` | Output LAWs, framework templates |
| Write brief | `WRITE` | `.agents/results/market/` |
| Report preview | `NOTIFY` | First 50 lines of brief |

### Tools and instruments
- `oma market detect-trap` (preflight gate)
- `oma market harvest` (delegates to `oma search fetch --only api`)
- `oma market score` (engagement weights, log1p, intent blends)
- `oma market fuse` (URL canonical, RRF, diversity guard)
- `oma market cluster` (entity overlap, MMR)
- `oma market render` (md/json, LAW self-check, file write)

### Canonical command path
```bash
TOPIC="VS Code pain points"
oma market detect-trap "$TOPIC" \
  && oma market harvest "vscode (broken OR bug OR migrate OR quit OR slow)" \
       --sources reddit,hn,bluesky,mastodon,github-issues --window 30d \
       --operator-pack pain \
  | oma market score --intent pain \
  | oma market fuse \
  | oma market cluster \
  | oma market render --format md --intent pain --frameworks auto
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `NETWORK` | Community sources via `oma search fetch` (reddit, hn, bluesky, mastodon, github-issues, grounding) |
| `LOCAL_FS` | Brief output at `.agents/results/market/`; cache at `~/.cache/oma/market/` |
| `PROCESS` | `oma market` subcommands; `oma search fetch` |
| `MEMORY` | Intent classification, operator pack selection, cluster summaries |

### Preconditions
- Topic is non-empty and passes detect-trap (not demographic-shopping, not single-noun-too-broad).
- At least one keyless source is reachable (reddit, hn, bluesky, mastodon, github-issues, or grounding).

### Effects and side effects
- Writes brief markdown to `.agents/results/market/{topic-slug}-{YYYYMMDD}.md`.
- Populates local cache at `~/.cache/oma/market/{sha1-hash}/result.json` (TTL 15m).
- Reads Serena `trust-registry-cache` (no write).

### Guardrails
1. **detect-trap first**: never harvest without preflight; `--force` bypasses only in test mode.
2. **Delegate all fetches**: `harvest` calls `oma search fetch --only api`; no direct platform HTTP.
3. **Trust labels read-only**: no re-scoring; Trust Registry ownership stays with `oma-search`.
4. **Paid sources auto-skip**: drop silently with `[INFO]` stderr if env key absent; never error.
5. **LAW self-check mandatory**: render runs self-check before file write; `--no-self-check` for debug only.
6. **No raw evidence dump**: cluster internals (scores, item counts) stay in JSON output; markdown body paraphrases.
7. **Stdout pure JSON per stage**: each pipe stage (except render) emits valid JSON only; stderr for warnings.
8. **No Serena writes in v1**: `trust-registry-cache` is read-only; write authority stays with `oma-search`.

### Routes

| Intent | Operator pack | Auto frameworks | Notes |
|--------|--------------|-----------------|-------|
| `pain` | `resources/operator-packs/pain.md` | SWOT | Weights: engagement 0.40, freshness 0.30, quality 0.30 |
| `trend` | none (optional: `resources/operator-packs/positive.md` for pain/positive contrast) | SWOT | Weights: freshness 0.50, engagement 0.30, quality 0.20 |
| `competitor` | `resources/operator-packs/competitor.md` | SWOT + Porter's 5F (v1.1 stub) | Weights: relevance 0.35, engagement 0.35, quality 0.30; `--vs` enables COMPARISON template |
| `discovery` | `resources/operator-packs/discovery.md` | SWOT + PESTEL (v1.1 stub) | Weights: relevance 0.45, engagement 0.30, quality 0.25 |

Porter's 5F and PESTEL: the CLI renders empty framework slots; the host LLM fills them using the analyst prompts in `resources/frameworks/porters-5f.md` and `pestel.md` (execution-protocol Step 6).

### Default Workflow
1. **Preflight**: `oma market detect-trap` exits 0 or halts.
2. **Harvest**: fan-out to keyless sources with operator-pack query; paid sources conditional on env keys.
3. **Score**: apply intent-specific engagement weights and log1p normalization.
4. **Fuse**: URL-canonicalize, deduplicate, RRF k=60, per-author cap <= 3.
5. **Cluster**: entity overlap coefficient >= 0.4, MMR lambda=0.75, <= 3 representatives.
6. **Render**: select frameworks, synthesize brief, run LAW self-check, write file.

### Invocation

#### Standalone
```
/oma-market "Next.js pain points" --intent pain --window 30d
/oma-market "AI coding tools trend" --intent trend
/oma-market "Cursor vs Windsurf" --intent competitor --vs Windsurf
/oma-market "developer productivity market" --intent discovery
```

#### Shared (from other skills or workflows)
Pass the rendered brief path (`.agents/results/market/{slug}-{YYYYMMDD}.md`) as a `--use-market-research` arg to Brainstorm or PM workflows. The brief is a static file; the calling skill reads it directly.

## References
- Intent classification: `resources/intent-rules.md`
- Operator packs: `resources/operator-packs/` (pain.md, positive.md, competitor.md, discovery.md)
- Frameworks: `resources/frameworks/` (swot.md, porters-5f.md, pestel.md — analyst prompts the host LLM fills into the rendered slots)
- Execution steps: `resources/execution-protocol.md`
- Output LAWs and self-check rules: `resources/output-laws.md`
- Input/output examples: `resources/examples.md`
- Pre-flight checklist: `resources/checklist.md`
- Error recovery: `resources/error-playbook.md`
- Context loading: `../_shared/core/context-loading.md`
- Lessons learned: `../_shared/core/lessons-learned.md`
