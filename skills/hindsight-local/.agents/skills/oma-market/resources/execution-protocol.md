# Step-by-step execution protocol for oma-market

The CLI handles deterministic compute (search, score, fuse, cluster, render
skeleton + cluster bank). The HOST LLM (Claude / Codex / Gemini reading
this skill) handles semantic work (intent detection, framework synthesis).
Steps below alternate between the two responsibilities.

## Step 0 — Trap Detection (CLI)

Run `oma market detect-trap "<topic>"`. On exit 2, return the REFUSE
message to the user and STOP.

## Step 1 — Intent Classification (LLM)

Read the user's prompt and classify intent into one of
`pain | trend | competitor | discovery` per `intent-rules.md`.

Mapping cues (non-exhaustive — read the prompt, do not pattern-match
blindly):

- "X 페인 / X 불편 / X 이탈 이유 / why do users leave X" → `pain`
- "X 트렌드 / X 부상 / hot in X / what's hot in X" → `trend`
- "X vs Y / X 대 Y / X와 Y 비교" → `competitor` (set `--vs Y`)
- "X 시장 / X 카테고리 / opportunities in X / unmet needs around X"
  → `discovery`

Explicit `--intent` flag from the user always wins. Record result as
`$INTENT`. Also derive `$LOCALE` (`ko` if topic contains Hangul, else
`en`) and `$SITES` (Naver / tistory / brunch domains for `ko`, none
otherwise).

## Step 2 — Operator Pack Selection (LLM)

| Intent | Pack |
|---|---|
| pain | `operator-packs/pain.md` |
| trend | none |
| competitor | `operator-packs/competitor.md` |
| discovery | `operator-packs/discovery.md` |

## Step 3 — Harvest (CLI)

```bash
oma market harvest "<topic>" \
  --sources <list> \
  --window <window> \
  --operator-pack <pain|positive|competitor|discovery|none> \
  --locale $LOCALE \
  [--vs <competitor>] \
  [--sites $SITES] \
  [--per-source-limit <n>] \
  [--query-strict] \
  [--no-cache]
```

Default `--sources reddit,hn,bluesky,mastodon,grounding` (paid sources
auto-added when env keys present). Default `--window 30d`.

## Step 4 — Score, Fuse, Cluster (CLI)

```bash
oma market score --intent $INTENT \
  | oma market fuse \
  | oma market cluster --overlap-threshold 0.2
```

Tune `--overlap-threshold` if clusters are over-fragmented for KR
queries (default 0.4 is too strict for n-gram tokens).

## Step 5 — Render Skeleton (CLI)

```bash
oma market render --format md --intent $INTENT --frameworks auto \
  --topic "<user-facing title>" \
  [--vs <competitor>]
```

This writes a skeleton brief: badge → body → KEY PATTERNS →
**Cluster Bank** → empty SWOT/5F/PESTEL slots.

## Step 6 — Analyst Synthesis (LLM)

Read the skeleton brief. For each framework section present:

1. Open the corresponding prompt under
   `.agents/skills/oma-market/resources/frameworks/<name>.md`.
2. Map clusters from the Cluster Bank into the framework slots per the
   classification rules in that prompt.
3. Replace every `_(fill from cluster bank)_` placeholder with concrete
   bullets, each citing a cluster representative as `[name](url)` and
   tagged with its cluster ID `(C#)`.
4. Apply the output LAWs (`output-laws.md`): no em-dash, no invented
   titles, inline `[name](url)` only, no trailing `Sources:` block.
5. Korean briefs: bullet text in Korean; structural labels stay as
   written in the framework prompt.

If a framework axis genuinely has no signal in the Cluster Bank, write
`_(no signal)_`. Do not invent quotes.

## Step 7 — Self-Check and Finalize (LLM)

1. Run mental checks from `checklist.md`.
2. Save the synthesized brief at
   `.agents/results/market/{slug}-{YYYYMMDD}.md` (render's default
   output path — overwrite the skeleton from Step 5).
3. Return the first 20 lines + file path to the user.

## Responsibility split

| Step | Owner | Why |
|---|---|---|
| 0 | CLI | Deterministic regex preflight |
| 1, 2 | LLM | Semantic understanding of user intent |
| 3, 4, 5 | CLI | Deterministic search + scoring + rendering |
| 6 | LLM | Semantic classification + writing |
| 7 | LLM | Self-check on final prose |

The CLI never auto-classifies clusters into SWOT/5F/PESTEL — keyword
classifiers do not generalize across domains and languages. The LLM
hosting this skill performs all semantic mapping.
