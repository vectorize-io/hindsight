# @vectorize-io/hindsight-mission-sandbox

Tune Hindsight's **retain (extraction)** and **observation (consolidation)** missions against your
own task, then verify with an **external validator** (a benchmark like LOCOMO, or your app's eval).

The tool is deliberately small and opinionated:

- **You bring** the documents and a way to score success (the validator). The tool does **not**
  measure accuracy or label facts — task success is decoupled from the tool.
- **You refine a mission with feedback.** After looking at validator results, you hand the tool
  _feedback_ (and optional failing examples); it rewrites the current mission. No good/bad labeling.
- **Retain iterates across versioned banks** (`<project>-v1`, `-v2`, …) so you can point the
  validator at any version and compare. **Observations iterate in place** (clear + re-consolidate),
  since they're re-derived from the same facts.

## The loop

```
init (bind docs)
  └─ retain mission  (feedback + examples → refine retain mission)
     └─ retain apply (ingest docs into a NEW bank <project>-vN)
        └─ VALIDATE EXTERNALLY against <project>-vN  ─┐
   ┌──────────────────────────────────────────────────┘  failures become the next feedback
   ▼
  retain mission (feedback) → retain apply → validate → …

observe mission (feedback → refine obs mission) → observe apply (clear obs + re-consolidate on current bank) → validate
```

The validator is never inside the tool. A typical round: run your eval against `<project>-vN`,
read what failed, then `retain mission <project> --feedback "<what to fix>" --example "<failing case>"`
→ `retain apply` (new version) → re-validate.

## Commands

```bash
# bind a project to its documents (no ingest yet)
mission-sandbox init <project> --documents <path> [--api-url URL]

# RETAIN loop — iterates across versioned banks
mission-sandbox retain mission <project> --feedback "<what to change>" [--example "<failing case>" ...]
mission-sandbox retain apply   <project>          # ingest docs → new bank <project>-vN, prints the bank id

# OBSERVE loop — iterates in place on the current bank
mission-sandbox observe mission <project> --feedback "<what to change>" [--example "<...>" ...]
mission-sandbox observe apply   <project>         # clear observations on current bank + re-consolidate

mission-sandbox status <project>                  # bound docs, current missions, versions (+ bank ids)
mission-sandbox ui <projects-dir>                 # minimal UI: project status + versions
```

- `retain mission` / `observe mission` refine the **current** mission from your feedback (+ examples);
  the first call (no prior mission) treats the feedback as the initial spec. The LLM sees the current
  mission + feedback + examples — nothing else, no labels.
- `retain apply` always creates the **next** version bank and ingests into it. Point your validator
  (e.g. LOCOMO `--template`/the bank id) at that version.
- `--model` overrides the Gemini model used for mission refinement (default `gemini-2.5-flash`, or
  `HINDSIGHT_API_LLM_MODEL`). Mission refinement is the **only** LLM call the tool makes; ingestion +
  consolidation run on the Hindsight deployment.

## Verifying with LOCOMO (example external validator)

The LOCOMO runner is unchanged and is the **only** thing that measures accuracy. Build a template
from a version's missions and point the runner at it (default mode — **no `--use-reflect`**):

```bash
# representative subset: trim the runner's input to N per category (data only — restore after)
cd hindsight-dev/benchmarks/locomo/datasets && cp locomo10.json locomo10.full.json
N=5   # widen to 10+ once a mission looks good, to confirm it generalises and surface weak categories
python3 - "$N" <<'PY'
import json, sys
n=int(sys.argv[1]); d=json.load(open("locomo10.json"))
for s in d:
    if s["sample_id"]!="<id>": continue
    s["qa"]=[q for c in (1,2,3,4) for q in [x for x in s["qa"] if x.get("category")==c and x.get("answer")][:n]]
json.dump(d,open("locomo10.json","w"))
PY

# verify a version's missions
python3 -c "import json;p=json.load(open('<project>/project.json'));v=p['versions'][-1]; \
  json.dump({'version':'1','bank':{'retain_mission':v['retainMission'],'observations_mission':v.get('observeMission')}}, \
  open('<project>/template.json','w'))"
set -a; source hindsight-api-slim/.env; set +a; export HINDSIGHT_API_LLM_MODEL=gemini-2.5-flash
uv run --project hindsight-dev python hindsight-dev/benchmarks/locomo/locomo_benchmark.py \
  --conversation <id> --wait-consolidation --template <project>/template.json
# results: hindsight-dev/benchmarks/locomo/results/benchmark_results.json (by-category is_correct)
mv hindsight-dev/benchmarks/locomo/datasets/locomo10.full.json hindsight-dev/benchmarks/locomo/datasets/locomo10.json
```

Read accuracy **by category**; a weak category is your next `--feedback`. Notes from real runs:
single-question swings between runs are **recall variance** (each apply re-ingests) — watch
category trends; and verify a "failure" against the transcript before chasing it (some benchmark
golds are wrong).

## Project model (`project.json`)

```jsonc
{
  "documents": "/path/to/docs", // bound at init
  "apiUrl": "http://localhost:8888",
  "retain": { "mission": "…", "feedback": ["…"] },
  "observe": { "mission": "…", "feedback": ["…"] },
  "versions": [
    {
      "n": 1,
      "bank": "<project>-v1",
      "retainMission": "…",
      "observeMission": "…",
      "createdAt": "…",
    },
  ],
  "currentVersion": 1,
}
```

## Setup

```bash
npm install
npm run build --workspace @vectorize-io/hindsight-mission-sandbox
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY, or a Gemini HINDSIGHT_API_LLM_* in your .env
```

## Development

```bash
npm run test       # vitest unit tests for core
npm run typecheck  # tsc for the lib + the Next app
npm run build      # build the core lib (dist) + the minimal Next UI
```
