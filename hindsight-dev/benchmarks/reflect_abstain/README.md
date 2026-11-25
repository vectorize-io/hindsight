# Reflect abstain benchmark

Does the decision model know when a coding agent's memory watcher should speak?

The watcher (`autoInject: "watch"` in `hindsight-integrations/coding-agents`) calls `reflect` with
`abstain` at each moment of the agent's work: its request (a heads-up before it starts), a plan, a
command, an edit, a failure. Before any LLM call, the decision model routes the moment:

| route | what happens | wrong when |
|---|---|---|
| **silent** | no note, no LLM call | the moment should speak: a **missed note** |
| **writer** | the memories settle it; a writer only words the note | the moment should stay silent: a **false note** |
| **check** | the reflect LLM reads the memories and decides | never; it only costs an LLM call |

A good decision model makes no mistakes and sends few moments to `check`.

## Dataset

`datasets/watcher_moments.jsonl`: 118 moments from sde-bench runs of the watcher, one JSON object per line.

| field | meaning |
|---|---|
| `id`, `split`, `task`, `trigger` | `trigger` is `task` (heads-up), `plan`, `command`, `edit` or `failure` |
| `query` | the question the watcher asks reflect at this moment |
| `context` | the agent's situation as reflect gets it: request, latest step, recent steps, code change, `ALREADY TOLD ME` |
| `memories` | the 8 memories reflect's search put in front of the decision model, best first |
| `label` | `speak` or `silent` |
| `label_source`, `note` | who labeled it, and why when it is not obvious |

The decision model's state is `context`, then `MEMORY:`, then the numbered memories
(`Moment.state()`), exactly as `pick_answering` builds it.

| split | moments | speak | labels | use |
|---|---|---|---|---|
| `tuned` | 42 | 13 | by hand, against the code | the 8 tasks the verdict wording was tuned on: train here |
| `heldout` | 76 | 32 | an LLM judge (gemini-3.1-pro-preview) | 19 tasks nothing was tuned on: evaluate here, never train |

What "speak" means, for both splits:

- a heads-up (`trigger: task`) speaks when a memory records a rule for the code the request is about;
- a later moment speaks when the latest step goes against, or misses, a recorded rule (or uses a rejected
  option), and `ALREADY TOLD ME` does not already point that problem out. A rule stated in a heads-up
  before the code existed does not count: when the agent breaks it anyway, speak.
- it stays silent when the agent follows the rules, was already told, only runs or looks for tests, or
  the memories are about other code. When a memory disagrees with `ALREADY TOLD ME`, the latter is the
  corrected version (it was written from the memory's source text).

Two things to know about the labels. The judge agreed with the hand labels on 38 of the 40 `tuned`
moments it answered, and its 4 disagreements there were judge mistakes or an ambiguous rule, so expect a
few wrong `heldout` labels. And the `tuned` labels are not the watcher's v5 decisions: a replay's
`ALREADY TOLD ME` holds the replay's own notes, so 2 of v5's notes no longer fit the state, and 2 more
misread a plan that followed the rule (see `note`).

## Run it

```bash
cd hindsight-dev
# whatever decision model the server would use; e.g. Jev:
export HINDSIGHT_API_DECISION_MODEL_PROVIDER=typesafe HINDSIGHT_API_DECISION_MODEL_API_KEY=...
uv run python -m benchmarks.reflect_abstain.reflect_abstain_benchmark                 # reflect's own routing
uv run python -m benchmarks.reflect_abstain.reflect_abstain_benchmark --mode verdict  # the candidate question
uv run python -m benchmarks.reflect_abstain.reflect_abstain_benchmark --split heldout --out /tmp/heldout.jsonl
```

Any System One endpoint works through the `typesafe` provider: set `..._BASE_URL` and `..._MODEL`
(for instance `hindsight-decision` on the dev inference endpoint, with `--concurrency 2`, since the pick
does not retry a 429). `--out` writes every moment's route, speak score and full probabilities.

- `--mode pick` runs `TypeSafeCrossEncoder.pick_answering` and the cut-offs in `engine/reflect/agent.py`
  (`none >= 0.9` silent, `< 0.1` writer, else check): what reflect ships.
- `--mode verdict` asks one 8-option question about the moment (`VERDICT_OPTIONS`) and routes on the
  summed probability of its two speak options (`--write-at 0.66`, `--check-at 0.5`).

The `AUC` column is the share of (speak, silent) pairs the speak score orders correctly: 1.0 means some
cut-off separates the split perfectly, 0.5 means the score says nothing. Read it before tuning cut-offs:
a model with a low AUC needs a better question or more training, not other thresholds.

## Results (2026-10-19)

| model, mode | split | writer | check | silent | missed | false | AUC |
|---|---|---|---|---|---|---|---|
| jev-latest, pick | heldout | 22 | 28 | 26 | 0 | 2 | 0.95 |
| jev-latest, pick | tuned | 14 | 18 | 10 | 0 | 2 | 0.94 |
| jev-latest, verdict | heldout | 28 | 6 | 42 | 3 | 0 | 0.96 |
| jev-latest, verdict | tuned | 13 | 2 | 27 | 0 | 0 | 1.00 |
| hindsight-decision, pick | heldout | 19 | 57 | 0 | 0 | 3 | 0.80 |
| hindsight-decision, verdict | heldout | 0 | 0 | 76 | 32 | 0 | 0.76 |

- On Jev, the verdict question sends a fifth as many moments to the LLM check as reflect's pick, with
  no false notes; its 3 `heldout` misses are a real bug (deploywave), a missed debug-log rule (drainplan)
  and a nitpick (an unquoted CSV header).
- hindsight-decision does not yet tell these moments apart, whatever the question: under the pick it
  never stays silent (every moment costs an LLM call), and the verdict's probabilities never reach the
  bands. That is what training on `tuned` should move.
- The verdict's bands were set on Jev and only fit Jev: speak moments summed `>= 0.68`, silent `<= 0.65`
  over three runs, a narrow gap. A choice-only routing (no bands) needed two questions per call, which
  doubles the decision model's cost.
