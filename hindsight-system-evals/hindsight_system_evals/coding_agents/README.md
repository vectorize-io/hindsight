# coding-agents: does the agent actually read memory?

One number: **searches per user turn** — how often a real Claude Code turn calls
`hindsight_search_knowledge_pages`.

It exists because of a measurement, not a hunch. Two days of one developer's
real Claude Code transcripts: **1360 user turns, 7 searches (0.5% of turns), 37
memory *writes***. The plugin injects a tool guide telling the agent to search
first; the agent mostly ignores it. Prompt changes meant to fix that are cheap to
write and impossible to judge by reading, so this harness judges them.

## Running

```bash
cd hindsight-system-evals
uv run python -m hindsight_system_evals.coding_agents --build
```

Defaults to 5 sessions × 8 turns = 40 measured turns on Opus 5. Options:
`--sessions`, `--turns`, `--model`, `--api-url` (reuse a server instead of
starting one), `--no-warmup`, `--out report.json`, `--work-dir` (keep the
sandbox), `--build` (npm-build the plugin first — do this after every prompt
edit, or you measure the last build).

Needs: a **macOS** login to Claude Code (the token is copied out of the
Keychain), `node` and `npm` (run `npm install` in
`hindsight-integrations/coding-agents` once per worktree — it is not an npm
workspace), and provider credentials for the Hindsight server itself.

Those come from the repo `.env` automatically, but check they still work — a
dead key does not fail the run, it just makes seeding and reflect quietly
useless:

```bash
export HINDSIGHT_EVAL_LLM_PROVIDER=gemini HINDSIGHT_EVAL_LLM_MODEL=gemini-2.5-flash-lite
export HINDSIGHT_EVAL_LLM_API_KEY=...
uv run python -m hindsight_system_evals.coding_agents --env-file /dev/null
```

`--env-file /dev/null` is worth knowing: the repo `.env` sets
`HINDSIGHT_API_LLM_BASE_URL` to an OpenAI URL while the provider is `gemini`, and
loading it drags that into the server's config.

## What a run does

1. **Server** — `hindsight-api` on its own pg0 instance, as the other evals do.
2. **Sandbox** — a temp HOME with the plugin installed into it *from the
   checkout* (`dist/installer.js install claude-code --server self-hosted`), plus
   a small git repo (`scenario.py`) for the agent to work in. Everything the
   plugin and the CLI touch is under `$HOME`, so a run cannot reach the
   developer's own memory, logs, banks or agent config — and the bank id is
   pinned per run, so one run's memory never warms the next one's score.
3. **Warm-up session** — one throwaway turn, then wait for the bank to settle.
   The plugin seeds memory from the repo at SessionStart; measuring before that
   lands would score an empty memory rather than the prompt. `--no-warmup`
   measures the cold case deliberately.
4. **Measured sessions** — each a fresh Claude session over the same scripted
   turns, run one at a time.
5. **Metric** — aggregated from the plugin's own `usage.jsonl`, the same counter
   `hindsight-coding-agents stats` reports, so a harness number and a field
   number mean the same thing.

## Three things that are load-bearing

**One process per session, fed over stream-json.** The obvious alternative,
`claude -p` once per turn with `--resume`, fires SessionStart on *every* turn, so
the tool guide is re-injected before each prompt and the search rate becomes an
artifact of the harness. Verified on Claude Code 2.1.274: stream-json gives 1
SessionStart, N UserPromptSubmit, N Stop — the shape a human session has.

**The sandbox path is `realpath`-ed.** On macOS `/tmp` is a symlink to
`/private/tmp`, and the MCP server the installer registers only runs itself when
`process.argv[1]` matches its own `import.meta.url`. Launched through the
unresolved path it fails that comparison and **exits 0 in silence** — no tools
registered, no error anywhere, and a run that cheerfully reports 0% because the
agent had nothing to call. The first smoke run did exactly that. Always confirm
`"status": "connected"` for the `hindsight` server in a session log before
believing a low number:

```bash
python3 -c "import json;[print(d.get('mcp_servers')) for l in open('session-1.jsonl') \
  if (d:=json.loads(l)).get('subtype')=='init'][:1]"
```

**The prompts are copied from real ones in shape.** They were rewritten from a
sample of 1168 real user turns (terse, lowercase, typo-laden, each continuing
the work; median 114 characters). The first draft asked interview questions —
"why does pricing use Decimal instead of float here?", "what's our convention
for reporting errors?" — and scored **27.5%** searches/turn against a field rate
of 0.5%, with 10 of 11 searches landing on exactly those two prompts. An
interview question is a search request in disguise: it measures instruction
-following, not whether the plugin gets the agent to reach for memory during
ordinary work.

## What the runs have established

Each row is 5 sessions × 8 turns = 40 measured turns on Opus 5.

| bank / plugin | searches | turns that retrieved | credit rate |
|---|---|---|---|
| empty (pages say `Generating content...`) | 3 (7.5%) | 3 (7.5%) | 0% |
| prefilled, 3 pages | **0** | 5 (12.5%) | **60%** |
| prefilled, 12 pages | **0** | 12 (30%) | 8.3% |
| 12 pages, roster withheld + search-first guide | **6 (15%)** | 7 (17.5%) | 14.3% |
| + tool guide re-injected every turn | **13 (32.5%)** | 14 (35%) | 14.3% |
| + per-turn synthesis switched off (`--auto-inject none`) | 12 (30%) | 15 (37.5%) | 6.7% |
| + guide names the trigger moments | **29 (72.5%)** | 32 (80%) | **34.4%** |
| all of the above shipped as defaults, + read/search fixes | 18 (45%) | **34 (85%)** | **88.2%** |

Credit rates in earlier rows come from the plugin's live flag and are understated
— see finding 9.

Three findings, in the order they changed what to build:

**1. An empty bank teaches the agent that memory is empty.** The plugin seeds its
pages at SessionStart and refreshes them on a `H * * * *` cron, which never fires
inside a session — so the first `<hindsight_memory>` block the model saw listed
pages whose body was the literal string `Generating content...`, or said
"Hindsight's synthesis was unavailable this turn" after a 20s reflect timeout.
That is an anti-advertisement, and `prefill.py` exists because of it.

**2. The injection cuts search both ways.** When reflect succeeds it injects a
synthesis that answers the turn outright — nothing left to search for. When it
fails the memory looks empty — nothing worth searching. Either way the per-turn
injection removes the reason to call the tool.

**3. The roster is a substitute for search.** With the bank prefilled, the one
turn whose answer lives only in memory ("the discount and the free shipping
threshold interact weirdly") triggered retrieval in **5 of 5 sessions** — every
time as `hindsight_read_knowledge_page`, straight from an id in the roster, and
**never** as a search. The SessionStart block lists every page by title and id;
with three pages, searching to find one of three titles you can already see is a
wasted call. The model is behaving correctly. This is why the seeded bank now
carries a dozen pages, the size of a real repo's.

**4. Withholding the index is what creates a search — and costs retrieval.**
Replacing the roster with a count plus "call `hindsight_search_knowledge_pages`
to find the ones that bear on this turn" took searches from 0 to 6 (15% of
turns), the first non-zero search rate on a prefilled bank. But total retrieval
*fell*, 30% → 17.5%: the index was doing real work, and a bare count is a weaker
prompt than a list of titles one of which obviously matches. The next move is to
give the model something search-shaped to act on without handing it ids.

**5. Repetition is the second lever, and it is bigger than it sounds.** The tool
guide is injected at SessionStart and then only every `pageRefreshEveryTurns`
turns (default 10), so in an eight-turn session the model is told once, on turn
1, and never again. Re-injecting it every turn took searches from 15% to
**32.5%** with no wording change at all.

**6. The per-turn synthesis is NOT what suppresses search.** Switching
`autoInject` off entirely — the obvious suspect, since a synthesis that answers
the turn leaves nothing to look up — moved searches 32.5% → 30.0%, inside noise,
and halved attribution (14.3% → 6.7%). The injection is earning credit, so it
stays on. What suppressed search was the roster, not the synthesis.

**7. Naming the moments is the third lever, and the largest.** "Search when the
question might be answered by accumulated knowledge" is a category the model has
to recognise mid-task, and it mostly doesn't. Replacing it with the specific
moments that go wrong silently — *the user reports a bug or a wrong response; you
are about to write or change a test; you are implementing something new; the user
asks why, or what is left; you are about to commit* — took searches from 32.5% to
**72.5% of turns**, and attribution from 14.3% to **34.4%**.

It moved exactly the turns it named. Per prompt, over 5 sessions, before → after:
"add a test for it" 0/5 → 4/5; "getting a 409 on reserve with qty 0" 0/5 → 5/5;
"ok commit this" 0/5 → 2/5. Those turns were never ambiguous about whether memory
held the answer — the bank had both the testing convention and the decided rule
that qty ≤ 0 is a 400 — the model simply did not recognise them as search
moments.

The credits that result are specific, not decorative:

> 🧠 **From Hindsight memory (Inventory and reservations)** — quantity ≤ 0 is a
> decided **400**, kept deliberately separate from a genuine shortage at **409**.

**8. A falling search count can mean the search got better.** The final run
searched on 45% of turns, down from 72.5% — and used memory on **85%** of them,
its best, with attribution at **38.2%**. Two tool fixes explain it: a search now
returns 10 hits instead of 3, and reading a page returns its body once instead of
body-plus-the-same-body-with-frontmatter. One search answers more, so fewer are
needed: 15 of the 27 read-turns issued no search that turn, reusing a page id
learned from an earlier one. Judge the harness on `turns that retrieved` and the
credit rate; the raw search count is a proxy, and a proxy that can be gamed by
making each search worse.

**9. The shipped stats undercount attribution by ~2.4x.** The final run recorded
13 credited turns; the plugin's OWN reader and summariser, re-run over the same
transcripts once they were complete, find 31. The Stop hook evaluates a turn
before the host has flushed that turn's final reply — and the credit blockquote
lives in exactly that reply — so the turn is written `credited: false` and the
cursor moves past it for good. Every credit line was checked: none used negative
phrasing ("no relevant results", "nothing found"), and every one's distinctive
wording appears in what the tools actually returned, so these are real credits
for real retrieved content.

Both halves are fixed: the plugin re-emits the previous turn on each Stop (the
report already keeps the last line per turn, so the corrected read wins), and
this harness counts credit from the finished transcripts rather than the live
flag (`transcript_credits`).

## The recipe

Three changes, each measured separately, together 0% → 72.5%:

1. **Withhold the index.** Tell the agent how many pages exist and how to find
   them; do not list titles and ids. (`core/knowledge-injection.ts`, `indexLine`.)
2. **Repeat the guide every turn**, not every 10th
   (`pageRefreshEveryTurns`). Free, and worth 15% → 32.5% on its own. Now the
   shipped default, so a run needs no `--config` for it; pass
   `--config pageRefreshEveryTurns=10` to measure the old behaviour.
3. **Name the trigger moments** rather than describing a category.
   (`core/knowledge-injection.ts`, `TOOL_GUIDE`.)

Leave `autoInject` alone: switching it off does not help search and halves
attribution.

That third finding also produced the first real attribution: **60% of retrieval
turns credited memory** with the `> 🧠 **From Hindsight memory**` blockquote, up
from 0%. Worth knowing: in 4 of 5 sessions the agent also filed a
`Correction: …` document arguing the page was wrong, because the prefilled
decision ("judge the threshold on the discounted total") contradicted code
written before any discount existed. The prefill wording now says so explicitly —
a memory that reads as a claim about current code invites the agent to
"correct" it.

## Arms

```bash
--auto-inject none                  # no per-turn injection at all
--config pageRefreshEveryTurns=1    # re-inject the tool guide every turn
--config KEY=VALUE                  # any plugin config field, repeatable
--no-prefill                        # measure the empty-bank case deliberately
```

## Reading the result

```
  turns                40  (5 sessions)
  searches             2
  searches / turn      0.050
  turns that searched  2  (5.0%)
  turns that retrieved 3  (7.5%)
  credit rate          33.3%
```

`search_turn_rate` is the headline. A single run at these sizes is noisy — 40
turns cannot resolve 1% from 3% — so treat one run as a data point, not a
verdict: re-run the baseline before believing a change helped. What 40 turns
*can* resolve is the difference the plugin is actually chasing, 1% against 10%+.

`credit_rate` is `None`, printed as `n/a`, when nothing retrieved. 0/0 is not 0%,
and printing 0% there reads as "retrieval was useless" when it means "retrieval
never ran".

## Layout

| file | what |
|---|---|
| `sandbox.py` | temp HOME, credentials, plugin install, per-run bank |
| `scenario.py` | the fixture repo (with a git history that carries the *why*) and the scripted turns |
| `session.py` | one Claude session over stream-json |
| `metric.py` | `usage.jsonl` → the rate |
| `run.py` | the CLI that wires the five steps together |

The metric itself is the one deterministic piece, so it has real tests:
`uv run pytest evals/test_00_coding_agents_metric.py` — no server, no model.

The session runs with `--strict-mcp-config` and a Hindsight-only MCP config, so
the account's connectors (Gmail, Drive, Claude Docs) stay out of the tool list.
Account-synced *skills* still land in the sandbox HOME; they are inert here, but
they are the remaining piece of the developer's account that a run can see.

## When a run dies with `401 OAuth access token has been revoked`

The sandbox copies the Keychain token at build time and holds it for the whole
run. If the host rotates or revokes that token meanwhile — which a long
back-to-back series of runs can provoke — every remaining session fails at once,
mid-run. It is not silent: the session reports `turn N returned an error` and the
report still prints, covering the sessions that completed.

Treat a partial report as valid for the turns it measured (`sessions` and `turns`
say how many), and re-run after logging in again with `claude`. A future
ANTHROPIC_API_KEY path would avoid this entirely — see below.

## Not built yet

- **No pass/fail gate and no CI job.** The baseline has to be stable before a
  threshold means anything, and a gate on a noisy rate fails for sampling.
- **macOS only.** A CI runner would authenticate with `ANTHROPIC_API_KEY`
  instead of the Keychain copy; `sandbox.py` is the one place that changes.
- **Sessions are sequential.** Parallel ones would share the sandbox HOME and the
  plugin's per-session cursors.
