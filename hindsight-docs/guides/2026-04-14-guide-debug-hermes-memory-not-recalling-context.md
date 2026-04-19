---
title: "How to Fix Hermes Memory When It Stops Recalling Context"
authors: [benfrank241]
date: 2026-04-14
tags: [how-to, hermes, memory, troubleshooting]
description: "Debug Hermes memory when it stops recalling context. Check mode, health, hooks, logs, and retention timing so auto-recall starts working again."
image: /img/blog/guide-debug-hermes-memory-not-recalling-context.png
hide_table_of_contents: true
---

![How to Fix Hermes Memory When It Stops Recalling Context](/img/blog/guide-debug-hermes-memory-not-recalling-context.png)

If you are trying to **debug Hermes memory not recalling context**, the fastest way to think about it is this: one of five things is usually wrong. Hermes is in the wrong memory mode, the backend is unhealthy, the native hooks are not active, the new memory has not finished retaining yet, or the model is still using the wrong memory path.

What makes this frustrating is that the system can look half-correct while still failing in practice. `hermes memory status` may show a configured provider, the Hindsight tools may appear in the tool list, and the config file may look fine, yet the agent still answers like it has amnesia. That usually means the issue is behavioral, not just configurational. The wrong mode, the wrong expectation, or the wrong timing is enough to make healthy infrastructure feel broken.

This guide gives you a reliable troubleshooting sequence. You will check the active mode, confirm the Hindsight backend is reachable, verify whether auto-recall hooks are available, run a real next-turn memory test, inspect logs, and fix the most common causes of missing recall. If you need the full config reference while working, keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [docs home](https://hindsight.vectorize.io/docs), and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open.

<!-- truncate -->

> **Quick answer**
>
> 1. Run `hermes memory status` and confirm Hindsight is active.
> 2. Check `memory_mode`, because `tools` disables automatic recall by design.
> 3. Confirm the Hindsight backend is healthy and the bank has real memories.
> 4. Test retain on one turn and recall on the next turn, not immediately.
> 5. Check whether Hermes supports the native lifecycle hooks.
> 6. Inspect logs and disable the built-in `memory` tool if it is confusing the model.

## Prerequisites

Before you debug anything, make sure you are actually using the native Hindsight provider and not an older mixed setup.

You should have:

- Hermes installed and launching normally.
- A configured Hindsight provider.
- Access to the machine where Hermes runs.
- At least one known fact you can test with.

Start with status:

```bash
hermes memory status
```

Then inspect the provider config file:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
print(json.dumps(json.loads(path.read_text()), indent=2))
PY
```

If you are new to Hindsight's retrieval and retention behavior, it is worth reading the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) once. Those two docs explain why a system can retain successfully but still not surface context when you expect it.

## Step by step

### 1. Check `memory_mode` first

This is the single most common cause of confusion.

Hermes supports three Hindsight memory modes:

- `hybrid`, automatic recall plus explicit tools
- `context`, automatic recall only
- `tools`, explicit tools only

If your config says `tools`, then auto-recall is not supposed to happen. The model must call `hindsight_recall` or `hindsight_reflect` explicitly. Many people see missing automatic context and assume recall is broken when the system is just following the chosen mode.

Print the mode directly:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
print("memory_mode:", cfg.get("memory_mode", "hybrid"))
print("prefetch_method:", cfg.get("prefetch_method", "recall"))
PY
```

If you want automatic recall, switch to `hybrid` or `context`:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
cfg["memory_mode"] = "hybrid"
cfg.setdefault("prefetch_method", "recall")
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Updated {path} to hybrid mode")
PY
```

If you are working through advanced mode choices, the dedicated guide on [Hermes memory modes with Hindsight](https://hindsight.vectorize.io/blog/2026/04/14/guide-hermes-memory-modes-with-hindsight-hybrid-context-tools) is the deeper explanation.

### 2. Confirm the backend is healthy

If Hermes cannot reach Hindsight, recall obviously cannot work.

For **local mode**, test the embedded server directly:

```bash
curl http://localhost:9077/health
```

You want a healthy response from the local API. If the connection is refused or the process is still starting, recall may fail even when Hermes itself is fine.

For **cloud mode**, the best quick signal is still `hermes memory status`, plus a check that your env file has the required values:

```bash
grep '^HINDSIGHT_' ~/.hermes/.env || true
```

If the API key or URL is missing, Hermes may initialize without a working backend.

If you are using local mode and want the broader setup reference, keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) and [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) handy while you debug.

### 3. Test whether retention is succeeding at all

Recall can only return what exists in the bank. A surprisingly common issue is that users test recall on a bank that has never retained anything useful.

Run a very controlled test.

In one Hermes turn, tell it something distinctive:

```text
Remember that the billing freeze ends Friday and the onboarding rewrite is blocked on analytics events.
```

Then wait for the response to finish. After that, on the **next turn**, ask:

```text
What do you remember about the billing freeze?
```

This next-turn requirement matters. Hindsight retains asynchronously after the assistant response, then makes the new memory available for future retrieval. If you retain and immediately try to verify in the exact same turn, you can mistake normal async behavior for a broken system.

If you want a stricter test, ask Hermes to use explicit memory retrieval when tools are available:

```text
Use hindsight_recall and tell me what you know about the billing freeze.
```

That helps you separate two different problems:

- recall data does not exist
- recall data exists, but auto-injection is not happening

### 4. Check whether Hermes has the native lifecycle hooks

The Hermes docs note an important caveat: automatic recall and automatic retain rely on lifecycle hooks such as `pre_llm_call` and `post_llm_call`. On older Hermes builds, the Hindsight tools may still register, but auto-injection can be skipped.

This creates a specific failure pattern:

- `hindsight_recall` exists as a tool
- explicit tool calls work
- automatic context injection does not happen

If that sounds like your situation, you are probably dealing with a Hermes version issue, not bad Hindsight configuration.

A quick symptom-based test is this:

- if explicit `hindsight_recall` works, but memory never shows up automatically in `hybrid` or `context`
- and your backend is healthy
- and your bank clearly contains data

then hook support is the likely missing piece.

### 5. Check whether the model is using the wrong memory path

Hermes still has a built-in `memory` tool that writes markdown notes locally. If that tool stays enabled, the model may keep using it instead of Hindsight. The result is confusing because the assistant looks like it is storing memory, but not in the path you are trying to debug.

Disable the built-in tool while testing:

```bash
hermes tools disable memory
```

Then repeat the controlled retain and recall test. This removes one of the messiest sources of false positives.

If you later decide you want the built-in tool back for a specific workflow, you can re-enable it:

```bash
hermes tools enable memory
```

### 6. Inspect logs instead of guessing

Logs turn a vague memory complaint into a concrete system problem.

For local mode startup issues, inspect the embedded daemon log:

```bash
cat ~/.hermes/logs/hindsight-embed.log
```

For runtime issues in Hindsight itself, inspect the active profile logs:

```bash
tail -f ~/.hindsight/profiles/*.log
```

If you want more verbosity from the provider, enable debug mode in the config file:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
cfg["debug"] = True
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Enabled debug logging in {path}")
PY
```

Then restart Hermes and reproduce the issue.

Logs are especially useful for telling apart these cases:

- provider never initialized
- backend is unreachable
- retention happened, but recall found nothing relevant
- prefetch ran, but the model still answered poorly

### 7. Verify the bank ID is the bank you think it is

Sometimes recall is healthy, but Hermes is pointed at the wrong bank. This happens most often after migration or when someone manually edits config.

Print the bank ID directly:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
print("bank_id:", cfg.get("bank_id"))
PY
```

If the bank is wrong, Hermes is not failing to recall. It is recalling from the wrong place.

This is also why migration guides emphasize keeping the same bank when moving from the older plugin path. If you need that walkthrough, see [Guide: Migrate hindsight-hermes to Native Hermes Memory](https://hindsight.vectorize.io/blog/2026/04/14/guide-migrate-hindsight-hermes-to-native-hermes-memory).

## Verifying it works

After you fix the suspected issue, verify in layers.

### Status

```bash
hermes memory status
```

This should show Hindsight as active.

### Config sanity

Confirm:

- `memory_mode` is what you intend
- `prefetch_method` is what you intend
- `bank_id` is correct
- your backend settings match reality

### Controlled next-turn test

Use a fresh fact, then query it on the next turn. Do not rely on old fuzzy impressions from previous conversations.

### Explicit recall test

If tools are visible, ask Hermes to call `hindsight_recall` directly. If that works while auto-recall does not, the problem is almost always mode selection or hook availability.

### UX test

Finally, test the behavior you actually care about. Ask a natural follow-up in a realistic conversation and see whether Hermes starts from the right context without you restating it.

## Troubleshooting common failure patterns

### Pattern: `hermes memory status` is fine, but auto-recall never happens

Most likely causes:

- `memory_mode` is `tools`
- Hermes lacks hook support
- the bank has no useful memories yet

### Pattern: explicit `hindsight_recall` works, but normal replies feel stateless

Most likely causes:

- auto-recall mode is disabled
- hook support is missing
- `prefetch_method` is configured, but you are expecting same-turn results

### Pattern: local mode fails only on first start

Most likely cause:

- embedded Hindsight and PostgreSQL are still initializing

Check `~/.hermes/logs/hindsight-embed.log` before assuming the setup is dead.

### Pattern: memory was working yesterday, now it feels inconsistent

Most likely causes:

- the config changed
- the bank ID changed
- the model is using the built-in Hermes `memory` tool again
- a backend credential disappeared from `~/.hermes/.env`

### Pattern: the model seems to ignore obviously relevant history

This can still happen even when recall is technically healthy. In that case, the issue may be relevance or prompt use, not provider failure. The [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) helps you reason about what Hindsight considers relevant.

## FAQ

### Why does recall fail when I test it right after storing a memory?

Because retention is asynchronous. The new memory becomes available on a later turn, not instantly in the same turn.

### Why do I see Hindsight tools, but no automatic context?

Usually because Hermes is on an older build without the native lifecycle hooks, or because `memory_mode` is set to `tools`.

### Does `tools` mode mean Hindsight is broken?

No. It means the model has to choose to use memory tools explicitly.

### Should I disable Hermes's built-in `memory` tool?

During debugging, yes. It removes ambiguity. Once recall is stable, you can decide whether you want that tool back.

### How do I know whether the bank is empty or recall is broken?

Run a controlled retain test, wait for the response to finish, then query the new fact on the next turn. If explicit `hindsight_recall` also finds nothing, the bank probably does not contain the expected data.

### Where can I learn more if I want to debug deeper than the CLI?

Start with the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), then read the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain). If you want another example of automatic memory injection in practice, the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) are a useful comparison.

## Next Steps

- [Create a Hindsight Cloud account](https://hindsight.vectorize.io) if you want the fastest path to testing memory without local daemon variables in the way.
- Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) open while you debug live issues.
- Read the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you want a simpler setup baseline before further troubleshooting.
- Use the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) to understand what should be returned when recall is healthy.
- Use the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) to reason about why expected information may never have been stored.
- If this started after a migration, compare your setup against [Guide: Migrate hindsight-hermes to Native Hermes Memory](https://hindsight.vectorize.io/blog/2026/04/14/guide-migrate-hindsight-hermes-to-native-hermes-memory).
