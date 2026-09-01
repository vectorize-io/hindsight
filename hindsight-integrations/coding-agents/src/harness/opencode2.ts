/**
 * opencode2 (opencode v2) harness adapter — full parity with the v1 opencode plugin.
 *
 * opencode v2 (`npm @opencode-ai/cli@beta`, binary `opencode2`) is a ground-up rewrite of the
 * plugin API, so NOTHING from harness/opencode.ts carries over. A v1 plugin is a function returning
 * a bag of named hooks; a v2 plugin is `{id, setup(ctx)}` where `ctx` hands out per-domain
 * registration calls. The mapping this file implements:
 *
 *   v1 `chat.message`                       -> `ctx.session.hook("prompt")`
 *   v1 `experimental.chat.system.transform` -> `ctx.session.hook("context")` (push a SystemPart)
 *   v1 `tool: {...}` (native tools)         -> `ctx.tool.transform(draft => draft.add(...))`
 *   v1 `event` (session.idle)               -> `ctx.event.subscribe()` async iterator
 *   v1 `client.session.messages()`          -> `ctx.session.context({sessionID})`
 *
 * Everything behind those five seams is the harness-agnostic RuntimeCore, exactly as for v1, so
 * opencode2 gets the same surface: per-turn recall + attribution/user-feedback injection, the
 * hindsight_* knowledge tools, cold-check auto-seed and rich write-back.
 *
 * Two deliberate differences from the v1 adapter, both forced by the host:
 *
 *  - NO survey agent. v1 taught the host about a read-only `hindsight-survey` agent through its
 *    `config` hook (see core/survey.ts). v2's `ctx.agent.transform` draft exposes only
 *    list/get/default/update/remove — a plugin cannot DEFINE an agent — and an agent declared in
 *    the config file is not a sandbox either: the user's own global `permissions` are appended
 *    AFTER an agent's rules (verified against 0.0.0-beta-18743, where `explore`'s `edit: deny`
 *    ends up followed by a user's `edit: allow`). The survey reads untrusted repo files, so
 *    without a guaranteed read-only boundary it must not run under this host — opencode2 falls
 *    back to another installed agent's CLI, the same treatment kilo/cline/cursor/dcode already
 *    get (core/survey.ts). Revisit if v2 gains plugin-defined agents.
 *
 *  - NO toast. v2's plugin client can OBSERVE `tui.toast.show` events but cannot publish one, so
 *    the seed banner is logged rather than shown. `RuntimeCore.notify` is optional and fail-open.
 *
 * One capability the v1 adapter has is deliberately absent because v2 makes it redundant, not
 * because it was forgotten: v1 also drove `RuntimeCore.onTranscript` from its messages.transform
 * hook, as a mid-session cadence, because its `session.idle` was the ONLY moment a completed reply
 * was readable and a long session would otherwise write back nothing until it ended. v2 raises
 * `session.execution.succeeded` after every assistant turn, so the idle path already fires once per
 * turn — from the authoritative post-reply transcript rather than the pre-reply list the context
 * hook sees. Adding the cadence path on top would only re-retain the same turns.
 *
 * This is the only opencode2-specific file besides the entrypoint; everything it uses is in ../core.
 */
import { z } from "zod";
import type { Plugin } from "@opencode-ai/plugin-v2";

import { resolveHostMemory } from "../core/host-client";
import { describeError } from "../core/log";
import { diag } from "../core/diag";
import { RuntimeCore } from "../core/runtime";
import type { ToolSpec } from "../core/knowledge-tools";
import { readOpencode2Messages, type Oc2Message } from "../core/transcript-opencode2";
import { resolveProjectDirectory } from "./plugin-entry";

/** The v2 SDK exports its plugin surface as a namespace (`export * as Plugin`), so the interface a
 *  default export must satisfy is `Plugin.Plugin`. */
type Opencode2Plugin = Plugin.Plugin;
/** The setup context the host hands a plugin — the half of it this adapter uses. */
type Opencode2Context = Parameters<Opencode2Plugin["setup"]>[0];

/**
 * Events that mean "the assistant has finished and the completed exchange is readable" — the
 * Stop-equivalent these hosts otherwise lack.
 *
 * Both are needed. `session.idle` is what the interactive TUI settles on; a one-shot `opencode2
 * run` exits before it ever fires and emits only `session.execution.succeeded` (verified against
 * 0.0.0-beta-18743). Listening to both is safe: `RuntimeCore.onSessionIdle` only retains when the
 * transcript actually grew past what it last wrote, so a turn that raises both costs one retain.
 */
const IDLE_EVENTS = new Set([
  "session.idle",
  "session.execution.succeeded",
  "session.execution.failed",
  "session.execution.interrupted",
]);

/** Structural subset of the v2 event envelope we act on. */
interface Oc2Event {
  type?: string;
  data?: { sessionID?: string };
}

/**
 * Adapt a harness-agnostic ToolSpec (the MCP-shaped spec every harness shares) to a v2 tool.
 *
 * Two v2 specifics:
 *  - `input` is a whole schema, not v1's raw shape, so the spec's Zod shape is wrapped in
 *    `z.object`. v2 accepts any Standard Schema, which zod v4 satisfies.
 *  - The spec's MCP safety `annotations` have no slot in v2's tool contract (its `Info` carries only
 *    name/description/input/execute/options, and `options.permission` names a permission action,
 *    not a read-only hint), so they are dropped here — the same as every other native-tool harness
 *    (opencode v1, Cline, dsh, Prime Agent), which publish them only over MCP. Nothing is gated on
 *    them here: the host asks the user about tool calls through its own permission rules.
 *  - `options.codemode: false` is REQUIRED, not decoration. A tool left on the default is offered
 *    to the model only through v2's `execute` code-execution tool, so a model that calls it by
 *    name gets "Unknown tool: hindsight_…" — which is exactly how every hindsight_* tool behaves
 *    without this flag, since the skill and the injected preamble both tell the model to call them
 *    by name.
 */
function toOpencode2Tool(spec: ToolSpec) {
  return {
    name: spec.name,
    description: spec.description,
    input: z.object(spec.inputSchema),
    options: { codemode: false as const },
    async execute(input: unknown) {
      const r = await spec.handler(input as Record<string, unknown>);
      return { content: r.content?.map((c) => c.text).join("\n") || "" };
    },
  };
}

/**
 * Wire a RuntimeCore onto one opencode2 plugin context and return the teardown.
 *
 * Split out of `createOpencode2PluginEntry` so the hook wiring can be exercised against a fake
 * context without a config file, a bank or a server — the same seam `createClineHooks` gives the
 * Cline adapter.
 */
export async function wireOpencode2Runtime(
  core: RuntimeCore,
  ctx: Opencode2Context
): Promise<() => void> {
  const harness = core.harness;

  // Let the runtime READ the transcript back once the assistant has finished. `session.context`
  // returns the session's messages in the same shape readOpencode2Messages consumes.
  core.setTranscriptSource(async (sessionID) =>
    readOpencode2Messages((await ctx.session.context({ sessionID })) as Oc2Message[])
  );

  // Register the full hindsight_* knowledge + recall suite natively (no MCP server needed).
  await ctx.tool.transform((draft) => {
    for (const spec of core.toolSpecs()) draft.add(toOpencode2Tool(spec));
  });

  // Sessions THIS plugin instance owns. See the event loop below for why the write-back cannot do
  // without it.
  const ownSessions = new Set<string>();

  // Each user turn: recall on the prompt; the injection it builds is pushed by the context hook.
  await ctx.session.hook("prompt", async (input) => {
    ownSessions.add(input.sessionID);
    await core.onPrompt(input.sessionID, input.prompt?.text ?? "");
  });

  // Push this turn's injection (recalled memories + attribution/user-feedback framing, plus the
  // knowledge preamble on turn 1 and the roster refresh on cadence) into the system prompt every
  // turn. Unlike v1's system.transform this hook DOES carry the session id, so the lookup is
  // always session-keyed.
  await ctx.session.hook("context", (input) => {
    const inj = core.getInjection(input.sessionID);
    if (inj) input.system.push({ type: "text", text: inj });
    diag(harness, inj ? "inject_ok" : "inject_empty", { chars: inj?.length ?? 0 });
  });

  // Write-back (on by default). Refetching on idle — rather than replaying the list the context
  // hook saw — is the whole point: that list is built BEFORE the reply, so it always lags a turn
  // and a session's final exchange would never be retained.
  //
  // `ownSessions` is a HARD BANK-ISOLATION GUARD, not an optimisation. v2's background service
  // hosts every open project at once, and unlike the session hooks — which fire only for the
  // location that loaded this plugin instance — `event.subscribe()` is GLOBAL: a plugin loaded for
  // project A receives session.idle for a session running in project B, carrying no location to
  // tell them apart (the envelope's `location` arrives undefined; verified against
  // 0.0.0-beta-18743). Acting on those ids would fetch another project's transcript and retain it
  // into THIS project's bank. So only sessions our own scoped prompt hook has admitted are eligible.
  const events = new AbortController();
  void (async () => {
    for await (const event of ctx.event.subscribe({ signal: events.signal })) {
      const e = event as Oc2Event;
      if (!e.type || !IDLE_EVENTS.has(e.type)) continue;
      const sessionID = e.data?.sessionID;
      if (!sessionID || !ownSessions.has(sessionID)) continue;
      diag(harness, "idle_signal", { signal: e.type, session: sessionID });
      // Deliberately NOT awaited: a retain must never stall delivery of the next event.
      void core.onSessionIdle(sessionID);
    }
    // Falling out of the loop (rather than throwing) means the host closed the stream — teardown.
    diag(harness, "idle_stream_closed", {});
    // A stream that DIES takes write-back with it silently, which is exactly the failure a
    // memory-less session must not be able to hide, so it goes in the diagnostic trail too.
  })().catch((e) => diag(harness, "idle_stream_failed", { error: describeError(e) }));

  return () => events.abort();
}

/**
 * Build the default export for an opencode2 host. `harness` is the name the host is known by, used
 * for config lookup (`harnesses.<name>`), `{harness}` bank templating and log scoping.
 */
export function createOpencode2PluginEntry(harness: string): Opencode2Plugin {
  return {
    id: `hindsight-${harness}`,
    async setup(ctx) {
      // v2 splits v1's single `{worktree, directory}` into a session cwd plus the project root;
      // `location.directory` is the worktree analogue, so it keeps precedence (and the same
      // filesystem-root guard) with the project root as the fallback.
      const projectDir = resolveProjectDirectory({
        worktree: ctx.location?.directory,
        directory: ctx.location?.project?.directory,
      });
      const dir = projectDir || process.cwd();
      const { cfg, bankId, client } = resolveHostMemory(harness, dir);
      // Global switch, per-bank opt-out and optInOnly all land here: inert plugin, same agent, no
      // memory (baseline parity).
      if (cfg.disabled) return;
      const core = new RuntimeCore(client, bankId, cfg, harness, dir);
      const cleanup = await wireOpencode2Runtime(core, ctx);

      // SessionStart-equivalent: cold-check the bank, kick off the background engine, compute the
      // knowledge preamble. Fire-and-forget — this host BLOCKS ITS BOOT on plugin setup, so it must
      // never gate startup on a network round-trip. onPrompt tolerates an empty preamble until it
      // resolves.
      void core.seedIfCold(projectDir);

      return cleanup;
    },
  };
}
