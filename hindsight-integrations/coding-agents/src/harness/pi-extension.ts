/**
 * Shared entrypoint factory for the pi-family EXTENSION hosts (pi and its fork Prime Agent).
 *
 * pi (`@earendil-works/pi-coding-agent`) loads extensions listed in `~/.pi/agent/settings.json`;
 * Prime Agent (PrimeIntellect) is a fork of pi and loads the same shape from
 * `~/.prime/agent/settings.json`. Both call an extension's default export with their `pi` API, and
 * both expose the identical surface this adapter needs: `before_agent_start` (recall +
 * user-turn injection), `agent_end` (transcript write-back), and `registerTool` for the native
 * `hindsight_*` knowledge tools. So neither host needs an adapter of its own — they differ only in
 * which harness name they report, which selects the `harnesses.<name>` config section, feeds
 * `{harness}` bank templating, and keeps their sessions attributable in diagnostics separately.
 *
 * The memory behaviour itself stays in RuntimeCore — the same reflect-and-inject core every
 * Hindsight harness uses; this file only adapts the pi extension API at its boundary.
 */
import { z } from "zod";
import { resolveHostMemory } from "../core/host-client";
import { diag } from "../core/diag";
import type { ToolSpec } from "../core/knowledge-tools";
import { RuntimeCore } from "../core/runtime";
import { type PiEntry, type PiMessage, readPiEntries, readPiMessages } from "../core/transcript-pi";

// ── Structural subset of the pi extension API ───────────────────────────────────────────────────
// Declared locally so this package takes no dependency on the fast-moving pi / Prime Agent SDKs;
// the real runtime passes a compatible object at load time.

interface BeforeAgentStartEvent {
  type: "before_agent_start";
  /** The raw user prompt text. */
  prompt: string;
}

interface AgentEndEvent {
  type: "agent_end";
  /** The conversation messages for the completed agent loop. */
  messages: readonly PiMessage[];
}

/** A custom message for the host to store after the user's prompt. It reaches the model as a
 *  user message; `display: false` keeps it out of the transcript view. */
interface BeforeAgentStartResult {
  message?: { customType: string; content: string; display: boolean };
}

interface SessionManagerLike {
  getSessionId(): string;
  /** Every entry from the session root to the current leaf, already holding this run's messages
   *  when `agent_end` fires. Optional: a host without it falls back to the event's messages. */
  getBranch?(): readonly PiEntry[];
}

/** Only what this adapter reads. Both hosts also expose a UI notifier, but the seed banner it would
 *  carry is raised inside seedIfCold, which runs when a workspace is first resolved (at extension
 *  load for the launch directory) rather than inside a handler holding a `ctx` to notify through —
 *  so it is logged rather than toasted here, and the field is not declared. */
interface ExtensionContext {
  /** The session's working directory. Optional so a host without it falls back to the process cwd. */
  cwd?: string;
  sessionManager: SessionManagerLike;
}

/** A JSON-Schema-shaped parameters object. The host forwards it to the model provider verbatim. */
type JsonSchema = Record<string, unknown>;

interface ToolDefinition {
  name: string;
  label: string;
  description: string;
  parameters: JsonSchema;
  execute(
    toolCallId: string,
    params: Record<string, unknown>,
    signal?: unknown,
    onUpdate?: unknown,
    ctx?: ExtensionContext
  ): Promise<{ content: { type: "text"; text: string }[]; details: unknown }>;
}

interface ExtensionAPI {
  on(event: "session_start", handler: (event: unknown, ctx: ExtensionContext) => void): void;
  on(
    event: "before_agent_start",
    handler: (
      event: BeforeAgentStartEvent,
      ctx: ExtensionContext
    ) => Promise<BeforeAgentStartResult | void> | BeforeAgentStartResult | void
  ): void;
  on(
    event: "agent_end",
    handler: (event: AgentEndEvent, ctx: ExtensionContext) => Promise<void> | void
  ): void;
  on(event: "session_shutdown", handler: () => Promise<void> | void): void;
  registerTool(definition: ToolDefinition): void;
}

export type ExtensionFactory = (pi: ExtensionAPI) => void;

/**
 * Adapt a harness-agnostic ToolSpec (MCP-shaped, shared by every harness) to a pi native tool. The
 * spec's Zod raw shape is converted to a JSON Schema for `parameters` — pi's documented type is a
 * TypeBox `TSchema`, which is itself a JSON Schema, and neither host validates tool arguments
 * against it (the agent loop only runs an optional `prepareArguments`), so the schema is passed
 * straight to the model provider. A plain JSON Schema is therefore exactly what the tool needs. The
 * spec's handler returns an MCP `{content:[{text}]}` result and never throws, so we surface the
 * joined text back to the model.
 *
 * `bind` picks the spec to run for the calling session (its bank), since one process can serve
 * sessions in several repositories; undefined means memory is off for that session's repository.
 */
export function toPiTool(
  spec: ToolSpec,
  bind: (ctx: ExtensionContext | undefined) => ToolSpec | undefined = () => spec
): ToolDefinition {
  const parameters = z.toJSONSchema(z.object(spec.inputSchema)) as JsonSchema;
  return {
    name: spec.name,
    label: spec.name,
    description: spec.description,
    parameters,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const bound = bind(ctx);
      if (!bound) {
        const text = "Hindsight memory is disabled for this workspace.";
        return { content: [{ type: "text", text }], details: null };
      }
      const r = await bound.handler(params);
      const text = r.content?.map((c) => c.text).join("\n") || "";
      return { content: [{ type: "text", text }], details: null };
    },
  };
}

/**
 * Make the host-specific pi hooks testable without importing either host's SDK. RuntimeCore is the
 * shared lifecycle implementation; this adapter only converts pi messages at its boundary and never
 * calls Hindsight directly.
 */
export function createPiHooks(
  core: Pick<RuntimeCore, "onPrompt" | "getInjection" | "onTranscript" | "flush">,
  harness: string,
  sessionStart?: Promise<void>
) {
  let sessionStartAwaited = false;
  return {
    async beforeAgentStart(
      event: Omit<BeforeAgentStartEvent, "type">,
      sessionId: string
    ): Promise<BeforeAgentStartResult | undefined> {
      if (!sessionStartAwaited) {
        sessionStartAwaited = true;
        // Awaiting the shared SessionStart lifecycle before the first prompt preserves the invariant
        // that a brand-new bank skips its first auto-reflect instead of spending that synthesis
        // before it has any knowledge (mirrors the other harnesses).
        await sessionStart;
      }
      // Called on an empty prompt too, so the core drops the previous turn's block (see onPrompt).
      await core.onPrompt(sessionId, event.prompt.trim());
      const injection = core.getInjection(sessionId);
      if (!injection) {
        diag(harness, "inject_empty", { session: sessionId });
        return undefined;
      }
      diag(harness, "inject_ok", { session: sessionId, chars: injection.length });
      // A message, not the system prompt: the block changes between turns, so a system prompt
      // carrying it broke the provider's prefix cache (#5914), while a stored message stays in the
      // turn it was made for. It also leaves other extensions' prompt sections alone, which a
      // returned systemPrompt did not (#4841).
      return { message: { customType: "hindsight", content: injection, display: false } };
    },
    /** `branch` is the session's entries (`sessionManager.getBranch()`): the event alone holds only
     *  the latest run, and writing that back replaced the session's earlier runs (#4963). */
    async agentEnd(
      event: { messages: readonly PiMessage[] },
      sessionId: string,
      branch?: readonly PiEntry[]
    ): Promise<void> {
      const turns = branch ? readPiEntries(branch) : readPiMessages(event.messages);
      if (turns.length) await core.onTranscript(sessionId, turns, true); // the run has ended
    },
    /** The host is exiting: the last agent_end's write-back is still in flight (#5081). */
    sessionShutdown: () => core.flush(),
  };
}

function createRuntime(harness: string, repoPath: string): RuntimeCore | undefined {
  const { cfg, bankId, client } = resolveHostMemory(harness, repoPath);
  if (cfg.disabled) return undefined; // global switch, per-bank opt-out or optInOnly
  return new RuntimeCore(client, bankId, cfg, harness, repoPath);
}

interface PiWorkspace {
  hooks: ReturnType<typeof createPiHooks>;
  tools: ToolSpec[];
}

/**
 * Build the default export for a pi-family extension host. `harness` is the name the host is known
 * by ("pi", "prime-agent"), used for config lookup, bank derivation and diagnostics scoping — it is
 * NOT config-chosen; the entrypoint the host loaded determines it.
 *
 * Config and bank are resolved per SESSION, from `ctx.cwd`, and cached per directory. They used to
 * be resolved once from `process.cwd()` at load, which is the session's repository only for the
 * CLI: pi's web UI serves every session from one process started elsewhere, so the launch directory
 * picked the bank — or, under `optInOnly`, disabled the extension for every session (#5039).
 *
 * The launch directory is still resolved at load, so the CLI registers its tools and starts its
 * cold seed exactly as before. Tools cannot be unregistered, so they are registered once, from the
 * first enabled workspace (at load for the CLI, at the first session in an enabled repository
 * otherwise), and each call runs against the calling session's bank.
 */
export function createPiExtension(
  harness: string,
  makeRuntime: (harness: string, repoPath: string) => RuntimeCore | undefined = createRuntime
): ExtensionFactory {
  return (pi) => {
    const launchDir = process.cwd();
    const dirOf = (ctx: ExtensionContext | undefined) => ctx?.cwd || launchDir;
    // `null` caches "memory is off here" so a disabled repository isn't re-resolved every turn.
    const workspaces = new Map<string, PiWorkspace | null>();
    const workspaceFor = (dir: string): PiWorkspace | undefined => {
      const cached = workspaces.get(dir);
      if (cached !== undefined) return cached ?? undefined;
      const core = makeRuntime(harness, dir);
      // Fire-and-forget cold seed (bank check + background git seed); the workspace's first
      // before_agent_start awaits it via createPiHooks.
      const ws = core
        ? { hooks: createPiHooks(core, harness, core.seedIfCold(dir)), tools: core.toolSpecs() }
        : null;
      workspaces.set(dir, ws);
      return ws ?? undefined;
    };

    let toolsRegistered = false;
    const at = (ctx: ExtensionContext | undefined): PiWorkspace | undefined => {
      const ws = workspaceFor(dirOf(ctx));
      if (ws && !toolsRegistered) {
        toolsRegistered = true;
        for (const spec of ws.tools)
          pi.registerTool(
            toPiTool(spec, (c) => workspaceFor(dirOf(c))?.tools.find((t) => t.name === spec.name))
          );
      }
      return ws;
    };

    at(undefined); // the launch directory: the CLI's session
    pi.on("session_start", (_event, ctx) => {
      at(ctx);
    });
    pi.on("before_agent_start", async (event, ctx) =>
      at(ctx)?.hooks.beforeAgentStart(event, ctx.sessionManager.getSessionId())
    );
    pi.on("agent_end", async (event, ctx) =>
      at(ctx)?.hooks.agentEnd(
        event,
        ctx.sessionManager.getSessionId(),
        ctx.sessionManager.getBranch?.()
      )
    );
    // Every workspace this process served may still have a write-back in flight.
    pi.on("session_shutdown", async () => {
      await Promise.all([...workspaces.values()].map((ws) => ws?.hooks.sessionShutdown()));
    });
  };
}
