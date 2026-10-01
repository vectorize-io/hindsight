/**
 * Shared entrypoint factory for the pi-family EXTENSION hosts (pi and its fork Prime Agent).
 *
 * pi (`@earendil-works/pi-coding-agent`) loads extensions listed in `~/.pi/agent/settings.json`;
 * Prime Agent (PrimeIntellect) is a fork of pi and loads the same shape from
 * `~/.prime/agent/settings.json`. Both call an extension's default export with their `pi` API, and
 * both expose the identical surface this adapter needs: `before_agent_start` (recall +
 * system-prompt injection), `agent_end` (transcript write-back), and `registerTool` for the native
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
import { type PiMessage, readPiMessages } from "../core/transcript-pi";

// ── Structural subset of the pi extension API ───────────────────────────────────────────────────
// Declared locally so this package takes no dependency on the fast-moving pi / Prime Agent SDKs;
// the real runtime passes a compatible object at load time.

interface BeforeAgentStartEvent {
  type: "before_agent_start";
  /** The raw user prompt text. */
  prompt: string;
  /** The fully assembled system prompt for this turn. */
  systemPrompt: string;
  /** Present on hosts that compose the prompt from extension sections. */
  systemPromptOptions?: { sections: Record<string, string> };
}

interface AgentEndEvent {
  type: "agent_end";
  /** The conversation messages for the completed agent loop. */
  messages: readonly PiMessage[];
}

interface BeforeAgentStartResult {
  systemPrompt?: string;
}

interface SessionManagerLike {
  getSessionId(): string;
}

/** Only what this adapter reads. Both hosts also expose a UI notifier, but the seed banner it would
 *  carry is raised inside seedIfCold at extension load, before any handler has a `ctx` to notify
 *  through — so it is logged rather than toasted here, and the field is not declared. */
interface ExtensionContext {
  /** The session's working directory. A long-lived host (pi-web-ui) serves many workspaces from one
   *  process, so this — not `process.cwd()` — says which repo a session belongs to. */
  cwd: string;
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
    ctx?: { cwd?: string }
  ): Promise<{ content: { type: "text"; text: string }[]; details: unknown }>;
}

interface ExtensionAPI {
  on(
    event: "session_start",
    handler: (event: { type: "session_start" }, ctx: ExtensionContext) => Promise<void> | void
  ): void;
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
 */
export function toPiTool(spec: ToolSpec): ToolDefinition {
  const parameters = z.toJSONSchema(z.object(spec.inputSchema)) as JsonSchema;
  return {
    name: spec.name,
    label: spec.name,
    description: spec.description,
    parameters,
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const r = await spec.handler(params);
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
  core: Pick<RuntimeCore, "onPrompt" | "getInjection" | "onTranscript">,
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
      const prompt = event.prompt.trim();
      if (prompt) await core.onPrompt(sessionId, prompt);
      const injection = core.getInjection(sessionId);
      if (!injection) {
        diag(harness, "inject_empty", { session: sessionId });
        return undefined;
      }
      diag(harness, "inject_ok", { session: sessionId, chars: injection.length });
      // Hosts that compose the prompt from sections get the memory as one: a returned systemPrompt
      // forces the whole prompt, so sections added by later extensions are dropped (#4841). Older
      // hosts without sections still take the appended full prompt below.
      if (event.systemPromptOptions?.sections) {
        event.systemPromptOptions.sections.hindsight = injection;
        return undefined;
      }
      return { systemPrompt: `${event.systemPrompt}\n\n${injection}` };
    },
    async agentEnd(event: { messages: readonly PiMessage[] }, sessionId: string): Promise<void> {
      const turns = readPiMessages(event.messages);
      if (turns.length) await core.onTranscript(sessionId, turns, true); // the run has ended
    },
  };
}

function createRuntime(harness: string, repoPath: string): RuntimeCore | undefined {
  const { cfg, bankId, client } = resolveHostMemory(harness, repoPath);
  if (cfg.disabled) return undefined; // global switch, per-bank opt-out or optInOnly
  return new RuntimeCore(client, bankId, cfg, harness, repoPath);
}

/**
 * Build the default export for a pi-family extension host. `harness` is the name the host is known
 * by ("pi", "prime-agent"), used for config lookup, bank derivation and diagnostics scoping — it is
 * NOT config-chosen; the entrypoint the host loaded determines it.
 *
 * The returned factory is called once when the extension loads. It cannot know the project yet: a
 * long-lived host such as pi-web-ui loads it once in a process whose `process.cwd()` is wherever
 * the server was launched, then serves sessions for many workspaces. So the repo, config and bank
 * are resolved per session from `ctx.cwd` (once per directory), and the knowledge tools and the
 * cold-seed are set up the first time a session in an enabled directory starts.
 */
export function createPiExtension(harness: string): ExtensionFactory {
  return (pi) => {
    interface Workspace {
      core: RuntimeCore;
      hooks: ReturnType<typeof createPiHooks>;
    }
    const workspaces = new Map<string, Workspace | undefined>();
    let toolsRegistered = false;

    const workspaceFor = (cwd: string): Workspace | undefined => {
      if (workspaces.has(cwd)) return workspaces.get(cwd);
      const core = createRuntime(harness, cwd);
      if (!core) {
        diag(harness, "disabled", { cwd });
        workspaces.set(cwd, undefined);
        return undefined;
      }
      // Fire-and-forget cold seed (bank check + background git seed); the first
      // before_agent_start awaits it via createPiHooks.
      const workspace = { core, hooks: createPiHooks(core, harness, core.seedIfCold(cwd)) };
      workspaces.set(cwd, workspace);
      return workspace;
    };

    const activate = (cwd: string): Workspace | undefined => {
      const workspace = workspaceFor(cwd);
      if (workspace && !toolsRegistered) {
        toolsRegistered = true;
        // Tool names are global to the host, so register them once; each call runs against the
        // workspace of the session that made it.
        for (const spec of workspace.core.toolSpecs()) {
          const tool = toPiTool(spec);
          pi.registerTool({
            ...tool,
            execute: async (toolCallId, params, _signal, _onUpdate, ctx) => {
              const target = workspaceFor(ctx?.cwd || cwd)?.core.toolSpecs();
              const targetSpec = target?.find((t) => t.name === spec.name);
              return toPiTool(targetSpec ?? spec).execute(toolCallId, params);
            },
          });
        }
      }
      return workspace;
    };

    const cwdOf = (ctx: ExtensionContext) => ctx.cwd || process.cwd();

    pi.on("session_start", (_event, ctx) => {
      activate(cwdOf(ctx));
    });
    pi.on("before_agent_start", (event, ctx) =>
      activate(cwdOf(ctx))?.hooks.beforeAgentStart(event, ctx.sessionManager.getSessionId())
    );
    pi.on("agent_end", (event, ctx) =>
      workspaceFor(cwdOf(ctx))?.hooks.agentEnd(event, ctx.sessionManager.getSessionId())
    );
  };
}
