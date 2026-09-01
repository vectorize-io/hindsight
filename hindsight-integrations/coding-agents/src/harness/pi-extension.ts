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
    params: Record<string, unknown>
  ): Promise<{ content: { type: "text"; text: string }[]; details: unknown }>;
}

interface ExtensionAPI {
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
      event: { prompt: string; systemPrompt: string },
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
      return { systemPrompt: `${event.systemPrompt}\n\n${injection}` };
    },
    async agentEnd(event: { messages: readonly PiMessage[] }, sessionId: string): Promise<void> {
      const turns = readPiMessages(event.messages);
      if (turns.length) await core.onTranscript(sessionId, turns);
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
 * The returned factory is called once per session in the project directory; it resolves config and
 * bank from `process.cwd()`, registers the knowledge tools, kicks off the shared cold-seed, and
 * wires the recall/retain hooks.
 */
export function createPiExtension(harness: string): ExtensionFactory {
  return (pi) => {
    const repoPath = process.cwd();
    const core = createRuntime(harness, repoPath);
    if (!core) return;

    for (const spec of core.toolSpecs()) pi.registerTool(toPiTool(spec));

    // Fire-and-forget cold seed (bank check + background git seed + knowledge preamble); the first
    // before_agent_start awaits it via createPiHooks.
    const sessionStart = core.seedIfCold(repoPath);
    const hooks = createPiHooks(core, harness, sessionStart);

    pi.on("before_agent_start", (event, ctx) =>
      hooks.beforeAgentStart(event, ctx.sessionManager.getSessionId())
    );
    pi.on("agent_end", (event, ctx) => hooks.agentEnd(event, ctx.sessionManager.getSessionId()));
  };
}
