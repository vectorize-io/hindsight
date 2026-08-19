/**
 * Prime Agent's extension entrypoint.
 *
 * Prime Agent (PrimeIntellect) loads distributed packages listed in `~/.prime/agent/settings.json`
 * and calls each extension's default export with its `pi` API. This wires Prime Agent's
 * `before_agent_start` (recall + system-prompt injection) and `agent_end` (transcript write-back)
 * onto the shared RuntimeCore — the same reflect-and-inject core every Hindsight harness uses — and
 * registers the `hindsight_*` knowledge tools natively via `pi.registerTool`. The memory behaviour
 * itself stays in RuntimeCore; this file only adapts Prime Agent's API at its boundary.
 */
import { z } from "zod";
import { deriveBankId } from "./core/bank";
import { applyBankConfig, loadConfig } from "./core/config";
import { diag } from "./core/diag";
import { HindsightClient } from "./core/hindsight";
import type { ToolSpec } from "./core/knowledge-tools";
import { RuntimeCore } from "./core/runtime";
import { type PaMessage, readPrimeAgentMessages } from "./core/transcript-prime-agent";

const HARNESS = "prime-agent";

// ── Structural subset of Prime Agent's extension API (@earendil-works/pi-coding-agent) ──────────
// Declared locally so this package takes no dependency on the fast-moving Prime Agent SDK; the real
// runtime passes a compatible object at load time.

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
  messages: readonly PaMessage[];
}

interface BeforeAgentStartResult {
  systemPrompt?: string;
}

interface SessionManagerLike {
  getSessionId(): string;
}

interface ExtensionContext {
  hasUI: boolean;
  ui: { notify(message: string, type?: "info" | "warning" | "error"): void };
  sessionManager: SessionManagerLike;
}

/** A JSON-Schema-shaped parameters object. Prime Agent forwards it to the model provider verbatim. */
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
 * Adapt a harness-agnostic ToolSpec (MCP-shaped, shared by every harness) to a Prime Agent native
 * tool. The spec's Zod raw shape is converted to a JSON Schema for `parameters` — Prime Agent passes
 * that straight to the model provider (it does not run TypeBox validation on tool args), so a plain
 * JSON Schema is exactly what the tool needs. The spec's handler returns an MCP `{content:[{text}]}`
 * result and never throws, so we surface the joined text back to the model.
 */
export function toPrimeAgentTool(spec: ToolSpec): ToolDefinition {
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
 * Make the host-specific Prime Agent hooks testable without importing Prime Agent's SDK. RuntimeCore
 * is the shared lifecycle implementation; this adapter only converts Prime Agent messages at its
 * boundary and never calls Hindsight directly.
 */
export function createPrimeAgentHooks(
  core: Pick<RuntimeCore, "onPrompt" | "getInjection" | "onTranscript">,
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
        diag(HARNESS, "inject_empty", { session: sessionId });
        return undefined;
      }
      diag(HARNESS, "inject_ok", { session: sessionId, chars: injection.length });
      return { systemPrompt: `${event.systemPrompt}\n\n${injection}` };
    },
    async agentEnd(event: { messages: readonly PaMessage[] }, sessionId: string): Promise<void> {
      const turns = readPrimeAgentMessages(event.messages);
      if (turns.length) await core.onTranscript(sessionId, turns);
    },
  };
}

function createRuntime(repoPath: string): RuntimeCore | undefined {
  let cfg = loadConfig({ harness: HARNESS });
  if (cfg.disabled) return undefined;
  const resolved = applyBankConfig(cfg, deriveBankId(cfg, repoPath, HARNESS), repoPath);
  cfg = resolved.cfg;
  if (cfg.disabled) return undefined;
  const client = new HindsightClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: resolved.bankId,
    observationScopes: cfg.observationScopes,
  });
  return new RuntimeCore(client, resolved.bankId, cfg, HARNESS, repoPath);
}

/**
 * Prime Agent extension factory. Loaded once per session in the project directory; resolves config
 * and bank from `process.cwd()`, registers the knowledge tools, kicks off the shared cold-seed, and
 * wires the recall/retain hooks.
 */
const extension: ExtensionFactory = (pi) => {
  const repoPath = process.cwd();
  const core = createRuntime(repoPath);
  if (!core) return;

  for (const spec of core.toolSpecs()) pi.registerTool(toPrimeAgentTool(spec));

  // Fire-and-forget cold seed (bank check + background git seed + knowledge preamble); the first
  // before_agent_start awaits it via createPrimeAgentHooks.
  const sessionStart = core.seedIfCold(repoPath);
  const hooks = createPrimeAgentHooks(core, sessionStart);

  pi.on("before_agent_start", (event, ctx) =>
    hooks.beforeAgentStart(event, ctx.sessionManager.getSessionId())
  );
  pi.on("agent_end", (event, ctx) => hooks.agentEnd(event, ctx.sessionManager.getSessionId()));
};

export { extension };
export default extension;
