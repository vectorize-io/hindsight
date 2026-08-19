/**
 * DeepSeek Harness (`dsh`) entrypoint — a native Cordis plugin.
 *
 * dsh is plugin-first: its canonical extension surface is a set of typed lifecycle events, and its
 * Claude Code / Codex hook bridges are just translators onto that same surface. So this integration
 * binds the events directly instead of installing hook binaries — a bridge would cost the same
 * install (neither bridge ships in a default profile) while losing the transcript, the session id
 * and the awaited stop boundary.
 *
 *   agent/session-start  -> RuntimeCore.seedIfCold      (cold-check + background git/codebase seed)
 *   agent/pre-step       -> RuntimeCore.onPrompt        (recall) + the injection as a sourced message
 *   agent/turn-stopping  -> RuntimeCore.onSessionIdle   (write-back of the completed exchange)
 *   ctx.tools            -> the hindsight_* knowledge suite, registered natively
 *
 * ONE dsh PROCESS SERVES MANY WORKSPACES. Unlike opencode/Kilo/Cline — one process, one project —
 * dsh's Web UI creates sessions in whatever directory the user picks, each with its own
 * `session.header.cwd`. Memory is per-repo, so a RuntimeCore (bank, client, seed) is built PER
 * WORKSPACE ROOT and cached, never once per process.
 *
 * The plugin imports NOTHING from dsh at all: every host shape is structurally typed here, and the
 * tool definitions are built in dsh's own registry shape rather than through its `defineTool`
 * builder (see `toDshParameters`). So there is no dsh package for pnpm to resolve inside a profile,
 * and no version to keep in step — any dsh whose event names still match can load this file.
 */
import { randomUUID } from "node:crypto";
import { deriveBankId } from "./core/bank";
import { applyBankConfig, loadConfig } from "./core/config";
import { diag } from "./core/diag";
import { HindsightClient } from "./core/hindsight";
import type { ToolSpec } from "./core/knowledge-tools";
import { log } from "./core/log";
import { RuntimeCore } from "./core/runtime";
import { HINDSIGHT_PLUGIN, readDshEvents, type DshSessionEvent } from "./core/transcript-dsh";

const HARNESS = "dsh";

/** Cordis plugin name (loader diagnostics, and the `source.plugin` on everything we inject). */
export const name = HINDSIGHT_PLUGIN;

/** The agent registry owns the lifecycle events below; without it there is nothing to bind. */
export const inject = ["agents"];

// ── host shapes (structural: this package must not depend on dsh's packages) ─────

interface DshSession {
  readonly header: { readonly id: string; readonly cwd?: string; readonly origin?: string };
  readonly events: readonly DshSessionEvent[];
}

interface DshAgent {
  readonly session: DshSession;
}

/** A `user/message` as dsh's pre-step decision carries it. */
interface DshUserMessage {
  role: "user";
  content: { type: string; text?: string }[];
  source: { kind: string; plugin?: string; form?: string };
  id?: string;
}

type PreStepDecision =
  | { kind: "enter"; messages: DshUserMessage[] }
  | { kind: "reject"; [key: string]: unknown };

interface PreStepPayload {
  agent: DshAgent;
  signal: AbortSignal;
}

/** The Cordis context surface this plugin uses. */
interface DshContext {
  on(
    event: "agent/session-start",
    listener: (payload: { agent: DshAgent }) => void
  ): (() => void) | void;
  on(
    event: "agent/pre-step",
    listener: (
      payload: PreStepPayload,
      next: () => Promise<PreStepDecision>
    ) => Promise<PreStepDecision>,
    options?: { prepend?: boolean }
  ): (() => void) | void;
  on(
    event: "agent/turn-stopping",
    listener: (payload: { agent: DshAgent }) => Promise<void> | void
  ): (() => void) | void;
  on(
    event: "agent/disposed",
    listener: (payload: { agent: DshAgent }) => void
  ): (() => void) | void;
  inject(names: string[], callback: (ctx: DshToolContext) => void): void;
}

interface DshToolContext {
  tools: {
    register(definition: unknown): void;
    /** Registry introspection; used only to log what the host ended up exposing. */
    schemas?(): { name: string }[];
  };
}

/** `exec` as a dsh tool body receives it (only the caller matters to us). */
interface DshToolRunContext {
  readonly agent?: DshAgent;
}

// ── per-workspace runtime cache ─────────────────────────────────────────────────

export interface Workspace {
  core: RuntimeCore;
  root: string;
  /** Started by the first SESSION in this repo, then shared by every later one. */
  seeded?: Promise<void>;
  tools?: ToolSpec[];
}

/** `null` caches "memory is off for this repo" so a disabled bank isn't re-derived every turn. */
const workspaces = new Map<string, Workspace | null>();
/**
 * sessionId -> agent, so a core's transcript source (which only receives the id) can read the live
 * log back on stop. `agent/disposed` is what bounds it: an agent holds its whole session log, so
 * dropping the entry when the host disposes the agent is what keeps a long-lived Web server from
 * pinning every session it ever opened.
 */
const liveAgents = new Map<string, DshAgent>();

/** Where a session is working. dsh records it on the session header; a session without one is rare. */
function workspaceRoot(agent: DshAgent): string {
  return agent.session.header.cwd || process.cwd();
}

/**
 * The memory runtime for one workspace root, built on first use.
 *
 * Bank derivation and the per-bank opt-out live here rather than in `apply`, because the plugin
 * loads before any session exists and the process may end up serving several unrelated
 * repositories. Building one has NO side effect: the seed is started by `ensureSeeded` when a
 * session actually opens, so merely registering the tools never seeds the launch directory.
 */
function workspaceFor(root: string): Workspace | undefined {
  const cached = workspaces.get(root);
  if (cached !== undefined) return cached ?? undefined;

  let cfg = loadConfig({ harness: HARNESS });
  if (cfg.disabled) {
    workspaces.set(root, null); // inert: same agent, no memory (baseline parity)
    return undefined;
  }
  const resolved = applyBankConfig(cfg, deriveBankId(cfg, root, HARNESS));
  cfg = resolved.cfg;
  if (cfg.disabled) {
    workspaces.set(root, null); // per-bank opt-out (banks.<id> override)
    return undefined;
  }
  const client = new HindsightClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: resolved.bankId,
    observationScopes: cfg.observationScopes,
  });
  // The 5th argument is this host's whole reason for existing here: RuntimeCore binds the
  // knowledge tools (and retain stamps) to the workspace it is given, and dsh's process cwd is the
  // launch directory, routinely a different repository from the session's.
  const core = new RuntimeCore(client, resolved.bankId, cfg, HARNESS, root);
  // The write-back path reads the log back off the live agent — dsh keeps the whole session in
  // memory, so unlike opencode there is nothing to refetch over HTTP.
  core.setTranscriptSource(async (sessionId) => {
    const agent = liveAgents.get(sessionId);
    return agent ? readDshEvents(agent.session.events) : [];
  });
  const workspace: Workspace = { core, root };
  workspaces.set(root, workspace);
  return workspace;
}

/**
 * Start this repo's cold-check + background seed once.
 *
 * Fire-and-forget by design: `agent/session-start` is an emit that no extension point awaits, and
 * the Web UI must not wait on a network round-trip to open a session. The first prompt tolerates an
 * empty knowledge preamble until this resolves.
 */
function ensureSeeded(workspace: Workspace): void {
  workspace.seeded ??= workspace.core.seedIfCold(workspace.root);
}

function workspaceForAgent(agent: DshAgent): Workspace | undefined {
  // A subagent session is a child of a conversation we already track: injecting into it would pay
  // for a second recall per delegation, and retaining it would file a fragment of a session that
  // the parent already stores in full.
  if (agent.session.header.origin === "subagent") return undefined;
  return workspaceFor(workspaceRoot(agent));
}

// ── message helpers ─────────────────────────────────────────────────────────────

/**
 * The human text in this step's claimed batch.
 *
 * `agent/pre-step` fires before EVERY step, including tool continuations that claimed no new input,
 * and other plugins contribute their own `kind: 'plugin'` messages to the same batch. Only a
 * `kind: 'user'` message is a prompt worth recalling on.
 */
function promptOf(messages: readonly DshUserMessage[]): string {
  return (messages || [])
    .filter((message) => message?.source?.kind === "user")
    .flatMap((message) => message.content || [])
    .filter((block) => block?.type === "text" && block.text)
    .map((block) => block.text!)
    .join("\n")
    .trim();
}

/**
 * Build the injected memory message.
 *
 * `form: 'recall'` is dsh's own vocabulary for retrieved context, so its UI renders the block as
 * recalled material rather than as something the user typed, and `plugin: 'hindsight'` names us in
 * the durable log. Neither is what keeps the block out of a write-back — transcript-dsh.ts drops
 * every plugin-sourced message, ours included.
 */
function injectionMessage(text: string): DshUserMessage {
  return {
    id: randomUUID(),
    role: "user",
    content: [{ type: "text", text }],
    source: { kind: "plugin", plugin: HINDSIGHT_PLUGIN, form: "recall" },
  };
}

// ── the lifecycle, testable without a dsh host ──────────────────────────────────

/**
 * The three lifecycle handlers, over an injectable workspace resolver. `apply` binds these to the
 * real Cordis events; the tests drive them with a stand-in agent and core.
 */
export function createDshHooks(resolve: (agent: DshAgent) => Workspace | undefined) {
  return {
    sessionStart({ agent }: { agent: DshAgent }): void {
      const workspace = resolve(agent);
      if (!workspace) return;
      liveAgents.set(agent.session.header.id, agent);
      ensureSeeded(workspace);
    },

    async preStep(
      { agent, signal }: PreStepPayload,
      next: () => Promise<PreStepDecision>
    ): Promise<PreStepDecision> {
      const decision = await next();
      if (decision.kind !== "enter" || signal.aborted) return decision;
      const workspace = resolve(agent);
      if (!workspace) return decision;
      const sessionId = agent.session.header.id;
      liveAgents.set(sessionId, agent);
      // A resumed session gets no `agent/session-start` in some compositions; seeding here too
      // keeps the cold-check exactly-once without depending on which events a host fired.
      ensureSeeded(workspace);
      // No new human input means a tool continuation: recall already ran for this turn, and its
      // block is already in the history the model is about to be sent.
      const prompt = promptOf(decision.messages);
      if (!prompt) return decision;
      await workspace.core.onPrompt(sessionId, prompt);
      const injection = workspace.core.getInjection(sessionId);
      if (!injection) {
        diag(HARNESS, "inject_empty", { session: sessionId });
        return decision;
      }
      diag(HARNESS, "inject_ok", { session: sessionId, chars: injection.length });
      return { kind: "enter", messages: [...decision.messages, injectionMessage(injection)] };
    },

    async turnStopping({ agent }: { agent: DshAgent }): Promise<void> {
      const workspace = resolve(agent);
      if (!workspace) return;
      const sessionId = agent.session.header.id;
      liveAgents.set(sessionId, agent);
      // The stop boundary is the only moment the completed exchange is readable, so this bypasses
      // the turn cadence exactly like the opencode idle path it shares (`onSessionIdle`).
      await workspace.core.onSessionIdle(sessionId);
    },

    disposed({ agent }: { agent: DshAgent }): void {
      liveAgents.delete(agent.session.header.id);
    },
  };
}

// ── native tools ────────────────────────────────────────────────────────────────

/**
 * Project one harness-agnostic ToolSpec's input onto the JSON Schema dsh's registry stores.
 *
 * dsh's own `defineTool` is a pure builder — it compiles this exact schema shape and wraps the
 * callbacks — and `ctx.tools.register` validates a definition structurally, not by module identity.
 * Building the schema here therefore keeps the plugin free of a `@deepseek-ai/dsh-tools` dependency
 * that pnpm would have to resolve inside the profile, at the cost of one small projection.
 *
 * Our specs state their input as a Zod raw shape (the MCP dialect every harness shares) and are
 * deliberately flat: required or optional strings. Optionality and the description are the only two
 * facts to carry across; a parameter that is ever anything but a string fails loudly here rather
 * than registering a mis-typed tool.
 */
export function toDshParameters(spec: ToolSpec): Record<string, unknown> {
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  for (const [key, schema] of Object.entries(spec.inputSchema)) {
    const zod = schema as {
      def?: { type?: string; innerType?: { def?: { type?: string } } };
      description?: string;
      isOptional?: () => boolean;
    };
    // `.optional()` wraps the real type, so unwrap before checking it — otherwise every optional
    // parameter would pass the guard below whatever it actually holds.
    const inner = zod.def?.type === "optional" ? zod.def.innerType?.def?.type : zod.def?.type;
    if (inner !== undefined && inner !== "string") {
      throw new Error(
        `hindsight tool ${spec.name}: dsh projection supports string parameters only`
      );
    }
    properties[key] = {
      type: "string",
      ...(zod.description ? { description: zod.description } : {}),
    };
    if (typeof zod.isOptional !== "function" || !zod.isOptional()) required.push(key);
  }
  return { type: "object", properties, ...(required.length ? { required } : {}) };
}

/** The tool bodies return MCP-shaped results and never throw; dsh wants a value or a throw. */
async function runSpec(spec: ToolSpec, args: Record<string, unknown>): Promise<string> {
  const result = await spec.handler(args);
  const text = (result.content || []).map((block) => block.text).join("\n");
  // dsh normalizes a thrown tool body into an isError result with this message, which is exactly
  // what our isError:true means — mapping it any other way would report a failure as a success.
  if (result.isError) throw new Error(text || "hindsight tool failed");
  return text;
}

/**
 * Register the hindsight_* suite on `ctx.tools`.
 *
 * The dsh tools registry is host+per-scope layered, so a host-plane registration is visible to
 * every agent — including the Web surface's per-session agent presets. Each call resolves the bank
 * from the CALLING agent's workspace, because one process serves many repositories.
 */
function registerTools(toolCtx: DshToolContext, fallbackRoot: string): void {
  // The registry is populated at plugin load, before any session exists, so the NAMES and schemas
  // come from the launch directory's workspace and each call rebinds to its own caller's bank. The
  // one consequence: if memory is off for the launch directory specifically (a `banks.<id>`
  // opt-out) the tools are absent even for other repositories this process later serves.
  const template = workspaceFor(fallbackRoot);
  if (!template) return; // memory disabled where this process was launched
  for (const spec of specsOf(template)) {
    toolCtx.tools.register({
      name: spec.name,
      description: spec.description,
      parameters: toDshParameters(spec),
      output: {
        schema: { type: "string" },
        render: (_args: unknown, value: string) => [{ type: "text", text: value }],
      },
      async execute(args: Record<string, unknown>, exec: DshToolRunContext): Promise<string> {
        // Bind to the caller's repository, not to whichever workspace happened to load first.
        const workspace = exec?.agent ? workspaceForAgent(exec.agent) : template;
        if (!workspace) return "Hindsight memory is disabled for this workspace.";
        const bound = specsOf(workspace).find((candidate) => candidate.name === spec.name);
        if (!bound) return `Hindsight tool ${spec.name} is unavailable.`;
        return runSpec(bound, args);
      },
    });
  }
  // Log what the HOST sees, not just what we handed it: a tool that registers cleanly but never
  // reaches a model is the failure mode worth catching, and this is the one line that shows it.
  const exposed = toolCtx.tools.schemas?.() ?? [];
  log.debug(HARNESS, "tools registered", {
    count: specsOf(template).length,
    hostCatalog: exposed.length,
    hindsight: exposed.filter((schema) => schema.name?.startsWith("hindsight_")).length,
  });
}

/** The tool specs for one workspace, built once (each closes over that bank's client and root). */
function specsOf(workspace: Workspace): ToolSpec[] {
  workspace.tools ??= workspace.core.toolSpecs();
  return workspace.tools;
}

// ── cordis entrypoint ───────────────────────────────────────────────────────────

/**
 * Bind the memory lifecycle for the lifetime of `ctx`. Every listener is fail-open: RuntimeCore
 * never throws, and a workspace we cannot resolve simply leaves that session unmemoried.
 */
export function apply(ctx: DshContext): void {
  const hooks = createDshHooks(workspaceForAgent);
  ctx.on("agent/session-start", hooks.sessionStart);
  // `prepend: true` puts this listener outermost, so the injection is appended AFTER every other
  // contributor's messages — the memory block reads last, closest to the model's turn.
  ctx.on("agent/pre-step", hooks.preStep, { prepend: true });
  ctx.on("agent/turn-stopping", hooks.turnStopping);
  ctx.on("agent/disposed", hooks.disposed);
  // The tools registry is optional: a composition without it (a bare headless assembly) still gets
  // recall, injection and write-back.
  ctx.inject(["tools"], (toolCtx) => {
    registerTools(toolCtx, process.cwd());
  });
}

export default { name, inject, apply };
