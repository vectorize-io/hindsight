/**
 * Cline CLI's native plugin entrypoint.
 *
 * Cline's external file hooks are intentionally not used here: their TaskStart and
 * UserPromptSubmit commands are observational and cannot alter the model request. The in-process
 * `beforeModel` hook can, so it is the only path that can provide the same reflect/page injection
 * guarantee as every other Hindsight harness. The memory behaviour itself stays in RuntimeCore.
 */
import { applyBankConfig, loadConfig } from "./core/config";
import { deriveBankId } from "./core/bank";
import { HindsightClient } from "./core/hindsight";
import { RuntimeCore } from "./core/runtime";
import type { TransportTurn } from "./core/chat";
import { diag } from "./core/diag";

const HARNESS = "cline-cli";
const INJECTION_METADATA_KEY = "hindsight_coding_agents_injection";
// Cline CLI 3.0.48 runs plugins in a sandbox and aborts every hook RPC at 3 seconds. Leave room
// for sandbox IPC/serialization; a reflect may take longer, but must never make Cline abort a run.
export const CLINE_HOOK_BUDGET_MS = 2_200;

interface ClineMessagePart {
  type: string;
  text?: string;
  toolName?: string;
  input?: unknown;
  output?: unknown;
}

interface ClineMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: ClineMessagePart[];
  createdAt: number;
  metadata?: Record<string, unknown>;
}

interface ClineSnapshot {
  agentId: string;
  conversationId?: string;
  messages: readonly ClineMessage[];
}

interface ClineBeforeModelContext {
  snapshot: ClineSnapshot;
  request: { messages: readonly ClineMessage[] };
}

interface ClinePlugin {
  name: string;
  manifest: { capabilities: string[] };
  setup: (
    _api: unknown,
    ctx: { workspaceInfo?: { rootPath?: string }; session?: { sessionId?: string } }
  ) => void;
  hooks: {
    beforeModel: (
      context: ClineBeforeModelContext
    ) => Promise<{ messages: ClineMessage[] } | undefined>;
    afterRun: (context: { snapshot: ClineSnapshot }) => Promise<void>;
  };
}

/**
 * Retain only the user-visible conversation. Tool arguments/results can contain bulky command
 * output, credentials, or unrelated file contents; they are execution trace data, not durable
 * conversational memory. Cline's tool-role messages and non-text parts are intentionally omitted.
 */
export function clineTranscript(messages: readonly ClineMessage[]): TransportTurn[] {
  return messages.flatMap((message) => {
    if (message.metadata?.[INJECTION_METADATA_KEY] === true || message.role === "tool") return [];
    const content = message.content
      .filter((part) => part.type === "text" && part.text)
      .map((part) => part.text!)
      .join("\n")
      .trim();
    if (!content) return [];
    return [
      {
        role: message.role,
        content,
        timestamp: new Date(message.createdAt).toISOString(),
      },
    ];
  });
}

function latestUserMessage(messages: readonly ClineMessage[]): ClineMessage | undefined {
  return [...messages].reverse().find((message) => message.role === "user");
}

function sessionId(snapshot: ClineSnapshot, configuredSessionId?: string): string {
  return snapshot.conversationId ?? configuredSessionId ?? snapshot.agentId;
}

async function settleWithinHookBudget(work: Promise<void>): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work.then(
        () => true,
        () => true // RuntimeCore is fail-open; an adapter failure must be equally non-fatal.
      ),
      new Promise<false>((resolve) => {
        timer = setTimeout(() => resolve(false), CLINE_HOOK_BUDGET_MS);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Make the host-specific Cline hooks testable without importing Cline's SDK into this package.
 * RuntimeCore is the shared lifecycle implementation; this adapter only converts Cline messages
 * at its boundary and never calls Hindsight directly.
 */
export function createClineHooks(
  core: Pick<RuntimeCore, "onPrompt" | "getInjection" | "onTranscript">,
  configuredSessionId?: string,
  sessionStart?: Promise<void>
) {
  const injectedPrompts = new Set<string>();
  let sessionStartAwaited = false;
  return {
    async beforeModel({
      snapshot,
      request,
    }: ClineBeforeModelContext): Promise<{ messages: ClineMessage[] } | undefined> {
      if (!sessionStartAwaited) {
        sessionStartAwaited = true;
        // `seedIfCold` sets the shared new-bank marker before the first prompt. Awaiting it here,
        // rather than racing plugin setup, preserves the invariant that a brand-new bank skips its
        // first auto-reflect instead of spending that one synthesis before it has any knowledge.
        // Bounded by the same budget as onPrompt below: seedIfCold now also waits for a cold local
        // daemon (up to DAEMON_WAIT_SESSION_START_MS), and this is the one host that awaits it
        // inside an RPC the sandbox aborts at 3s. It stays running either way, so a start that
        // outlasts the budget just lands for a later turn.
        await settleWithinHookBudget(sessionStart ?? Promise.resolve());
      }
      const user = latestUserMessage(snapshot.messages);
      if (!user) return undefined;
      const id = sessionId(snapshot, configuredSessionId);
      const promptKey = `${id}:${user.id}`;
      if (!injectedPrompts.has(promptKey)) {
        injectedPrompts.add(promptKey);
        const prompt = user.content
          .filter((part) => part.type === "text" && part.text)
          .map((part) => part.text!)
          .join("\n")
          .trim();
        if (prompt) {
          const completed = await settleWithinHookBudget(core.onPrompt(id, prompt));
          if (!completed) {
            // Keep the shared operation alive in the sandbox: a subsequent model call or user turn
            // can consume the completed injection, while this first call remains safely fail-open.
            diag(HARNESS, "inject_deferred_sandbox_budget", { session: id });
          }
        }
      }
      const context = core.getInjection(id);
      if (!context) {
        diag(HARNESS, "inject_empty", { session: id });
        return undefined;
      }
      diag(HARNESS, "inject_ok", { session: id, chars: context.length });
      return {
        messages: [
          ...request.messages,
          {
            id: `hindsight:${user.id}`,
            role: "user",
            content: [{ type: "text", text: context }],
            createdAt: Date.now(),
            // This marker is defensive: Cline currently does not commit beforeModel's replacement
            // messages to its snapshot, but retaining it would store our own prompt as user intent.
            metadata: { [INJECTION_METADATA_KEY]: true },
          },
        ],
      };
    },
    async afterRun({ snapshot }: { snapshot: ClineSnapshot }): Promise<void> {
      const turns = clineTranscript(snapshot.messages);
      if (turns.length) await core.onTranscript(sessionId(snapshot, configuredSessionId), turns);
    },
  };
}

function createRuntime(workspaceRoot: string | undefined): RuntimeCore | undefined {
  let cfg = loadConfig({ harness: HARNESS });
  if (cfg.disabled) return undefined;
  const resolved = applyBankConfig(
    cfg,
    deriveBankId(cfg, workspaceRoot || process.cwd(), HARNESS),
    workspaceRoot || process.cwd()
  );
  cfg = resolved.cfg;
  if (cfg.disabled) return undefined;
  const client = new HindsightClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: resolved.bankId,
    maxParallelRetains: cfg.maxParallelRetains,
    observationScopes: cfg.observationScopes,
  });
  return new RuntimeCore(client, resolved.bankId, cfg, HARNESS, workspaceRoot || process.cwd());
}

const plugin: ClinePlugin = {
  name: "hindsight-coding-agents",
  manifest: { capabilities: ["hooks"] },
  setup(_api, ctx) {
    const core = createRuntime(ctx.workspaceInfo?.rootPath);
    if (!core) return;
    const sessionStart = core.seedIfCold(ctx.workspaceInfo?.rootPath);
    const hooks = createClineHooks(core, ctx.session?.sessionId, sessionStart);
    plugin.hooks.beforeModel = hooks.beforeModel;
    plugin.hooks.afterRun = hooks.afterRun;
    // This starts the shared SessionStart lifecycle once for the Cline session: bank cold-check,
    // knowledge pages, and background seed. Do not await it in setup because Cline loads plugins
    // before rendering its TUI; the first beforeModel waits for this shared lifecycle work.
  },
  hooks: {
    async beforeModel() {
      return undefined;
    },
    async afterRun() {
      return undefined;
    },
  },
};

export { plugin };
export default plugin;
