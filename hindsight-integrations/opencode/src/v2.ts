/**
 * Hindsight OpenCode V2 plugin.
 *
 * OpenCode V2 uses a different plugin API from V1: a default export with an
 * `id` and a `setup(ctx)` function, where behaviour is registered as hooks,
 * transforms, and subscriptions on the plugin context. This module implements
 * the V2 entrypoint; `index.ts` exposes it as the default export alongside the
 * legacy V1 `server()` implementation so one package supports both.
 *
 * V1 -> V2 mapping used here:
 *   - V1 `tool` map            -> `ctx.tool.transform(editor => editor.add(...))`
 *   - V1 `chat.system.transform`-> `ctx.session.hook("context", ...)`
 *   - V1 `session.compacting`  -> `ctx.session.hook("compaction", ...)`
 *   - V1 `event` (session.idle)-> `ctx.event.subscribe()`
 *   - V1 `input.directory`     -> `ctx.location.directory`
 */

import { Plugin } from "@opencode/plugin";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { loadConfig, type HindsightConfig } from "./config.js";
import { deriveBankId, ensureBankMission } from "./bank.js";
import {
  formatMemories,
  formatCurrentTime,
  composeRecallQuery,
  truncateRecallQuery,
  prepareRetentionTranscript,
  sliceLastTurnsByUserBoundary,
  type Message,
} from "./content.js";

interface RecallOutcome {
  context: string | null;
  ok: boolean;
}

/** Normalize OpenCode V2 session messages into `{ role, content }`.
 *  V2 shapes: user `{ type: "user", text }`, assistant `{ type: "assistant", content: [{ type: "text", text }] }`.
 *  Also tolerates the V1 `{ role, parts: [{ type: "text", text }] }` shape. */
function normalizeMessages(raw: unknown): Message[] {
  const out: Message[] = [];
  if (!Array.isArray(raw)) return out;
  for (const m of raw as Array<Record<string, unknown>>) {
    const info = (m?.info ?? {}) as Record<string, unknown>;
    const role = (info.role ?? m?.role ?? m?.type) as string | undefined;
    if (role !== "user" && role !== "assistant") continue;
    let text = "";
    if (typeof m?.text === "string" && m.text) {
      text = m.text;
    } else {
      const parts = (m?.parts ?? m?.content) as unknown;
      if (Array.isArray(parts)) {
        text = (parts as Array<Record<string, unknown>>)
          .filter((p) => p && p.type === "text" && typeof p.text === "string")
          .map((p) => p.text as string)
          .join("\n");
      } else if (typeof parts === "string") {
        text = parts;
      }
    }
    if (text) out.push({ role, content: text });
  }
  return out;
}

const RETENTION_SCHEMA = (properties: Record<string, unknown>, required: string[]) => ({
  type: "object",
  properties,
  required,
  additionalProperties: false,
});

export const HindsightV2Plugin = Plugin.define({
  id: "hindsight",

  async setup(ctx) {
    const config = loadConfig((ctx.options ?? {}) as Record<string, unknown>);
    const debug = config.debug;
    const log = (level: string, message: string, extra?: Record<string, unknown>) => {
      if (level === "debug" && !debug) return;
      console.error(extra ? `[Hindsight] ${message} ${JSON.stringify(extra)}` : `[Hindsight] ${message}`);
    };

    const client = new HindsightClient({
      baseUrl: config.hindsightApiUrl!,
      apiKey: config.hindsightApiToken || undefined,
    });
    const bankId = deriveBankId(config, ctx.location?.directory ?? process.cwd());
    const missionsSet = new Set<string>();
    const recalledSessions = new Set<string>();
    const lastRetainedTurn = new Map<string, number>();

    log("info", "Hindsight plugin initialized", {
      api: config.hindsightApiUrl,
      bank: bankId,
      authenticated: Boolean(config.hindsightApiToken),
      autoRecall: config.autoRecall,
      autoRetain: config.autoRetain,
    });

    // ---- Tools ----------------------------------------------------------

    await ctx.tool.transform((editor) => {
      editor.add({
        name: "hindsight_retain",
        description:
          "Store information in long-term memory. Use this to remember important facts, " +
          "user preferences, project context, decisions, and anything worth recalling in " +
          "future sessions. Be specific — include who, what, when, and why.",
        input: RETENTION_SCHEMA(
          {
            content: {
              type: "string",
              description: "The information to remember. Be specific and self-contained.",
            },
            context: {
              type: "string",
              description: "Optional context about where this information came from.",
            },
          },
          ["content"]
        ),
        async execute(input) {
          const args = input as { content: string; context?: string };
          await ensureBankMission(client, bankId, config, missionsSet);
          await client.retain(bankId, args.content, {
            context: args.context || config.retainContext,
            tags: config.retainTags.length ? config.retainTags : undefined,
            metadata: Object.keys(config.retainMetadata).length ? config.retainMetadata : undefined,
          });
          return { content: "Memory stored successfully." };
        },
      });

      editor.add({
        name: "hindsight_recall",
        description:
          "Search long-term memory for relevant information. Use this proactively before " +
          "answering questions about past conversations, user preferences, project history, " +
          "or any topic where prior context would help. When in doubt, recall first.",
        input: RETENTION_SCHEMA(
          {
            query: {
              type: "string",
              description: "Natural language search query. Be specific about what you need.",
            },
          },
          ["query"]
        ),
        async execute(input) {
          const args = input as { query: string };
          const response = await client.recall(bankId, args.query, {
            budget: config.recallBudget as "low" | "mid" | "high",
            maxTokens: config.recallMaxTokens,
            types: config.recallTypes,
            tags: config.recallTags.length ? config.recallTags : undefined,
            tagsMatch: config.recallTags.length ? config.recallTagsMatch : undefined,
          });
          const results = response.results || [];
          if (!results.length) return { content: "No relevant memories found." };
          return {
            content:
              `Found ${results.length} relevant memories (as of ${formatCurrentTime()} UTC):\n\n` +
              formatMemories(results),
          };
        },
      });

      editor.add({
        name: "hindsight_reflect",
        description:
          "Generate a thoughtful answer using long-term memory. Unlike recall (which returns " +
          "raw memories), reflect synthesizes memories into a coherent answer. Use for " +
          'questions like "What do you know about this user?" or "Summarize our project decisions."',
        input: RETENTION_SCHEMA(
          {
            query: {
              type: "string",
              description: "The question to answer using long-term memory.",
            },
            context: {
              type: "string",
              description: "Optional additional context to guide the reflection.",
            },
          },
          ["query"]
        ),
        async execute(input) {
          const args = input as { query: string; context?: string };
          await ensureBankMission(client, bankId, config, missionsSet);
          const response = await client.reflect(bankId, args.query, {
            context: args.context,
            budget: config.recallBudget as "low" | "mid" | "high",
          });
          return { content: response.text || "No relevant information found to reflect on." };
        },
      });
    });

    // ---- Helpers --------------------------------------------------------

    async function getMessages(sessionID: string): Promise<Message[]> {
      try {
        const raw = await ctx.session.context({ sessionID });
        return normalizeMessages(raw);
      } catch (e) {
        log("error", "Failed to get session messages", { error: String(e) });
        return [];
      }
    }

    async function recallForContext(query: string): Promise<RecallOutcome> {
      try {
        const response = await client.recall(bankId, query, {
          budget: config.recallBudget as "low" | "mid" | "high",
          maxTokens: config.recallMaxTokens,
          types: config.recallTypes,
          tags: config.recallTags.length ? config.recallTags : undefined,
          tagsMatch: config.recallTags.length ? config.recallTagsMatch : undefined,
        });
        const results = response.results || [];
        if (!results.length) return { context: null, ok: true };
        const context =
          `<hindsight_memories>\n` +
          `${config.recallPromptPreamble}\n` +
          `Current time: ${formatCurrentTime()} UTC\n\n` +
          `${formatMemories(results)}\n` +
          `</hindsight_memories>`;
        return { context, ok: true };
      } catch (e) {
        log("error", "Recall failed", { error: String(e) });
        return { context: null, ok: false };
      }
    }

    async function retainSession(sessionID: string, messages: Message[]): Promise<void> {
      const retainFullWindow = config.retainMode === "full-session";
      let targetMessages: Message[];
      let documentId: string;

      if (retainFullWindow) {
        targetMessages = messages;
        documentId = sessionID;
      } else {
        const windowTurns = config.retainEveryNTurns + config.retainOverlapTurns;
        targetMessages = sliceLastTurnsByUserBoundary(messages, windowTurns);
        documentId = `${sessionID}-${Date.now()}`;
      }

      const { transcript } = prepareRetentionTranscript(targetMessages, true);
      if (!transcript) return;

      await ensureBankMission(client, bankId, config, missionsSet);
      await client.retain(bankId, transcript, {
        documentId,
        context: config.retainContext,
        tags: config.retainTags.length ? config.retainTags : undefined,
        metadata: Object.keys(config.retainMetadata).length
          ? { ...config.retainMetadata, session_id: sessionID }
          : { session_id: sessionID },
        async: true,
      });
    }

    async function handleSessionIdle(sessionID: string): Promise<void> {
      if (!config.autoRetain) return;
      const messages = await getMessages(sessionID);
      if (!messages.length) return;

      const userTurns = messages.filter((m) => m.role === "user").length;
      const lastRetained = lastRetainedTurn.get(sessionID) || 0;
      if (userTurns - lastRetained < config.retainEveryNTurns) return;

      try {
        await retainSession(sessionID, messages);
        lastRetainedTurn.set(sessionID, userTurns);
        log("info", `Auto-retained ${messages.length} messages`, { session: sessionID, bank: bankId });
      } catch (e) {
        log("error", "Auto-retain failed", { error: String(e) });
      }
    }

    // ---- Hooks ----------------------------------------------------------

    // Auto-recall into the first model context of each session.
    await ctx.session.hook("context", async (event) => {
      if (!config.autoRecall) return;
      const sessionID = event.sessionID;
      if (!sessionID) return;
      if (recalledSessions.has(sessionID)) return;

      try {
        await ensureBankMission(client, bankId, config, missionsSet);
        const messages = await getMessages(sessionID);
        const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
        let query = "project context and recent work";
        if (lastUserMsg && lastUserMsg.content.trim()) {
          const composed = composeRecallQuery(lastUserMsg.content, messages, config.recallContextTurns);
          query = truncateRecallQuery(composed, lastUserMsg.content, config.recallMaxQueryChars);
        }
        const { context, ok } = await recallForContext(query);
        if (ok) {
          recalledSessions.add(sessionID);
          if (recalledSessions.size > 1000) {
            const first = recalledSessions.values().next().value;
            if (first) recalledSessions.delete(first);
          }
        }
        if (context) {
          event.system.push({ type: "text", text: context });
        }
      } catch (e) {
        log("error", "Context hook error", { error: String(e) });
      }
    });

    // Retain before compaction, and inject memories into the compaction context.
    await ctx.session.hook("compaction", async (event) => {
      const sessionID = event.sessionID;
      if (!sessionID) return;
      try {
        const messages = await getMessages(sessionID);
        if (messages.length && config.autoRetain) {
          try {
            await retainSession(sessionID, messages);
            lastRetainedTurn.delete(sessionID);
          } catch (e) {
            log("error", "Pre-compaction retain failed", { error: String(e) });
          }
        }
        if (messages.length) {
          const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
          if (lastUserMsg) {
            const query = composeRecallQuery(lastUserMsg.content, messages, config.recallContextTurns);
            const truncated = truncateRecallQuery(query, lastUserMsg.content, config.recallMaxQueryChars);
            const { context } = await recallForContext(truncated);
            if (context) event.system.push({ type: "text", text: context });
          }
        }
      } catch (e) {
        log("error", "Compaction hook error", { error: String(e) });
      }
    });

    // Auto-retain on end-of-turn.
    // OpenCode v1 emitted `session.idle`; v2 delivers the durable
    // `session.execution.succeeded` event to plugins instead (`session.idle` is a
    // client/status event plugins never receive). Accept both.
    const RETAIN_EVENTS = new Set(["session.idle", "session.execution.succeeded"]);
    const controller = new AbortController();
    void (async () => {
      try {
        for await (const evt of ctx.event.subscribe({ signal: controller.signal })) {
          const e = evt as { type?: string; properties?: { sessionID?: string }; data?: { sessionID?: string } };
          if (e?.type && RETAIN_EVENTS.has(e.type)) {
            const sessionID = e.properties?.sessionID || e.data?.sessionID;
            if (sessionID) await handleSessionIdle(sessionID);
          }
        }
      } catch {
        /* stream aborted on unload */
      }
    })();

    return () => controller.abort();
  },
});

export type { HindsightConfig };
