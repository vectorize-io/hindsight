/**
 * The agent tools the Hermes integration offers, offered the same way here (#5635):
 * hindsight_retain, hindsight_recall and hindsight_reflect, with Hermes's names,
 * descriptions and parameters, gated by the same `memoryMode` (Hermes's
 * `memory_mode`): every mode but "context" offers them.
 *
 * No tool takes a bank. Each call works on the bank the session's automatic recall
 * and retain use (agentBankMap, static bank or per-agent/channel/user routing), so
 * a model cannot read or write another agent's or user's memory.
 */
import { HindsightError, type HindsightClient } from "@vectorize-io/hindsight-client";
import { compileSessionPatterns, matchesSessionPattern } from "./session-patterns.js";
import type { PluginConfig, PluginTool, PluginToolContext, PluginToolResult } from "./types.js";

export const HINDSIGHT_TOOL_NAMES = ["hindsight_retain", "hindsight_recall", "hindsight_reflect"];

/** What the tools need from the plugin; index.ts supplies the live implementations. */
export interface HindsightToolDeps {
  /** The bank this session's automatic recall and retain use, or why the session has none. */
  bankFor(ctx: PluginToolContext): { bankId: string } | { error: string };
  /** The ready client, with the configured bank defaults applied to `bankId`. */
  clientFor(bankId: string): Promise<HindsightClient>;
  config(): PluginConfig;
}

const NOTHING_FOUND = "No relevant memories found.";

function text(value: string): PluginToolResult {
  return { content: [{ type: "text", text: value }], details: { result: value } };
}

function requiredString(params: Record<string, unknown>, name: string): string {
  const value = params[name];
  if (typeof value !== "string" || !value) throw new Error(`Missing required parameter: ${name}`);
  return value;
}

// Hindsight answers 404 for a bank nothing has been retained to yet: nothing to find, not an error.
function bankNotWrittenYet(error: unknown): boolean {
  return error instanceof HindsightError && error.statusCode === 404;
}

export function createHindsightTools(
  deps: HindsightToolDeps,
  ctx: PluginToolContext
): PluginTool[] | null {
  const config = deps.config();
  if (config.memoryMode === "context") return null;
  // The session filters auto-recall and auto-retain honour apply here too: an
  // ignored session gets no tools, and a stateless one never writes (no
  // hindsight_retain) and reads only when skipStatelessSessions is false.
  let readOnly = false;
  const sessionKey = ctx.sessionKey;
  if (sessionKey) {
    if (
      matchesSessionPattern(sessionKey, compileSessionPatterns(config.ignoreSessionPatterns ?? []))
    ) {
      return null;
    }
    if (
      matchesSessionPattern(
        sessionKey,
        compileSessionPatterns(config.statelessSessionPatterns ?? [])
      )
    ) {
      if (config.skipStatelessSessions !== false) return null;
      readOnly = true;
    }
  }

  // Resolved per call, not when the tools are built, so each call reads the session
  // identity the hooks cached most recently, exactly as auto-recall does.
  async function bank(): Promise<{ bankId: string; client: HindsightClient }> {
    const resolved = deps.bankFor(ctx);
    if ("error" in resolved) {
      throw new Error(`Hindsight memory is not available in this session: ${resolved.error}`);
    }
    return { bankId: resolved.bankId, client: await deps.clientFor(resolved.bankId) };
  }

  const tools: PluginTool[] = [
    {
      name: "hindsight_retain",
      label: "Hindsight Retain",
      description:
        "Store information to long-term memory. Hindsight automatically " +
        "extracts structured facts, resolves entities, and indexes for retrieval.",
      parameters: {
        type: "object",
        properties: {
          content: { type: "string", description: "The information to store." },
          context: {
            type: "string",
            description: "Short label (e.g. 'user preference', 'project decision').",
          },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Optional per-call tags to merge with configured default retain tags.",
          },
          occurred_at: {
            type: "string",
            description:
              "When the remembered event actually happened, as an ISO-8601 date " +
              "or datetime (e.g. '2026-08-20' or '2026-08-20T14:30:00+02:00'). " +
              "Pass this whenever the memory references a specific event time " +
              "('yesterday', 'last Tuesday', 'on March 3rd') so Hindsight can " +
              "anchor it on the timeline. Omit for timeless facts/preferences.",
          },
        },
        required: ["content"],
      },
      async execute(_id, params, signal) {
        const content = requiredString(params, "content");
        const config = deps.config();
        const { bankId, client } = await bank();
        const callTags = Array.isArray(params.tags)
          ? params.tags.filter((t): t is string => typeof t === "string").map((t) => t.trim())
          : [];
        const tags = [...new Set([...(config.retainTags ?? []), ...callTags].filter(Boolean))];
        const occurredAt = typeof params.occurred_at === "string" ? params.occurred_at.trim() : "";
        await client.retain(bankId, content, {
          context: typeof params.context === "string" ? params.context : undefined,
          tags: tags.length > 0 ? tags : undefined,
          timestamp: occurredAt || undefined,
          metadata: {
            source: config.retainSource || "openclaw",
            ...(ctx.sessionKey ? { session_key: ctx.sessionKey } : {}),
            ...(ctx.agentId ? { agent_id: ctx.agentId } : {}),
          },
          // Like auto-retain: accepted now, searchable once the server has processed it.
          async: true,
          signal,
        });
        return text("Memory queued for storage.");
      },
    },
    {
      name: "hindsight_recall",
      label: "Hindsight Recall",
      description:
        "Search long-term memory. Returns memories ranked by relevance using " +
        "semantic search, keyword matching, entity graph traversal, and reranking.",
      parameters: {
        type: "object",
        required: ["query"],
        properties: { query: { type: "string", description: "What to search for." } },
      },
      async execute(_id, params, signal) {
        const query = requiredString(params, "query");
        const config = deps.config();
        const { bankId, client } = await bank();
        try {
          // The same settings auto-recall uses, so the tool sees what auto-recall would.
          const response = await client.recall(bankId, query, {
            maxTokens: config.recallMaxTokens,
            budget: config.recallBudget,
            types: config.recallTypes,
            preferObservations: config.preferObservations,
            minScores: config.recallMinScores,
            signal,
          });
          const lines = (response.results ?? []).map((r, i) => `${i + 1}. ${r.text}`);
          return text(lines.join("\n") || NOTHING_FOUND);
        } catch (error) {
          if (bankNotWrittenYet(error)) return text(NOTHING_FOUND);
          throw error;
        }
      },
    },
    {
      name: "hindsight_reflect",
      label: "Hindsight Reflect",
      description:
        "Synthesize a reasoned answer from long-term memories. Unlike recall, " +
        "this reasons across all stored memories to produce a coherent response.",
      parameters: {
        type: "object",
        required: ["query"],
        properties: { query: { type: "string", description: "The question to reflect on." } },
      },
      async execute(_id, params, signal) {
        const query = requiredString(params, "query");
        const config = deps.config();
        const { bankId, client } = await bank();
        try {
          const response = await client.reflect(bankId, query, {
            budget: config.recallBudget,
            signal,
          });
          return text(response.text || NOTHING_FOUND);
        } catch (error) {
          if (bankNotWrittenYet(error)) return text(NOTHING_FOUND);
          throw error;
        }
      },
    },
  ];
  return readOnly ? tools.filter((tool) => tool.name !== "hindsight_retain") : tools;
}
