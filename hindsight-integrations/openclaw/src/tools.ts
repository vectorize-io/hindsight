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

// The knowledge-base tools mirror Hindsight's own MCP tools of the same names
// (hindsight_api/mcp_tools.py), on the session's bank. Deleting needs
// allowKnowledgeDelete: it removes a whole subtree and cannot be undone.
export const HINDSIGHT_TOOL_NAMES = [
  "hindsight_retain",
  "hindsight_recall",
  "hindsight_reflect",
  "hindsight_get_knowledge_base_tree",
  "hindsight_search_knowledge_base",
  "hindsight_get_knowledge_page",
  "hindsight_create_knowledge_folder",
  "hindsight_create_knowledge_page",
  "hindsight_update_knowledge_node",
  "hindsight_delete_knowledge_node",
];

// What a read-only (stateless) session must not get.
const WRITE_TOOLS = new Set([
  "hindsight_retain",
  "hindsight_create_knowledge_folder",
  "hindsight_create_knowledge_page",
  "hindsight_update_knowledge_node",
  "hindsight_delete_knowledge_node",
]);

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

function json(value: unknown): PluginToolResult {
  return text(JSON.stringify(value, null, 2));
}

function optionalString(params: Record<string, unknown>, name: string): string | undefined {
  const value = params[name];
  return typeof value === "string" && value ? value : undefined;
}

function optionalTags(params: Record<string, unknown>): string[] | undefined {
  return Array.isArray(params.tags)
    ? params.tags.filter((t): t is string => typeof t === "string")
    : undefined;
}

function optionalNumber(params: Record<string, unknown>, name: string): number | undefined {
  const value = params[name];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
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
    {
      name: "hindsight_get_knowledge_base_tree",
      label: "Hindsight Knowledge Base Tree",
      description:
        "Browse the knowledge base as a nested tree of folders and pages. Each page is a " +
        "living markdown document synthesized from memories; folders group them. Start here " +
        "to discover what is documented, then read a page with hindsight_get_knowledge_page. " +
        "Pages report `is_stale`: true means something was written since the last refresh.",
      parameters: { type: "object", properties: {} },
      async execute(_id, _params, signal) {
        const { bankId, client } = await bank();
        try {
          return json(await client.getKnowledgeBaseTree(bankId, { signal }));
        } catch (error) {
          if (bankNotWrittenYet(error)) return json({ roots: [] });
          throw error;
        }
      },
    },
    {
      name: "hindsight_search_knowledge_base",
      label: "Hindsight Knowledge Base Search",
      description:
        "Find knowledge pages by relevance (keyword + semantic search over page names and " +
        "content), with a short snippet each. Read a hit in full with " +
        "hindsight_get_knowledge_page. Use hindsight_recall to search raw memories instead.",
      parameters: {
        type: "object",
        required: ["query"],
        properties: {
          query: { type: "string", description: "What to search for." },
          limit: { type: "number", description: "Maximum pages to return (1-50, default 10)." },
        },
      },
      async execute(_id, params, signal) {
        const query = requiredString(params, "query");
        const limit = optionalNumber(params, "limit");
        const { bankId, client } = await bank();
        try {
          return json(
            await client.searchKnowledgeBase(bankId, query, {
              // Clamped like the MCP tool: an agent asking for 500 pages wants results.
              limit: limit === undefined ? undefined : Math.max(1, Math.min(Math.trunc(limit), 50)),
              signal,
            })
          );
        } catch (error) {
          if (bankNotWrittenYet(error)) return json({ results: [], total: 0 });
          throw error;
        }
      },
    },
    {
      name: "hindsight_get_knowledge_page",
      label: "Hindsight Knowledge Page",
      description:
        "Read a knowledge page as a markdown document: YAML frontmatter (id, type, title, " +
        "description, tags, timestamp) followed by its synthesized body. Find page ids with " +
        "hindsight_get_knowledge_base_tree or hindsight_search_knowledge_base.",
      parameters: {
        type: "object",
        required: ["page_id"],
        properties: {
          page_id: { type: "string", description: "The page to read (a `kp-...` node id)." },
        },
      },
      async execute(_id, params, signal) {
        const pageId = requiredString(params, "page_id");
        const { bankId, client } = await bank();
        return json(await client.getKnowledgePage(bankId, pageId, { signal }));
      },
    },
    {
      name: "hindsight_create_knowledge_folder",
      label: "Hindsight Create Knowledge Folder",
      description:
        "Create a folder in the knowledge base. Folders group pages and hold no content.",
      parameters: {
        type: "object",
        required: ["name"],
        properties: {
          name: { type: "string", description: "Folder name." },
          parent_id: {
            type: "string",
            description: "Parent folder (a `kf-...` node id). Omit for the top level.",
          },
        },
      },
      async execute(_id, params, signal) {
        const name = requiredString(params, "name");
        const { bankId, client } = await bank();
        return json(
          await client.createKnowledgeFolder(bankId, name, {
            parentId: optionalString(params, "parent_id"),
            signal,
          })
        );
      },
    },
    {
      name: "hindsight_create_knowledge_page",
      label: "Hindsight Create Knowledge Page",
      description:
        "Create a knowledge page: a living document Hindsight writes and keeps up to date " +
        "from memories, answering `source_query`. Content is generated in the background " +
        "after this returns.",
      parameters: {
        type: "object",
        required: ["name", "source_query"],
        properties: {
          name: { type: "string", description: "Page title." },
          source_query: {
            type: "string",
            description:
              "The question the page answers, e.g. 'What are the user's food preferences?'.",
          },
          parent_id: {
            type: "string",
            description: "Folder to put it in (a `kf-...` node id). Omit for the top level.",
          },
          tags: {
            type: "array",
            items: { type: "string" },
            description:
              "Only build the page from memories carrying ALL these tags. Tags the memories " +
              "don't carry build an empty page.",
          },
          max_tokens: { type: "number", description: "Maximum length of the page body." },
        },
      },
      async execute(_id, params, signal) {
        const name = requiredString(params, "name");
        const sourceQuery = requiredString(params, "source_query");
        const { bankId, client } = await bank();
        return json(
          await client.createKnowledgePage(bankId, name, sourceQuery, {
            parentId: optionalString(params, "parent_id"),
            tags: optionalTags(params),
            maxTokens: optionalNumber(params, "max_tokens"),
            signal,
          })
        );
      },
    },
    {
      name: "hindsight_update_knowledge_node",
      label: "Hindsight Update Knowledge Node",
      description:
        "Rename or move a knowledge folder or page, or change a page's question, tags or " +
        "length. Only the fields you pass change. A new `source_query` rebuilds the page.",
      parameters: {
        type: "object",
        required: ["node_id"],
        properties: {
          node_id: {
            type: "string",
            description: "The folder (`kf-...`) or page (`kp-...`) to update.",
          },
          name: { type: "string", description: "New name." },
          parent_id: {
            type: "string",
            description: "New parent folder (a `kf-...` node id), or `root` for the top level.",
          },
          source_query: { type: "string", description: "Pages only: the new question." },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Pages only: replaces the page's tags ([] clears them).",
          },
          max_tokens: { type: "number", description: "Pages only: maximum body length." },
        },
      },
      async execute(_id, params, signal) {
        const nodeId = requiredString(params, "node_id");
        const parentId = optionalString(params, "parent_id");
        const update = {
          name: optionalString(params, "name"),
          // The client sends only the keys present, and `null` moves to the root.
          ...(parentId === undefined ? {} : { parentId: parentId === "root" ? null : parentId }),
          sourceQuery: optionalString(params, "source_query"),
          tags: optionalTags(params),
          maxTokens: optionalNumber(params, "max_tokens"),
        };
        if (Object.values(update).every((value) => value === undefined)) {
          throw new Error(
            "Provide name, parent_id, source_query, tags and/or max_tokens to update"
          );
        }
        const { bankId, client } = await bank();
        return json(await client.updateKnowledgeNode(bankId, nodeId, { ...update, signal }));
      },
    },
    {
      name: "hindsight_delete_knowledge_node",
      label: "Hindsight Delete Knowledge Node",
      description:
        "Delete a knowledge folder or page and everything under it. This cannot be undone.",
      parameters: {
        type: "object",
        required: ["node_id"],
        properties: {
          node_id: {
            type: "string",
            description: "The folder (`kf-...`) or page (`kp-...`) to delete.",
          },
        },
      },
      async execute(_id, params, signal) {
        const nodeId = requiredString(params, "node_id");
        const { bankId, client } = await bank();
        await client.deleteKnowledgeNode(bankId, nodeId, { signal });
        return json({ status: "deleted", node_id: nodeId });
      },
    },
  ];
  return tools.filter(
    (tool) =>
      !(readOnly && WRITE_TOOLS.has(tool.name)) &&
      (tool.name !== "hindsight_delete_knowledge_node" || config.allowKnowledgeDelete === true)
  );
}
