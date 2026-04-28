/**
 * Agent knowledge tools for OpenClaw.
 *
 * Registered as a ToolFactory — called per session with agent context.
 * Resolves bank ID using deriveBankId, creates tools scoped to that bank.
 *
 * Tools use Hindsight mental models as the storage layer.
 * Pages are created with opinionated defaults for self-learning agents.
 */

import type { PluginConfig, PluginToolContext } from "./types.js";
import { deriveBankId, detectExternalApi } from "./index.js";

const PAGE_DEFAULTS = {
  mode: "delta",
  refresh_after_consolidation: true,
  exclude_mental_models: true,
  fact_types: ["observation"],
};

interface ToolDeps {
  pluginConfig: PluginConfig;
  getApiUrl: () => string;
  getApiToken: () => string | undefined;
}

function req(
  apiUrl: string,
  path: string,
  method: string,
  body?: unknown,
  token?: string,
): Promise<unknown> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const opts: RequestInit = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  return fetch(`${apiUrl}${path}`, opts).then(async (resp) => {
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }
    return resp.json();
  });
}

function ok(data: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }], details: {} };
}

export function createWikiToolFactory(deps: ToolDeps) {
  return (ctx: PluginToolContext) => {
    const bankId = deriveBankId(ctx as any, deps.pluginConfig);
    const apiUrl = deps.getApiUrl();
    const token = deps.getApiToken();
    const bp = `/v1/default/banks/${bankId}`;

    return [
      {
        name: "agent_knowledge_list_pages",
        label: "List knowledge pages",
        description: "List all your knowledge pages. Each page has an ID, name, source query, and synthesized content that auto-updates from your conversations.",
        parameters: { type: "object", properties: {} },
        async execute() {
          return ok(await req(apiUrl, `${bp}/mental-models`, "GET", undefined, token));
        },
      },
      {
        name: "agent_knowledge_get_page",
        label: "Read a knowledge page",
        description: "Read a specific knowledge page by its ID. Returns the full synthesized content.",
        parameters: {
          type: "object",
          properties: { page_id: { type: "string", description: "Page ID (e.g. 'user-preferences')" } },
          required: ["page_id"],
        },
        async execute(_id: string, params: { page_id: string }) {
          return ok(await req(apiUrl, `${bp}/mental-models/${params.page_id}`, "GET", undefined, token));
        },
      },
      {
        name: "agent_knowledge_create_page",
        label: "Create a knowledge page",
        description:
          "Create a new knowledge page. The source_query is a question the system re-asks after each consolidation to rebuild the page from conversation observations. " +
          "Pages auto-update as you have more conversations. Use for: user preferences, procedures, performance data, best practices.",
        parameters: {
          type: "object",
          properties: {
            page_id: { type: "string", description: "Unique page ID, lowercase with hyphens (e.g. 'editorial-preferences')" },
            name: { type: "string", description: "Human-readable page name" },
            source_query: { type: "string", description: "The question that rebuilds this page (e.g. 'What are the user\\'s editorial preferences?')" },
          },
          required: ["page_id", "name", "source_query"],
        },
        async execute(_id: string, params: { page_id: string; name: string; source_query: string }) {
          return ok(await req(apiUrl, `${bp}/mental-models`, "POST", {
            id: params.page_id, name: params.name, source_query: params.source_query,
            max_tokens: 4096, trigger: PAGE_DEFAULTS,
          }, token));
        },
      },
      {
        name: "agent_knowledge_update_page",
        label: "Update a knowledge page",
        description: "Update a page's name or source query. The content will re-synthesize on next consolidation.",
        parameters: {
          type: "object",
          properties: {
            page_id: { type: "string", description: "Page ID to update" },
            name: { type: "string", description: "New name (optional)" },
            source_query: { type: "string", description: "New source query (optional)" },
          },
          required: ["page_id"],
        },
        async execute(_id: string, params: { page_id: string; name?: string; source_query?: string }) {
          const body: Record<string, string> = {};
          if (params.name) body.name = params.name;
          if (params.source_query) body.source_query = params.source_query;
          return ok(await req(apiUrl, `${bp}/mental-models/${params.page_id}`, "PATCH", body, token));
        },
      },
      {
        name: "agent_knowledge_delete_page",
        label: "Delete a knowledge page",
        description: "Permanently delete a knowledge page.",
        parameters: {
          type: "object",
          properties: { page_id: { type: "string", description: "Page ID to delete" } },
          required: ["page_id"],
        },
        async execute(_id: string, params: { page_id: string }) {
          await req(apiUrl, `${bp}/mental-models/${params.page_id}`, "DELETE", undefined, token);
          return ok({ success: true });
        },
      },
      {
        name: "agent_knowledge_recall",
        label: "Search memories",
        description: "Search across all retained conversations and documents for specific facts, numbers, or details not covered by your knowledge pages.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string", description: "What to search for" },
            max_results: { type: "number", description: "Max results (default 10)" },
          },
          required: ["query"],
        },
        async execute(_id: string, params: { query: string; max_results?: number }) {
          return ok(await req(apiUrl, `${bp}/memories/recall`, "POST", {
            query: params.query, max_results: params.max_results ?? 10,
          }, token));
        },
      },
      {
        name: "agent_knowledge_ingest",
        label: "Ingest a document",
        description:
          "Upload a document into your memory bank. Pass the full raw content — never summarize before ingesting. " +
          "The system handles chunking and fact extraction. The title becomes the document ID (re-ingesting replaces it).",
        parameters: {
          type: "object",
          properties: {
            title: { type: "string", description: "Document title (becomes the document ID)" },
            content: { type: "string", description: "Full raw document content" },
          },
          required: ["title", "content"],
        },
        async execute(_id: string, params: { title: string; content: string }) {
          const docId = params.title.toLowerCase().replace(/ /g, "-");
          return ok(await req(apiUrl, `${bp}/memories`, "POST", {
            items: [{ content: params.content, document_id: docId }], async: true,
          }, token));
        },
      },
    ];
  };
}
