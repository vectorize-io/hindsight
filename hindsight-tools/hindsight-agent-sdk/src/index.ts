/**
 * Hindsight Agent SDK — harness-agnostic knowledge tools.
 *
 * Provides agent_knowledge_* tool definitions that any harness can register.
 * Uses @vectorize-io/hindsight-client for all API calls.
 */

import {
  HindsightClient,
  sdk,
  createClient,
  createConfig,
  type HindsightClientOptions,
} from "@vectorize-io/hindsight-client";

// ── Types ──────────────────────────────────────────────

export interface KnowledgeTool {
  name: string;
  label: string;
  description: string;
  parameters: Record<string, unknown>;
  execute: (params: Record<string, unknown>) => Promise<KnowledgeToolResult>;
}

export interface KnowledgeToolResult {
  content: Array<{ type: "text"; text: string }>;
}

export interface CreateKnowledgeToolsOptions {
  /** Hindsight API base URL */
  apiUrl: string;
  /** Optional API token */
  apiToken?: string;
  /** Memory bank ID for this agent */
  bankId: string;
}

// ── Constants ──────────────────────────────────────────

const PAGE_DEFAULTS = {
  mode: "delta" as const,
  refresh_after_consolidation: true,
  exclude_mental_models: true,
  fact_types: ["observation" as const],
};

export const TOOL_NAMES = [
  "agent_knowledge_list_pages",
  "agent_knowledge_get_page",
  "agent_knowledge_create_page",
  "agent_knowledge_update_page",
  "agent_knowledge_delete_page",
  "agent_knowledge_recall",
  "agent_knowledge_ingest",
] as const;

export type KnowledgeToolName = (typeof TOOL_NAMES)[number];

// ── Helpers ────────────────────────────────────────────

function ok(data: unknown): KnowledgeToolResult {
  return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
}

// ── Factory ────────────────────────────────────────────

/**
 * Create the full set of agent_knowledge_* tools for a given bank.
 *
 * Returns harness-agnostic tool definitions. Each harness adapter
 * wraps these into its native tool format.
 */
export function createKnowledgeTools(opts: CreateKnowledgeToolsOptions): KnowledgeTool[] {
  const client = new HindsightClient({
    baseUrl: opts.apiUrl,
    apiKey: opts.apiToken,
    userAgent: "hindsight-agent-sdk/0.1.0",
  });
  const lowLevel = createClient(
    createConfig({
      baseUrl: opts.apiUrl,
      headers: {
        ...(opts.apiToken ? { Authorization: `Bearer ${opts.apiToken}` } : {}),
        "User-Agent": "hindsight-agent-sdk/0.1.0",
      },
    })
  );
  const bankId = opts.bankId;

  return [
    {
      name: "agent_knowledge_list_pages",
      label: "List knowledge pages",
      description:
        "List all your knowledge pages (IDs and names only). Use agent_knowledge_get_page to read the full content of a specific page.",
      parameters: { type: "object", properties: {} },
      async execute() {
        const resp = await sdk.listMentalModels({
          client: lowLevel,
          path: { bank_id: bankId },
          query: { detail: "metadata" },
        });
        return ok(resp.data);
      },
    },
    {
      name: "agent_knowledge_get_page",
      label: "Read a knowledge page",
      description:
        "Read a specific knowledge page by its ID. Returns the full synthesized content.",
      parameters: {
        type: "object",
        properties: {
          page_id: { type: "string", description: "Page ID (e.g. 'user-preferences')" },
        },
        required: ["page_id"],
      },
      async execute(params: Record<string, unknown>) {
        const result = await client.getMentalModel(bankId, params.page_id as string);
        return ok(result);
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
          page_id: {
            type: "string",
            description: "Unique page ID, lowercase with hyphens (e.g. 'editorial-preferences')",
          },
          name: { type: "string", description: "Human-readable page name" },
          source_query: {
            type: "string",
            description:
              "The question that rebuilds this page (e.g. 'What are the user\\'s editorial preferences?')",
          },
        },
        required: ["page_id", "name", "source_query"],
      },
      async execute(params: Record<string, unknown>) {
        const resp = await sdk.createMentalModel({
          client: lowLevel,
          path: { bank_id: bankId },
          body: {
            id: params.page_id as string,
            name: params.name as string,
            source_query: params.source_query as string,
            max_tokens: 4096,
            trigger: PAGE_DEFAULTS,
          },
        });
        return ok(resp.data);
      },
    },
    {
      name: "agent_knowledge_update_page",
      label: "Update a knowledge page",
      description:
        "Update a page's name or source query. The content will re-synthesize on next consolidation.",
      parameters: {
        type: "object",
        properties: {
          page_id: { type: "string", description: "Page ID to update" },
          name: { type: "string", description: "New name (optional)" },
          source_query: { type: "string", description: "New source query (optional)" },
        },
        required: ["page_id"],
      },
      async execute(params: Record<string, unknown>) {
        const result = await client.updateMentalModel(bankId, params.page_id as string, {
          name: params.name as string | undefined,
          sourceQuery: params.source_query as string | undefined,
        });
        return ok(result);
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
      async execute(params: Record<string, unknown>) {
        await client.deleteMentalModel(bankId, params.page_id as string);
        return ok({ success: true });
      },
    },
    {
      name: "agent_knowledge_recall",
      label: "Search memories",
      description:
        "Search across all retained conversations and documents for specific facts, numbers, or details not covered by your knowledge pages.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "What to search for" },
          max_results: { type: "number", description: "Max results (default 10)" },
        },
        required: ["query"],
      },
      async execute(params: Record<string, unknown>) {
        const result = await client.recall(bankId, params.query as string, {
          maxTokens: (params.max_results as number | undefined) ?? 10,
        });
        return ok(result);
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
      async execute(params: Record<string, unknown>) {
        const docId = (params.title as string).toLowerCase().replace(/ /g, "-");
        const result = await client.retainBatch(
          bankId,
          [{ content: params.content as string, document_id: docId }],
          { async: true }
        );
        return ok(result);
      },
    },
  ];
}
