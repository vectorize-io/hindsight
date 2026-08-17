/**
 * Knowledge-page MCP tool specs — SDK-free so this stays unit-testable without a real MCP host.
 *
 * `src/mcp-server.ts` is the only file that imports the MCP SDK; it wires the specs returned here
 * into an `McpServer`. Each tool wraps one `HindsightClient` knowledge-page/recall method: it never
 * throws — a thrown client error is caught and turned into an `isError:true` text result so the
 * calling LLM sees the failure instead of the process crashing.
 *
 * The agent-facing surface is intentionally curated: grounding + capture only. Raw page CRUD
 * (create/update/delete) is deliberately NOT exposed — agents never author page structure; they
 * capture initiatives (`hindsight_capture_initiative`) and pages are synthesized/maintained by the
 * server. `hindsight_ingest_document` (raw-content retain) also backs the codebase survey
 * (core/survey.ts).
 */
import { z } from "zod";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ZodRawShape } from "zod";
import type { HindsightClient } from "./hindsight";
import { syncStatus } from "./status";
import { loadConfig } from "./config";
import { describeError } from "./log";
import type { RetainStamp } from "./retain-stamp";
import type { PageTrigger } from "./missions";

export interface ToolResult {
  // Index signature so this structurally satisfies the MCP SDK's CallToolResult (which carries
  // extra optional fields we don't set) when passed to `McpServer.tool()` in src/mcp-server.ts —
  // the only file that imports the SDK; this file stays SDK-free.
  [x: string]: unknown;
  content: { type: "text"; text: string }[];
  isError?: boolean;
}

export interface ToolSpec {
  name: string;
  description: string;
  inputSchema: ZodRawShape;
  handler: (args: any) => Promise<ToolResult>;
}

function ok(value: unknown): ToolResult {
  return { content: [{ type: "text", text: JSON.stringify(value, null, 2) }] };
}

function err(e: unknown): ToolResult {
  const message = describeError(e);
  return { content: [{ type: "text", text: JSON.stringify({ error: message }) }], isError: true };
}

/** Wrap a handler body so a thrown client error always becomes an isError:true result, never a throw. */
function guarded(fn: (args: any) => Promise<unknown>): (args: any) => Promise<ToolResult> {
  return async (args: any) => {
    try {
      return ok(await fn(args));
    } catch (e) {
      return err(e);
    }
  };
}

/** Build the knowledge-page + recall MCP tool specs, bound to one client/bank. */
export function buildKnowledgeTools(
  client: HindsightClient,
  bankId: string,
  opts: {
    repoDir?: string;
    harness?: string;
    stampFor?: () => RetainStamp;
    /** Refresh policy for a page `hindsight_capture_initiative` creates (core/missions.ts). */
    pageTrigger?: PageTrigger;
  } = {}
): ToolSpec[] {
  return [
    {
      name: "hindsight_sync_status",
      description:
        "Report whether this repo's memory bank is in sync: gitlog seed present, how much recent " +
        "history has been deepened with full diffs, conversations ingested, knowledge pages " +
        "created, and extractions still running. `synced: true` means the seeded memory is fully " +
        "queryable. Ingestion is automatic and background — if not synced, it is in progress; " +
        "nothing to run.",
      inputSchema: {},
      handler: async () => {
        try {
          return ok(await syncStatus(client, bankId, opts.repoDir ?? process.cwd()));
        } catch (e) {
          return err(e);
        }
      },
    },
    {
      name: "hindsight_diagnose",
      description:
        "Report safe Hindsight runtime diagnostics for this coding-agent session: resolved bank, " +
        "workspace, harness, config location, API endpoint, and non-secret environment overrides. " +
        "Use this when memory, hooks, MCP tools, or configuration appear not to work. Tokens and " +
        "other secret values are never returned.",
      inputSchema: {},
      handler: async (_args: Record<string, never>) => {
        const configPath =
          process.env.HINDSIGHT_CONFIG || join(homedir(), ".hindsight", "coding-agent.json");
        const harness = opts.harness ?? "unknown";
        const cfg = loadConfig({ harness: opts.harness });
        return ok({
          bank_id: bankId,
          harness,
          workspace: opts.repoDir ?? process.cwd(),
          config: {
            path: configPath,
            exists: existsSync(configPath),
            api_url: cfg.apiUrl,
            api_token_configured: Boolean(cfg.apiToken),
            disabled: cfg.disabled,
          },
          environment: {
            config_override: Boolean(process.env.HINDSIGHT_CONFIG),
            hooks_disabled: Boolean(process.env.HINDSIGHT_DISABLE_HOOKS),
            log_level: process.env.HINDSIGHT_LOG_LEVEL ?? null,
            diagnostics_file: process.env.HINDSIGHT_DIAG_FILE ?? "/tmp/hindsight-plugin.log",
            channel_id_configured: Boolean(process.env.HINDSIGHT_CHANNEL_ID),
            user_id_configured: Boolean(process.env.HINDSIGHT_USER_ID),
          },
        });
      },
    },
    {
      name: "hindsight_search_knowledge_pages",
      description:
        "Search this repository's Hindsight knowledge pages for content relevant to a query — " +
        "hybrid full-text + semantic search, server-side. Call this when the user's question may " +
        "be answered by the project's accumulated knowledge (architecture, conventions, decisions, " +
        "initiatives) rather than by reading code. Returns ranked pages with a relevance snippet; " +
        "read a full page with hindsight_read_knowledge_page. When a result informs your answer, " +
        "credit it visibly: start that part with a markdown blockquote header " +
        '"> 🧠 **From Hindsight memory (<page name>)** — <the facts you drew on>".',
      inputSchema: { query: z.string().describe("what to look for") },
      handler: async (args: { query: string }) => {
        try {
          const hits = await client.searchKnowledgePages(args.query, 3);
          return ok(
            hits.map((h) => ({
              page: h.name,
              page_id: h.id,
              snippet: h.snippet,
              score: h.score,
            }))
          );
        } catch (e) {
          return err(e);
        }
      },
    },
    {
      name: "hindsight_list_knowledge_pages",
      description:
        "List this repository's Hindsight knowledge pages — curated, continuously-updated " +
        "summaries of the project's durable knowledge (architecture, components, conventions, key " +
        "decisions, and in-flight initiatives). Returns each page's id, title, and a one-line " +
        "description of what it covers. Call this at the start of any non-trivial task, and again " +
        "periodically in long sessions, to see what the project already knows before you read code " +
        "or ask the user. The list changes as work is captured, so re-check it occasionally.",
      inputSchema: {},
      handler: guarded(async () => client.listPages()),
    },
    {
      name: "hindsight_read_knowledge_page",
      description:
        "Read the full content of one knowledge page by its id (from " +
        "hindsight_list_knowledge_pages). Call this whenever a listed page is relevant to what " +
        "you're about to do — e.g. read Conventions before writing new code, Component map before " +
        "changing a subsystem, or an initiative's page before continuing that feature. A page may " +
        "contain [[page:<id>]] links to related pages; follow one by calling this tool again with " +
        "that id. Prefer reading a page over re-deriving the same understanding from source.",
      inputSchema: { page_id: z.string() },
      handler: guarded(async ({ page_id }) => client.getPage(page_id)),
    },
    {
      name: "hindsight_reflect",
      description:
        "Deep memory reasoning: an agentic synthesis over this repository's FULL memory (git " +
        "decisions, past sessions, ingested knowledge) that answers WHY questions — the past " +
        "decision and exact rule/values that explain a behavior, bug, or convention. Slower than " +
        "hindsight_search_knowledge_pages (several seconds): reach for it when pages are too " +
        "shallow and you need the root cause or the decided literals. When the answer informs " +
        'your reply, credit it visibly with a blockquote header: "> 🧠 **From Hindsight memory** — <summary>".',
      inputSchema: { query: z.string().describe("the question to reason over memory about") },
      handler: guarded(async ({ query }: { query: string }) =>
        client.reflect(query, { budget: "high" })
      ),
    },
    {
      name: "hindsight_capture_initiative",
      description:
        "Record a new feature or initiative as a tracked knowledge page, so future sessions know it " +
        "exists and can build on it.\n\n" +
        "WHEN TO CALL: right after the user approves a plan or finishes brainstorming a new feature/" +
        "capability and you are about to start implementing — BEFORE you write any code. Call it ONCE, EARLY.\n\n" +
        "WHEN TO SKIP: bug fixes, small tweaks, refactors, and chores are not initiatives — skip " +
        "them.\n\n" +
        "- title: short, specific name (e.g. 'Newsletter refinement chat').\n" +
        "- summary: 2-3 sentences on what you're building and why (the intent, not a code diff).\n" +
        "- relates_to_page_id: leave empty for a new initiative. Set it only to attach to an " +
        "initiative that already has a page — pass that page's id from hindsight_list_knowledge_pages.\n\n" +
        "Returns the page id. The page is generated for you — you never format one yourself.",
      inputSchema: {
        title: z.string(),
        summary: z.string(),
        relates_to_page_id: z.string().optional(),
      },
      handler: guarded(async ({ title, summary, relates_to_page_id }) =>
        client.captureInitiative({
          title,
          summary,
          relatesToPageId: relates_to_page_id,
          ...(opts.stampFor ? { stamp: opts.stampFor() } : {}),
          ...(opts.pageTrigger ? { pageTrigger: opts.pageTrigger } : {}),
        })
      ),
    },
    {
      name: "hindsight_ingest_document",
      description:
        "Save an external document or a block of durable notes/findings into this repository's " +
        "memory so it informs future recall and pages. Use for design notes, research, or " +
        "reference material you want remembered — not for the conversation you're already in " +
        "(that's captured automatically at session end). This is ALSO the correction mechanism: " +
        "when you verify that a retrieved memory is wrong or outdated, ingest a document titled " +
        "'Correction: <topic>' stating what memory claimed, what is actually true, and the " +
        "evidence — the newer fact supersedes the stale one in future retrieval.",
      inputSchema: { title: z.string(), content: z.string() },
      handler: guarded(async ({ title, content }) => {
        const docId =
          title
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "") || "doc";
        const stamp = opts.stampFor?.();
        const metadata = {
          ...stamp?.metadata,
          ...(opts.harness ? { harness: opts.harness } : {}),
        };
        await client.retain(
          content,
          "ingested document",
          docId,
          [
            ...new Set([
              ...(stamp?.tags ?? []),
              "source:upload",
              ...(opts.harness ? [`harness:${opts.harness}`] : []),
            ]),
          ],
          "document",
          Object.keys(metadata).length ? { metadata } : {}
        );
        return { ok: true, doc_id: docId };
      }),
    },
  ];
}
