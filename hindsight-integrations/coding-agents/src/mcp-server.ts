#!/usr/bin/env node
/**
 * Native TS MCP (stdio) server exposing the `hindsight_*` knowledge-page + recall + capture tools.
 *
 * Bank resolution MUST mirror the hooks exactly (`resolveHostMemory`, i.e. loadConfig +
 * deriveBankId + the banks section, harness from the REQUIRED HINDSIGHT_MCP_HARNESS) so knowledge
 * pages, recall, and retain all land in ONE per-repo bank — this is
 * why this is a native TS server and not a reuse of the Python MCP (whose bank derivation
 * differs). MCP servers launch with the project dir as cwd; the env override is an optional
 * escape hatch (not currently set by the plugin).
 */
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { type Config } from "./core/config";
import { resolveHostMemory } from "./core/host-client";
import { HindsightClient } from "./core/hindsight";
import { buildKnowledgeTools, type ToolSpec } from "./core/knowledge-tools";
import { buildPageTrigger } from "./core/missions";
import { buildRetainStamp } from "./core/retain-stamp";

/**
 * Which tools this server should expose for a given config. Pure + SDK-free so the
 * disabled-flag behavior is unit-testable without spinning up a real MCP host: a disabled
 * Hindsight (mirrors the hooks' `cfg.disabled` check) exposes NO tools — the server still
 * connects, it just has nothing registered.
 */
export function selectTools(
  cfg: Config,
  client: HindsightClient,
  bankId: string,
  opts: { cwd?: string; harness?: string } = {}
): ToolSpec[] {
  const cwd = opts.cwd ?? process.cwd();
  const harness = opts.harness ?? cfg.harness;
  return cfg.disabled
    ? []
    : buildKnowledgeTools(client, bankId, {
        repoDir: cwd,
        harness,
        pageTrigger: buildPageTrigger(cfg),
        reflectTimeoutMs: cfg.reflectToolTimeoutMs,
        reflectBudget: cfg.reflectBudget,
        stampFor: () => buildRetainStamp(cfg, { directory: cwd, harness, bankId }),
      });
}

/**
 * The harness this server is running for, from the environment its registration declares.
 *
 * REQUIRED, deliberately with no fallback. This used to default to "claude-code", which read as a
 * safe convenience and was not: every host launches this same binary, so a registration that named
 * no harness was silently served as Claude Code — Codex ingests came back tagged
 * `harness:claude-code`, indistinguishable from Claude Code's own, and resolved Claude Code's bank
 * (#3603). A wrong answer here corrupts stored data; refusing to start is recoverable.
 */
export function resolveHarness(env: NodeJS.ProcessEnv = process.env): string {
  const harness = env.HINDSIGHT_MCP_HARNESS;
  if (!harness) {
    throw new Error(
      "HINDSIGHT_MCP_HARNESS is not set. Every coding agent launches this same mcp-server.js, so " +
        "only that variable identifies the caller — it decides the harness:<id> stamp on everything " +
        "ingested and which bank this session resolves. Re-run `npx " +
        "@vectorize-io/hindsight-coding-agents install <harness>` to repair a registration written " +
        "before the installer set it."
    );
  }
  return harness;
}

/**
 * Build the MCP surface for the resolved project.
 *
 * An opted-out project intentionally has no Hindsight tools, but some clients probe `tools/list`
 * without first checking the initialize capabilities. Advertising an empty, queryable tool list
 * keeps that privacy boundary intact while preventing one optional probe from failing startup.
 */
export function buildMcpServer(tools: ToolSpec[]): McpServer {
  const server = new McpServer({ name: "hindsight", version: "0.1.0" });
  if (tools.length === 0) {
    server.server.registerCapabilities({ tools: { listChanged: false } });
    server.server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: [] }));
    return server;
  }

  for (const tool of tools) {
    // registerTool (not the deprecated `tool()`) so the safety annotations reach the client:
    // Dcode rejects unannotated MCP calls outright in headless mode, and Codex Auto-review treats
    // them as unverified external access. See ToolSpec.annotations.
    server.registerTool(
      tool.name,
      {
        description: tool.description,
        inputSchema: tool.inputSchema,
        annotations: tool.annotations,
      },
      tool.handler
    );
  }
  return server;
}

async function main() {
  const cwd = process.env.HINDSIGHT_MCP_PROJECT_CWD || process.cwd();
  // Mirrors that harness's hooks: it selects the config `harnesses.<name>` section and feeds the
  // `{harness}` bank template, so both routes into a repo land in ONE bank.
  const harness = resolveHarness();
  const { cfg, bankId, client } = resolveHostMemory(harness, cwd);

  const server = buildMcpServer(selectTools(cfg, client, bankId, { cwd, harness }));

  await server.connect(new StdioServerTransport());
}

// Only auto-run when this file is executed directly (e.g. `node dist/mcp-server.js`) — importing
// it (as src/mcp-server.test.ts does) must not start a real server.
if (process.argv[1] && process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error("hindsight mcp-server failed:", e);
    process.exit(1);
  });
}
