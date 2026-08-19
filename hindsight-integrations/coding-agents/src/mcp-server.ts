#!/usr/bin/env node
/**
 * Native TS MCP (stdio) server exposing the `hindsight_*` knowledge-page + recall + capture tools.
 *
 * Bank resolution MUST mirror the hooks exactly (loadConfig + deriveBankId, harness
 * "claude-code") so knowledge pages, recall, and retain all land in ONE per-repo bank — this is
 * why this is a native TS server and not a reuse of the Python MCP (whose bank derivation
 * differs). MCP servers launch with the project dir as cwd; the env override is an optional
 * escape hatch (not currently set by the plugin).
 */
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { applyBankConfig, loadConfig, type Config } from "./core/config";
import { deriveBankId } from "./core/bank";
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
        stampFor: () => buildRetainStamp(cfg, { directory: cwd, harness, bankId }),
      });
}

async function main() {
  const cwd = process.env.HINDSIGHT_MCP_PROJECT_CWD || process.cwd();
  // Harness is set per wrapper (Claude default; codex sets HINDSIGHT_MCP_HARNESS=codex) so bank
  // resolution mirrors that harness's hooks — config `harnesses.<name>` section + `{harness}` template.
  const harness = process.env.HINDSIGHT_MCP_HARNESS || "claude-code";
  const cfg0 = loadConfig({ harness });
  const { cfg, bankId } = applyBankConfig(cfg0, deriveBankId(cfg0, cwd, harness), cwd);
  const client = new HindsightClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: bankId,
    maxParallelRetains: cfg.maxParallelRetains,
    observationScopes: cfg.observationScopes,
  });

  const server = new McpServer({ name: "hindsight", version: "0.1.0" });
  for (const t of selectTools(cfg, client, bankId, { cwd, harness })) {
    server.tool(t.name, t.description, t.inputSchema, t.handler);
  }

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
