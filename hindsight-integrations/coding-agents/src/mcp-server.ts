#!/usr/bin/env node
/**
 * Native TS MCP (stdio) server exposing the `hindsight_*` knowledge-page + recall + capture tools.
 *
 * Bank resolution MUST mirror the hooks exactly (`resolveHostMemory`, i.e. loadConfig +
 * deriveBankId + the banks section, harness from HINDSIGHT_MCP_HARNESS) so knowledge pages,
 * recall, and retain all land in ONE per-repo bank — this is
 * why this is a native TS server and not a reuse of the Python MCP (whose bank derivation
 * differs). MCP servers launch with the project dir as cwd; the env override is an optional
 * escape hatch (not currently set by the plugin).
 */
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
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

async function main() {
  const cwd = process.env.HINDSIGHT_MCP_PROJECT_CWD || process.cwd();
  // Every host launches this same binary, so who is calling is knowable only from the environment
  // the installer wrote (`HINDSIGHT_MCP_HARNESS`). It decides both the retain stamp — the
  // `harness:<id>` tag and metadata on anything ingested — and bank resolution (config
  // `harnesses.<name>` section + `{harness}` template), so it must mirror that harness's hooks.
  // The "claude-code" fallback only covers registrations written before every installer set the
  // variable; re-running `install <harness>` repairs those.
  const harness = process.env.HINDSIGHT_MCP_HARNESS || "claude-code";
  const { cfg, bankId, client } = resolveHostMemory(harness, cwd);

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
