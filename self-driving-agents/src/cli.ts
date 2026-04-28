#!/usr/bin/env node
/**
 * self-driving-agents — install a self-driving agent from a directory.
 *
 * npx @vectorize-io/self-driving-agents install <dir> --harness openclaw|hermes|claude-code [--agent <name>]
 *
 * Directory layout:
 *   bank-template.json   — optional: bank config + mental models + directives
 *   content/             — optional: reference docs to ingest (.md, .txt, etc.)
 *
 * The agent name defaults to the directory name.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from "fs";
import { join, resolve, extname, basename } from "path";
import { homedir } from "os";
import { createInterface } from "readline";

const HINDSIGHT_CLOUD_URL = "https://api.hindsight.vectorize.io";

// ── Interactive prompts ─────────────────────────────────

function prompt(question: string): Promise<string> {
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  return new Promise((resolve) => {
    rl.question(question, (answer) => { rl.close(); resolve(answer.trim()); });
  });
}

async function promptCloudSetup(): Promise<{ apiUrl: string; apiToken: string } | null> {
  console.error("\nHindsight API not reachable locally.");
  console.error("You can use Hindsight Cloud instead.\n");
  console.error("  Sign up at: https://ui.hindsight.vectorize.io/signup\n");

  const useCloud = await prompt("Use Hindsight Cloud? (y/n) ");
  if (useCloud.toLowerCase() !== "y") return null;

  const token = await prompt("API token: ");
  if (!token) { console.error("No token provided."); return null; }

  return { apiUrl: HINDSIGHT_CLOUD_URL, apiToken: token };
}

// ── Skill ───────────────────────────────────────────────

const SKILL_MD = `---
name: agent-knowledge
description: Your long-term knowledge pages. Read them at session start. Create new pages for recurring topics. Pages auto-update from your conversations.
---

# Agent Knowledge

You have knowledge pages that persist across sessions and auto-update from your conversations.

**How it works:** Conversations are retained into Hindsight. The system extracts observations and rebuilds each page via its "source query." You create pages; the system maintains them.

## At session start

Call \`agent_knowledge_list_pages\` to see what pages exist, then \`agent_knowledge_get_page\` for each one you need.

## Tools

- \`agent_knowledge_list_pages()\` — list page IDs and names (no content)
- \`agent_knowledge_get_page(page_id)\` — read the full content of a page
- \`agent_knowledge_create_page(page_id, name, source_query)\` — create a page
- \`agent_knowledge_update_page(page_id, name?, source_query?)\` — update a page
- \`agent_knowledge_delete_page(page_id)\` — delete a page
- \`agent_knowledge_recall(query)\` — search all memories
- \`agent_knowledge_ingest(title, content)\` — upload raw content (never summarize)

## Creating pages

Create when you learn something durable — preferences, procedures, performance data.
The source_query is a question the system re-asks to rebuild the page.

Examples:
- "What are the user's preferences for tone, length, and formatting?"
- "What strategies have performed well or poorly? Include numbers."
- "What are the best practices for [topic], preferring our data over generic advice?"

## Rules

- Pages update automatically — don't edit content directly
- State preferences clearly in responses so the system captures them
- Create pages silently
- Prefer fewer broad pages over many narrow ones
`;

// ── HTTP ────────────────────────────────────────────────

async function api(
  baseUrl: string, path: string, method: string,
  body?: unknown, token?: string,
): Promise<unknown> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const opts: RequestInit = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(`${baseUrl}${path}`, opts);
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${text.slice(0, 200)}`);
  }
  return resp.json().catch(() => ({}));
}

// ── Harness resolvers ───────────────────────────────────

interface HarnessInfo {
  apiUrl: string;
  bankId: string;
  apiToken?: string;
  workspaceDir: string;
  startupFile?: string;
  startupPatch?: string;
}

function resolveOpenclaw(agentId: string): HarnessInfo {
  const configPath = join(homedir(), ".openclaw", "openclaw.json");
  if (!existsSync(configPath)) throw new Error("~/.openclaw/openclaw.json not found. Is OpenClaw installed?");
  const config = JSON.parse(readFileSync(configPath, "utf-8"));
  const pc = config.plugins?.entries?.["hindsight-openclaw"]?.config || {};

  const apiUrl = pc.hindsightApiUrl || `http://localhost:${pc.apiPort || 9077}`;
  const apiToken = pc.hindsightApiToken || undefined;

  let bankId: string;
  if (pc.dynamicBankId === false && pc.bankId) {
    bankId = pc.bankId;
  } else {
    const granularity: string[] = pc.dynamicBankGranularity || ["agent", "channel", "user"];
    const fieldMap: Record<string, string> = { agent: agentId, channel: "unknown", user: "anonymous", provider: "unknown" };
    const base = granularity.map((f) => encodeURIComponent(fieldMap[f] || "unknown")).join("::");
    bankId = pc.bankIdPrefix ? `${pc.bankIdPrefix}-${base}` : base;
  }

  const workspace = join(homedir(), ".hindsight-agents", "openclaw", agentId);
  return {
    apiUrl, bankId, apiToken, workspaceDir: workspace,
    startupFile: join(workspace, "AGENTS.md"),
    startupPatch: '5. Read `skills/agent-knowledge/SKILL.md` and **execute its mandatory startup sequence**',
  };
}

function resolveHermes(agentId: string): HarnessInfo {
  return {
    apiUrl: "http://localhost:8888",
    bankId: agentId,
    workspaceDir: join(homedir(), ".hermes"),
    startupFile: agentId === "default"
      ? join(homedir(), ".hermes", "SOUL.md")
      : join(homedir(), ".hermes", "profiles", agentId, "SOUL.md"),
    startupPatch: "## Mandatory: Agent Knowledge\n\nAt session start, load the `agent-knowledge` skill and execute its startup sequence.",
  };
}

function resolveClaudeCode(_agentId: string): HarnessInfo {
  return {
    apiUrl: "http://localhost:8888",
    bankId: "claude_code",
    workspaceDir: join(homedir(), ".claude"),
  };
}

function resolveHarness(harness: string, agentId: string, apiUrlOverride?: string, apiTokenOverride?: string): HarnessInfo {
  let info: HarnessInfo;
  switch (harness) {
    case "openclaw": info = resolveOpenclaw(agentId); break;
    case "hermes": info = resolveHermes(agentId); break;
    case "claude-code": info = resolveClaudeCode(agentId); break;
    default: throw new Error(`Unknown harness: ${harness}. Supported: openclaw, hermes, claude-code`);
  }
  if (apiUrlOverride) info.apiUrl = apiUrlOverride;
  if (apiTokenOverride) info.apiToken = apiTokenOverride;
  return info;
}

// ── Main ────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 1 || args[0] === "--help" || args[0] === "-h") {
    console.log(`Usage: npx @vectorize-io/self-driving-agents install <dir> --harness <harness> [--agent <name>]

Arguments:
  <dir>              Agent directory (contains optional bank-template.json + content/)

Options:
  --harness <h>      Required. openclaw | hermes | claude-code
  --agent <name>     Agent name (defaults to directory name)
  --api-url <url>    Override Hindsight API URL
  --api-token <t>    Override API token`);
    process.exit(0);
  }

  let dirArg = args[0] === "install" ? args[1] : args[0];
  const restArgs = args[0] === "install" ? args.slice(2) : args.slice(1);

  if (!dirArg) {
    console.error("Error: directory argument required");
    process.exit(1);
  }

  let harness: string | undefined;
  let agentName: string | undefined;
  let apiUrlOverride: string | undefined;
  let apiTokenOverride: string | undefined;

  for (let i = 0; i < restArgs.length; i++) {
    if (restArgs[i] === "--harness" && restArgs[i + 1]) harness = restArgs[++i];
    else if (restArgs[i] === "--agent" && restArgs[i + 1]) agentName = restArgs[++i];
    else if (restArgs[i] === "--api-url" && restArgs[i + 1]) apiUrlOverride = restArgs[++i];
    else if (restArgs[i] === "--api-token" && restArgs[i + 1]) apiTokenOverride = restArgs[++i];
  }

  if (!harness) {
    console.error("Error: --harness is required (openclaw | hermes | claude-code)");
    process.exit(1);
  }

  const dir = resolve(dirArg);
  if (!existsSync(dir)) {
    console.error(`Error: directory not found: ${dir}`);
    process.exit(1);
  }

  const agentId = agentName || basename(dir);
  const info = resolveHarness(harness, agentId, apiUrlOverride, apiTokenOverride);

  // Health check — if local fails, offer cloud
  let healthy = false;
  try {
    await api(info.apiUrl, "/health", "GET", undefined, info.apiToken);
    healthy = true;
  } catch {
    // Local not reachable — try cloud interactively if no explicit --api-url
    if (!apiUrlOverride && process.stdin.isTTY) {
      const cloud = await promptCloudSetup();
      if (cloud) {
        info.apiUrl = cloud.apiUrl;
        info.apiToken = cloud.apiToken;
        try {
          await api(info.apiUrl, "/health", "GET", undefined, info.apiToken);
          healthy = true;
        } catch {
          console.error(`Error: Cannot reach Hindsight Cloud. Check your token.`);
          process.exit(1);
        }
      }
    }
  }

  if (!healthy) {
    console.error(`Error: Cannot reach Hindsight at ${info.apiUrl}`);
    console.error("Start the server, or use --api-url / --api-token for cloud.");
    process.exit(1);
  }

  console.log(`Installing '${agentId}' on ${harness}`);
  console.log(`  Directory: ${dir}`);
  console.log(`  Bank:      ${info.bankId}`);
  console.log(`  API:       ${info.apiUrl}`);
  console.log(`  Workspace: ${info.workspaceDir}`);
  console.log();

  // Import bank template
  const templatePath = join(dir, "bank-template.json");
  if (existsSync(templatePath)) {
    console.log("Importing bank template...");
    const template = JSON.parse(readFileSync(templatePath, "utf-8"));
    await api(info.apiUrl, `/v1/default/banks/${info.bankId}/import`, "POST", template, info.apiToken);
    console.log("  Done.");
  }

  // Ingest content
  const contentDir = join(dir, "content");
  if (existsSync(contentDir)) {
    const exts = new Set([".md", ".txt", ".html", ".json", ".csv", ".xml"]);
    const files = readdirSync(contentDir).filter((f) => exts.has(extname(f).toLowerCase())).sort();
    if (files.length > 0) {
      console.log(`Ingesting ${files.length} file(s)...`);
      for (const file of files) {
        const content = readFileSync(join(contentDir, file), "utf-8");
        if (!content.trim()) continue;
        const docId = file.replace(/\.[^.]+$/, "");
        await api(info.apiUrl, `/v1/default/banks/${info.bankId}/memories`, "POST", {
          items: [{ content, document_id: docId }], async: true,
        }, info.apiToken);
        console.log(`  ${file} → queued`);
      }
    }
  }

  // Create harness agent (openclaw)
  if (harness === "openclaw") {
    try {
      const { execSync } = await import("child_process");
      const listOut = execSync("openclaw agents list --json 2>/dev/null", { encoding: "utf-8" });
      const agents = JSON.parse(listOut).agents || [];
      if (!agents.some((a: any) => a.name === agentId)) {
        mkdirSync(info.workspaceDir, { recursive: true });
        execSync(`openclaw agents add ${agentId} --workspace ${info.workspaceDir} --non-interactive`, { stdio: "pipe" });
        console.log(`Created OpenClaw agent '${agentId}'.`);
      } else {
        console.log(`Agent '${agentId}' already exists.`);
      }
    } catch {
      console.log(`Note: create agent manually: openclaw agents add ${agentId} --workspace ${info.workspaceDir} --non-interactive`);
    }
  }

  // Install skill
  const skillDir = join(info.workspaceDir, "skills", "agent-knowledge");
  mkdirSync(skillDir, { recursive: true });
  writeFileSync(join(skillDir, "SKILL.md"), SKILL_MD);
  console.log("Skill installed.");

  // Patch startup file
  if (info.startupFile && info.startupPatch && existsSync(info.startupFile)) {
    let text = readFileSync(info.startupFile, "utf-8");
    if (!text.includes("agent-knowledge")) {
      if (text.includes("Don't ask permission.")) {
        text = text.replace("Don't ask permission. Just do it.", `${info.startupPatch}\n\nDon't ask permission. Just do it.`);
      } else {
        text += `\n\n${info.startupPatch}\n`;
      }
      writeFileSync(info.startupFile, text);
      console.log("Startup patched.");
    }
  }

  console.log();
  console.log(`'${agentId}' is ready.`);
  if (harness === "openclaw") console.log("  Restart gateway: openclaw gateway restart");
  if (harness === "hermes") console.log(`  Chat: hermes${agentId !== "default" ? ` --profile ${agentId}` : ""}`);
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
