#!/usr/bin/env node
/**
 * self-driving-agents — install a self-driving agent.
 *
 * npx @vectorize-io/self-driving-agents install <agent> --harness openclaw [--agent <name>]
 *
 * Agent resolution:
 *   marketing-agent            → vectorize-io/self-driving-agents/marketing-agent (default repo)
 *   my-org/my-repo/my-agent   → my-org/my-repo/my-agent on GitHub
 *   ./local-dir                → local directory
 *   /absolute/path             → local directory
 *
 * Directory layout (recursive):
 *   bank-template.json   — optional: bank config at this level
 *   *.md, *.txt, ...     — content files (found recursively, excluding bank-template.json)
 */

import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  existsSync,
  readdirSync,
  statSync,
  rmSync,
} from "fs";
import { join, resolve, extname, basename, relative, dirname } from "path";
import { homedir, tmpdir } from "os";
import { fileURLToPath } from "url";
import { execSync } from "child_process";
import * as p from "@clack/prompts";
import color from "picocolors";
import { HindsightClient, sdk, createClient, createConfig } from "@vectorize-io/hindsight-client";

const DEFAULT_REPO = "vectorize-io/self-driving-agents";

// ── Content discovery ──────────────────────────────────

const CONTENT_EXTS = new Set([".md", ".txt", ".html", ".json", ".csv", ".xml"]);
const IGNORED_FILES = new Set(["bank-template.json"]);

/** Recursively find all content files under `dir`, returning paths relative to `dir`. */
function findContentFiles(dir: string): string[] {
  const results: string[] = [];
  function walk(current: string) {
    for (const entry of readdirSync(current)) {
      const full = join(current, entry);
      if (statSync(full).isDirectory()) {
        walk(full);
      } else if (CONTENT_EXTS.has(extname(entry).toLowerCase()) && !IGNORED_FILES.has(entry)) {
        results.push(relative(dir, full));
      }
    }
  }
  walk(dir);
  return results.sort();
}

// ── Agent resolution ───────────────────────────────────

function isLocalPath(input: string): boolean {
  return (
    input.startsWith("./") ||
    input.startsWith("../") ||
    input.startsWith("/") ||
    input.startsWith("~")
  );
}

/**
 * Resolve the agent specifier to a local directory.
 *
 * - Local paths (./foo, /foo, ~/foo) → resolve directly
 * - "name"                          → GitHub: vectorize-io/self-driving-agents/name
 * - "org/repo/path"                 → GitHub: org/repo/path
 */
async function resolveAgentDir(
  input: string,
  spinner: ReturnType<typeof p.spinner>
): Promise<{ dir: string; source: string; cleanup?: () => void }> {
  if (isLocalPath(input)) {
    const dir = resolve(input.replace(/^~/, homedir()));
    if (!existsSync(dir)) throw new Error(`Directory not found: ${dir}`);
    return { dir, source: dir };
  }

  // Parse GitHub reference: "name" or "org/repo/path/to/agent"
  const parts = input.split("/");
  let org: string, repo: string, subpath: string;

  if (parts.length === 1) {
    // Just a name → default repo
    org = "vectorize-io";
    repo = "self-driving-agents";
    subpath = parts[0];
  } else if (parts.length >= 3) {
    // org/repo/path...
    org = parts[0];
    repo = parts[1];
    subpath = parts.slice(2).join("/");
  } else {
    throw new Error(
      `Invalid agent reference: '${input}'\n` +
        `  Use: <name>, <org>/<repo>/<path>, or a local path (./dir)`
    );
  }

  spinner.start(`Fetching ${color.cyan(`${org}/${repo}/${subpath}`)} from GitHub...`);

  const tmp = join(tmpdir(), `sda-${Date.now()}`);
  mkdirSync(tmp, { recursive: true });

  try {
    // Download repo tarball and extract the specific subdirectory
    const tarballUrl = `https://github.com/${org}/${repo}/archive/refs/heads/main.tar.gz`;
    execSync(
      `curl -sL "${tarballUrl}" | tar xz -C "${tmp}" --strip-components=1 "${repo}-main/${subpath}"`,
      { stdio: "pipe" }
    );
  } catch {
    rmSync(tmp, { recursive: true, force: true });
    throw new Error(
      `Failed to fetch ${org}/${repo}/${subpath}\n` +
        `  Make sure the repository and path exist on GitHub.`
    );
  }

  const dir = join(tmp, subpath);
  if (!existsSync(dir)) {
    rmSync(tmp, { recursive: true, force: true });
    throw new Error(`Path '${subpath}' not found in ${org}/${repo}`);
  }

  const source = `github.com/${org}/${repo}/${subpath}`;
  spinner.stop(`Fetched ${color.cyan(source)}`);

  return { dir, source, cleanup: () => rmSync(tmp, { recursive: true, force: true }) };
}

// ── Skill ───────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const SKILL_PATH = join(__dirname, "..", "skill", "SKILL.md");
const SKILL_MD = readFileSync(SKILL_PATH, "utf-8");

// ── Plugin management ───────────────────────────────────

const OPENCLAW_CONFIG_PATH = join(homedir(), ".openclaw", "openclaw.json");

function readOpenClawConfig(): any {
  if (!existsSync(OPENCLAW_CONFIG_PATH)) return null;
  return JSON.parse(readFileSync(OPENCLAW_CONFIG_PATH, "utf-8"));
}

function enableKnowledgeTools(): void {
  const config = readOpenClawConfig();
  if (!config) return;
  const pc = config.plugins?.entries?.["hindsight-openclaw"]?.config;
  if (!pc) return;
  if (pc.enableKnowledgeTools === true) return;
  pc.enableKnowledgeTools = true;
  writeFileSync(OPENCLAW_CONFIG_PATH, JSON.stringify(config, null, 2) + "\n");
}

function isPluginInstalled(): boolean {
  const config = readOpenClawConfig();
  if (!config) return false;
  return (
    config.plugins?.entries?.["hindsight-openclaw"]?.enabled !== false &&
    config.plugins?.entries?.["hindsight-openclaw"] !== undefined
  );
}

function isPluginConfigured(): boolean {
  const config = readOpenClawConfig();
  if (!config) return false;
  const pc = config.plugins?.entries?.["hindsight-openclaw"]?.config || {};
  return !!(pc.hindsightApiUrl || pc.embedVersion || pc.llmProvider);
}

function resolveFromPlugin(agentId: string): { apiUrl: string; bankId: string; apiToken?: string } {
  const config = readOpenClawConfig();
  if (!config) throw new Error("OpenClaw config not found");
  const pc = config.plugins?.entries?.["hindsight-openclaw"]?.config || {};

  const apiUrl = pc.hindsightApiUrl || `http://localhost:${pc.apiPort || 9077}`;
  const apiToken = pc.hindsightApiToken || undefined;

  let bankId: string;
  if (pc.dynamicBankId === false && pc.bankId) {
    bankId = pc.bankId;
  } else {
    const granularity: string[] = pc.dynamicBankGranularity || ["agent", "channel", "user"];
    const fieldMap: Record<string, string> = {
      agent: agentId,
      channel: "unknown",
      user: "anonymous",
      provider: "unknown",
    };
    const base = granularity.map((f) => encodeURIComponent(fieldMap[f] || "unknown")).join("::");
    bankId = pc.bankIdPrefix ? `${pc.bankIdPrefix}-${base}` : base;
  }

  return { apiUrl, bankId, apiToken };
}

function getPluginSummary(): string {
  const config = readOpenClawConfig();
  if (!config) return "Not found";
  const pc = config.plugins?.entries?.["hindsight-openclaw"]?.config || {};
  if (pc.hindsightApiUrl) return `External: ${pc.hindsightApiUrl}`;
  if (pc.embedVersion) return `Embedded v${pc.embedVersion}`;
  return "Not configured";
}

function parseAgentsJson(raw: string): any[] {
  const clean = raw.replace(/\n?\x1b\[[0-9;]*m[^\n]*/g, "").trim();
  const arrStart = clean.indexOf("\n[");
  const jsonStr = arrStart >= 0 ? clean.slice(arrStart + 1) : clean.startsWith("[") ? clean : "[]";
  return JSON.parse(jsonStr);
}

async function ensurePlugin(): Promise<void> {
  if (!isPluginInstalled()) {
    p.log.warn("Hindsight plugin not found. Installing...");
    try {
      execSync("openclaw plugins install @vectorize-io/hindsight-openclaw", { stdio: "inherit" });
    } catch {
      p.cancel(
        "Failed to install plugin. Run manually:\n  openclaw plugins install @vectorize-io/hindsight-openclaw"
      );
      process.exit(1);
    }
  }

  if (!isPluginConfigured()) {
    p.log.warn("Hindsight plugin needs configuration.");
    try {
      execSync("npx --yes --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup", {
        stdio: "inherit",
      });
    } catch {
      p.cancel(
        "Run the wizard manually:\n  npx --yes --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup"
      );
      process.exit(1);
    }
  } else {
    const summary = getPluginSummary();
    if (process.stdin.isTTY) {
      const ok = await p.confirm({
        message: `Hindsight: ${color.cyan(summary)}. Use this?\n${color.dim("  Changing this will affect all existing agents — one OpenClaw instance shares a single Hindsight instance.")}`,
      });
      if (p.isCancel(ok)) {
        p.cancel("Cancelled.");
        process.exit(0);
      }
      if (!ok) {
        p.log.info("Launching configuration wizard...");
        try {
          execSync(
            "npx --yes --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup",
            { stdio: "inherit" }
          );
        } catch {
          p.cancel(
            "Configuration failed. Run manually:\n  npx --yes --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup"
          );
          process.exit(1);
        }
      }
    } else {
      p.log.info(`Hindsight: ${color.cyan(summary)}`);
    }
  }
}

// ── Main ────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 1 || args[0] === "--help" || args[0] === "-h") {
    console.log(`
  ${color.bold("self-driving-agents")} — install a self-driving agent

  ${color.dim("Usage:")}
    npx @vectorize-io/self-driving-agents install <agent> --harness <harness> [--agent <name>]

  ${color.dim("Agent sources:")}
    ${color.cyan("marketing-agent")}             → ${DEFAULT_REPO}/marketing-agent
    ${color.cyan("org/repo/my-agent")}           → org/repo/my-agent on GitHub
    ${color.cyan("./local-dir")}                 → local directory

  ${color.dim("Options:")}
    ${color.cyan("--harness <h>")}      Required. openclaw | hermes | claude-code
    ${color.cyan("--agent <name>")}     Agent name (defaults to directory name)
`);
    process.exit(0);
  }

  let dirArg = args[0] === "install" ? args[1] : args[0];
  const restArgs = args[0] === "install" ? args.slice(2) : args.slice(1);

  if (!dirArg) {
    p.cancel("Agent argument required.");
    process.exit(1);
  }

  let harness: string | undefined;
  let agentName: string | undefined;

  for (let i = 0; i < restArgs.length; i++) {
    if (restArgs[i] === "--harness" && restArgs[i + 1]) harness = restArgs[++i];
    else if (restArgs[i] === "--agent" && restArgs[i + 1]) agentName = restArgs[++i];
  }

  if (!harness) {
    p.cancel("--harness required (openclaw | hermes | claude-code)");
    process.exit(1);
  }

  p.intro(color.bgCyan(color.black(` self-driving-agents `)));

  // Step 0: Resolve agent directory (local or GitHub)
  const spin = p.spinner();
  const { dir, source, cleanup } = await resolveAgentDir(dirArg, spin);

  try {
    const agentId = agentName || basename(dir);

    // Step 1: Ensure plugin
    if (harness === "openclaw") {
      await ensurePlugin();
      enableKnowledgeTools();
    }

    // Step 2: Resolve bank + API from plugin config
    const { apiUrl, bankId, apiToken } = resolveFromPlugin(agentId);

    const workspaceDir = join(homedir(), ".self-driving-agents", "openclaw", agentId);

    p.log.info(
      [
        `Agent:     ${color.bold(agentId)}`,
        `Source:    ${color.dim(source)}`,
        `Bank:      ${color.dim(bankId)}`,
        `API:       ${color.dim(apiUrl)}`,
        `Workspace: ${color.dim(workspaceDir)}`,
      ].join("\n")
    );

    // Step 3: Create client + health check
    const client = new HindsightClient({
      baseUrl: apiUrl,
      apiKey: apiToken,
      userAgent: "self-driving-agents/0.1.0",
    });
    const lowLevel = createClient(
      createConfig({
        baseUrl: apiUrl,
        headers: {
          ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
          "User-Agent": "self-driving-agents/0.1.0",
        },
      })
    );

    spin.start("Connecting to Hindsight...");
    try {
      await sdk.healthEndpointHealthGet({ client: lowLevel });
      spin.stop("Connected to Hindsight");
    } catch {
      spin.stop("Connection failed");
      p.cancel(`Cannot reach Hindsight at ${apiUrl}\nStart the server or reconfigure the plugin.`);
      process.exit(1);
    }

    // Step 4: Import bank template
    const templatePath = join(dir, "bank-template.json");
    if (existsSync(templatePath)) {
      spin.start("Importing bank template...");
      const template = JSON.parse(readFileSync(templatePath, "utf-8"));
      await sdk.importBankTemplate({
        client: lowLevel,
        path: { bank_id: bankId },
        body: template,
      });
      spin.stop("Bank template imported");
    }

    // Step 5: Ingest content (recursive — all text files except bank-template.json)
    const contentFiles = findContentFiles(dir);
    if (contentFiles.length > 0) {
      spin.start(`Ingesting ${contentFiles.length} file(s)...`);
      for (const relPath of contentFiles) {
        const content = readFileSync(join(dir, relPath), "utf-8");
        if (!content.trim()) continue;
        // Use relative path (without extension) as document ID, e.g. "seo/keyword-research"
        const docId = relPath.replace(/\.[^.]+$/, "");
        await client.retainBatch(bankId, [{ content, document_id: docId }], { async: true });
        spin.message(`Ingesting ${relPath}...`);
      }
      spin.stop(`Ingested ${contentFiles.length} file(s)`);
    }

    // Step 6: Create agent + install skill
    mkdirSync(workspaceDir, { recursive: true });

    const skillDir = join(workspaceDir, "skills", "agent-knowledge");
    mkdirSync(skillDir, { recursive: true });
    writeFileSync(join(skillDir, "SKILL.md"), SKILL_MD);
    p.log.success("Knowledge skill installed");

    if (harness === "openclaw") {
      try {
        const listOut = execSync("openclaw agents list --json 2>/dev/null", { encoding: "utf-8" });
        const agents = parseAgentsJson(listOut);
        if (!agents.some((a: any) => a.name === agentId || a.id === agentId)) {
          execSync(`openclaw agents add ${agentId} --workspace ${workspaceDir} --non-interactive`, {
            stdio: "pipe",
          });
          p.log.success(`Agent '${agentId}' created`);
        } else {
          p.log.info(`Agent '${agentId}' already exists`);
        }
      } catch {
        p.log.warn(
          `Create agent manually:\n  openclaw agents add ${agentId} --workspace ${workspaceDir} --non-interactive`
        );
      }
    }

    // Step 7: Patch startup
    const startupFile = join(workspaceDir, "AGENTS.md");
    if (existsSync(startupFile)) {
      let text = readFileSync(startupFile, "utf-8");
      if (!text.includes("agent-knowledge")) {
        text = text.replace(
          "Don't ask permission. Just do it.",
          "5. Read `skills/agent-knowledge/SKILL.md` and **execute its mandatory startup sequence**\n\nDon't ask permission. Just do it."
        );
        writeFileSync(startupFile, text);
        p.log.success("Startup patched");
      }
    }

    p.note(
      [
        `${color.dim("1.")} openclaw gateway restart`,
        `${color.dim("2.")} openclaw tui --session agent:${agentId}:main:session1`,
      ].join("\n"),
      "Next steps"
    );

    p.outro(color.green(`'${agentId}' is ready`));
  } finally {
    cleanup?.();
  }
}

main().catch((err) => {
  p.cancel(err.message);
  process.exit(1);
});
