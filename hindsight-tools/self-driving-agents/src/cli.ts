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
): Promise<{ dir: string; source: string; defaultName: string; cleanup?: () => void }> {
  if (isLocalPath(input)) {
    const dir = resolve(input.replace(/^~/, homedir()));
    if (!existsSync(dir)) throw new Error(`Directory not found: ${dir}`);
    return { dir, source: dir, defaultName: basename(dir) };
  }

  // Parse GitHub reference: "name" or "org/repo/path/to/agent"
  const parts = input.split("/");
  let org: string, repo: string, subpath: string;

  if (parts.length <= 2) {
    // "name" or "name/subpath" → default repo
    org = "vectorize-io";
    repo = "self-driving-agents";
    subpath = input;
  } else {
    // org/repo/path...
    org = parts[0];
    repo = parts[1];
    subpath = parts.slice(2).join("/");
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
  const defaultName = subpath.replace(/\//g, "-");
  spinner.stop(`Fetched ${color.cyan(source)}`);

  return { dir, source, defaultName, cleanup: () => rmSync(tmp, { recursive: true, force: true }) };
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

const MIN_PLUGIN_VERSION = "0.7.2";

function getInstalledPluginVersion(): string | null {
  try {
    // Check the installed plugin's package.json
    const extDir = join(homedir(), ".openclaw", "extensions", "hindsight-openclaw");
    const pkgPath = join(extDir, "package.json");
    if (!existsSync(pkgPath)) return null;
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
    return pkg.version || null;
  } catch {
    return null;
  }
}

function versionGte(current: string, required: string): boolean {
  const [aMaj, aMin, aPat] = current.split(".").map(Number);
  const [bMaj, bMin, bPat] = required.split(".").map(Number);
  if (aMaj !== bMaj) return aMaj > bMaj;
  if (aMin !== bMin) return aMin > bMin;
  return aPat >= bPat;
}

function isPluginInstalled(): boolean {
  const config = readOpenClawConfig();
  if (!config) return false;
  const hasConfig =
    config.plugins?.entries?.["hindsight-openclaw"]?.enabled !== false &&
    config.plugins?.entries?.["hindsight-openclaw"] !== undefined;
  // Also check the extension dir actually exists (may have been deleted during a failed upgrade)
  const extDir = join(homedir(), ".openclaw", "extensions", "hindsight-openclaw");
  return hasConfig && existsSync(extDir);
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
  const installed = isPluginInstalled();
  const currentVersion = installed ? getInstalledPluginVersion() : null;
  const needsInstall = !installed;
  const needsUpgrade =
    installed && currentVersion && !versionGte(currentVersion, MIN_PLUGIN_VERSION);

  if (needsInstall || needsUpgrade) {
    if (needsUpgrade) {
      p.log.warn(
        `Hindsight plugin v${currentVersion} is outdated (need >=${MIN_PLUGIN_VERSION}). Upgrading...`
      );
    } else {
      p.log.warn("Hindsight plugin not found. Installing...");
    }
    try {
      // Remove old extension if present — openclaw doesn't support in-place upgrade
      const extDir = join(homedir(), ".openclaw", "extensions", "hindsight-openclaw");
      rmSync(extDir, { recursive: true, force: true });

      // Temporarily clear plugins.slots.memory so openclaw doesn't reject
      // the config while the extension is missing
      const cfg = readOpenClawConfig();
      if (cfg?.plugins?.slots?.memory === "hindsight-openclaw") {
        delete cfg.plugins.slots.memory;
        writeFileSync(OPENCLAW_CONFIG_PATH, JSON.stringify(cfg, null, 2) + "\n");
      }

      execSync("openclaw plugins install @vectorize-io/hindsight-openclaw", { stdio: "inherit" });
      const newVersion = getInstalledPluginVersion();
      p.log.success(`Hindsight plugin v${newVersion} installed`);
    } catch {
      p.cancel(
        "Failed to install plugin. Run manually:\n  openclaw plugins install @vectorize-io/hindsight-openclaw"
      );
      process.exit(1);
    }
  } else if (currentVersion) {
    p.log.info(`Hindsight plugin v${currentVersion}`);
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

// ── NemoClaw plugin management ─────────────────────────

function listNemoClawSandboxes(): string[] {
  try {
    const out = execSync("nemoclaw list", { encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] });
    return out
      .split("\n")
      .filter((l) => /^\s{4}\S/.test(l) && !l.includes("model:") && !l.includes("dashboard:"))
      .map((l) => l.trim().replace(/\s*\*$/, ""));
  } catch {
    return [];
  }
}

async function detectNemoClawSandbox(): Promise<string> {
  const sandboxes = listNemoClawSandboxes();

  if (sandboxes.length === 0) {
    p.cancel("No NemoClaw sandboxes found. Create one with: nemoclaw onboard");
    process.exit(1);
  }

  if (sandboxes.length === 1) {
    p.log.info(`Using sandbox: ${color.cyan(sandboxes[0])}`);
    return sandboxes[0];
  }

  const selected = await p.select({
    message: "Select a NemoClaw sandbox:",
    options: sandboxes.map((s) => ({ value: s, label: s })),
  });

  if (p.isCancel(selected)) {
    p.cancel("Cancelled.");
    process.exit(0);
  }

  return selected as string;
}

async function ensureNemoClawPlugin(sandboxName: string, agentId: string): Promise<void> {
  // Check nemoclaw is installed
  try {
    execSync("which nemoclaw", { stdio: "pipe" });
  } catch {
    p.cancel(
      "nemoclaw not found. Install it: curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash"
    );
    process.exit(1);
  }

  // Check sandbox exists
  try {
    execSync(`nemoclaw ${sandboxName} status`, { stdio: "pipe" });
  } catch {
    p.cancel(`Sandbox '${sandboxName}' not found. Create one with: nemoclaw onboard`);
    process.exit(1);
  }

  // NemoClaw runs OpenClaw inside a sandbox with read-only config (Landlock).
  // hindsight-nemoclaw setup handles everything:
  //   1. Installs the openclaw plugin
  //   2. Writes plugin config to host ~/.openclaw/openclaw.json
  //   3. Adds the Hindsight network policy to the sandbox
  //   4. Restarts the gateway
  // We always run it — it's idempotent and ensures the sandbox has the
  // network policy even if the host already has the plugin configured.
  const config = readOpenClawConfig();
  const pc = config?.plugins?.entries?.["hindsight-openclaw"]?.config || {};

  if (!pc.hindsightApiUrl || !pc.hindsightApiToken) {
    // No Hindsight config at all — run interactive setup
    p.log.warn("Hindsight plugin needs configuration for NemoClaw.");
    try {
      execSync(
        `npx --yes --package @vectorize-io/hindsight-nemoclaw hindsight-nemoclaw setup --sandbox ${sandboxName}`,
        { stdio: "inherit" }
      );
    } catch {
      p.cancel(
        "Plugin setup failed. Run manually:\n  npx --yes --package @vectorize-io/hindsight-nemoclaw hindsight-nemoclaw setup --sandbox " +
          sandboxName
      );
      process.exit(1);
    }
  } else {
    // Config exists — run non-interactive setup to ensure network policy + plugin are in place
    const apiUrl = pc.hindsightApiUrl;
    const apiToken = pc.hindsightApiToken;
    const bankPrefix = pc.bankIdPrefix || "nemoclaw";
    p.log.info(`Hindsight: ${color.cyan(`External: ${apiUrl}`)}`);
    try {
      execSync(
        `npx --yes --package @vectorize-io/hindsight-nemoclaw hindsight-nemoclaw setup` +
          ` --sandbox ${sandboxName}` +
          ` --api-url ${apiUrl}` +
          ` --api-token ${apiToken}` +
          ` --bank-prefix ${bankPrefix}` +
          ` --skip-plugin-install`,
        { stdio: "inherit" }
      );
    } catch {
      p.cancel(
        `Failed to apply sandbox network policy.\n` +
          `  The sandbox may have been destroyed. Check with: nemoclaw list\n` +
          `  Recreate with: nemoclaw onboard`
      );
      process.exit(1);
    }
  }

  enableKnowledgeTools();

  // Rebuild sandbox so it picks up the latest host config
  p.log.info("Rebuilding sandbox to apply config...");
  try {
    execSync(`nemoclaw ${sandboxName} rebuild --yes`, { stdio: "inherit" });
    p.log.success("Sandbox rebuilt");
  } catch {
    p.cancel(
      `Failed to rebuild sandbox '${sandboxName}'.\n` +
        `  Check with: nemoclaw list\n` +
        `  Recreate with: nemoclaw onboard`
    );
    process.exit(1);
  }
}

// ── Hermes plugin management ───────────────────────────

function ensureHermesPlugin(
  agentId: string,
  apiUrl: string,
  bankId: string,
  apiToken?: string
): void {
  // Check hermes is installed
  try {
    execSync("which hermes", { stdio: "pipe" });
  } catch {
    p.cancel(
      "hermes not found. Install it: curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash"
    );
    process.exit(1);
  }

  // Create a Hermes profile for this agent (or reuse if exists)
  let profileHome: string;
  try {
    const showOut = execSync(`hermes profile show ${agentId}`, {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    const pathMatch = showOut.match(/Path:\s+(\S+)/);
    profileHome = pathMatch ? pathMatch[1] : join(homedir(), ".hermes", "profiles", agentId);
    p.log.info(`Hermes profile '${agentId}' already exists`);
  } catch {
    // Profile doesn't exist — create it (clone config from default)
    try {
      execSync(`hermes profile create ${agentId} --clone`, {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      });
      const showOut = execSync(`hermes profile show ${agentId}`, {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      });
      const pathMatch = showOut.match(/Path:\s+(\S+)/);
      profileHome = pathMatch ? pathMatch[1] : join(homedir(), ".hermes", "profiles", agentId);
      p.log.success(`Hermes profile '${agentId}' created`);
    } catch (err: any) {
      const msg = err?.stderr?.toString?.()?.trim() || err?.message || String(err);
      p.cancel(`Failed to create Hermes profile: ${msg}`);
      process.exit(1);
    }
  }

  // Install plugin into the profile
  const pluginDir = join(profileHome, "plugins", "hindsight-sda");
  const sdaPluginSrc = join(__dirname, "..", "hermes-plugin");
  mkdirSync(pluginDir, { recursive: true });
  for (const file of ["plugin.yaml", "__init__.py"]) {
    const src = join(sdaPluginSrc, file);
    if (existsSync(src)) {
      writeFileSync(join(pluginDir, file), readFileSync(src, "utf-8"));
    }
  }
  p.log.success("Hermes plugin installed");

  // Write hindsight/config.json in the profile — single source of truth for
  // both the bundled hindsight provider (auto-retain) and our tool plugin.
  // Static bank_id, no template — both read the same bank.
  const hindsightCfgDir = join(profileHome, "hindsight");
  mkdirSync(hindsightCfgDir, { recursive: true });
  writeFileSync(
    join(hindsightCfgDir, "config.json"),
    JSON.stringify(
      {
        mode: "cloud",
        api_url: apiUrl,
        api_key: apiToken,
        bank_id: bankId,
        bank_id_template: "",
        recall_budget: "mid",
        memory_mode: "hybrid",
      },
      null,
      2
    ) + "\n"
  );

  // Set the bundled hindsight as memory provider + enable our plugin in config.yaml
  const hermesConfigPath = join(profileHome, "config.yaml");
  if (existsSync(hermesConfigPath)) {
    let config = readFileSync(hermesConfigPath, "utf-8");

    // Set memory.provider to hindsight (bundled provider for auto-retain)
    const lines = config.split("\n");
    let inMemory = false;
    for (let i = 0; i < lines.length; i++) {
      if (/^memory:/.test(lines[i])) inMemory = true;
      else if (/^\S/.test(lines[i]) && inMemory) inMemory = false;
      if (inMemory && /^\s+provider:\s/.test(lines[i])) {
        lines[i] = lines[i].replace(/provider:\s*\S+/, "provider: hindsight");
        break;
      }
    }
    config = lines.join("\n");

    // Enable our tool plugin in plugins.enabled
    if (!config.includes("hindsight-sda")) {
      if (/plugins:\s*\n\s+enabled:/.test(config)) {
        config = config.replace(/(plugins:\s*\n\s+enabled:\s*\n)/, "$1    - hindsight-sda\n");
      } else if (/plugins:/.test(config)) {
        config = config.replace(/(plugins:)/, "$1\n  enabled:\n    - hindsight-sda");
      } else {
        config += "\nplugins:\n  enabled:\n    - hindsight-sda\n";
      }
    }

    writeFileSync(hermesConfigPath, config);
  }
  p.log.success("Hindsight memory + knowledge tools configured");

  // Install skill into the profile
  const skillDir = join(profileHome, "skills", "agent-knowledge");
  mkdirSync(skillDir, { recursive: true });
  writeFileSync(join(skillDir, "SKILL.md"), SKILL_MD);
  p.log.success("Knowledge skill installed");
}

// ── Claude skill generation ────────────────────────────

const CLAUDE_SKILLS_DIR = join(homedir(), "self-driving-agents", "claude");
const HINDSIGHT_CLOUD_API_URL = "https://api.hindsight.vectorize.io";

async function generateClaudeSkill(
  agentId: string,
  apiUrl: string,
  bankId: string,
  apiToken?: string
): Promise<string> {
  const authHeader = apiToken ? `-H "Authorization: Bearer ${apiToken}" \\\n      ` : "";
  const skillMd = `# Hindsight Memory — ${agentId}

## Mandatory Startup Sequence

On every new conversation, run these commands **before doing anything else**:

### 1. List all knowledge pages
\`\`\`bash
curl -s ${apiUrl}/v1/default/banks/${bankId}/knowledge/pages \\
  ${authHeader}-H "Content-Type: application/json"
\`\`\`

### 2. Read each page (replace PAGE_ID)
\`\`\`bash
curl -s ${apiUrl}/v1/default/banks/${bankId}/knowledge/pages/PAGE_ID \\
  ${authHeader}-H "Content-Type: application/json"
\`\`\`

Read **every** page listed in step 1 to load the agent's full knowledge base.

## Creating Knowledge Pages

\`\`\`bash
curl -s -X POST ${apiUrl}/v1/default/banks/${bankId}/knowledge/pages \\
  ${authHeader}-H "Content-Type: application/json" \\
  -d '{"title": "Page Title", "content": "Markdown content here"}'
\`\`\`

## Searching Memories

\`\`\`bash
curl -s -X POST ${apiUrl}/v1/default/banks/${bankId}/memories/recall \\
  ${authHeader}-H "Content-Type: application/json" \\
  -d '{"query": "your search query"}'
\`\`\`

## Ingesting Documents

\`\`\`bash
curl -s -X POST ${apiUrl}/v1/default/banks/${bankId}/memories/retain \\
  ${authHeader}-H "Content-Type: application/json" \\
  -d '{"content": "Document content to remember"}'
\`\`\`

## Updating Knowledge Pages

\`\`\`bash
curl -s -X PUT ${apiUrl}/v1/default/banks/${bankId}/knowledge/pages/PAGE_ID \\
  ${authHeader}-H "Content-Type: application/json" \\
  -d '{"title": "Updated Title", "content": "Updated content"}'
\`\`\`

## Deleting Knowledge Pages

\`\`\`bash
curl -s -X DELETE ${apiUrl}/v1/default/banks/${bankId}/knowledge/pages/PAGE_ID \\
  ${authHeader}-H "Content-Type: application/json"
\`\`\`

## Retaining Conversations

Claude Chat and Cowork do not have automatic hooks. You must **self-retain** important conversation content.

After any significant exchange (decisions, new information, task outcomes), retain it:

\`\`\`bash
curl -s -X POST ${apiUrl}/v1/default/banks/${bankId}/memories/retain \\
  ${authHeader}-H "Content-Type: application/json" \\
  -d '{"content": "Summary of the conversation or key information learned"}'
\`\`\`

**When to self-retain:**
- After learning user preferences or project context
- After completing a task (retain the outcome)
- After receiving corrections or feedback
- After discovering important facts during research

## Important Notes

- Always run the startup sequence at the beginning of every conversation
- Knowledge pages are persistent — they survive across conversations
- Use recall to search existing memories before creating duplicates
- Self-retain is essential since there are no automatic hooks in Claude Chat/Cowork
`;

  const outDir = join(CLAUDE_SKILLS_DIR, agentId);
  mkdirSync(outDir, { recursive: true });
  const skillPath = join(outDir, "SKILL.md");
  writeFileSync(skillPath, skillMd);

  // Create a zip of the skill directory
  const zipPath = join(outDir, `${agentId}-skill.zip`);
  execSync(`cd "${outDir}" && zip -j "${zipPath}" SKILL.md`, { stdio: "pipe" });

  return zipPath;
}

async function promptClaudeConfig(
  agentId: string
): Promise<{ apiUrl: string; bankId: string; apiToken?: string }> {
  const deploymentType = await p.select({
    message: "Hindsight deployment:",
    options: [
      { value: "cloud", label: "Cloud (api.hindsight.vectorize.io)" },
      { value: "self-hosted", label: "Self-hosted" },
    ],
  });
  if (p.isCancel(deploymentType)) {
    p.cancel("Cancelled.");
    process.exit(0);
  }

  let apiUrl: string;
  if (deploymentType === "cloud") {
    apiUrl = HINDSIGHT_CLOUD_API_URL;
  } else {
    const urlInput = await p.text({
      message: "Hindsight API URL:",
      placeholder: "https://your-hindsight.example.com",
      validate: (val) => {
        if (!val) return "URL is required";
        if (val.startsWith("http://localhost") || val.startsWith("http://127.0.0.1")) {
          return "Claude cannot reach localhost. Use a publicly accessible URL.";
        }
        return undefined;
      },
    });
    if (p.isCancel(urlInput)) {
      p.cancel("Cancelled.");
      process.exit(0);
    }
    apiUrl = urlInput as string;
    p.log.warn("Make sure your Hindsight instance is publicly accessible from Claude's servers.");
  }

  const tokenInput = await p.text({ message: "Hindsight API token:" });
  if (p.isCancel(tokenInput)) {
    p.cancel("Cancelled.");
    process.exit(0);
  }
  const apiToken = (tokenInput as string) || undefined;

  const bankInput = await p.text({
    message: "Bank ID:",
    initialValue: agentId,
  });
  if (p.isCancel(bankInput)) {
    p.cancel("Cancelled.");
    process.exit(0);
  }
  const bankId = (bankInput as string).trim() || agentId;

  return { apiUrl, bankId, apiToken };
}

// ── Claude Code plugin management ─────────────────────

const CLAUDE_CODE_USER_CONFIG_DIR = join(homedir(), ".hindsight");
const CLAUDE_CODE_USER_CONFIG_PATH = join(CLAUDE_CODE_USER_CONFIG_DIR, "claude-code.json");

function readClaudeCodeConfig(): any {
  if (!existsSync(CLAUDE_CODE_USER_CONFIG_PATH)) return null;
  return JSON.parse(readFileSync(CLAUDE_CODE_USER_CONFIG_PATH, "utf-8"));
}

function writeClaudeCodeConfig(config: any): void {
  mkdirSync(CLAUDE_CODE_USER_CONFIG_DIR, { recursive: true });
  writeFileSync(CLAUDE_CODE_USER_CONFIG_PATH, JSON.stringify(config, null, 2) + "\n");
}

function resolveFromClaudeCode(agentId: string): {
  apiUrl: string;
  bankId: string;
  apiToken?: string;
} {
  const config = readClaudeCodeConfig();
  if (!config) throw new Error("Claude Code config not found at " + CLAUDE_CODE_USER_CONFIG_PATH);

  const apiUrl = config.apiUrl || HINDSIGHT_CLOUD_API_URL;
  const apiToken = config.apiToken || undefined;

  let bankId: string;
  if (config.dynamicBankId === false && config.bankId) {
    bankId = config.bankId;
  } else {
    const granularity: string[] = config.dynamicBankGranularity || ["agent"];
    const fieldMap: Record<string, string> = {
      agent: agentId,
    };
    const base = granularity.map((f) => encodeURIComponent(fieldMap[f] || "unknown")).join("::");
    bankId = config.bankIdPrefix ? `${config.bankIdPrefix}-${base}` : base;
  }

  return { apiUrl, bankId, apiToken };
}

async function ensureClaudeCodePlugin(agentId: string): Promise<{ apiUrl: string; bankId: string; apiToken?: string }> {
  // Write or update config
  const config = readClaudeCodeConfig() || {};
  config.agentName = agentId;
  config.enableKnowledgeTools = true;

  // Always prompt for Hindsight connection — each agent install needs correct API + token
  let resolvedApiUrl = config.hindsightApiUrl || `http://localhost:${config.apiPort || 9077}`;
  let resolvedBankId = agentId;
  let resolvedApiToken = config.hindsightApiToken || undefined;

  if (process.stdin.isTTY) {
    const claudeConfig = await promptClaudeConfig(agentId);
    config.hindsightApiUrl = claudeConfig.apiUrl;
    config.hindsightApiToken = claudeConfig.apiToken;
    config.dynamicBankId = false;
    resolvedApiUrl = claudeConfig.apiUrl;
    resolvedBankId = claudeConfig.bankId;
    resolvedApiToken = claudeConfig.apiToken;
  }

  writeClaudeCodeConfig(config);

  return { apiUrl: resolvedApiUrl, bankId: resolvedBankId, apiToken: resolvedApiToken };
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
    ${color.cyan("--harness <h>")}      Required. openclaw | nemoclaw | hermes | claude | claude-code
    ${color.cyan("--agent <name>")}     Agent name (defaults to directory name)
    ${color.cyan("--sandbox <name>")}   NemoClaw sandbox (auto-detected if only one exists)
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
  let sandbox: string | undefined;

  for (let i = 0; i < restArgs.length; i++) {
    if (restArgs[i] === "--harness" && restArgs[i + 1]) harness = restArgs[++i];
    else if (restArgs[i] === "--agent" && restArgs[i + 1]) agentName = restArgs[++i];
    else if (restArgs[i] === "--sandbox" && restArgs[i + 1]) sandbox = restArgs[++i];
  }

  if (!harness) {
    p.cancel("--harness required (openclaw | nemoclaw | hermes | claude | claude-code)");
    process.exit(1);
  }

  if (harness === "nemoclaw" && !sandbox) {
    sandbox = await detectNemoClawSandbox();
  }

  p.intro(color.bgCyan(color.black(` self-driving-agents `)));

  // Step 0: Resolve agent directory (local or GitHub)
  const spin = p.spinner();
  const { dir, source, defaultName, cleanup } = await resolveAgentDir(dirArg, spin);

  try {
    let agentId: string;
    if (agentName) {
      agentId = agentName;
    } else if (process.stdin.isTTY) {
      const nameInput = await p.text({
        message: "Agent name:",
        initialValue: defaultName,
      });
      if (p.isCancel(nameInput)) {
        p.cancel("Cancelled.");
        process.exit(0);
      }
      agentId = (nameInput as string).trim() || defaultName;
    } else {
      agentId = defaultName;
    }

    // Step 1: Ensure plugin
    let apiUrl: string;
    let bankId: string;
    let apiToken: string | undefined;
    let claudeSkillZip: string | undefined;

    if (harness === "openclaw") {
      await ensurePlugin();
      enableKnowledgeTools();
      ({ apiUrl, bankId, apiToken } = resolveFromPlugin(agentId));
    } else if (harness === "nemoclaw") {
      await ensureNemoClawPlugin(sandbox!, agentId);
      ({ apiUrl, bankId, apiToken } = resolveFromPlugin(agentId));
    } else if (harness === "hermes") {
      // Resolve Hindsight connection: hermes config > openclaw config > prompt
      const hermesHsCfgPath = join(homedir(), ".hermes", "hindsight", "config.json");
      let defaultUrl = "https://api.hindsight.vectorize.io";
      let defaultToken = "";

      // Check existing hermes hindsight config first
      if (existsSync(hermesHsCfgPath)) {
        try {
          const hsCfg = JSON.parse(readFileSync(hermesHsCfgPath, "utf-8"));
          if (hsCfg.api_url) defaultUrl = hsCfg.api_url;
          if (hsCfg.api_key) defaultToken = hsCfg.api_key;
        } catch {
          /* ignore */
        }
      }

      // Fall back to openclaw config
      if (!defaultToken) {
        const config = readOpenClawConfig();
        const pc = config?.plugins?.entries?.["hindsight-openclaw"]?.config;
        if (pc?.hindsightApiUrl) defaultUrl = pc.hindsightApiUrl;
        if (pc?.hindsightApiToken) defaultToken = pc.hindsightApiToken;
      }

      // If we have credentials, confirm; otherwise prompt
      if (defaultUrl && defaultToken) {
        const ok = await p.confirm({
          message: `Hindsight: ${color.cyan(defaultUrl)}. Use this?`,
        });
        if (p.isCancel(ok)) {
          p.cancel("Cancelled.");
          process.exit(0);
        }
        if (ok) {
          apiUrl = defaultUrl;
          apiToken = defaultToken;
        } else {
          const urlInput = await p.text({
            message: "Hindsight API URL:",
            initialValue: defaultUrl,
          });
          if (p.isCancel(urlInput)) {
            p.cancel("Cancelled.");
            process.exit(0);
          }
          const tokenInput = await p.text({ message: "Hindsight API token:" });
          if (p.isCancel(tokenInput)) {
            p.cancel("Cancelled.");
            process.exit(0);
          }
          apiUrl = urlInput as string;
          apiToken = tokenInput as string;
        }
      } else {
        const urlInput = await p.text({ message: "Hindsight API URL:", initialValue: defaultUrl });
        if (p.isCancel(urlInput)) {
          p.cancel("Cancelled.");
          process.exit(0);
        }
        const tokenInput = await p.text({ message: "Hindsight API token:" });
        if (p.isCancel(tokenInput)) {
          p.cancel("Cancelled.");
          process.exit(0);
        }
        apiUrl = urlInput as string;
        apiToken = tokenInput as string;
      }

      bankId = agentId;
      ensureHermesPlugin(agentId, apiUrl, bankId, apiToken);
    } else if (harness === "claude") {
      const claudeConfig = await promptClaudeConfig(agentId);
      apiUrl = claudeConfig.apiUrl;
      bankId = claudeConfig.bankId;
      apiToken = claudeConfig.apiToken;
    } else if (harness === "claude-code") {
      ({ apiUrl, bankId, apiToken } = await ensureClaudeCodePlugin(agentId));
    } else {
      p.cancel(`Unknown harness: ${harness}`);
      process.exit(1);
    }

    const workspaceDir = join(homedir(), ".self-driving-agents", harness, agentId);

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

    // Step 6: Create agent + install skill (hermes handled in ensureHermesPlugin)
    if (harness === "claude") {
      claudeSkillZip = await generateClaudeSkill(agentId, apiUrl, bankId, apiToken);
    } else if (harness === "claude-code") {
      // Install per-agent subagent at ~/.claude/agents/<agentId>.md
      const agentsDir = join(homedir(), ".claude", "agents");
      mkdirSync(agentsDir, { recursive: true });
      const agentFile = join(agentsDir, `${agentId}.md`);
      writeFileSync(
        agentFile,
        `---
name: ${agentId}
description: ${agentId} agent with long-term memory. Delegate to this agent for tasks related to ${agentId.replace(/-/g, " ")}. It has access to knowledge pages and memory search via Hindsight.
skills:
  - agent-knowledge
mcpServers:
  - hindsight
hooks:
  Stop:
    - hooks:
        - type: command
          command: HINDSIGHT_BANK_ID="${bankId}" python3 "\${CLAUDE_PLUGIN_ROOT}/scripts/subagent_retain.py"
          timeout: 15
          async: true
---

You are the **${agentId}** agent with long-term memory powered by Hindsight.

## Startup — run these steps immediately

1. Call \`agent_knowledge_list_pages(bank_id="${bankId}")\` to see your knowledge pages.
2. Call \`agent_knowledge_get_page(page_id, bank_id="${bankId}")\` for each page to load your knowledge.
3. Use this knowledge to inform everything you do in this conversation.

## Creating pages

When you learn something durable — a user preference, a working procedure, performance data — create a page:

\`agent_knowledge_create_page(page_id, name, source_query, bank_id="${bankId}")\`

- \`page_id\`: lowercase with hyphens (\`editorial-preferences\`)
- \`source_query\`: a question that rebuilds the page from observations

## Searching memories

\`agent_knowledge_recall(query, bank_id="${bankId}")\` — search conversations and documents for specific facts.

## Ingesting documents

\`agent_knowledge_ingest(title, content, bank_id="${bankId}")\` — upload raw content into memory.

## Updating and deleting

- \`agent_knowledge_update_page(page_id, name?, source_query?, bank_id="${bankId}")\`
- \`agent_knowledge_delete_page(page_id, bank_id="${bankId}")\`

## Important

- Always pass \`bank_id="${bankId}"\` on every agent_knowledge_* tool call
- Pages update automatically — don't edit content directly
- Create pages silently — don't announce it to the user
- Prefer fewer broad pages over many narrow ones
`
      );
      p.log.success(`Subagent installed at ${color.dim(agentFile)}`);
    } else if (harness === "hermes") {
      // Skill + plugin already installed by ensureHermesPlugin
    } else if (harness === "nemoclaw") {
      // NemoClaw: install skill into the sandbox via nemoclaw CLI
      const tmpSkillDir = join(tmpdir(), `sda-skill-${Date.now()}`);
      const tmpSkill = join(tmpSkillDir, "agent-knowledge");
      mkdirSync(tmpSkill, { recursive: true });
      writeFileSync(join(tmpSkill, "SKILL.md"), SKILL_MD);
      try {
        execSync(`nemoclaw ${sandbox} skill install ${tmpSkill}`, { stdio: "inherit" });
        p.log.success("Knowledge skill installed in sandbox");
      } catch (err: any) {
        const stderr = err?.stderr?.toString?.()?.trim() || "";
        const msg = stderr || err?.message || String(err);
        p.log.warn(
          `Failed to install skill: ${msg}\n  Install manually:\n  nemoclaw ${sandbox} skill install <skill-dir>`
        );
      } finally {
        rmSync(tmpSkillDir, { recursive: true, force: true });
      }
    } else {
      // OpenClaw: install skill locally + create agent
      mkdirSync(workspaceDir, { recursive: true });

      const skillDir = join(workspaceDir, "skills", "agent-knowledge");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), SKILL_MD);
      p.log.success("Knowledge skill installed");

      try {
        const listOut = execSync("openclaw agents list --json", {
          encoding: "utf-8",
          stdio: ["pipe", "pipe", "pipe"],
        });
        const agents = parseAgentsJson(listOut);
        if (!agents.some((a: any) => a.name === agentId || a.id === agentId)) {
          execSync(`openclaw agents add ${agentId} --workspace ${workspaceDir} --non-interactive`, {
            encoding: "utf-8",
            stdio: ["pipe", "pipe", "pipe"],
          });
          p.log.success(`Agent '${agentId}' created`);
        } else {
          p.log.info(`Agent '${agentId}' already exists`);
        }
      } catch (err: any) {
        const stderr = err?.stderr?.toString?.()?.trim() || "";
        const msg = stderr || err?.message || String(err);
        p.log.warn(
          `Failed to manage agent: ${msg}\n  Create manually:\n  openclaw agents add ${agentId} --workspace ${workspaceDir} --non-interactive`
        );
      }

      // Patch startup
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
    }

    // Next steps
    let nextSteps: string[];
    if (harness === "claude") {
      const apiHost = new URL(apiUrl).hostname;
      nextSteps = [
        `${color.dim("1.")} Open Claude → Customize → Skills → Upload skill`,
        `${color.dim("2.")} Select: ${color.cyan(claudeSkillZip!)}`,
        `${color.dim("3.")} Allowlist the API host: Settings → Capabilities → add ${color.cyan(apiHost)}`,
        `${color.dim("4.")} Start a conversation and type ${color.cyan(`/${agentId}`)} to activate the agent`,
      ];
    } else if (harness === "claude-code") {
      nextSteps = [
        `${color.dim("1.")} Start Claude Code: ${color.cyan("claude")}`,
        `${color.dim("2.")} Claude will auto-delegate to ${color.cyan(agentId)} when relevant, or mention ${color.cyan(`@${agentId}`)}`,
        `${color.dim("3.")} Conversations are automatically retained via hooks`,
      ];
    } else if (harness === "hermes") {
      nextSteps = [`${color.dim("1.")} hermes -p ${agentId} chat`];
    } else if (harness === "nemoclaw") {
      nextSteps = [
        `${color.dim("1.")} nemoclaw ${sandbox} connect`,
        `${color.dim("2.")} openclaw tui --session agent:main:main:session1`,
      ];
    } else {
      nextSteps = [
        `${color.dim("1.")} openclaw gateway restart`,
        `${color.dim("2.")} openclaw tui --session agent:${agentId}:main:session1`,
      ];
    }

    p.note(nextSteps.join("\n"), "Next steps");

    p.outro(color.green(`'${agentId}' is ready`));
  } finally {
    cleanup?.();
  }
}

main().catch((err) => {
  p.cancel(err.message);
  process.exit(1);
});
