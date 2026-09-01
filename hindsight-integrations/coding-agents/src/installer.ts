#!/usr/bin/env node
/**
 * hindsight-coding-agents install|uninstall [harness...]
 *
 * ONE setup command for every supported coding agent. With no harness arguments it detects which
 * agents exist on this machine (binary on PATH or config dir present) and wires each one's NATIVE
 * integration — hooks + MCP where the host wants them:
 *
 *   opencode     add this package to `plugin` in ~/.config/opencode/opencode.json
 *   claude-code  3 hooks in ~/.claude/settings.json + `claude mcp add` (user scope)
 *   codex        3 hooks in ~/.codex/hooks.json + [features]/[mcp_servers] in config.toml
 *   antigravity-cli PreInvocation + Stop hooks, Hindsight status line + MCP
 *   devin-cli     SessionStart + UserPromptSubmit + Stop hooks in ~/.config/devin/config.json + MCP
 *   cursor-cli   sessionStart + beforeSubmitPrompt + stop hooks in ~/.cursor/hooks.json + ~/.cursor/mcp.json
 *   copilot-cli  sessionStart + userPromptTransformed + agentStop hooks in ~/.copilot/hooks/ + MCP
 *   cline-cli    native in-process plugin + MCP
 *   dcode        native Agent Plugin via `dcode plugin marketplace/install`
 *   dsh          native Cordis plugin row in $DSH_HOME/cordis.patch.yml (DeepSeek Harness)
 *
 * IDEMPOTENT: our entries are recognized by the package path in their command ("hindsight-coding-
 * agents"), replaced on re-install (so moving the package just needs `install` again) and removed
 * on `uninstall`. Everything else in the host's config files is preserved; files are created when
 * missing. Backups: the first time we touch an existing file we write `<file>.hindsight-backup`.
 */
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { isatty } from "node:tty";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { applyEdits, modify } from "jsonc-parser";
import { HOOK_HARNESSES, type HookHarnessName } from "./harness/hook-lifecycle";
import { importLocalHistory } from "./core/history";
import { detectLlm, hasRustToolchain, hasUvx, type LlmChoice } from "./core/daemon";
import { readLegacyEndpoint } from "./core/legacy";
import { createInstallerUi, type SelectOption } from "./install-ui";

/**
 * Substring that identifies OUR entries in a host's config, so a re-install replaces them and
 * `uninstall` removes exactly what we added.
 *
 * It must appear in the package path under BOTH layouts: npm
 * (`node_modules/@vectorize-io/hindsight-coding-agents/...`) and a repo checkout
 * (`hindsight-integrations/coding-agents/...`). The old value was the full package name, which the
 * repo path stopped containing when the directory dropped its `hindsight-` prefix — silently
 * breaking dedupe-on-reinstall and uninstall for anyone running from a checkout.
 */
export const MARKER = "coding-agents";

export interface InstallCtx {
  home: string;
  pkgRoot: string; // package root (opencode plugin entry)
  dist: string; // built entry points
  /** Runs `claude mcp ...`; injectable for tests. Returns false when the CLI isn't usable. */
  claudeMcp?: (args: string[]) => boolean;
  /** Runs `qwen mcp ...`; injectable for tests. Returns false when the CLI isn't usable. */
  qwenMcp?: (args: string[]) => boolean;
  /** Runs `cline plugin ...`; injectable for tests. Returns false when the CLI isn't usable. */
  clinePlugin?: (args: string[]) => boolean;
  /** Runs native Dcode plugin commands; injectable for tests. Returns false when unusable. */
  dcodePlugin?: (args: string[]) => boolean;
  /** Reports whether `node:sqlite` works in the node that runs hooks; injectable for tests. */
  nodeSqlite?: () => boolean;
  /** Whether stdin can be prompted. Defaults to the real TTY check at the CLI entry; tests set it
   *  explicitly so the suite never blocks on a read. */
  interactive?: boolean;
  /** Daemon prerequisite probes; injectable for tests. */
  hasUvx?: () => boolean;
  detectLlm?: () => LlmChoice | undefined;
  hasRust?: () => boolean;
  /** Reads an old per-agent plugin's endpoint; injectable for tests. */
  readLegacy?: (home: string, prefer: readonly string[]) => ReturnType<typeof readLegacyEndpoint>;
  log?: (m: string) => void;
  /** Styles an interactive readLineSync prompt (the CLI passes the InstallerUi rail style). */
  promptStyle?: (q: string) => string;
  /** Arrow-key picker (the CLI passes InstallerUi.select). Chosen index; null = cancelled;
   *  undefined = raw TTY unavailable, fall back to the numeric prompt. */
  selectPrompt?: (
    question: string,
    options: SelectOption[],
    defaultIndex: number
  ) => number | null | undefined;
}

function readJson(path: string): Record<string, any> {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return {};
  }
}

/**
 * Parse a JSONC file (JSON with comments), as Kilo's `kilo.jsonc` and opencode's `opencode.jsonc`
 * may be. Applied to `opencode.json` too: comments and trailing commas turn up in the wild under
 * that name as well, and the host reads it either way — so what decides the parser here is the
 * content the installer might meet, not the extension.
 *
 * Returns null — NOT {} — when the file exists but can't be parsed. readJson's {} fallback is safe
 * for a strict-JSON host (an unparseable file is a broken file), but here a config we merely failed
 * to understand must never be overwritten with just our own key: the caller aborts instead.
 */
export function parseJsonc(text: string): Record<string, any> | null {
  const stripped = text
    // Blank out comments, preserving anything inside string literals.
    .replace(/"(?:\\.|[^"\\])*"|\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, (m) => (m[0] === '"' ? m : ""))
    // Trailing commas are legal in JSONC, not in JSON.
    .replace(/,(\s*[}\]])/g, "$1");
  try {
    const v = JSON.parse(stripped);
    return v && typeof v === "object" ? (v as Record<string, any>) : null;
  } catch {
    return null;
  }
}

function writeJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  backupOnce(path);
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
}

/** The first time we touch an existing file, keep a copy of what the user had. */
function backupOnce(path: string): void {
  if (existsSync(path) && !existsSync(`${path}.hindsight-backup`)) {
    copyFileSync(path, `${path}.hindsight-backup`);
  }
}

/**
 * Set (or, with `undefined`, delete) ONE top-level key, leaving the rest of the file's text alone.
 *
 * `writeJson` round-trips through `JSON.stringify`, which can only emit strict JSON — so writing a
 * JSONC host's config with it silently strips every comment and trailing comma the user wrote, and
 * reflows their formatting. Parsing the file was only half the problem: even a config we read
 * correctly came back stripped. `jsonc-parser` computes a minimal text edit instead, so everything
 * we did not touch survives byte for byte.
 *
 * Limited to a single key on purpose. A whole-object write is what forces a re-serialize, and both
 * callers only ever mutate `plugin`; a general "merge this object" helper could not preserve
 * anything. Comments INSIDE the edited value are still lost — the array is re-emitted — but the
 * rest of the document, which is all a user notices, is not.
 */
function writeJsonc(path: string, key: string, value: unknown): void {
  const text = existsSync(path) ? readFileSync(path, "utf8") : "";
  const edits = modify(text, [key], value, {
    formattingOptions: {
      tabSize: 2,
      insertSpaces: true,
      // Keep a CRLF file CRLF: rewriting every line ending would turn a one-line change into a
      // whole-file diff for anyone on Windows.
      eol: text.includes("\r\n") ? "\r\n" : "\n",
    },
  });
  const next = applyEdits(text, edits);
  mkdirSync(dirname(path), { recursive: true });
  backupOnce(path);
  writeFileSync(path, next.endsWith("\n") ? next : `${next}\n`);
}

/** Hook-array merge for claude/codex-style files: drop our old entries, append the new one. */
function mergeHookEvent(existing: any[] | undefined, entry: unknown): any[] {
  const kept = (existing ?? []).filter((e) => !JSON.stringify(e).includes(MARKER));
  return [...kept, entry];
}

function stripOurs(existing: any[] | undefined): any[] {
  return (existing ?? []).filter((e) => !JSON.stringify(e).includes(MARKER));
}

/** Remove the event key entirely when nothing else remains (leave the host file tidy). */
function setOrDelete(obj: Record<string, any>, key: string, arr: any[]): void {
  if (arr.length) obj[key] = arr;
  else delete obj[key];
}

const cmdHook = (dist: string, file: string, timeout: number) => ({
  hooks: [{ type: "command", command: `node "${join(dist, file)}"`, timeout }],
});

/**
 * The MCP registration for a JSON-configured host.
 *
 * Every harness launches the SAME `dist/mcp-server.js`, so the command line alone cannot say who
 * is calling. `HINDSIGHT_MCP_HARNESS` is what the server reads to answer that; without it it falls
 * back to "claude-code", which is why a Codex `hindsight_ingest_document` landed tagged
 * `harness:claude-code` (and derived its bank as Claude Code's) on machines running both. Every
 * registration MUST name its own harness — pass it here, never rely on the fallback.
 */
const mcpServerEntry = (dist: string, harness: HookHarnessName | "cline-cli") => ({
  command: "node",
  args: [join(dist, "mcp-server.js")],
  env: { HINDSIGHT_MCP_HARNESS: harness },
});

/** Install/uninstall consume the same lifecycle declaration as the runtime entrypoints. Keeping
 * event names and command files here would allow a host to run a different lifecycle than it installs. */
function mergeHarnessHooks(
  hooks: Record<string, any>,
  harness: HookHarnessName,
  dist: string
): void {
  const spec = HOOK_HARNESSES[harness];
  const installedEvents = new Set<string>();
  for (const hook of Object.values(spec.install)) {
    // Antigravity has no SessionStart event. Its first PreInvocation performs the same seed guard,
    // so both conceptual lifecycle names intentionally resolve to one native hook entry.
    if (installedEvents.has(hook.event)) continue;
    installedEvents.add(hook.event);
    const entry =
      spec.configStyle === "nested"
        ? cmdHook(dist, hook.entry, hook.timeout!)
        : {
            command: `node "${join(dist, hook.entry)}"`,
            ...(hook.timeout ? { timeout: hook.timeout } : {}),
          };
    hooks[hook.event] = mergeHookEvent(hooks[hook.event], entry);
  }
}

function stripHarnessHooks(hooks: Record<string, any>, harness: HookHarnessName): void {
  const strippedEvents = new Set<string>();
  for (const hook of Object.values(HOOK_HARNESSES[harness].install)) {
    if (strippedEvents.has(hook.event)) continue;
    strippedEvents.add(hook.event);
    setOrDelete(hooks, hook.event, stripOurs(hooks[hook.event]));
  }
}

/** Copy the packaged companion SKILL into a host's skills directory (idempotent overwrite).
 * The log line carries the harness prefix like every adapter message: several adapters install
 * the skill before their first own log, and an unprefixed line would render under the PREVIOUS
 * harness's group in the CLI output. */
function installSkill(c: InstallCtx, harness: string, skillsBase: string): void {
  const src = join(c.pkgRoot, "skill");
  if (!existsSync(join(src, "SKILL.md"))) return;
  const dst = join(skillsBase, "hindsight-coding-agent");
  mkdirSync(skillsBase, { recursive: true });
  cpSync(src, dst, { recursive: true });
  c.log?.(`${harness}: skill installed at ${dst}`);
}

function uninstallSkill(c: InstallCtx, skillsBase: string): void {
  rmSync(join(skillsBase, "hindsight-coding-agent"), { recursive: true, force: true });
}

// ── per-harness adapters ────────────────────────────────────────────────────────

export interface HarnessInstaller {
  name: string;
  detect(ctx: InstallCtx): boolean;
  /**
   * Blocking environment check, run before anything is written. Returns the reason when this
   * machine cannot support the harness, so a doomed setup fails at install time instead of
   * reporting success and then never retaining anything.
   */
  preflight?(ctx: InstallCtx): string | undefined;
  install(ctx: InstallCtx): void | boolean;
  uninstall(ctx: InstallCtx): void | boolean;
}

function onPath(bin: string): boolean {
  try {
    execFileSync("which", [bin], { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

function runClinePlugin(args: string[]): boolean {
  try {
    execFileSync("cline", args, { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

/**
 * opencode reads its global config from EITHER `opencode.json` or `opencode.jsonc` under
 * `~/.config/opencode`; both names are documented. So the installer must edit the one that is
 * already there: hardcoding `opencode.json` meant a user whose config is `opencode.jsonc` got a
 * SECOND config file created next to their real one — their settings in the file they wrote, our
 * plugin entry in a file they never made.
 *
 * `.jsonc` is probed first because that is the name that carries comments, and a machine holding
 * both is far likelier to have the commented one as the real config.
 */
export const OPENCODE_CONFIG_CANDIDATES = ["opencode.jsonc", "opencode.json"];

function opencodeConfigPath(c: InstallCtx): string {
  const dir = join(c.home, ".config", "opencode");
  const existing = OPENCODE_CONFIG_CANDIDATES.map((f) => join(dir, f)).find((p) => existsSync(p));
  return existing ?? join(dir, "opencode.json");
}

function runDcodePlugin(args: string[]): boolean {
  try {
    execFileSync("dcode", args, { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

/**
 * Register/unregister the package root in `~/.config/opencode`'s `plugin` array. Shared verbatim by
 * `opencode` and `opencode2`: the two CLIs read the SAME config file, and v1 REJECTS the whole file
 * on v2's `plugins` key ("Configuration is invalid … Unrecognized key: plugins"), so a v2-specific
 * entry is not an option. One entry serves both because they resolve a plugin directory
 * differently — v1 follows package.json `main` (dist/index.js), v2 loads `<dir>/index.js` — and v2
 * migrates the legacy `plugin` key to `plugins` for itself. See src/opencode2.ts.
 *
 * Consequences worth knowing: installing either harness wires the other one too (the MARKER filter
 * keeps that idempotent), and uninstalling either removes the shared entry.
 */
function registerOpencodePlugin(c: InstallCtx, name: string): void {
  const path = opencodeConfigPath(c);
  let cfg: Record<string, any> = {};
  if (existsSync(path)) {
    const parsed = parseJsonc(readFileSync(path, "utf8"));
    if (!parsed) {
      c.log?.(`${name}: SKIPPED — could not parse ${path}; add the plugin entry manually`);
      return;
    }
    cfg = parsed;
  }
  const plugins: unknown[] = Array.isArray(cfg.plugin) ? cfg.plugin : [];
  // writeJsonc, not writeJson: this config routinely carries comments, and re-serializing the
  // parsed object would strip every one of them even though the read succeeded.
  writeJsonc(path, "plugin", [...plugins.filter((p) => !String(p).includes(MARKER)), c.pkgRoot]);
  c.log?.(`${name}: plugin registered in ${path}`);
}

function unregisterOpencodePlugin(c: InstallCtx, name: string): void {
  const path = opencodeConfigPath(c);
  if (!existsSync(path)) return;
  const cfg = parseJsonc(readFileSync(path, "utf8"));
  if (!cfg || !Array.isArray(cfg.plugin)) return;
  const kept = cfg.plugin.filter((p: unknown) => !String(p).includes(MARKER));
  writeJsonc(path, "plugin", kept.length ? kept : undefined);
  c.log?.(`${name}: plugin entry removed`);
}

const opencode: HarnessInstaller = {
  name: "opencode",
  detect: (c) => onPath("opencode") || existsSync(join(c.home, ".config", "opencode")),
  install: (c) => registerOpencodePlugin(c, "opencode"),
  uninstall: (c) => unregisterOpencodePlugin(c, "opencode"),
};

/**
 * opencode v2 — the `opencode2` binary (npm `@opencode-ai/cli@beta`), which installs ALONGSIDE v1
 * rather than replacing it.
 *
 * Detection is the binary only, deliberately NOT `~/.config/opencode`: that directory is v1's too,
 * so keying on it would make `install` (which wires every detected agent) claim opencode2 on every
 * v1-only machine.
 */
const opencode2: HarnessInstaller = {
  name: "opencode2",
  detect: () => onPath("opencode2"),
  install: (c) => registerOpencodePlugin(c, "opencode2"),
  uninstall: (c) => unregisterOpencodePlugin(c, "opencode2"),
};

/**
 * Prime Agent (PrimeIntellect) — a persistent plugin loaded as an extension. Register the built
 * `dist/prime-agent.js` in the `extensions` array of `~/.prime/agent/settings.json`; Prime Agent
 * loads that file's default export at session start. The entry path contains MARKER (the package is
 * `hindsight-coding-agents`), so uninstall's MARKER filter removes exactly what install added.
 *
 * The skill goes to Prime Agent's OWN `~/.prime/agent/skills`, not the shared `~/.agents/skills`
 * root it also reads: `uninstallSkill` removes a fixed directory name, so installing to the shared
 * root would make `uninstall prime-agent` delete Codex's and dsh's copy too (#3772).
 */
const primeAgent: HarnessInstaller = {
  name: "prime-agent",
  detect: (c) => onPath("prime-agent") || existsSync(join(c.home, ".prime", "agent")),
  install(c) {
    const path = join(c.home, ".prime", "agent", "settings.json");
    const cfg = readJson(path);
    const entry = join(c.pkgRoot, "dist", "prime-agent.js");
    const exts: string[] = Array.isArray(cfg.extensions) ? cfg.extensions : [];
    cfg.extensions = [...exts.filter((p) => !String(p).includes(MARKER)), entry];
    writeJson(path, cfg);
    installSkill(c, "prime-agent", join(c.home, ".prime", "agent", "skills"));
    c.log?.(`prime-agent: extension registered in ${path}`);
  },
  uninstall(c) {
    const path = join(c.home, ".prime", "agent", "settings.json");
    if (existsSync(path)) {
      const cfg = readJson(path);
      if (Array.isArray(cfg.extensions)) {
        cfg.extensions = cfg.extensions.filter((p: string) => !String(p).includes(MARKER));
        if (!cfg.extensions.length) delete cfg.extensions;
        writeJson(path, cfg);
      }
    }
    // Outside the settings guard on purpose: a hand-deleted settings.json must not strand the skill.
    uninstallSkill(c, join(c.home, ".prime", "agent", "skills"));
    c.log?.("prime-agent: extension entry + skill removed");
  },
};

/**
 * Kilo Code CLI — an opencode fork, so registration is opencode's: append our entry to the config's
 * `plugin` array. Two Kilo-specific wrinkles:
 *
 *  - The config may be `kilo.jsonc` OR `kilo.json` (Kilo also honours legacy `opencode.json`). We
 *    edit whichever already exists and only create `kilo.json` when none does — writing a second
 *    file would leave the user with two configs and no obvious winner.
 *  - JSONC allows comments, which JSON.parse rejects. Falling back to `{}` there would overwrite a
 *    real config with just our plugin key, so an unparseable file aborts the install instead.
 *
 * We register `dist/kilo.js` explicitly rather than the package root: the root resolves via
 * package.json main to dist/index.js, which reports the harness as "opencode".
 *
 * The entry MUST be a `file://` URL. Unlike opencode — which accepts a bare filesystem path — Kilo
 * treats a bare absolute path as an npm module specifier, fails to resolve it, and SILENTLY skips
 * the plugin: the session starts normally with no memory and no error anywhere. Verified against
 * Kilo 7.4.17: the same entry loads as `file://…` and is ignored as `/…`.
 */
export const KILO_CONFIG_CANDIDATES = ["kilo.jsonc", "kilo.json", "opencode.json"];

function kiloConfigPath(c: InstallCtx): string {
  const dir = join(c.home, ".config", "kilo");
  const existing = KILO_CONFIG_CANDIDATES.map((f) => join(dir, f)).find((p) => existsSync(p));
  return existing ?? join(dir, "kilo.json");
}

const kilo: HarnessInstaller = {
  name: "kilo",
  detect: (c) => onPath("kilo") || existsSync(join(c.home, ".config", "kilo")),
  install(c) {
    const path = kiloConfigPath(c);
    let cfg: Record<string, any> = {};
    if (existsSync(path)) {
      const parsed = parseJsonc(readFileSync(path, "utf8"));
      if (!parsed) {
        c.log?.(`kilo: SKIPPED — could not parse ${path}; add the plugin entry manually`);
        return;
      }
      cfg = parsed;
    }
    const entry = pathToFileURL(join(c.dist, "kilo.js")).href;
    const plugins: unknown[] = Array.isArray(cfg.plugin) ? cfg.plugin : [];
    // writeJsonc for the same reason the read uses parseJsonc: kilo.jsonc is a commented file, and
    // re-serializing the parsed object would strip what we just took care to read.
    writeJsonc(path, "plugin", [...plugins.filter((p) => !String(p).includes(MARKER)), entry]);
    c.log?.(`kilo: plugin registered in ${path}`);
  },
  uninstall(c) {
    const path = kiloConfigPath(c);
    if (!existsSync(path)) return;
    const cfg = parseJsonc(readFileSync(path, "utf8"));
    if (!cfg || !Array.isArray(cfg.plugin)) return;
    const kept = cfg.plugin.filter((p: unknown) => !String(p).includes(MARKER));
    writeJsonc(path, "plugin", kept.length ? kept : undefined);
    c.log?.("kilo: plugin entry removed");
  },
};

const claudeCode: HarnessInstaller = {
  name: "claude-code",
  detect: (c) => onPath("claude") || existsSync(join(c.home, ".claude")),
  install(c) {
    const path = join(c.home, ".claude", "settings.json");
    const settings = readJson(path);
    settings.hooks = settings.hooks ?? {};
    mergeHarnessHooks(settings.hooks, "claude-code", c.dist);
    writeJson(path, settings);
    c.log?.(`claude-code: hooks merged into ${path}`);
    // Companion SKILL: every skills-capable host gets it (claude/antigravity/cursor native dirs;
    // codex via the ~/.agents/skills standard).
    installSkill(c, "claude-code", join(c.home, ".claude", "skills"));
    const mcp = c.claudeMcp ?? defaultClaudeMcp;
    // `claude mcp add` REFUSES when the name is taken ("MCP server hindsight already exists in
    // user config") — so on a machine that already had Hindsight, a re-install could never
    // repoint the server. After the package moved, the stale registration survived and Claude
    // Code reported "Failed to connect — Connection closed" with the hindsight_* tools dead.
    // Remove first (a no-op when absent) so the add always lands, matching how every other host
    // wiring is replaced rather than skipped.
    mcp(["mcp", "remove", "--scope", "user", "hindsight"]);
    if (
      mcp([
        "mcp",
        "add",
        "--scope",
        "user",
        "hindsight",
        "--env",
        "HINDSIGHT_MCP_HARNESS=claude-code",
        "--",
        "node",
        join(c.dist, "mcp-server.js"),
      ])
    ) {
      c.log?.("claude-code: MCP server registered (claude mcp add, user scope)");
    } else {
      c.log?.(
        `claude-code: could not run \`claude mcp add\` — register the tools manually:\n` +
          `  claude mcp add --scope user hindsight --env HINDSIGHT_MCP_HARNESS=claude-code -- node "${join(c.dist, "mcp-server.js")}"`
      );
    }
  },
  uninstall(c) {
    const path = join(c.home, ".claude", "settings.json");
    if (existsSync(path)) {
      const settings = readJson(path);
      if (settings.hooks) {
        stripHarnessHooks(settings.hooks, "claude-code");
        if (!Object.keys(settings.hooks).length) delete settings.hooks;
        writeJson(path, settings);
      }
    }
    const mcp = c.claudeMcp ?? defaultClaudeMcp;
    mcp(["mcp", "remove", "--scope", "user", "hindsight"]);
    uninstallSkill(c, join(c.home, ".claude", "skills"));
    c.log?.("claude-code: hooks + MCP registration + skill removed");
  },
};

function defaultClaudeMcp(args: string[]): boolean {
  try {
    execFileSync("claude", args, { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

/**
 * Our `[mcp_servers.hindsight]` table (plus any sub-table of it): the header line through to the
 * next table header or EOF. Shared by install — which REPLACES the block — and uninstall.
 */
const CODEX_MCP_BLOCK_RE = /^\[mcp_servers\.hindsight(?:\.[^\]]+)?\][^\n]*\n(?:(?!\[)[^\n]*\n?)*/gm;

/** Inline `env`, so CODEX_MCP_BLOCK_RE never has to straddle a `[mcp_servers.hindsight.env]` table. */
const codexMcpBlock = (dist: string) =>
  `[mcp_servers.hindsight]\ncommand = "node"\nargs = [${JSON.stringify(join(dist, "mcp-server.js"))}]\n` +
  `env = { HINDSIGHT_MCP_HARNESS = "codex" }`;

const codex: HarnessInstaller = {
  name: "codex",
  detect: (c) => onPath("codex") || existsSync(join(c.home, ".codex")),
  install(c) {
    const hooksPath = join(c.home, ".codex", "hooks.json");
    const cfg = readJson(hooksPath);
    cfg.hooks = cfg.hooks ?? {};
    mergeHarnessHooks(cfg.hooks, "codex", c.dist);
    writeJson(hooksPath, cfg);
    c.log?.(`codex: hooks merged into ${hooksPath}`);

    // config.toml: append-only for anything that is not ours (TOML round-tripping is not worth the
    // risk). Our OWN mcp block is stripped and rewritten instead of skipped when present: appending
    // only when absent made this install-once-only, so the harness-less registration that
    // attributed every Codex write to claude-code could never be repaired by re-running install.
    const tomlPath = join(c.home, ".codex", "config.toml");
    const existing = existsSync(tomlPath) ? readFileSync(tomlPath, "utf8") : "";
    const toml = existing.replace(CODEX_MCP_BLOCK_RE, "");
    const additions: string[] = [];
    // Codex ≥ 0.145 deprecates `codex_hooks` for `[features].hooks`; accept either as "already
    // enabled", write the modern name for new installs.
    if (!/^\s*(codex_hooks|hooks)\s*=/m.test(toml)) {
      if (/^\[features\]/m.test(toml)) {
        c.log?.(
          "codex: add `hooks = true` under your existing [features] section in ~/.codex/config.toml"
        );
      } else {
        additions.push("[features]\nhooks = true");
      }
    }
    additions.push(codexMcpBlock(c.dist));
    const next = `${toml.replace(/\n*$/, "\n\n")}${additions.join("\n\n")}\n`;
    if (next !== existing) {
      if (existsSync(tomlPath) && !existsSync(`${tomlPath}.hindsight-backup`)) {
        copyFileSync(tomlPath, `${tomlPath}.hindsight-backup`);
      }
      mkdirSync(dirname(tomlPath), { recursive: true });
      writeFileSync(tomlPath, next);
      c.log?.(`codex: wrote ${additions.length} section(s) to ${tomlPath}`);
    }
    installSkill(c, "codex", join(c.home, ".agents", "skills")); // agentskills-standard shared dir
  },
  uninstall(c) {
    const hooksPath = join(c.home, ".codex", "hooks.json");
    if (existsSync(hooksPath)) {
      const cfg = readJson(hooksPath);
      if (cfg.hooks) {
        stripHarnessHooks(cfg.hooks, "codex");
        writeJson(hooksPath, cfg);
      }
    }
    uninstallSkill(c, join(c.home, ".agents", "skills"));
    const tomlPath = join(c.home, ".codex", "config.toml");
    if (existsSync(tomlPath)) {
      const toml = readFileSync(tomlPath, "utf8");
      const cleaned = toml.replace(CODEX_MCP_BLOCK_RE, "");
      if (cleaned !== toml) writeFileSync(tomlPath, cleaned);
    }
    c.log?.(
      "codex: hooks + MCP section + skill removed ([features] hooks flag left as-is — other hooks may use it)"
    );
  },
};

const antigravity: HarnessInstaller = {
  name: "antigravity-cli",
  // `agy` is the supported Antigravity CLI executable.  Do not infer support from the
  // legacy Gemini CLI's ~/.gemini state: both clients may leave files there, but only agy
  // can consume this integration.
  detect: (c) => onPath("agy"),
  install(c) {
    const hooksPath = join(c.home, ".gemini", "config", "hooks.json");
    const hooks = readJson(hooksPath);
    // Antigravity's hooks.json is a map of named customizations, not a direct event map. Keep
    // Hindsight in its own namespace so it cannot collide with other global hook bundles.
    // Drop any namespace we wrote under a PREVIOUS marker value. This file is keyed BY the marker,
    // so unlike every other host — where entries are matched by substring — a changed marker
    // orphans the old namespace instead of replacing it, leaving both registered and every hook
    // firing twice. Match on our own name rather than an exact string so past and future renames
    // are both cleaned up.
    for (const key of Object.keys(hooks)) {
      if (key !== MARKER && key.includes(MARKER)) delete hooks[key];
    }
    hooks[MARKER] = hooks[MARKER] ?? {};
    mergeHarnessHooks(hooks[MARKER], "antigravity-cli", c.dist);
    // Clean up the short-lived root-level shape written by the first Antigravity adapter release.
    // stripOurs only removes commands bearing our marker, preserving a user's own root hooks.
    for (const event of ["PreInvocation", "Stop"]) {
      setOrDelete(hooks, event, stripOurs(hooks[event]));
    }
    writeJson(hooksPath, hooks);
    const mcpPath = join(c.home, ".gemini", "config", "mcp_config.json");
    const mcp = readJson(mcpPath);
    mcp.mcpServers = {
      ...(mcp.mcpServers ?? {}),
      hindsight: mcpServerEntry(c.dist, "antigravity-cli"),
    };
    writeJson(mcpPath, mcp);
    const settingsPath = join(c.home, ".gemini", "antigravity-cli", "settings.json");
    const settings = readJson(settingsPath);
    // A custom status-line command owns the whole rendered line. Do not overwrite a user's
    // formatter; Hindsight can only add its native indicator when no formatter is already set.
    if (!settings.statusLine) {
      settings.statusLine = {
        type: "command",
        command: `node "${join(c.dist, "antigravity-statusline.js")}"`,
      };
      writeJson(settingsPath, settings);
      c.log?.(`antigravity-cli: Hindsight status line enabled in ${settingsPath}`);
    } else if (JSON.stringify(settings.statusLine).includes(MARKER)) {
      settings.statusLine = {
        type: "command",
        command: `node "${join(c.dist, "antigravity-statusline.js")}"`,
      };
      writeJson(settingsPath, settings);
    } else {
      c.log?.(
        "antigravity-cli: existing custom status line preserved (Hindsight indicator not added)"
      );
    }
    c.log?.(`antigravity-cli: hooks merged into ${hooksPath}, MCP into ${mcpPath}`);
    installSkill(c, "antigravity-cli", join(c.home, ".gemini", "config", "skills"));
  },
  uninstall(c) {
    const hooksPath = join(c.home, ".gemini", "config", "hooks.json");
    if (existsSync(hooksPath)) {
      const hooks = readJson(hooksPath);
      if (hooks[MARKER]) {
        stripHarnessHooks(hooks[MARKER], "antigravity-cli");
        if (!Object.keys(hooks[MARKER]).length) delete hooks[MARKER];
      }
      // Also remove any root-level entries from the first adapter release.
      for (const event of ["PreInvocation", "Stop"]) {
        setOrDelete(hooks, event, stripOurs(hooks[event]));
      }
      writeJson(hooksPath, hooks);
    }
    const mcpPath = join(c.home, ".gemini", "config", "mcp_config.json");
    if (existsSync(mcpPath)) {
      const mcp = readJson(mcpPath);
      if (mcp.mcpServers?.hindsight) {
        delete mcp.mcpServers.hindsight;
        writeJson(mcpPath, mcp);
      }
    }
    const settingsPath = join(c.home, ".gemini", "antigravity-cli", "settings.json");
    if (existsSync(settingsPath)) {
      const settings = readJson(settingsPath);
      if (JSON.stringify(settings.statusLine ?? {}).includes(MARKER)) {
        delete settings.statusLine;
        writeJson(settingsPath, settings);
      }
    }
    uninstallSkill(c, join(c.home, ".gemini", "config", "skills"));
    c.log?.("antigravity-cli: hooks + MCP entry + status line + skill removed");
  },
};

/**
 * Probe the `node` on PATH — not this process — because that is the interpreter the installed hook
 * command (`node "<dist>/devin-stop-hook.js"`) will actually run under. An installer started through
 * npx, a version manager or a wrapper script is easily a different build than the one the agent
 * later uses. `-e` runs as CommonJS regardless of the surrounding package type, so `require` here
 * is safe.
 */
function pathNodeHasSqlite(): boolean {
  try {
    execFileSync("node", ["-e", "require('node:sqlite')"], { stdio: "pipe", timeout: 10_000 });
    return true;
  } catch {
    return false;
  }
}

function pathNodeVersion(): string {
  try {
    return execFileSync("node", ["-v"], { encoding: "utf8", stdio: "pipe" }).trim();
  } catch {
    return "not found";
  }
}

// ── runtime staging ─────────────────────────────────────────────────────────────

/**
 * Where the runtime is copied to, and therefore what every hook command points at.
 *
 * Under ~/.hindsight, which this package already owns (the config file lives there), and named
 * `coding-agents` on purpose: MARKER matching is what lets a re-install replace our entries and
 * `uninstall` remove them, and it looks for exactly that substring in the command path.
 */
export function runtimeDir(home: string): string {
  return join(home, ".hindsight", "coding-agents");
}

/** Names the directory `stageRuntime` copied from — read by core/auto-update.ts. */
export const ORIGIN_FILE = ".install-origin.json";

/**
 * Copy the runtime out of wherever this was executed from and into a stable location, then point
 * the wiring at the copy.
 *
 * Both fields matter: `dist` is baked into every hook command and MCP registration, and `pkgRoot`
 * is what opencode and Kilo load as a plugin directory. Repointing them here means no per-harness
 * installer needs to know staging happened.
 *
 * Copying is skipped when there is nothing to copy — running from a checkout whose dist has not
 * been built, and in tests — so the wiring falls back to the source path rather than a directory
 * that does not exist. It is also skipped when already running from the staged copy, which is what
 * makes re-running `install` cheap.
 */
function stageRuntime(c: InstallCtx): InstallCtx {
  const target = runtimeDir(c.home);
  // Compared through realpath: re-running the STAGED installer must not reach the copy below, which
  // deletes the very dist it is executing from. A symlinked or differently-spelled path to the same
  // directory would slip past a string compare.
  const same = (a: string, b: string): boolean => {
    try {
      return realpathSync(a) === realpathSync(b);
    } catch {
      return a === b;
    }
  };
  if (same(c.pkgRoot, target)) return c;
  if (!existsSync(join(c.dist, "installer.js"))) return c;
  try {
    // Replaced wholesale rather than merged: a stale entry point left behind by an older version
    // would still be reachable from a host config that references it by name.
    rmSync(join(target, "dist"), { recursive: true, force: true });
    mkdirSync(target, { recursive: true });
    cpSync(c.dist, join(target, "dist"), { recursive: true });
    const skill = join(c.pkgRoot, "skill");
    if (existsSync(skill)) cpSync(skill, join(target, "skill"), { recursive: true });
    const pkgJson = join(c.pkgRoot, "package.json");
    if (existsSync(pkgJson)) copyFileSync(pkgJson, join(target, "package.json"));
    // The Dcode Agent Plugin surface (its manifest and the hooks that manifest points at) plus the
    // root index.js opencode2 loads a plugin directory from — see src/opencode2.ts. Without that
    // file here, the staged copy is a v1-only plugin and opencode2 logs "configured plugin
    // directory has no index entrypoint".
    // (All are listed in package.json#files, so they exist in a published install too.)
    for (const resource of ["plugin.json", "hooks", "index.js"]) {
      const source = join(c.pkgRoot, resource);
      if (existsSync(source)) cpSync(source, join(target, resource), { recursive: true });
    }
    // Record WHERE this copy came from, LAST — the marker means "a complete copy landed here".
    // core/auto-update.ts will only replace a runtime it can prove was downloaded by npx: a copy
    // staged from a global `npm i -g`, a project dependency or a local checkout belongs to whoever
    // manages that source, and silently overwriting it would either fight their package manager
    // (leaving `npm ls -g` reporting a version that is no longer what runs) or destroy a
    // developer's locally-built dist.
    writeFileSync(
      join(target, ORIGIN_FILE),
      JSON.stringify({ source: c.pkgRoot, stagedAt: new Date().toISOString() })
    );
    c.log?.(`runtime staged at ${target}`);
    return { ...c, pkgRoot: target, dist: join(target, "dist") };
  } catch (error) {
    // A failed copy must not wire hooks at a half-written directory.
    c.log?.(`could not stage the runtime at ${target}: ${String(error)}`);
    return c;
  }
}

// ── server setup (cloud / self-hosted / local daemon) ───────────────────────────

export type ServerMode = "cloud" | "self-hosted" | "daemon";

const SERVER_MODES: ServerMode[] = ["cloud", "self-hosted", "daemon"];

/** Value of `--name value` or `--name=value`. */
export function flagValue(args: string[], name: string): string | undefined {
  const inline = args.find((a) => a.startsWith(`--${name}=`));
  if (inline) return inline.slice(name.length + 3);
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Args that are VALUES of a preceding flag, so they aren't mistaken for harness names. */
export function flagValueArgs(args: string[], names: string[]): Set<string> {
  const taken = new Set<string>();
  for (const name of names) {
    const i = args.indexOf(`--${name}`);
    if (i >= 0 && args[i + 1] && !args[i + 1].startsWith("--")) taken.add(args[i + 1]);
  }
  return taken;
}

const CONFIG_RELATIVE = [".hindsight", "coding-agent.json"];

/**
 * Ask which of the three connection modes to use, once.
 *
 * Only ever asked on a TTY and only when the config doesn't already answer it: `install` is
 * idempotent and routinely re-run (after an upgrade, or to add another agent), and re-prompting
 * then would be noise — worse, it would silently rewrite a working setup in CI, where there is no
 * one to answer. Non-interactive callers pass `--server`.
 */
const SERVER_CHOICES: { mode: ServerMode; label: string; hint: string }[] = [
  { mode: "cloud", label: "Hindsight Cloud", hint: "hosted, needs an API token" },
  { mode: "self-hosted", label: "Self-hosted server", hint: "a Hindsight server you already run" },
  {
    mode: "daemon",
    label: "Local daemon (on-device)",
    hint: "runs hindsight-embed here; no account, needs uv + an LLM key",
  },
];

function promptServerMode(c: InstallCtx): ServerMode | undefined {
  // Preferred UX: arrow-key picker (digits still submit directly). It reports undefined when a
  // raw TTY isn't available (no stty, exotic shell) — then the plain numbered menu below still
  // works everywhere a line can be read.
  if (c.selectPrompt) {
    const picked = c.selectPrompt(
      "Where should memory live?",
      SERVER_CHOICES.map(({ label, hint }) => ({ label, hint })),
      0
    );
    if (picked === null) {
      c.log?.("no server chosen — leaving the server config unchanged");
      return undefined;
    }
    if (picked !== undefined) return SERVER_CHOICES[picked].mode;
  }
  c.log?.(
    `\nWhere should memory live?\n` +
      SERVER_CHOICES.map((o, i) => `  ${i + 1}) ${o.label.padEnd(28)}— ${o.hint}`).join("\n") +
      `\n`
  );
  const answer = readLineSync(c, "Choose [1-3] (default 1): ").trim();
  if (answer === "") return "cloud";
  const digit = Number.parseInt(answer, 10);
  if (digit >= 1 && digit <= SERVER_CHOICES.length) return SERVER_CHOICES[digit - 1].mode;
  c.log?.(`unrecognised choice "${answer}" — leaving the server config unchanged`);
  return undefined;
}

/**
 * Read one line from stdin synchronously.
 *
 * `run()` is synchronous and called that way from both the CLI entry and the test suite, so an
 * async readline would mean making the whole installer async. readSync on the TTY blocks until
 * Enter, which is exactly the semantics wanted here.
 */
function readLineSync(c: InstallCtx, prompt: string): string {
  process.stdout.write((c.promptStyle ?? ((s: string) => s))(prompt));
  const buf = Buffer.alloc(1024);
  // Touching process.stdin ANYWHERE flips a TTY fd 0 to non-blocking (its getter initializes the
  // TTY stream), after which readSync throws EAGAIN instead of waiting for Enter. The old
  // blanket catch treated that as "no stdin" — so the picker printed its menu and every answer
  // silently became the default. The CLI entry now probes the TTY with tty.isatty (no stream
  // init), and EAGAIN here waits for input rather than fabricating an empty answer.
  for (;;) {
    try {
      const n = readSync(0, buf, 0, buf.length, null);
      return buf.subarray(0, n).toString("utf8");
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== "EAGAIN") {
        return ""; // genuinely no readable stdin — treated as the default
      }
      // Sleep 50ms without spinning; Atomics.wait is allowed on Node's main thread.
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50);
    }
  }
}

/**
 * Resolve and persist the connection mode into ~/.hindsight/coding-agent.json.
 *
 * Returns false to abort the install. Everything else here is advisory: a daemon whose
 * prerequisites are missing is still worth configuring, because `uv` or an API key can be
 * installed right after — unlike the harness preflights, which gate wiring that could never work.
 */
function configureServer(c: InstallCtx, args: string[], installing: readonly string[]): boolean {
  const explicit = flagValue(args, "server");
  if (explicit && !SERVER_MODES.includes(explicit as ServerMode)) {
    c.log?.(`unknown --server "${explicit}" — expected one of: ${SERVER_MODES.join(", ")}`);
    return false;
  }
  // The runtime resolves its config as HINDSIGHT_CONFIG || ~/.hindsight/coding-agent.json
  // (core/config.ts CONFIG_PATH). The wizard must honor the same override, or a user with that
  // var set would be configured into a file their sessions never read.
  const configPath = process.env.HINDSIGHT_CONFIG || join(c.home, ...CONFIG_RELATIVE);
  const existing = readJson(configPath);
  const alreadyConfigured = !!(existing.serverMode || existing.apiUrl);

  let mode = explicit as ServerMode | undefined;
  if (!mode) {
    if (alreadyConfigured) return true; // respect what's already there
    // Someone coming from the old per-agent plugin already chose where their memory lives.
    // Adopt it rather than asking again — and above all rather than defaulting to Cloud, which
    // would quietly redirect their prompts and transcripts to a different server.
    const legacy = (c.readLegacy ?? readLegacyEndpoint)(c.home, installing);
    if (legacy) {
      const carried: Record<string, unknown> = { ...existing, serverMode: legacy.serverMode };
      if (legacy.apiUrl) carried.apiUrl = legacy.apiUrl;
      if (legacy.apiToken) carried.apiToken = legacy.apiToken;
      if (legacy.apiPort) carried.apiPort = legacy.apiPort;
      writeJson(configPath, carried);
      c.log?.(
        `server: ${legacy.serverMode}${legacy.apiUrl ? ` (${legacy.apiUrl})` : ""} — carried over ` +
          `from the ${legacy.harness} plugin (${legacy.source})\n` +
          `        Only the endpoint moves; conversations do not. To bring this repo's history\n` +
          `        across, re-run here with --import-conversations.`
      );
      if (legacy.serverMode === "daemon") reportDaemonPrereqs(c);
      return true;
    }
    if (!c.interactive) {
      c.log?.(
        `\nserver: defaulting to Hindsight Cloud. Re-run with --server self-hosted|daemon to change,\n` +
          `        or edit ${configPath}.`
      );
      return true;
    }
    mode = promptServerMode(c);
    if (!mode) return true;
  }

  const next: Record<string, unknown> = { ...existing, serverMode: mode };
  if (mode === "cloud") {
    delete next.apiUrl; // fall back to the built-in Cloud URL rather than pinning a stale one
    // Cloud always authenticates — a config without a token only surfaces later as 401s on the
    // first session, so the token is REQUIRED here (it used to be "blank to set later").
    const token = flagValue(args, "api-token") ?? (c.interactive ? askToken(c) : undefined);
    if (!token) {
      c.log?.(
        `❌ Hindsight Cloud needs an API token — pass --api-token <token> (or set apiToken in ${configPath}).`
      );
      return false;
    }
    next.apiToken = token;
  } else if (mode === "self-hosted") {
    const url =
      flagValue(args, "api-url") ??
      (c.interactive
        ? readLineSync(c, "Server URL (e.g. http://localhost:8888): ").trim()
        : undefined);
    if (!url) {
      c.log?.(
        `❌ self-hosted mode needs a server URL — pass --api-url <url> (or set apiUrl in ${configPath}).`
      );
      return false;
    }
    next.apiUrl = url;
    const token = flagValue(args, "api-token");
    if (token) next.apiToken = token;
  } else {
    delete next.apiUrl; // daemon mode derives its URL from apiPort
    reportDaemonPrereqs(c);
  }

  writeJson(configPath, next);
  c.log?.(`server: ${mode} (${configPath})`);
  return true;
}

/** Cloud tokens are mandatory: re-ask a couple of times rather than writing a config that 401s
 *  on the first session. Three blank answers mean the user doesn't have one at hand — give up
 *  and let the cloud branch abort with the actionable message. */
function askToken(c: InstallCtx): string | undefined {
  for (let attempt = 0; attempt < 3; attempt++) {
    const token = readLineSync(c, "API token (required for Hindsight Cloud): ").trim();
    if (token) return token;
    c.log?.("  a token is required — find yours in the Hindsight Cloud dashboard");
  }
  return undefined;
}

/**
 * Daemon mode has two prerequisites the plugin can't supply. Report both up front rather than
 * letting the first session fail quietly with nothing but a diagnostic line.
 */
function reportDaemonPrereqs(c: InstallCtx): void {
  if (!(c.hasUvx ?? hasUvx)()) {
    c.log?.(
      `⚠️  \`uv\` is not on PATH. The daemon is fetched and run with it, so memory stays inert\n` +
        `    until you install it: https://docs.astral.sh/uv/`
    );
  }
  if (!(c.hasRust ?? hasRustToolchain)()) {
    c.log?.(
      `⚠️  macOS needs a current Rust toolchain to build the daemon's dependencies (litellm\n` +
        `    publishes no macOS wheel). Install from https://rustup.rs, then\n` +
        `    \`rustup default stable && rustup update\` — an OUT-OF-DATE toolchain fails too.`
    );
  }
  const llm = (c.detectLlm ?? detectLlm)();
  if (llm) {
    c.log?.(`   local extraction will use ${llm.provider} (from ${llm.source})`);
  } else {
    c.log?.(
      `⚠️  No LLM available for local fact extraction. Set OPENAI_API_KEY, ANTHROPIC_API_KEY or\n` +
        `    GEMINI_API_KEY (or install the Claude Code CLI, which needs no key).`
    );
  }
}

const devin: HarnessInstaller = {
  name: "devin-cli",
  detect: (c) => onPath("devin") || existsSync(join(c.home, ".config", "devin")),
  // Devin is the ONLY harness whose hooks never hand over a transcript: they carry a session id and
  // nothing else, so retain can only work by reading the CLI's own sessions.db. That makes SQLite
  // support a hard prerequisite here — and one worth checking now, because the alternative is an
  // install that looks perfectly healthy and stores nothing, forever (#3125).
  preflight(c) {
    if ((c.nodeSqlite ?? pathNodeHasSqlite)()) return undefined;
    return (
      `\`node:sqlite\` is unavailable in the \`node\` on PATH (${pathNodeVersion()}).\n` +
      `   Devin keeps its conversations in ~/.local/share/devin/cli/sessions.db and its hooks pass\n` +
      `   only a session id, so without SQLite nothing could ever be retained.\n` +
      `   Upgrade to Node 22.5 or newer (24 LTS recommended) and re-run this command.`
    );
  },
  install(c) {
    const configPath = join(c.home, ".config", "devin", "config.json");
    const config = readJson(configPath);
    config.hooks = config.hooks ?? {};
    mergeHarnessHooks(config.hooks, "devin-cli", c.dist);
    writeJson(configPath, config);
    const mcpPath = join(c.home, ".config", "devin", "mcp_config.json");
    const mcp = readJson(mcpPath);
    mcp.mcpServers = {
      ...(mcp.mcpServers ?? {}),
      hindsight: mcpServerEntry(c.dist, "devin-cli"),
    };
    writeJson(mcpPath, mcp);
    c.log?.(`devin-cli: hooks merged into ${configPath}, MCP into ${mcpPath}`);
  },
  uninstall(c) {
    const configPath = join(c.home, ".config", "devin", "config.json");
    if (existsSync(configPath)) {
      const config = readJson(configPath);
      if (config.hooks) {
        stripHarnessHooks(config.hooks, "devin-cli");
        if (!Object.keys(config.hooks).length) delete config.hooks;
        writeJson(configPath, config);
      }
    }
    const mcpPath = join(c.home, ".config", "devin", "mcp_config.json");
    if (existsSync(mcpPath)) {
      const mcp = readJson(mcpPath);
      if (mcp.mcpServers?.hindsight) {
        delete mcp.mcpServers.hindsight;
        writeJson(mcpPath, mcp);
      }
    }
    c.log?.("devin-cli: hooks + MCP entry removed");
  },
};

const cursor: HarnessInstaller = {
  name: "cursor-cli",
  detect: (c) => onPath("cursor-agent") || existsSync(join(c.home, ".cursor")),
  install(c) {
    const hooksPath = join(c.home, ".cursor", "hooks.json");
    const cfg = readJson(hooksPath);
    cfg.hooks = cfg.hooks ?? {};
    mergeHarnessHooks(cfg.hooks, "cursor-cli", c.dist);
    writeJson(hooksPath, cfg);
    const mcpPath = join(c.home, ".cursor", "mcp.json");
    const mcp = readJson(mcpPath);
    mcp.mcpServers = {
      ...(mcp.mcpServers ?? {}),
      hindsight: mcpServerEntry(c.dist, "cursor-cli"),
    };
    writeJson(mcpPath, mcp);
    c.log?.(`cursor-cli: hooks merged into ${hooksPath}, MCP into ${mcpPath}`);
    installSkill(c, "cursor-cli", join(c.home, ".cursor", "skills"));
  },
  uninstall(c) {
    const hooksPath = join(c.home, ".cursor", "hooks.json");
    if (existsSync(hooksPath)) {
      const cfg = readJson(hooksPath);
      if (cfg.hooks) {
        stripHarnessHooks(cfg.hooks, "cursor-cli");
        if (!Object.keys(cfg.hooks).length) delete cfg.hooks;
        writeJson(hooksPath, cfg);
      }
    }
    const mcpPath = join(c.home, ".cursor", "mcp.json");
    if (existsSync(mcpPath)) {
      const mcp = readJson(mcpPath);
      if (mcp.mcpServers?.hindsight) {
        delete mcp.mcpServers.hindsight;
        writeJson(mcpPath, mcp);
      }
    }
    uninstallSkill(c, join(c.home, ".cursor", "skills"));
    c.log?.("cursor-cli: hooks + MCP entry + skill removed");
  },
};

const copilot: HarnessInstaller = {
  name: "copilot-cli",
  detect: (c) => onPath("copilot") || existsSync(join(c.home, ".copilot")),
  install(c) {
    const hooksPath = join(c.home, ".copilot", "hooks", "hindsight-coding-agents.json");
    const hooks: Record<string, any> = {};
    mergeHarnessHooks(hooks, "copilot-cli", c.dist);
    writeJson(hooksPath, { version: 1, hooks });

    const mcpPath = join(c.home, ".copilot", "mcp-config.json");
    const mcp = readJson(mcpPath);
    mcp.mcpServers = {
      ...(mcp.mcpServers ?? {}),
      hindsight: mcpServerEntry(c.dist, "copilot-cli"),
    };
    writeJson(mcpPath, mcp);
    installSkill(c, "copilot-cli", join(c.home, ".copilot", "skills"));
    c.log?.(`copilot-cli: hooks installed at ${hooksPath}, MCP into ${mcpPath}`);
  },
  uninstall(c) {
    rmSync(join(c.home, ".copilot", "hooks", "hindsight-coding-agents.json"), {
      force: true,
    });
    const mcpPath = join(c.home, ".copilot", "mcp-config.json");
    if (existsSync(mcpPath)) {
      const mcp = readJson(mcpPath);
      if (mcp.mcpServers?.hindsight) {
        delete mcp.mcpServers.hindsight;
        writeJson(mcpPath, mcp);
      }
    }
    uninstallSkill(c, join(c.home, ".copilot", "skills"));
    c.log?.("copilot-cli: hooks + MCP entry + skill removed");
  },
};

const GROK_MARKER_START = "# HINDSIGHT_CODING_AGENTS_GROK_START";
const GROK_MARKER_END = "# HINDSIGHT_CODING_AGENTS_GROK_END";
/** Our sentinel-delimited block; shared by install (replace) and uninstall (strip). */
const GROK_BLOCK_RE = new RegExp(`\\n?${GROK_MARKER_START}[\\s\\S]*?${GROK_MARKER_END}\\n?`);

const grok: HarnessInstaller = {
  name: "grok-build",
  detect: (c) => onPath("grok") || existsSync(join(c.home, ".grok")),
  install(c) {
    const path = join(c.home, ".grok", "config.toml");
    const existing = existsSync(path) ? readFileSync(path, "utf8") : "";
    // REPLACE any previous block rather than skipping when one exists. Skipping made this
    // install-once-only: after the package moved, a re-install silently left the old (now dead)
    // paths in place, which is exactly the case `install` is meant to repair.
    const withoutOurs = existing.replace(GROK_BLOCK_RE, "\n");
    // Grok executes this shell command verbatim. Quote the absolute script path so a globally
    // installed package still works when its installation directory contains spaces.
    const command = (entry: string) => JSON.stringify(`node "${join(c.dist, entry)}"`);
    const tomlString = (value: string) => JSON.stringify(value);
    const block =
      `\n${GROK_MARKER_START}\n` +
      `[[hooks.SessionStart]]\n  [[hooks.SessionStart.hooks]]\n  type = \"command\"\n  command = ${command("grok-sessionstart-hook.js")}\n  timeout = 30\n\n` +
      `[[hooks.UserPromptSubmit]]\n  [[hooks.UserPromptSubmit.hooks]]\n  type = \"command\"\n  command = ${command("grok-hook.js")}\n  timeout = 30\n\n` +
      `[[hooks.Stop]]\n  [[hooks.Stop.hooks]]\n  type = \"command\"\n  command = ${command("grok-stop-hook.js")}\n  timeout = 60\n\n` +
      `[mcp_servers.hindsight]\ncommand = \"node\"\nargs = [${tomlString(join(c.dist, "mcp-server.js"))}]\n` +
      `env = { HINDSIGHT_MCP_HARNESS = \"grok-build\" }\n${GROK_MARKER_END}\n`;
    if (existsSync(path) && !existsSync(`${path}.hindsight-backup`))
      copyFileSync(path, `${path}.hindsight-backup`);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${withoutOurs.replace(/\n*$/, "\n")}${block}`);
    installSkill(c, "grok-build", join(c.home, ".grok", "skills"));
    c.log?.(`grok-build: native hooks + MCP installed in ${path}`);
  },
  uninstall(c) {
    const path = join(c.home, ".grok", "config.toml");
    if (existsSync(path)) {
      const existing = readFileSync(path, "utf8");
      const cleaned = existing.replace(GROK_BLOCK_RE, "\n");
      if (cleaned !== existing) writeFileSync(path, cleaned);
    }
    uninstallSkill(c, join(c.home, ".grok", "skills"));
    c.log?.("grok-build: native hooks + MCP + skill removed");
  },
};

const CLINE_HOOK_MARKER = "HINDSIGHT_CODING_AGENTS_CLINE";
const CLINE_OLD_HOOK_EVENTS = ["TaskStart", "UserPromptSubmit", "TaskComplete"];
const CLINE_PLUGIN_NAME = "@vectorize-io/hindsight-coding-agents";

/** Remove only wrappers from the short-lived file-hook implementation. Cline file hooks cannot
 * mutate a prompt; keeping them would falsely imply that old installs still inject memory. */
function removeLegacyClineHooks(c: InstallCtx): void {
  const hooksDir = join(c.home, "Documents", "Cline", "Hooks");
  for (const event of CLINE_OLD_HOOK_EVENTS) {
    const path = join(hooksDir, event);
    if (existsSync(path) && readFileSync(path, "utf8").includes(CLINE_HOOK_MARKER)) {
      rmSync(path, { force: true });
    }
  }
}

const cline: HarnessInstaller = {
  name: "cline-cli",
  detect: (c) => onPath("cline") || existsSync(join(c.home, ".cline")),
  install(c) {
    removeLegacyClineHooks(c);
    const installed = (c.clinePlugin ?? runClinePlugin)([
      "plugin",
      "install",
      "--force",
      c.pkgRoot,
    ]);
    const mcpPath = join(c.home, ".cline", "data", "settings", "cline_mcp_settings.json");
    const mcp = readJson(mcpPath);
    mcp.mcpServers = {
      ...(mcp.mcpServers ?? {}),
      hindsight: mcpServerEntry(c.dist, "cline-cli"),
    };
    writeJson(mcpPath, mcp);
    installSkill(c, "cline-cli", join(c.home, ".cline", "data", "settings", "skills"));
    c.log?.(
      installed
        ? "cline-cli: native plugin + MCP + skill installed"
        : `cline-cli: MCP + skill installed; run: cline plugin install --force "${c.pkgRoot}"`
    );
  },
  uninstall(c) {
    removeLegacyClineHooks(c);
    // Cline's uninstall command receives an installed plugin name, not the original local source
    // path. The package name is stable across global npm updates, unlike the package directory.
    (c.clinePlugin ?? runClinePlugin)(["plugin", "uninstall", CLINE_PLUGIN_NAME]);
    const mcpPath = join(c.home, ".cline", "data", "settings", "cline_mcp_settings.json");
    if (existsSync(mcpPath)) {
      const mcp = readJson(mcpPath);
      if (mcp.mcpServers?.hindsight) {
        delete mcp.mcpServers.hindsight;
        writeJson(mcpPath, mcp);
      }
    }
    uninstallSkill(c, join(c.home, ".cline", "data", "settings", "skills"));
    c.log?.("cline-cli: native plugin + MCP + skill removed");
  },
};

/** Dcode's plugin manager owns its cache and enablement state. Keep this installer as a thin
 * native CLI invocation: the package's bundled marketplace points back at the staged package,
 * while Dcode performs the copy, validation, enablement, and foreign-state preservation. */
const DCODE_MARKETPLACE = "hindsight-coding-agents";
const DCODE_PLUGIN_ID = "hindsight-coding-agents@hindsight-coding-agents";
const DCODE_MARKETPLACE_RELATIVE_PATH = join(".agents", "plugins", "marketplace.json");
const DCODE_FALLBACK_MARKETPLACE = "hindsight-coding-agents-marketplace.json";

/**
 * Prepare a local marketplace without taking ownership of a foreign manifest at the conventional
 * path. A custom JSON file still has the same ~/.hindsight root, so ./coding-agents remains valid,
 * while Dcode records it as its own marketplace and the foreign top-level name is untouched.
 */
function prepareDcodeMarketplace(c: InstallCtx): string | false {
  const root = join(c.home, ".hindsight");
  const conventionalPath = join(root, DCODE_MARKETPLACE_RELATIVE_PATH);
  let path = conventionalPath;
  let registrationSource = root;
  let marketplace: Record<string, any> = { plugins: [] };
  if (existsSync(conventionalPath)) {
    try {
      const parsed = JSON.parse(readFileSync(conventionalPath, "utf8"));
      if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.plugins)) {
        path = join(root, DCODE_FALLBACK_MARKETPLACE);
      } else if (
        typeof (parsed as Record<string, unknown>).name === "string" &&
        parsed.name !== DCODE_MARKETPLACE
      ) {
        // Do not mutate the identity of a marketplace that another Dcode plugin may already use.
        path = join(root, DCODE_FALLBACK_MARKETPLACE);
      } else {
        marketplace = parsed;
      }
    } catch {
      path = join(root, DCODE_FALLBACK_MARKETPLACE);
    }
  }
  if (path !== conventionalPath && existsSync(path)) {
    try {
      const parsed = JSON.parse(readFileSync(path, "utf8"));
      if (
        !parsed ||
        typeof parsed !== "object" ||
        !Array.isArray(parsed.plugins) ||
        parsed.name !== DCODE_MARKETPLACE
      ) {
        c.log?.(`dcode: refusing to replace malformed marketplace file ${path}`);
        return false;
      }
      marketplace = parsed;
    } catch {
      c.log?.(`dcode: refusing to replace unreadable marketplace file ${path}`);
      return false;
    }
  }
  if (path !== conventionalPath) registrationSource = path;
  const foreign = marketplace.plugins.filter(
    (entry: unknown) =>
      !entry ||
      typeof entry !== "object" ||
      (entry as Record<string, unknown>).name !== "hindsight-coding-agents"
  );
  marketplace.name = DCODE_MARKETPLACE;
  marketplace.plugins = [
    ...foreign,
    {
      name: "hindsight-coding-agents",
      source: { source: "local", path: "./coding-agents" },
    },
  ];
  writeJson(path, marketplace);
  return registrationSource;
}

const dcode: HarnessInstaller = {
  name: "dcode",
  detect: (c) => onPath("dcode") || existsSync(join(c.home, ".deepagents")),
  preflight: (c) =>
    c.dcodePlugin || onPath("dcode") ? undefined : "the `dcode` CLI is not on PATH",
  install(c) {
    const plugin = c.dcodePlugin ?? runDcodePlugin;
    const marketplacePath = prepareDcodeMarketplace(c);
    const installed =
      marketplacePath !== false &&
      plugin(["plugin", "marketplace", "add", marketplacePath]) &&
      plugin(["plugin", "install", DCODE_PLUGIN_ID]);
    if (installed) {
      c.log?.(`dcode: native Agent Plugin installed (${DCODE_PLUGIN_ID})`);
    } else {
      const source = marketplacePath === false ? join(c.home, ".hindsight") : marketplacePath;
      c.log?.(
        `dcode: could not install the native plugin — run \
` + `  dcode plugin marketplace add "${source}" && dcode plugin install ${DCODE_PLUGIN_ID}`
      );
    }
    return installed;
  },
  uninstall(c) {
    const plugin = c.dcodePlugin ?? runDcodePlugin;
    const removed = plugin(["plugin", "uninstall", DCODE_PLUGIN_ID]);
    // Also retire the marketplace WE registered. `plugin uninstall` alone leaves it listed by
    // `dcode plugin marketplace list` and the plugin itself listed as `disabled` — our own
    // leftovers in Dcode's state, not foreign state we are obliged to preserve. Only ours is
    // named DCODE_MARKETPLACE, so a foreign marketplace at the conventional path is untouched.
    const marketplaceRemoved = plugin(["plugin", "marketplace", "remove", DCODE_MARKETPLACE]);
    if (removed) {
      c.log?.("dcode: native Agent Plugin removed (foreign ~/.deepagents state preserved)");
      if (!marketplaceRemoved) {
        c.log?.(
          `dcode: the ${DCODE_MARKETPLACE} marketplace is still registered — remove it with \`dcode plugin marketplace remove ${DCODE_MARKETPLACE}\``
        );
      }
    } else {
      c.log?.(`dcode: could not run native uninstall for ${DCODE_PLUGIN_ID}`);
    }
    return removed;
  },
};

const DSH_MARKER_START = "# HINDSIGHT_CODING_AGENTS_DSH_START";
const DSH_MARKER_END = "# HINDSIGHT_CODING_AGENTS_DSH_END";
const DSH_BLOCK_RE = new RegExp(`\\n?${DSH_MARKER_START}[\\s\\S]*?${DSH_MARKER_END}\\n?`);

/** DeepSeek Harness home — `$DSH_HOME`, else `~/.dsh` (its own `home-paths` resolution order). */
function dshHome(c: InstallCtx): string {
  return process.env.DSH_HOME || join(c.home, ".dsh");
}

/**
 * DeepSeek Harness — a native Cordis plugin, wired through the HOME-level patch layer
 * (`$DSH_HOME/cordis.patch.yml`), which every profile composes after its bundles.
 *
 * The alternative is `dsh plugin --profile <name> add @vectorize-io/hindsight-coding-agents`, which
 * pnpm-installs the package into ONE profile and picks up the `dsh.bundle.patch` this package
 * ships. That is the right route for a published install and is what the docs recommend, but it
 * needs pnpm, a network, and a repeat per profile — so the installer takes the path that always
 * works: patch the home layer to load the dist entry that is already on this machine.
 *
 * The row's `name` MUST be a `file://` URL. Cordis resolves it as an ES module specifier, and a
 * bare absolute path is not one — the same trap Kilo has, where it fails to resolve and the plugin
 * is silently skipped.
 */
const dsh: HarnessInstaller = {
  name: "dsh",
  detect: (c) => onPath("dsh") || existsSync(dshHome(c)),
  install(c) {
    const path = join(dshHome(c), "cordis.patch.yml");
    const existing = existsSync(path) ? readFileSync(path, "utf8") : "";
    // REPLACE any previous block rather than skipping: a re-install after the package moved must
    // repair the now-dead path, which is exactly what `install` is for.
    const others = existing.replace(DSH_BLOCK_RE, "\n").trim();
    const entry = pathToFileURL(join(c.dist, "dsh.js")).href;
    const block =
      `${DSH_MARKER_START}\n` +
      `- insert:\n` +
      `    - id: hindsight\n` +
      `      name: ${JSON.stringify(entry)}\n` +
      `${DSH_MARKER_END}\n`;
    if (existsSync(path) && !existsSync(`${path}.hindsight-backup`))
      copyFileSync(path, `${path}.hindsight-backup`);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, others ? `${others}\n\n${block}` : block);
    // dsh's skill provider scans the shared agentskills root, the same one Codex reads.
    installSkill(c, "dsh", join(c.home, ".agents", "skills"));
    c.log?.(`dsh: plugin registered in ${path} (applies to every dsh profile)`);
  },
  uninstall(c) {
    const path = join(dshHome(c), "cordis.patch.yml");
    if (existsSync(path)) {
      const existing = readFileSync(path, "utf8");
      const others = existing.replace(DSH_BLOCK_RE, "\n").trim();
      // `[]`, not an empty file: dsh requires this file to parse to a top-level ARRAY and fails
      // BOOT on anything else, so removing the last block must leave an empty list behind.
      if (others !== existing.trim()) writeFileSync(path, others ? `${others}\n` : "[]\n");
    }
    uninstallSkill(c, join(c.home, ".agents", "skills"));
    c.log?.("dsh: plugin entry + skill removed");
  },
};

const qwen: HarnessInstaller = {
  name: "qwen-code",
  detect: (c) => onPath("qwen") || existsSync(join(c.home, ".qwen")),
  install(c) {
    const path = join(c.home, ".qwen", "settings.json");
    const settings = readJson(path);
    settings.hooks = settings.hooks ?? {};
    // Qwen's settings.json hook shape is Claude Code's, so the shared nested writer applies —
    // the ONLY difference is the unit of `timeout`, which HOOK_HARNESSES already carries in ms.
    mergeHarnessHooks(settings.hooks, "qwen-code", c.dist);
    writeJson(path, settings);
    c.log?.(`qwen-code: hooks merged into ${path}`);
    installSkill(c, "qwen-code", join(c.home, ".qwen", "skills"));
    const mcp = c.qwenMcp ?? defaultQwenMcp;
    // Same rationale as claude-code: `qwen mcp add` refuses a name that already exists, so a
    // re-install could never repoint a stale server. Remove first (a no-op when absent).
    mcp(["mcp", "remove", "hindsight"]);
    if (
      mcp([
        "mcp",
        "add",
        "-s",
        "user",
        "-e",
        "HINDSIGHT_MCP_HARNESS=qwen-code",
        "hindsight",
        "node",
        join(c.dist, "mcp-server.js"),
      ])
    ) {
      c.log?.("qwen-code: MCP server registered (qwen mcp add, user scope)");
    } else {
      c.log?.(
        `qwen-code: could not run \`qwen mcp add\` — register the tools manually:\n` +
          `  qwen mcp add -s user -e HINDSIGHT_MCP_HARNESS=qwen-code hindsight node "${join(c.dist, "mcp-server.js")}"`
      );
    }
  },
  uninstall(c) {
    const path = join(c.home, ".qwen", "settings.json");
    if (existsSync(path)) {
      const settings = readJson(path);
      if (settings.hooks) {
        stripHarnessHooks(settings.hooks, "qwen-code");
        if (!Object.keys(settings.hooks).length) delete settings.hooks;
        writeJson(path, settings);
      }
    }
    const mcp = c.qwenMcp ?? defaultQwenMcp;
    mcp(["mcp", "remove", "hindsight"]);
    uninstallSkill(c, join(c.home, ".qwen", "skills"));
    c.log?.("qwen-code: hooks + MCP registration + skill removed");
  },
};

function defaultQwenMcp(args: string[]): boolean {
  try {
    execFileSync("qwen", args, { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

export const INSTALLERS: HarnessInstaller[] = [
  opencode,
  opencode2,
  kilo,
  primeAgent,
  claudeCode,
  codex,
  antigravity,
  devin,
  cursor,
  copilot,
  grok,
  qwen,
  cline,
  dcode,
  dsh,
];

// The public executable was renamed from Gemini CLI to Antigravity's `agy`. Keep the
// integration's stable internal name for bank identity and hook compatibility, while accepting
// the executable name users naturally type at the installer prompt.
const HARNESS_ALIASES: Record<string, string> = { agy: "antigravity-cli" };

// ── CLI ─────────────────────────────────────────────────────────────────────────

/**
 * Backfill this repo's past sessions for one harness, by handing them to the same deepen engine a
 * session start uses. Deepen dedups by document id, so re-running is safe.
 *
 * Scoped to the current directory: history is per-repo, and a machine can hold thousands of
 * sessions across unrelated projects that nobody wants extracted.
 */
function importConversations(harness: string, ctx: InstallCtx): void {
  const repo = process.cwd();
  const found = importLocalHistory(harness, repo);
  if (!found.supported) {
    ctx.log?.(`${harness}: --import-conversations skipped — ${found.reason}`);
    return;
  }
  if (found.unattributed) {
    // Never silently: a session we cannot attribute is one the user might expect to see imported.
    ctx.log?.(
      `${harness}: skipped ${found.unattributed} session(s) that do not record which directory ` +
        `they ran in — importing them could file another repo's conversation into this bank`
    );
  }
  if (!found.sessions.length) {
    ctx.log?.(`${harness}: no past sessions found on disk for ${repo}`);
    return;
  }
  const turns = found.sessions.reduce((n, s) => n + s.turns.length, 0);
  const file = join(mkdtempSync(join(tmpdir(), "hindsight-import-")), "conversations.json");
  writeFileSync(file, JSON.stringify(found.sessions));
  ctx.log?.(
    `${harness}: importing ${found.sessions.length} past sessions (${turns} turns) from ${repo} — ` +
      `this runs extraction and may take a while`
  );
  try {
    execFileSync("node", [join(ctx.dist, "deepen.js"), "--repo", repo, "--conversations", file], {
      stdio: "inherit",
    });
  } catch {
    // The wiring is already in place; a failed backfill must not make `install` look failed.
    ctx.log?.(`${harness}: conversation import did not finish — re-run it any time with:`);
    ctx.log?.(`  node "${join(ctx.dist, "deepen.js")}" --repo "${repo}" --conversations "${file}"`);
  }
}

export function run(argv: string[], ctxIn: InstallCtx): number {
  let ctx = ctxIn;
  const [command, ...rawArgs] = argv;
  // `--import-conversations` backfills this repo's PAST sessions for the harness being installed —
  // the migration path off the older per-agent plugins, whose banks the server cannot merge into
  // this one. Opt-in: it re-extracts history and therefore costs tokens.
  const importHistory = rawArgs.includes("--import-conversations");
  // A flag's VALUE (`--server daemon`) is a bare word too — excluding it keeps "daemon" from being
  // read as a harness name and rejected.
  const valueArgs = flagValueArgs(rawArgs, ["server", "api-url", "api-token"]);
  const names = rawArgs.filter((a) => !a.startsWith("--") && !valueArgs.has(a));
  // Everything we write into a host's config is an ABSOLUTE path into this package. Run straight
  // from an npx cache those paths die on the first eviction and every hook stops SILENTLY, which is
  // why installing from a cache used to be refused outright. Copying the runtime somewhere stable
  // first removes the problem instead of pushing it onto the user: `npx` now works, and nobody has
  // to keep a global install of a tool whose only job is to set other tools up.
  if (command === "install" || command === "update") ctx = stageRuntime(ctx);
  // `update` is `install`'s staging half and nothing else: it replaces the runtime every wired
  // agent already points at (~/.hindsight/coding-agents/dist, a path that is stable across
  // versions) and writes to NO host config. That separation is what makes it safe to run
  // unattended from `core/auto-update.ts` — an `install` would need a harness list, and choosing
  // one on the user's behalf would rewire agents they never asked us to touch.
  //
  // The cost of not rewiring: a version that introduces a NEW hook entry point is staged but not
  // referenced, so that one feature waits for a manual `install`. Every existing entry point picks
  // the new code up on its next spawn.
  if (command === "update") {
    ctx.log?.(
      ctx.pkgRoot === ctxIn.pkgRoot
        ? "runtime already up to date — nothing staged"
        : "runtime updated — every wired agent picks it up on its next session"
    );
    return 0;
  }
  if (command !== "install" && command !== "uninstall") {
    ctx.log?.(
      `usage: hindsight-coding-agents <install|uninstall> <all|harness...>\n` +
        `       hindsight-coding-agents update\n` +
        `       [--server cloud|self-hosted|daemon] [--api-url <url>] [--api-token <token>]\n` +
        `       [--import-conversations]\n` +
        `  all      every agent detected on this machine\n` +
        `  harness  ${INSTALLERS.map((i) => i.name).join(", ")} (agy aliases antigravity-cli)\n` +
        `  update   re-stage the runtime only, leaving every host config untouched\n` +
        `  agents/CI: without a TTY nothing ever prompts — pass --server (and --api-url/--api-token) to choose`
    );
    return command ? 1 : 0;
  }
  let targets: HarnessInstaller[];
  // `all` is spelled out rather than implied by a bare command: this rewrites the config of EVERY
  // detected agent — hooks, MCP registration and the companion skill — which is too much of a
  // machine to change by accident from `install` alone.
  if (names.includes("all")) {
    targets = INSTALLERS.filter((i) => i.detect(ctx));
    if (!targets.length) {
      ctx.log?.("no supported coding agents detected — name one explicitly to wire it anyway");
      return 1;
    }
    ctx.log?.(`detected: ${targets.map((t) => t.name).join(", ")}`);
  } else if (names.length) {
    targets = [];
    for (const n of names) {
      const hit = INSTALLERS.find((i) => i.name === (HARNESS_ALIASES[n] ?? n));
      if (!hit) {
        ctx.log?.(
          `unknown harness "${n}" — expected "all" or one of: ${INSTALLERS.map((i) => i.name).join(", ")} (agy aliases antigravity-cli)`
        );
        return 1;
      }
      targets.push(hit);
    }
  } else {
    ctx.log?.(
      `${command}: name a harness, or "all" for every agent detected on this machine.\n` +
        `  hindsight-coding-agents ${command} claude-code\n` +
        `  hindsight-coding-agents ${command} all\n` +
        `harnesses: ${INSTALLERS.map((i) => i.name).join(", ")} (agy aliases antigravity-cli)`
    );
    return 1;
  }
  // Which server the agents will talk to. Resolved BEFORE any harness is wired so the very first
  // session already has a config to read.
  if (
    command === "install" &&
    !configureServer(
      ctx,
      rawArgs,
      targets.map((t) => t.name)
    )
  )
    return 1;

  // Preflight runs BEFORE any config is written, and only blocks the harness that failed: on
  // `install all` the other agents are still worth wiring. The non-zero exit keeps the failure
  // visible to whatever script invoked this.
  const blocked = new Set<string>();
  if (command === "install") {
    for (const t of targets) {
      const problem = t.preflight?.(ctx);
      if (!problem) continue;
      ctx.log?.(`\n❌ ${t.name}: ${problem}`);
      blocked.add(t.name);
    }
  }
  const runnable = targets.filter((t) => !blocked.has(t.name));
  const failed: string[] = [];
  for (const t of runnable) {
    if (t[command](ctx) === false) failed.push(t.name);
  }
  if (command === "install" && importHistory) {
    for (const t of runnable) importConversations(t.name, ctx);
  }
  if (blocked.size || failed.length) {
    const notInstalled = [...blocked, ...failed];
    ctx.log?.(`\n❌ not installed: ${notInstalled.join(", ")} — see the messages above.`);
    return 1;
  }
  // No completion message here: the CLI entry's InstallerUi outro reports success (and where the
  // settings live), so run() stays a silent-on-success engine for programmatic use.
  return 0;
}

/* c8 ignore start */
// npm exposes `bin` entries through a symlink. Node leaves argv[1] at that symlink path, so a
// direct URL comparison incorrectly makes the installed CLI a silent no-op. Resolve it first;
// source/dev invocations still work through the installer.js suffix fallback.
const mainPath = process.argv[1]
  ? (() => {
      try {
        return realpathSync(process.argv[1]);
      } catch {
        return process.argv[1];
      }
    })()
  : undefined;
const isMain = mainPath && import.meta.url === pathToFileURL(mainPath).href;
if (isMain || mainPath?.endsWith("installer.js")) {
  const dist = dirname(fileURLToPath(import.meta.url));
  const pkgRoot = dirname(dist);
  let version: string | undefined;
  try {
    version = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8")).version;
  } catch {
    version = undefined; // cosmetic only — a header without a version beats a crashed installer
  }
  const ui = createInstallerUi({
    home: homedir(),
    command: process.argv[2],
    version,
    harnessNames: INSTALLERS.map((i) => i.name),
    // configureServer's messages render as their own "server" group, but don't count as an agent.
    auxNames: ["server"],
    configPath: process.env.HINDSIGHT_CONFIG || undefined,
  });
  ui.intro();
  const code = run(process.argv.slice(2), {
    home: homedir(),
    pkgRoot,
    dist,
    // Only a real terminal gets prompted; piped/CI installs take the documented default.
    // tty.isatty, NOT process.stdin.isTTY: the process.stdin getter initializes the stdin TTY
    // stream, which puts fd 0 in non-blocking mode and breaks readLineSync (see there).
    interactive: Boolean(isatty(0) && isatty(1)),
    log: ui.log,
    promptStyle: ui.prompt,
    selectPrompt: ui.select,
  });
  ui.outro(code);
  process.exit(code);
}
/* c8 ignore stop */
