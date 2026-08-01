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
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { HOOK_HARNESSES, type HookHarnessName } from "./harness/hook-lifecycle";

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
  /** Runs `cline plugin ...`; injectable for tests. Returns false when the CLI isn't usable. */
  clinePlugin?: (args: string[]) => boolean;
  /** Probes sqlite3's required JSON query mode; injectable so installs stay deterministic in tests. */
  sqlite3Probe?: () => boolean;
  log?: (m: string) => void;
}

function readJson(path: string): Record<string, any> {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return {};
  }
}

/**
 * Parse a JSONC file (JSON with comments), as Kilo's `kilo.jsonc` may be.
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
  if (existsSync(path) && !existsSync(`${path}.hindsight-backup`)) {
    copyFileSync(path, `${path}.hindsight-backup`);
  }
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
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

/** Copy the packaged companion SKILL into a host's skills directory (idempotent overwrite). */
function installSkill(c: InstallCtx, skillsBase: string): void {
  const src = join(c.pkgRoot, "skill");
  if (!existsSync(join(src, "SKILL.md"))) return;
  const dst = join(skillsBase, "hindsight-coding-agent");
  mkdirSync(skillsBase, { recursive: true });
  cpSync(src, dst, { recursive: true });
  c.log?.(`skill installed at ${dst}`);
}

function uninstallSkill(c: InstallCtx, skillsBase: string): void {
  rmSync(join(skillsBase, "hindsight-coding-agent"), { recursive: true, force: true });
}

// ── per-harness adapters ────────────────────────────────────────────────────────

export interface HarnessInstaller {
  name: string;
  detect(ctx: InstallCtx): boolean;
  install(ctx: InstallCtx): void;
  uninstall(ctx: InstallCtx): void;
}

function onPath(bin: string): boolean {
  try {
    execFileSync("which", [bin], { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

/** Check the exact sqlite3 mode used by the Devin transcript reader, not just binary presence. */
export function canUseSqlite3(): boolean {
  try {
    const raw = execFileSync("sqlite3", ["-json", ":memory:", "SELECT 1 AS probe;"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 2000,
    });
    const parsed: unknown = JSON.parse(raw.trim());
    if (!Array.isArray(parsed) || parsed.length !== 1) return false;
    const row = parsed[0];
    return typeof row === "object" && row !== null && (row as { probe?: unknown }).probe === 1;
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

const opencode: HarnessInstaller = {
  name: "opencode",
  detect: (c) => onPath("opencode") || existsSync(join(c.home, ".config", "opencode")),
  install(c) {
    const path = join(c.home, ".config", "opencode", "opencode.json");
    const cfg = readJson(path);
    const plugins: string[] = Array.isArray(cfg.plugin) ? cfg.plugin : [];
    cfg.plugin = [...plugins.filter((p) => !String(p).includes(MARKER)), c.pkgRoot];
    writeJson(path, cfg);
    c.log?.(`opencode: plugin registered in ${path}`);
  },
  uninstall(c) {
    const path = join(c.home, ".config", "opencode", "opencode.json");
    if (!existsSync(path)) return;
    const cfg = readJson(path);
    if (Array.isArray(cfg.plugin)) {
      cfg.plugin = cfg.plugin.filter((p: string) => !String(p).includes(MARKER));
      if (!cfg.plugin.length) delete cfg.plugin;
      writeJson(path, cfg);
    }
    c.log?.("opencode: plugin entry removed");
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
    cfg.plugin = [...plugins.filter((p) => !String(p).includes(MARKER)), entry];
    writeJson(path, cfg);
    c.log?.(`kilo: plugin registered in ${path}`);
  },
  uninstall(c) {
    const path = kiloConfigPath(c);
    if (!existsSync(path)) return;
    const cfg = parseJsonc(readFileSync(path, "utf8"));
    if (!cfg || !Array.isArray(cfg.plugin)) return;
    cfg.plugin = cfg.plugin.filter((p: unknown) => !String(p).includes(MARKER));
    if (!cfg.plugin.length) delete cfg.plugin;
    writeJson(path, cfg);
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
    installSkill(c, join(c.home, ".claude", "skills"));
    const mcp = c.claudeMcp ?? defaultClaudeMcp;
    if (
      mcp([
        "mcp",
        "add",
        "--scope",
        "user",
        "hindsight",
        "--",
        "node",
        join(c.dist, "mcp-server.js"),
      ])
    ) {
      c.log?.("claude-code: MCP server registered (claude mcp add, user scope)");
    } else {
      c.log?.(
        `claude-code: could not run \`claude mcp add\` — register the tools manually:\n` +
          `  claude mcp add --scope user hindsight -- node "${join(c.dist, "mcp-server.js")}"`
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

    // config.toml: append-only, never rewrite (TOML round-tripping is not worth the risk).
    const tomlPath = join(c.home, ".codex", "config.toml");
    let toml = existsSync(tomlPath) ? readFileSync(tomlPath, "utf8") : "";
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
    if (!toml.includes("[mcp_servers.hindsight]")) {
      additions.push(
        `[mcp_servers.hindsight]\ncommand = "node"\nargs = ["${join(c.dist, "mcp-server.js")}"]`
      );
    }
    if (additions.length) {
      if (existsSync(tomlPath) && !existsSync(`${tomlPath}.hindsight-backup`)) {
        copyFileSync(tomlPath, `${tomlPath}.hindsight-backup`);
      }
      mkdirSync(dirname(tomlPath), { recursive: true });
      writeFileSync(tomlPath, `${toml.replace(/\n*$/, "\n\n")}${additions.join("\n\n")}\n`);
      c.log?.(`codex: appended ${additions.length} section(s) to ${tomlPath}`);
    }
    installSkill(c, join(c.home, ".agents", "skills")); // agentskills-standard shared dir
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
      const cleaned = toml.replace(
        /\n?\[mcp_servers\.hindsight\]\ncommand = "node"\nargs = \[[^\]]*\]\n?/g,
        "\n"
      );
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
      hindsight: {
        command: "node",
        args: [join(c.dist, "mcp-server.js")],
        env: { HINDSIGHT_MCP_HARNESS: "antigravity-cli" },
      },
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
    installSkill(c, join(c.home, ".gemini", "config", "skills"));
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

const devin: HarnessInstaller = {
  name: "devin-cli",
  detect: (c) => onPath("devin") || existsSync(join(c.home, ".config", "devin")),
  install(c) {
    if (!(c.sqlite3Probe ?? canUseSqlite3)()) {
      c.log?.(
        "devin-cli: warning: sqlite3 was not found or cannot run the required -json query; hooks " +
          "will be installed, but session retain requires a compatible sqlite3"
      );
    }
    const configPath = join(c.home, ".config", "devin", "config.json");
    const config = readJson(configPath);
    config.hooks = config.hooks ?? {};
    mergeHarnessHooks(config.hooks, "devin-cli", c.dist);
    writeJson(configPath, config);
    const mcpPath = join(c.home, ".config", "devin", "mcp_config.json");
    const mcp = readJson(mcpPath);
    mcp.mcpServers = {
      ...(mcp.mcpServers ?? {}),
      hindsight: {
        command: "node",
        args: [join(c.dist, "mcp-server.js")],
        env: { HINDSIGHT_MCP_HARNESS: "devin-cli" },
      },
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
      hindsight: { command: "node", args: [join(c.dist, "mcp-server.js")] },
    };
    writeJson(mcpPath, mcp);
    c.log?.(`cursor-cli: hooks merged into ${hooksPath}, MCP into ${mcpPath}`);
    installSkill(c, join(c.home, ".cursor", "skills"));
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
      hindsight: { command: "node", args: [join(c.dist, "mcp-server.js")] },
    };
    writeJson(mcpPath, mcp);
    installSkill(c, join(c.home, ".copilot", "skills"));
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
      `[mcp_servers.hindsight]\ncommand = \"node\"\nargs = [${tomlString(join(c.dist, "mcp-server.js"))}]\n${GROK_MARKER_END}\n`;
    if (existsSync(path) && !existsSync(`${path}.hindsight-backup`))
      copyFileSync(path, `${path}.hindsight-backup`);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${withoutOurs.replace(/\n*$/, "\n")}${block}`);
    installSkill(c, join(c.home, ".grok", "skills"));
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
      hindsight: {
        command: "node",
        args: [join(c.dist, "mcp-server.js")],
        env: { HINDSIGHT_MCP_HARNESS: "cline-cli" },
      },
    };
    writeJson(mcpPath, mcp);
    installSkill(c, join(c.home, ".cline", "data", "settings", "skills"));
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

export const INSTALLERS: HarnessInstaller[] = [
  opencode,
  kilo,
  claudeCode,
  codex,
  antigravity,
  devin,
  cursor,
  copilot,
  grok,
  cline,
];

// The public executable was renamed from Gemini CLI to Antigravity's `agy`. Keep the
// integration's stable internal name for bank identity and hook compatibility, while accepting
// the executable name users naturally type at the installer prompt.
const HARNESS_ALIASES: Record<string, string> = { agy: "antigravity-cli" };

// ── CLI ─────────────────────────────────────────────────────────────────────────

export function run(argv: string[], ctx: InstallCtx): number {
  const [command, ...names] = argv;
  // The wiring we write is ABSOLUTE paths into this package's dist. From an npx/pnpm-dlx cache
  // those paths die on cache eviction — every hook silently stops. Refuse and say what to do.
  if (command === "install" && /\/(_npx|\.npm\/_npx|dlx-)\/|\/_cacache\//.test(ctx.pkgRoot)) {
    ctx.log?.(
      "refusing to install from an npx/dlx cache: the hook wiring would point into a cache npm can " +
        "evict, silently breaking every session.\nInstall the package permanently, then re-run:\n" +
        "  npm install -g hindsight-coding-agents && hindsight-coding-agents install"
    );
    return 1;
  }
  if (command !== "install" && command !== "uninstall") {
    ctx.log?.(
      `usage: hindsight-coding-agents <install|uninstall> [harness...]\n` +
        `harnesses: ${INSTALLERS.map((i) => i.name).join(", ")} (agy is an alias for antigravity-cli; default: every one detected on this machine)`
    );
    return command ? 1 : 0;
  }
  let targets: HarnessInstaller[];
  if (names.length) {
    targets = [];
    for (const n of names) {
      const hit = INSTALLERS.find((i) => i.name === (HARNESS_ALIASES[n] ?? n));
      if (!hit) {
        ctx.log?.(
          `unknown harness "${n}" — expected one of: ${INSTALLERS.map((i) => i.name).join(", ")} (agy aliases antigravity-cli)`
        );
        return 1;
      }
      targets.push(hit);
    }
  } else {
    targets = INSTALLERS.filter((i) => i.detect(ctx));
    if (!targets.length) {
      ctx.log?.("no supported coding agents detected — pass harness names explicitly");
      return 1;
    }
    ctx.log?.(`detected: ${targets.map((t) => t.name).join(", ")}`);
  }
  for (const t of targets) t[command](ctx);
  ctx.log?.(
    command === "install"
      ? `\n✅ installed. Configure the server in ~/.hindsight/coding-agent.json (apiUrl/apiToken) and start a session.`
      : `\n✅ uninstalled.`
  );
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
  process.exit(
    run(process.argv.slice(2), {
      home: homedir(),
      pkgRoot: dirname(dist),
      dist,
      log: (m) => console.log(m),
    })
  );
}
/* c8 ignore stop */
