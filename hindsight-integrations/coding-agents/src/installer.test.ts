import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { INSTALLERS, MARKER, run, type InstallCtx } from "./installer";

// Every test gets a FRESH temp dir as ctx.home (never the real $HOME) and a stubbed
// claudeMcp so the real `claude` CLI is never executed. run() is always called with
// explicit harness names so detect() (which probes PATH) never runs.

const homes: string[] = [];

function makeCtx(): InstallCtx & {
  claudeMcp: ReturnType<typeof vi.fn>;
  clinePlugin: ReturnType<typeof vi.fn>;
  nodeSqlite: ReturnType<typeof vi.fn>;
} {
  const home = mkdtempSync(join(tmpdir(), "hindsight-installer-test-"));
  homes.push(home);
  const pkgRoot = join("/opt", MARKER); // contains the marker, like the real package path
  return {
    home,
    pkgRoot,
    dist: join(pkgRoot, "dist"),
    claudeMcp: vi.fn(() => true),
    clinePlugin: vi.fn(() => true),
    // Stubbed like the CLI seams above, so the suite never depends on the Node running it.
    nodeSqlite: vi.fn(() => true),
  };
}

function readJson(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, "utf8"));
}

function writeJsonAt(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
}

afterEach(() => {
  while (homes.length) rmSync(homes.pop()!, { recursive: true, force: true });
  vi.clearAllMocks();
});

describe("claude-code installer", () => {
  const settingsPath = (ctx: InstallCtx) => join(ctx.home, ".claude", "settings.json");

  it("install writes the 3 hook events with our dist commands and timeouts 30/30/60", () => {
    const ctx = makeCtx();
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    const settings = readJson(settingsPath(ctx));
    const hooks = settings.hooks;
    expect(Object.keys(hooks).sort()).toEqual(["SessionStart", "Stop", "UserPromptSubmit"]);
    const inner = (ev: string) => hooks[ev][0].hooks[0];
    expect(inner("SessionStart").command).toContain(join(ctx.dist, "claude-sessionstart-hook.js"));
    expect(inner("UserPromptSubmit").command).toContain(join(ctx.dist, "claude-hook.js"));
    expect(inner("Stop").command).toContain(join(ctx.dist, "claude-stop-hook.js"));
    expect(inner("SessionStart").timeout).toBe(30);
    expect(inner("UserPromptSubmit").timeout).toBe(30);
    expect(inner("Stop").timeout).toBe(60);
    for (const ev of ["SessionStart", "UserPromptSubmit", "Stop"]) {
      expect(inner(ev).type).toBe("command");
    }
  });

  it("preserves pre-existing foreign hook entries and appends ours", () => {
    const ctx = makeCtx();
    const foreign = { hooks: [{ type: "command", command: "echo other-tool", timeout: 5 }] };
    writeJsonAt(settingsPath(ctx), { hooks: { SessionStart: [foreign] } });
    run(["install", "claude-code"], ctx);
    const events = readJson(settingsPath(ctx)).hooks.SessionStart;
    expect(events).toHaveLength(2);
    expect(events[0]).toEqual(foreign);
    expect(JSON.stringify(events[1])).toContain(MARKER);
  });

  it("re-install is idempotent — exactly ONE of our entries per event", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    run(["install", "claude-code"], ctx);
    const hooks = readJson(settingsPath(ctx)).hooks;
    for (const ev of ["SessionStart", "UserPromptSubmit", "Stop"]) {
      const ours = hooks[ev].filter((e: unknown) => JSON.stringify(e).includes(MARKER));
      expect(ours).toHaveLength(1);
      expect(hooks[ev]).toHaveLength(1);
    }
  });

  it("removes before adding so an existing registration is REPOINTED, not skipped", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    const calls = ctx.claudeMcp.mock.calls.map((c) => c[0]);
    const removeAt = calls.findIndex((a) => a[1] === "remove");
    const addAt = calls.findIndex((a) => a[1] === "add");
    // `claude mcp add` refuses a name that already exists, so without the remove a re-install
    // silently leaves a stale (possibly dead) server path registered.
    expect(removeAt).toBeGreaterThanOrEqual(0);
    expect(removeAt).toBeLessThan(addAt);
    expect(calls[removeAt]).toEqual(["mcp", "remove", "--scope", "user", "hindsight"]);
  });

  it("registers the MCP server via `claude mcp add` (user scope)", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    expect(ctx.claudeMcp).toHaveBeenCalledWith([
      "mcp",
      "add",
      "--scope",
      "user",
      "hindsight",
      "--",
      "node",
      join(ctx.dist, "mcp-server.js"),
    ]);
  });

  it("still succeeds when claudeMcp reports the CLI is unusable (manual instructions)", () => {
    const ctx = makeCtx();
    ctx.claudeMcp.mockReturnValue(false);
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    expect(logs.join("\n")).toContain("claude mcp add");
    // hooks were still written despite the MCP failure
    expect(existsSync(settingsPath(ctx))).toBe(true);
  });

  it("uninstall strips our entries, keeps foreign ones, and calls `claude mcp remove`", () => {
    const ctx = makeCtx();
    const foreign = { hooks: [{ type: "command", command: "echo other-tool", timeout: 5 }] };
    writeJsonAt(settingsPath(ctx), { hooks: { Stop: [foreign] } });
    run(["install", "claude-code"], ctx);
    run(["uninstall", "claude-code"], ctx);
    const settings = readJson(settingsPath(ctx));
    expect(settings.hooks.Stop).toEqual([foreign]);
    expect(settings.hooks.SessionStart).toBeUndefined();
    expect(settings.hooks.UserPromptSubmit).toBeUndefined();
    expect(JSON.stringify(settings)).not.toContain(MARKER);
    expect(ctx.claudeMcp).toHaveBeenCalledWith(["mcp", "remove", "--scope", "user", "hindsight"]);
  });

  it("uninstall removes the hooks object entirely when nothing else remains", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    run(["uninstall", "claude-code"], ctx);
    expect(readJson(settingsPath(ctx)).hooks).toBeUndefined();
  });
});

describe("codex installer", () => {
  const hooksPath = (ctx: InstallCtx) => join(ctx.home, ".codex", "hooks.json");
  const tomlPath = (ctx: InstallCtx) => join(ctx.home, ".codex", "config.toml");

  it("install writes the 3 hook events into hooks.json", () => {
    const ctx = makeCtx();
    expect(run(["install", "codex"], ctx)).toBe(0);
    const hooks = readJson(hooksPath(ctx)).hooks;
    expect(Object.keys(hooks).sort()).toEqual(["SessionStart", "Stop", "UserPromptSubmit"]);
    expect(hooks.SessionStart[0].hooks[0].command).toContain("codex-sessionstart-hook.js");
    expect(hooks.UserPromptSubmit[0].hooks[0].command).toContain("codex-hook.js");
    expect(hooks.Stop[0].hooks[0].command).toContain("codex-stop-hook.js");
  });

  it("creates config.toml with the features flag and mcp_servers section when missing", () => {
    const ctx = makeCtx();
    run(["install", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml).toContain("[features]\nhooks = true");
    expect(toml).toContain("[mcp_servers.hindsight]");
    expect(toml).toContain(join(ctx.dist, "mcp-server.js"));
  });

  it("does NOT duplicate an existing [features] section (only appends mcp) and backs up the toml", () => {
    const ctx = makeCtx();
    const original = "[features]\nsome_flag = true\n";
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    writeFileSync(tomlPath(ctx), original);
    run(["install", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml.match(/^\[features\]/gm)).toHaveLength(1);
    expect(toml).not.toContain("hooks = true"); // user is told to add it manually
    expect(toml).toContain("[mcp_servers.hindsight]");
    expect(readFileSync(`${tomlPath(ctx)}.hindsight-backup`, "utf8")).toBe(original);
  });

  it("appends nothing features-related when hooks is already present", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    writeFileSync(tomlPath(ctx), "[features]\nhooks = true\n");
    run(["install", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml.match(/hooks/g)).toHaveLength(1);
    expect(toml.match(/^\[features\]/gm)).toHaveLength(1);
    expect(toml).toContain("[mcp_servers.hindsight]");
  });

  it("uninstall removes the mcp_servers.hindsight block and leaves the rest of the toml", () => {
    const ctx = makeCtx();
    run(["install", "codex"], ctx);
    run(["uninstall", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml).not.toContain("[mcp_servers.hindsight]");
    expect(toml).toContain("hooks = true"); // flag deliberately left in place
    const hooks = readJson(hooksPath(ctx)).hooks;
    expect(Object.keys(hooks)).toHaveLength(0);
  });
});

describe("cline-cli installer", () => {
  const hooksDir = (ctx: InstallCtx) => join(ctx.home, "Documents", "Cline", "Hooks");
  const mcpPath = (ctx: InstallCtx) =>
    join(ctx.home, ".cline", "data", "settings", "cline_mcp_settings.json");

  it("installs the native plugin, MCP, and the companion skill", () => {
    const ctx = makeCtx();
    expect(run(["install", "cline-cli"], ctx)).toBe(0);
    expect(ctx.clinePlugin).toHaveBeenCalledWith(["plugin", "install", "--force", ctx.pkgRoot]);
    expect(readJson(mcpPath(ctx)).mcpServers.hindsight).toEqual({
      command: "node",
      args: [join(ctx.dist, "mcp-server.js")],
      env: { HINDSIGHT_MCP_HARNESS: "cline-cli" },
    });
  });

  it("removes legacy wrappers but preserves foreign hooks and uninstalls the native plugin", () => {
    const ctx = makeCtx();
    const foreign = join(hooksDir(ctx), "TaskStart");
    mkdirSync(dirname(foreign), { recursive: true });
    writeFileSync(foreign, "#!/usr/bin/env sh\necho foreign\n");
    const legacy = join(hooksDir(ctx), "UserPromptSubmit");
    writeFileSync(legacy, "#!/usr/bin/env sh\n# HINDSIGHT_CODING_AGENTS_CLINE\n");
    run(["install", "cline-cli"], ctx);
    expect(readFileSync(foreign, "utf8")).toContain("foreign");
    expect(existsSync(legacy)).toBe(false);
    run(["uninstall", "cline-cli"], ctx);
    expect(existsSync(foreign)).toBe(true);
    expect(ctx.clinePlugin).toHaveBeenLastCalledWith([
      "plugin",
      "uninstall",
      "@vectorize-io/hindsight-coding-agents",
    ]);
    expect(readJson(mcpPath(ctx)).mcpServers.hindsight).toBeUndefined();
  });
});

describe("antigravity-cli installer", () => {
  it("removes a namespace written under a PREVIOUS marker instead of leaving both live", () => {
    const ctx = makeCtx();
    const hooksPath = join(ctx.home, ".gemini", "config", "hooks.json");
    // Exactly what a marker rename produced: our old namespace, still pointing at a stale path.
    writeJsonAt(hooksPath, {
      "hindsight-coding-agents": {
        PreInvocation: [{ command: 'node "/old/path/coding-agents/dist/antigravity-hook.js"' }],
      },
      "someone-elses-bundle": { PreInvocation: [{ command: "echo other" }] },
    });
    run(["install", "antigravity-cli"], ctx);

    const hooks = readJson(hooksPath);
    // The stale namespace is gone — otherwise Antigravity fires every hook twice.
    expect(hooks["hindsight-coding-agents"]).toBeUndefined();
    expect(hooks[MARKER]).toBeDefined();
    expect(JSON.stringify(hooks)).not.toContain("/old/path/");
    // An unrelated bundle is untouched.
    expect(hooks["someone-elses-bundle"]).toBeDefined();
  });

  const hooksPath = (ctx: InstallCtx) => join(ctx.home, ".gemini", "config", "hooks.json");
  const mcpPath = (ctx: InstallCtx) => join(ctx.home, ".gemini", "config", "mcp_config.json");
  const settingsPath = (ctx: InstallCtx) =>
    join(ctx.home, ".gemini", "antigravity-cli", "settings.json");

  it("installs PreInvocation and Stop hooks plus mcpServers.hindsight", () => {
    const ctx = makeCtx();
    expect(run(["install", "antigravity-cli"], ctx)).toBe(0);
    const hooks = readJson(hooksPath(ctx));
    expect(hooks[MARKER].PreInvocation[0].command).toContain("antigravity-hook.js");
    expect(hooks[MARKER].PreInvocation[0].timeout).toBe(30);
    expect(hooks[MARKER].Stop[0].command).toContain("antigravity-stop-hook.js");
    expect(hooks[MARKER].Stop[0].timeout).toBe(30);
    expect(readJson(mcpPath(ctx)).mcpServers.hindsight).toEqual({
      command: "node",
      args: [join(ctx.dist, "mcp-server.js")],
      env: { HINDSIGHT_MCP_HARNESS: "antigravity-cli" },
    });
    expect(readJson(settingsPath(ctx)).statusLine).toEqual({
      type: "command",
      command: `node "${join(ctx.dist, "antigravity-statusline.js")}"`,
    });
  });

  it("accepts agy as the supported CLI name", () => {
    const ctx = makeCtx();
    expect(run(["install", "agy"], ctx)).toBe(0);
    expect(readJson(mcpPath(ctx)).mcpServers.hindsight.env.HINDSIGHT_MCP_HARNESS).toBe(
      "antigravity-cli"
    );
  });

  it("preserves existing unrelated settings keys", () => {
    const ctx = makeCtx();
    writeJsonAt(hooksPath(ctx), { foreign: { enabled: false } });
    run(["install", "antigravity-cli"], ctx);
    expect(readJson(hooksPath(ctx)).foreign).toEqual({ enabled: false });
  });

  it("preserves an existing custom status line", () => {
    const ctx = makeCtx();
    writeJsonAt(settingsPath(ctx), { statusLine: { type: "command", command: "my-statusline" } });
    run(["install", "antigravity-cli"], ctx);
    expect(readJson(settingsPath(ctx)).statusLine.command).toBe("my-statusline");
  });

  it("uninstall removes only our hooks and mcp server", () => {
    const ctx = makeCtx();
    writeJsonAt(mcpPath(ctx), {
      mcpServers: { other: { command: "other-tool" } },
    });
    run(["install", "antigravity-cli"], ctx);
    run(["uninstall", "antigravity-cli"], ctx);
    expect(readJson(hooksPath(ctx))).toEqual({});
    const mcp = readJson(mcpPath(ctx));
    expect(mcp.mcpServers.hindsight).toBeUndefined();
    expect(mcp.mcpServers.other).toEqual({ command: "other-tool" });
    expect(JSON.stringify(mcp)).not.toContain(MARKER);
    expect(readJson(settingsPath(ctx)).statusLine).toBeUndefined();
  });
});

/** Kilo is an opencode fork: same `plugin` array, but a JSONC-capable config that may already
 *  exist under any of several names, and it must load dist/kilo.js (not the package root, which
 *  resolves to the opencode entry and would report the wrong harness). */
describe("kilo installer", () => {
  const kiloDir = (ctx: InstallCtx) => join(ctx.home, ".config", "kilo");
  const entryOf = (ctx: InstallCtx) => pathToFileURL(join(ctx.dist, "kilo.js")).href;

  it("registers dist/kilo.js — NOT the package root, which is the opencode entry", () => {
    const ctx = makeCtx();
    expect(run(["install", "kilo"], ctx)).toBe(0);
    const cfg = readJson(join(kiloDir(ctx), "kilo.json"));
    expect(cfg.plugin).toEqual([entryOf(ctx)]);
    expect(cfg.plugin).not.toContain(ctx.pkgRoot);
  });

  it("registers a file:// URL — a bare path is silently ignored by Kilo", () => {
    const ctx = makeCtx();
    run(["install", "kilo"], ctx);
    const [entry] = readJson(join(kiloDir(ctx), "kilo.json")).plugin;
    // Verified against Kilo 7.4.17: a bare absolute path is treated as an npm module specifier,
    // fails to resolve, and the plugin is skipped with NO error — the session just has no memory.
    expect(entry).toMatch(/^file:\/\//);
    expect(entry).not.toBe(join(ctx.dist, "kilo.js"));
  });

  it("is idempotent across reinstalls and preserves other plugin entries", () => {
    const ctx = makeCtx();
    writeJsonAt(join(kiloDir(ctx), "kilo.json"), { plugin: ["some-other-plugin"] });
    run(["install", "kilo"], ctx);
    run(["install", "kilo"], ctx);
    expect(readJson(join(kiloDir(ctx), "kilo.json")).plugin).toEqual([
      "some-other-plugin",
      entryOf(ctx),
    ]);
  });

  it("edits an existing kilo.jsonc instead of creating a competing kilo.json", () => {
    const ctx = makeCtx();
    const jsonc = join(kiloDir(ctx), "kilo.jsonc");
    mkdirSync(kiloDir(ctx), { recursive: true });
    writeFileSync(jsonc, '{\n  // my config\n  "$schema": "https://app.kilo.ai/config.json"\n}\n');
    run(["install", "kilo"], ctx);
    expect(existsSync(join(kiloDir(ctx), "kilo.json"))).toBe(false);
    const cfg = readJson(jsonc);
    expect(cfg.plugin).toEqual([entryOf(ctx)]);
    expect(cfg.$schema).toBe("https://app.kilo.ai/config.json"); // comments dropped, DATA kept
  });

  it("refuses to clobber a config it cannot parse", () => {
    const ctx = makeCtx();
    const jsonc = join(kiloDir(ctx), "kilo.jsonc");
    mkdirSync(kiloDir(ctx), { recursive: true });
    const broken = '{ "provider": { unquoted } }';
    writeFileSync(jsonc, broken);
    run(["install", "kilo"], ctx);
    // A naive JSON.parse->{} fallback would have replaced the whole file with just our plugin key.
    expect(readFileSync(jsonc, "utf8")).toBe(broken);
  });

  it("uninstall removes our entry and deletes the plugin key when empty", () => {
    const ctx = makeCtx();
    run(["install", "kilo"], ctx);
    run(["uninstall", "kilo"], ctx);
    expect(readJson(join(kiloDir(ctx), "kilo.json")).plugin).toBeUndefined();
  });
});

describe("opencode installer", () => {
  const cfgPath = (ctx: InstallCtx) => join(ctx.home, ".config", "opencode", "opencode.json");

  it("install adds ctx.pkgRoot to the plugin array exactly once, even across reinstalls", () => {
    const ctx = makeCtx();
    expect(run(["install", "opencode"], ctx)).toBe(0);
    run(["install", "opencode"], ctx);
    const cfg = readJson(cfgPath(ctx));
    expect(cfg.plugin).toEqual([ctx.pkgRoot]);
  });

  it("preserves other plugin entries", () => {
    const ctx = makeCtx();
    writeJsonAt(cfgPath(ctx), { plugin: ["some-other-plugin"] });
    run(["install", "opencode"], ctx);
    expect(readJson(cfgPath(ctx)).plugin).toEqual(["some-other-plugin", ctx.pkgRoot]);
  });

  it("uninstall removes our entry and deletes the plugin key when empty", () => {
    const ctx = makeCtx();
    run(["install", "opencode"], ctx);
    run(["uninstall", "opencode"], ctx);
    expect(readJson(cfgPath(ctx)).plugin).toBeUndefined();
  });

  it("uninstall keeps the plugin key when other entries remain", () => {
    const ctx = makeCtx();
    writeJsonAt(cfgPath(ctx), { plugin: ["some-other-plugin"] });
    run(["install", "opencode"], ctx);
    run(["uninstall", "opencode"], ctx);
    expect(readJson(cfgPath(ctx)).plugin).toEqual(["some-other-plugin"]);
  });
});

describe("cursor-cli installer", () => {
  const hooksPath = (ctx: InstallCtx) => join(ctx.home, ".cursor", "hooks.json");
  const mcpPath = (ctx: InstallCtx) => join(ctx.home, ".cursor", "mcp.json");

  it("install writes sessionStart, beforeSubmitPrompt, and stop hooks plus the mcp.json server entry", () => {
    const ctx = makeCtx();
    expect(run(["install", "cursor-cli"], ctx)).toBe(0);
    const hooks = readJson(hooksPath(ctx)).hooks;
    expect(hooks.sessionStart).toHaveLength(1);
    expect(hooks.sessionStart[0].command).toContain(join(ctx.dist, "cursor-sessionstart-hook.js"));
    expect(hooks.sessionStart[0].timeout).toBe(30);
    expect(hooks.beforeSubmitPrompt).toHaveLength(1);
    expect(hooks.beforeSubmitPrompt[0].command).toContain(join(ctx.dist, "cursor-hook.js"));
    expect(hooks.stop).toHaveLength(1);
    expect(hooks.stop[0].command).toContain(join(ctx.dist, "cursor-stop-hook.js"));
    expect(hooks.stop[0].timeout).toBe(30);
    const mcp = readJson(mcpPath(ctx));
    expect(mcp.mcpServers.hindsight).toEqual({
      command: "node",
      args: [join(ctx.dist, "mcp-server.js")],
    });
  });

  it("uninstall cleans both files", () => {
    const ctx = makeCtx();
    run(["install", "cursor-cli"], ctx);
    run(["uninstall", "cursor-cli"], ctx);
    expect(readJson(hooksPath(ctx)).hooks).toBeUndefined();
    expect(readJson(mcpPath(ctx)).mcpServers.hindsight).toBeUndefined();
  });
});

describe("grok-build installer", () => {
  const configPath = (ctx: InstallCtx) => join(ctx.home, ".grok", "config.toml");

  it("re-install REPLACES the block so a moved package is repointed, not left stale", () => {
    const ctx = makeCtx();
    run(["install", "grok-build"], ctx);
    // Simulate the package moving (the exact case that broke: the install used to skip whenever a
    // block already existed, leaving dead paths behind and no way to repair them but by hand).
    const moved = { ...ctx, dist: join("/opt", MARKER, "moved-dist") };
    run(["install", "grok-build"], moved);

    const toml = readFileSync(configPath(ctx), "utf8");
    expect(toml).toContain(join("/opt", MARKER, "moved-dist"));
    expect(toml).not.toContain(ctx.dist); // the old path is gone, not merely appended past
    // Exactly one block — a replace, not an accumulation.
    expect(toml.match(/HINDSIGHT_CODING_AGENTS_GROK_START/g)).toHaveLength(1);
  });

  it("installs native Grok lifecycle hooks and MCP without Claude configuration", () => {
    const ctx = makeCtx();
    expect(run(["install", "grok-build"], ctx)).toBe(0);
    const config = readFileSync(configPath(ctx), "utf8");
    expect(config).toContain("[[hooks.SessionStart]]");
    expect(config).toContain("[[hooks.UserPromptSubmit]]");
    expect(config).toContain("[[hooks.Stop]]");
    expect(config).toContain(join(ctx.dist, "grok-sessionstart-hook.js"));
    expect(config).toContain(join(ctx.dist, "grok-hook.js"));
    expect(config).toContain(join(ctx.dist, "grok-stop-hook.js"));
    expect(config).toContain("[mcp_servers.hindsight]");
    expect(config).toContain(join(ctx.dist, "mcp-server.js"));
    expect(existsSync(join(ctx.home, ".claude"))).toBe(false);
  });

  it("removes only its marked Grok TOML block", () => {
    const ctx = makeCtx();
    mkdirSync(dirname(configPath(ctx)), { recursive: true });
    writeFileSync(configPath(ctx), '[ui]\ntheme = "dark"\n');
    run(["install", "grok-build"], ctx);
    run(["uninstall", "grok-build"], ctx);
    const config = readFileSync(configPath(ctx), "utf8");
    expect(config).toContain('[ui]\ntheme = "dark"');
    expect(config).not.toContain("HINDSIGHT_CODING_AGENTS_GROK");
    expect(config).not.toContain("[mcp_servers.hindsight]");
  });
});

describe("run() CLI behavior", () => {
  it("returns 1 for an unknown harness name and touches nothing", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "not-a-harness"], ctx)).toBe(1);
    expect(logs.join("\n")).toContain('unknown harness "not-a-harness"');
    expect(existsSync(join(ctx.home, ".claude"))).toBe(false);
    expect(ctx.claudeMcp).not.toHaveBeenCalled();
  });

  it("returns 0 with usage when no command is given", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run([], ctx)).toBe(0);
    expect(logs.join("\n")).toContain("usage:");
  });

  it("returns 1 for an unknown command", () => {
    const ctx = makeCtx();
    expect(run(["frobnicate"], ctx)).toBe(1);
  });

  it("explicit harness names bypass detection — installs into an empty home", () => {
    const ctx = makeCtx();
    // nothing pre-exists in this fresh home, yet the named harness installs fine
    expect(run(["install", "antigravity-cli", "opencode"], ctx)).toBe(0);
    expect(existsSync(join(ctx.home, ".gemini", "config", "hooks.json"))).toBe(true);
    expect(existsSync(join(ctx.home, ".config", "opencode", "opencode.json"))).toBe(true);
  });

  it("first write to a pre-existing json creates <file>.hindsight-backup with the original content", () => {
    const ctx = makeCtx();
    const path = join(ctx.home, ".gemini", "config", "hooks.json");
    writeJsonAt(path, { auth: { selectedType: "oauth" } });
    const original = readFileSync(path, "utf8");
    run(["install", "antigravity-cli"], ctx);
    run(["install", "antigravity-cli"], ctx); // second write must NOT overwrite the backup
    expect(readFileSync(`${path}.hindsight-backup`, "utf8")).toBe(original);
  });

  it("MARKER identifies our entries under BOTH the npm and repo-checkout layouts", () => {
    // Dedupe-on-reinstall and uninstall both key off this substring appearing in the package path.
    // It silently stopped matching a checkout when the directory dropped its `hindsight-` prefix.
    expect("/usr/lib/node_modules/@vectorize-io/hindsight-coding-agents/dist").toContain(MARKER);
    expect("/repo/hindsight-integrations/coding-agents/dist").toContain(MARKER);
  });

  it("re-install from a repo-checkout path leaves exactly one hook entry per event", () => {
    const ctx = makeCtx();
    const repoStyle = {
      ...ctx,
      pkgRoot: "/repo/hindsight-integrations/coding-agents",
      dist: "/repo/hindsight-integrations/coding-agents/dist",
    };
    run(["install", "claude-code"], repoStyle);
    run(["install", "claude-code"], repoStyle);
    const hooks = readJson(join(ctx.home, ".claude", "settings.json")).hooks;
    for (const ev of ["SessionStart", "UserPromptSubmit", "Stop"]) {
      expect(hooks[ev]).toHaveLength(1);
    }
  });

  it("exposes the supported harnesses", () => {
    expect(INSTALLERS.map((i) => i.name)).toEqual([
      "opencode",
      "kilo",
      "claude-code",
      "codex",
      "antigravity-cli",
      "devin-cli",
      "cursor-cli",
      "copilot-cli",
      "grok-build",
      "cline-cli",
    ]);
  });
});

/**
 * `all` is an explicit target rather than the default for a bare command: wiring every detected
 * agent rewrites a lot of a machine's config and should not happen by accident.
 */
describe("all vs named harnesses", () => {
  it("`install all` wires every DETECTED agent", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    expect(run(["install", "all"], ctx)).toBe(0);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(true);
    expect(existsSync(join(ctx.home, ".codex", "hooks.json"))).toBe(true);
  });

  it("a bare `install` changes NOTHING and explains the choice", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    expect(run(["install"], ctx)).toBe(1);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(false);
    expect(logs.join("\n")).toContain("all");
  });

  it("a named harness wires only that one", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    run(["install", "claude-code"], ctx);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(true);
    expect(existsSync(join(ctx.home, ".codex", "hooks.json"))).toBe(false);
  });

  it("`uninstall all` is accepted too, so the pair stays symmetric", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    run(["install", "all"], ctx);
    expect(run(["uninstall", "all"], ctx)).toBe(0);
    expect(readJson(join(ctx.home, ".claude", "settings.json")).hooks).toBeUndefined();
  });
});

describe("npx-cache guard", () => {
  it("refuses install when pkgRoot is inside an npx cache (wiring would die on eviction)", () => {
    const logs: string[] = [];
    const ctx = {
      home: "/tmp/never-touched",
      pkgRoot: "/Users/x/.npm/_npx/abc123/node_modules/hindsight-coding-agents",
      dist: "/Users/x/.npm/_npx/abc123/node_modules/hindsight-coding-agents/dist",
      claudeMcp: () => true,
      log: (m: string) => logs.push(m),
    };
    expect(run(["install", "claude-code"], ctx)).toBe(1);
    expect(logs.join("\n")).toContain("npm install -g");
  });
});

describe("skill install across skills-capable hosts", () => {
  it("copies the packaged skill for claude/codex(~/.agents)/antigravity/cursor and uninstall removes each", () => {
    const home = mkdtempSync(join(tmpdir(), "hs-inst-skill-"));
    const pkgRoot = mkdtempSync(join(tmpdir(), "hs-pkg-"));
    mkdirSync(join(pkgRoot, "skill"), { recursive: true });
    writeFileSync(
      join(pkgRoot, "skill", "SKILL.md"),
      "---\nname: hindsight-coding-agent\n---\nbody"
    );
    const ctx = { home, pkgRoot, dist: join(pkgRoot, "dist"), claudeMcp: vi.fn(() => true) };
    const targets: [string, string][] = [
      ["claude-code", join(home, ".claude", "skills")],
      ["codex", join(home, ".agents", "skills")],
      ["antigravity-cli", join(home, ".gemini", "config", "skills")],
      ["cursor-cli", join(home, ".cursor", "skills")],
    ];
    run(["install", ...targets.map(([h]) => h)], ctx);
    for (const [, base] of targets) {
      expect(existsSync(join(base, "hindsight-coding-agent", "SKILL.md"))).toBe(true);
    }
    run(["uninstall", ...targets.map(([h]) => h)], ctx);
    for (const [, base] of targets) {
      expect(existsSync(join(base, "hindsight-coding-agent"))).toBe(false);
    }
    rmSync(home, { recursive: true, force: true });
    rmSync(pkgRoot, { recursive: true, force: true });
  });
});

/**
 * Devin is the only harness that cannot fall back to a transcript file: its hooks pass a session id
 * and the conversation lives in a SQLite database. Without `node:sqlite` the install used to
 * succeed and then retain nothing, forever (#3125).
 */
describe("devin-cli preflight", () => {
  it("refuses to install when the hook node can't read SQLite", () => {
    const ctx = makeCtx();
    ctx.nodeSqlite = vi.fn(() => false);
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);

    expect(run(["install", "devin-cli"], ctx)).toBe(1);
    // Nothing written: a blocked harness must leave the machine untouched.
    expect(existsSync(join(ctx.home, ".config", "devin", "config.json"))).toBe(false);
    const output = logs.join("\n");
    expect(output).toContain("node:sqlite");
    expect(output).toContain("Node 22.5");
    expect(output).not.toContain("✅ installed");
  });

  it("installs normally when SQLite is available", () => {
    const ctx = makeCtx();
    expect(run(["install", "devin-cli"], ctx)).toBe(0);
    expect(existsSync(join(ctx.home, ".config", "devin", "config.json"))).toBe(true);
    expect(ctx.nodeSqlite).toHaveBeenCalled();
  });

  it("blocks only the failing harness, still wiring the rest", () => {
    const ctx = makeCtx();
    ctx.nodeSqlite = vi.fn(() => false);
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);

    // Non-zero exit so a script notices, but Claude Code is still set up.
    expect(run(["install", "claude-code", "devin-cli"], ctx)).toBe(1);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(true);
    expect(existsSync(join(ctx.home, ".config", "devin", "config.json"))).toBe(false);
    expect(logs.join("\n")).toContain("not installed: devin-cli");
  });

  it("does not block uninstall", () => {
    const ctx = makeCtx();
    run(["install", "devin-cli"], ctx);
    ctx.nodeSqlite = vi.fn(() => false);
    expect(run(["uninstall", "devin-cli"], ctx)).toBe(0);
  });
});
