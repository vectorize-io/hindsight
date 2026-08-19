import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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
    // Never let a developer's real ~/.hindsight/claude-code.json steer the tests.
    readLegacy: () => undefined,
  };
}

function readJson(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, "utf8"));
}

function writeJsonAt(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
}

// configureServer honors HINDSIGHT_CONFIG — a developer shell exporting it must not leak the
// suite's --server writes into their real config file ("" is falsy → the per-test home is used).
beforeEach(() => {
  vi.stubEnv("HINDSIGHT_CONFIG", "");
});

afterEach(() => {
  vi.unstubAllEnvs();
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

describe("dsh installer", () => {
  const patchPath = (ctx: InstallCtx) => join(ctx.home, ".dsh", "cordis.patch.yml");

  // The harness home is env-driven; pin it to the test home so a developer's real $DSH_HOME
  // (or a CI runner's) can never be the thing this suite writes to.
  beforeEach(() => vi.stubEnv("DSH_HOME", ""));
  afterEach(() => vi.unstubAllEnvs());

  it("registers the plugin as a file:// row in the home patch layer", () => {
    const ctx = makeCtx();
    expect(run(["install", "dsh"], ctx)).toBe(0);
    const patch = readFileSync(patchPath(ctx), "utf8");
    // A bare absolute path is not a module specifier: Cordis would fail to resolve it and skip
    // the plugin silently, which is exactly the Kilo trap this asserts against.
    expect(patch).toContain(`name: "${pathToFileURL(join(ctx.dist, "dsh.js")).href}"`);
    expect(patch).toContain("- id: hindsight");
  });

  it("replaces its own block on re-install and preserves the user's other patches", () => {
    const ctx = makeCtx();
    mkdirSync(dirname(patchPath(ctx)), { recursive: true });
    writeFileSync(patchPath(ctx), "- id: llm\n  config:\n    provider: deepseek\n");
    run(["install", "dsh"], ctx);
    run(["install", "dsh"], ctx);
    const patch = readFileSync(patchPath(ctx), "utf8");
    expect(patch).toContain("provider: deepseek");
    expect(patch.match(/- id: hindsight/g)).toHaveLength(1);
  });

  it("uninstall leaves a valid empty patch list rather than an unparsable file", () => {
    const ctx = makeCtx();
    run(["install", "dsh"], ctx);
    run(["uninstall", "dsh"], ctx);
    // dsh REQUIRES this file to parse to a top-level array and fails BOOT otherwise, so an
    // emptied file must still be `[]`.
    expect(readFileSync(patchPath(ctx), "utf8").trim()).toBe("[]");
  });

  it("uninstall keeps the user's own patches", () => {
    const ctx = makeCtx();
    run(["install", "dsh"], ctx);
    const kept = `- id: llm\n  config:\n    provider: deepseek\n`;
    writeFileSync(patchPath(ctx), kept + readFileSync(patchPath(ctx), "utf8"));
    run(["uninstall", "dsh"], ctx);
    const patch = readFileSync(patchPath(ctx), "utf8");
    expect(patch).toContain("provider: deepseek");
    expect(patch).not.toContain("hindsight");
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

describe("prime-agent installer", () => {
  const cfgPath = (ctx: InstallCtx) => join(ctx.home, ".prime", "agent", "settings.json");
  const entry = (ctx: InstallCtx) => join(ctx.pkgRoot, "dist", "prime-agent.js");

  it("install adds the built extension to the extensions array exactly once, even across reinstalls", () => {
    const ctx = makeCtx();
    expect(run(["install", "prime-agent"], ctx)).toBe(0);
    run(["install", "prime-agent"], ctx);
    expect(readJson(cfgPath(ctx)).extensions).toEqual([entry(ctx)]);
  });

  it("preserves other extension entries", () => {
    const ctx = makeCtx();
    writeJsonAt(cfgPath(ctx), { extensions: ["/some/other/ext.js"] });
    run(["install", "prime-agent"], ctx);
    expect(readJson(cfgPath(ctx)).extensions).toEqual(["/some/other/ext.js", entry(ctx)]);
  });

  it("uninstall removes our entry and deletes the extensions key when empty", () => {
    const ctx = makeCtx();
    run(["install", "prime-agent"], ctx);
    run(["uninstall", "prime-agent"], ctx);
    expect(readJson(cfgPath(ctx)).extensions).toBeUndefined();
  });

  it("uninstall keeps the extensions key when other entries remain", () => {
    const ctx = makeCtx();
    writeJsonAt(cfgPath(ctx), { extensions: ["/some/other/ext.js"] });
    run(["install", "prime-agent"], ctx);
    run(["uninstall", "prime-agent"], ctx);
    expect(readJson(cfgPath(ctx)).extensions).toEqual(["/some/other/ext.js"]);
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
      "prime-agent",
      "claude-code",
      "codex",
      "antigravity-cli",
      "devin-cli",
      "cursor-cli",
      "copilot-cli",
      "grok-build",
      "cline-cli",
      "dsh",
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

/**
 * Running from an npx cache used to be refused: the wiring is absolute paths into this package, and
 * a cache eviction would break every hook silently. The runtime is now copied somewhere stable
 * first, so `npx` works and nobody needs a global install of a tool that only sets other tools up.
 */
describe("runtime staging", () => {
  /** A package layout convincing enough to be staged: staging keys off a built dist. */
  function fakePackage(root: string): { pkgRoot: string; dist: string } {
    const dist = join(root, "dist");
    mkdirSync(dist, { recursive: true });
    writeFileSync(join(dist, "installer.js"), "// built");
    writeFileSync(join(dist, "claude-hook.js"), "// built");
    writeFileSync(join(root, "package.json"), JSON.stringify({ name: "x", main: "dist/index.js" }));
    mkdirSync(join(root, "skill"), { recursive: true });
    writeFileSync(join(root, "skill", "SKILL.md"), "# skill");
    return { pkgRoot: root, dist };
  }

  it("installs from an npx cache, wiring the stable copy instead of the cache", () => {
    const ctx = makeCtx();
    const cache = mkdtempSync(join(tmpdir(), "npx-cache-"));
    homes.push(cache);
    Object.assign(ctx, fakePackage(join(cache, "_npx", "abc123", "node_modules", "coding-agents")));

    expect(run(["install", "claude-code"], ctx)).toBe(0);
    const staged = join(ctx.home, ".hindsight", "coding-agents");
    const command = readJson(join(ctx.home, ".claude", "settings.json")).hooks.SessionStart[0]
      .hooks[0].command as string;
    expect(command).toContain(join(staged, "dist"));
    // The whole point: nothing in a host config may reference the evictable cache.
    expect(command).not.toContain("_npx");
    expect(existsSync(join(staged, "dist", "claude-hook.js"))).toBe(true);
  });

  // MARKER matching is what makes re-install replace and uninstall remove, and it looks for this
  // substring in the command path — so the staged location must keep it.
  it("stages somewhere the marker still matches", () => {
    const ctx = makeCtx();
    const src = mkdtempSync(join(tmpdir(), "pkg-"));
    homes.push(src);
    Object.assign(ctx, fakePackage(src));

    run(["install", "claude-code"], ctx);
    expect(join(ctx.home, ".hindsight", "coding-agents")).toContain(MARKER);
    run(["uninstall", "claude-code"], ctx);
    expect(readJson(join(ctx.home, ".claude", "settings.json")).hooks).toBeUndefined();
  });

  it("copies the plugin entry point too, since opencode loads the directory", () => {
    const ctx = makeCtx();
    const src = mkdtempSync(join(tmpdir(), "pkg-"));
    homes.push(src);
    Object.assign(ctx, fakePackage(src));

    run(["install", "opencode"], ctx);
    const staged = join(ctx.home, ".hindsight", "coding-agents");
    expect(existsSync(join(staged, "package.json"))).toBe(true);
    expect(existsSync(join(staged, "skill", "SKILL.md"))).toBe(true);
    const cfg = readJson(join(ctx.home, ".config", "opencode", "opencode.json"));
    expect(cfg.plugin).toContain(staged);
  });

  it("upgrading replaces the staged runtime, stale files and all", () => {
    const ctx = makeCtx();
    const v1 = mkdtempSync(join(tmpdir(), "v1-"));
    const v2 = mkdtempSync(join(tmpdir(), "v2-"));
    homes.push(v1, v2);
    fakePackage(v1);
    writeFileSync(join(v1, "dist", "old-only.js"), "// dropped in the next version");
    fakePackage(v2);
    writeFileSync(join(v2, "dist", "new-only.js"), "// added in the next version");

    Object.assign(ctx, { pkgRoot: v1, dist: join(v1, "dist") });
    run(["install", "claude-code"], ctx);
    Object.assign(ctx, { pkgRoot: v2, dist: join(v2, "dist") });
    run(["install", "claude-code"], ctx);

    const staged = join(ctx.home, ".hindsight", "coding-agents", "dist");
    expect(existsSync(join(staged, "new-only.js"))).toBe(true);
    // Merging instead of replacing would leave an entry point a host config could still name.
    expect(existsSync(join(staged, "old-only.js"))).toBe(false);
    const events = readJson(join(ctx.home, ".claude", "settings.json")).hooks.SessionStart;
    expect(events).toHaveLength(1);
  });

  // Re-running the STAGED installer must not delete the dist it is executing from.
  it("is a no-op when run from the staged copy itself", () => {
    const ctx = makeCtx();
    const src = mkdtempSync(join(tmpdir(), "pkg-"));
    homes.push(src);
    fakePackage(src);
    Object.assign(ctx, { pkgRoot: src, dist: join(src, "dist") });
    run(["install", "claude-code"], ctx);

    const staged = join(ctx.home, ".hindsight", "coding-agents");
    Object.assign(ctx, { pkgRoot: staged, dist: join(staged, "dist") });
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    expect(existsSync(join(staged, "dist", "installer.js"))).toBe(true);
  });

  // A checkout whose dist was never built has nothing to copy; wiring the source path is better
  // than pointing every hook at a directory that does not exist.
  it("wires in place when there is nothing to stage", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    const command = readJson(join(ctx.home, ".claude", "settings.json")).hooks.SessionStart[0]
      .hooks[0].command as string;
    expect(command).toContain(ctx.dist);
    expect(existsSync(join(ctx.home, ".hindsight", "coding-agents", "dist"))).toBe(false);
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

/**
 * Choosing where memory lives — Cloud, a self-hosted server, or a local daemon. Asked once, at
 * install time; `--server` is the non-interactive form and the only one the suite uses (a prompt
 * would block on stdin).
 */
describe("server setup", () => {
  const configPath = (ctx: InstallCtx) => join(ctx.home, ".hindsight", "coding-agent.json");

  // The runtime reads HINDSIGHT_CONFIG first (core/config.ts CONFIG_PATH); the wizard must write
  // that same file, or a user with the var set is configured into a file sessions never read.
  it("honors HINDSIGHT_CONFIG for both the already-configured check and the write", () => {
    const ctx = makeCtx();
    const override = join(ctx.home, "elsewhere", "config.json");
    vi.stubEnv("HINDSIGHT_CONFIG", override);
    try {
      expect(run(["install", "claude-code", "--server", "daemon"], ctx)).toBe(0);
      expect(readJson(override).serverMode).toBe("daemon");
      expect(existsSync(configPath(ctx))).toBe(false); // the default path stays untouched
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("uses the injected arrow-key picker when interactive, mapping index → mode", () => {
    const ctx = makeCtx();
    ctx.interactive = true;
    ctx.hasUvx = () => true;
    ctx.hasRust = () => true;
    ctx.detectLlm = () => ({ provider: "openai", apiKey: "sk", source: "OPENAI_API_KEY" });
    ctx.selectPrompt = vi.fn(() => 2); // third row = daemon
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    expect(ctx.selectPrompt).toHaveBeenCalledOnce();
    expect(readJson(configPath(ctx)).serverMode).toBe("daemon");
  });

  it("a cancelled picker leaves the server config untouched but still installs", () => {
    const ctx = makeCtx();
    ctx.interactive = true;
    ctx.selectPrompt = () => null;
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    expect(existsSync(configPath(ctx))).toBe(false);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(true);
  });

  it("--server daemon records the mode and leaves apiUrl to the port", () => {
    const ctx = makeCtx();
    ctx.hasUvx = () => true;
    ctx.detectLlm = () => ({ provider: "openai", apiKey: "sk", source: "OPENAI_API_KEY" });
    expect(run(["install", "claude-code", "--server", "daemon"], ctx)).toBe(0);
    const cfg = readJson(configPath(ctx));
    expect(cfg.serverMode).toBe("daemon");
    expect(cfg.apiUrl).toBeUndefined();
  });

  it("--server self-hosted stores the URL", () => {
    const ctx = makeCtx();
    expect(
      run(
        ["install", "claude-code", "--server", "self-hosted", "--api-url", "http://box:8888"],
        ctx
      )
    ).toBe(0);
    expect(readJson(configPath(ctx)).apiUrl).toBe("http://box:8888");
  });

  // Without a URL the mode is unusable, and silently falling back to Cloud would send this user's
  // prompts somewhere they did not choose.
  it("self-hosted without a URL fails instead of falling back to Cloud", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code", "--server", "self-hosted"], ctx)).toBe(1);
    expect(logs.join("\n")).toContain("--api-url");
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(false);
  });

  it("rejects an unknown mode", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code", "--server", "hybrid"], ctx)).toBe(1);
    expect(logs.join("\n")).toContain("cloud, self-hosted, daemon");
  });

  // `--server daemon` puts a bare word in argv; without value-aware parsing it reads as a harness.
  it("does not mistake a flag value for a harness name", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code", "--server", "daemon"], ctx)).toBe(0);
    expect(logs.join("\n")).not.toContain('unknown harness "daemon"');
  });

  // install is idempotent and routinely re-run; it must not silently rewrite a working setup.
  it("leaves an existing server config alone", () => {
    const ctx = makeCtx();
    writeJsonAt(configPath(ctx), { serverMode: "self-hosted", apiUrl: "http://mine:8888" });
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    expect(readJson(configPath(ctx)).apiUrl).toBe("http://mine:8888");
  });

  it("warns when daemon prerequisites are missing, but still configures it", () => {
    const ctx = makeCtx();
    ctx.hasUvx = () => false;
    ctx.hasRust = () => true;
    ctx.detectLlm = () => undefined;
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    // Advisory, not blocking: uv and an API key can both be installed after the fact.
    expect(run(["install", "claude-code", "--server", "daemon"], ctx)).toBe(0);
    const out = logs.join("\n");
    expect(out).toContain("uv");
    expect(out).toContain("OPENAI_API_KEY");
    expect(readJson(configPath(ctx)).serverMode).toBe("daemon");
  });

  // Coming from the old per-agent plugin, the endpoint is already a decision the user made.
  // Defaulting to Cloud instead would quietly redirect their prompts to a different server.
  it("adopts the old plugin's endpoint instead of asking or defaulting to Cloud", () => {
    const ctx = makeCtx();
    ctx.readLegacy = () => ({
      harness: "claude-code",
      serverMode: "self-hosted" as const,
      apiUrl: "http://legacy:8888",
      apiToken: "tok",
      source: "/home/u/.hindsight/claude-code.json",
    });
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    const cfg = readJson(configPath(ctx));
    expect(cfg.serverMode).toBe("self-hosted");
    expect(cfg.apiUrl).toBe("http://legacy:8888");
    expect(cfg.apiToken).toBe("tok");
    // Conversations are a separate, opt-in step — say so rather than implying a full migration.
    expect(logs.join("\n")).toContain("--import-conversations");
  });

  it("an explicit --server still overrides what the old plugin used", () => {
    const ctx = makeCtx();
    ctx.readLegacy = () => ({
      harness: "claude-code",
      serverMode: "self-hosted" as const,
      apiUrl: "http://legacy:8888",
      source: "/x",
    });
    expect(run(["install", "claude-code", "--server", "cloud", "--api-token", "tok"], ctx)).toBe(0);
    const cfg = readJson(configPath(ctx));
    expect(cfg.serverMode).toBe("cloud");
    expect(cfg.apiUrl).toBeUndefined();
  });

  it("--server cloud stores the required token", () => {
    const ctx = makeCtx();
    expect(
      run(["install", "claude-code", "--server", "cloud", "--api-token", "sk-cloud"], ctx)
    ).toBe(0);
    const cfg = readJson(configPath(ctx));
    expect(cfg.serverMode).toBe("cloud");
    expect(cfg.apiToken).toBe("sk-cloud");
  });

  // A Cloud config without a token only surfaces later as 401s on the first session — refuse
  // up front instead, like self-hosted without a URL.
  it("--server cloud without a token fails instead of writing a config that 401s", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code", "--server", "cloud"], ctx)).toBe(1);
    expect(logs.join("\n")).toContain("--api-token");
    expect(existsSync(configPath(ctx))).toBe(false);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(false);
  });

  // litellm publishes no macOS wheel, so a Mac compiles it from source and needs cargo. Without
  // this the failure surfaces minutes later, deep in a pip build log.
  it("flags a missing Rust toolchain", () => {
    const ctx = makeCtx();
    ctx.hasUvx = () => true;
    ctx.hasRust = () => false;
    ctx.detectLlm = () => ({ provider: "openai", apiKey: "sk", source: "OPENAI_API_KEY" });
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code", "--server", "daemon"], ctx)).toBe(0);
    expect(logs.join("\n")).toContain("rustup");
  });
});
