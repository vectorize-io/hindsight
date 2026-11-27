import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { INSTALLERS, run, type InstallCtx } from "./installer";

// #6958: a server the USER registered as "hindsight" (another deployment, another bank) must
// survive install --mcp-name + uninstall byte-for-byte, on every kind of host.
const USERS_SERVER = { command: "uvx", args: ["hindsight-local-mcp"], env: { BANK: "support" } };

let home: string;
const ctx = (): InstallCtx & {
  claudeMcp: ReturnType<typeof vi.fn>;
  qwenMcp: ReturnType<typeof vi.fn>;
} => ({
  home,
  pkgRoot: "/opt/hindsight-coding-agents",
  dist: "/opt/hindsight-coding-agents/dist",
  claudeMcp: vi.fn(() => true),
  qwenMcp: vi.fn(() => true),
  nodeSqlite: () => true,
  readLegacy: () => undefined,
  interactive: false,
});
const write = (p: string, v: string) => {
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, v);
};
const read = (p: string) => readFileSync(p, "utf8");
const install = (c: InstallCtx, ...h: string[]) =>
  run(
    [
      "install",
      ...h,
      "--server",
      "self-hosted",
      "--api-url",
      "http://x",
      "--mcp-name",
      "hindsight-code",
    ],
    c
  );

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "mcp-name-"));
  vi.stubEnv("HINDSIGHT_CONFIG", "");
});
afterEach(() => {
  rmSync(home, { recursive: true, force: true });
  vi.unstubAllEnvs();
});

describe("--mcp-name leaves the user's own 'hindsight' server alone", () => {
  it("JSON host (cursor): ours goes under the new name; theirs survives install + uninstall", () => {
    const path = join(home, ".cursor", "mcp.json");
    const before = JSON.stringify({ mcpServers: { hindsight: USERS_SERVER } }, null, 2) + "\n";
    write(path, before);
    const c = ctx();
    expect(install(c, "cursor-cli")).toBe(0);
    const servers = JSON.parse(read(path)).mcpServers;
    expect(servers.hindsight).toEqual(USERS_SERVER);
    expect(servers["hindsight-code"].args[0]).toBe(
      "/opt/hindsight-coding-agents/dist/mcp-server.js"
    );
    // the name is remembered: uninstall needs no flag
    expect(JSON.parse(read(join(home, ".hindsight", "coding-agent.json"))).mcpServerName).toBe(
      "hindsight-code"
    );
    expect(run(["uninstall", "cursor-cli"], c)).toBe(0);
    expect(JSON.parse(read(path)).mcpServers).toEqual({ hindsight: USERS_SERVER });
  });

  it("JSON host: plain uninstall never deletes a same-named server we did not write", () => {
    const path = join(home, ".copilot", "mcp-config.json");
    write(path, JSON.stringify({ mcpServers: { hindsight: USERS_SERVER } }));
    expect(run(["uninstall", "copilot-cli"], ctx())).toBe(0);
    expect(JSON.parse(read(path)).mcpServers.hindsight).toEqual(USERS_SERVER);
  });

  it("re-install under a new name moves our entry instead of leaving a second copy", () => {
    const path = join(home, ".cursor", "mcp.json");
    const c = ctx();
    run(["install", "cursor-cli", "--server", "self-hosted", "--api-url", "http://x"], c);
    expect(Object.keys(JSON.parse(read(path)).mcpServers)).toEqual(["hindsight"]);
    install(c, "cursor-cli");
    expect(Object.keys(JSON.parse(read(path)).mcpServers)).toEqual(["hindsight-code"]);
  });

  it("TOML host (codex): the user's [mcp_servers.hindsight] table survives byte-for-byte", () => {
    const path = join(home, ".codex", "config.toml");
    const users = `[mcp_servers.hindsight]\ncommand = "uvx"\nargs = ["hindsight-local-mcp"]\n`;
    write(path, users);
    const c = ctx();
    expect(install(c, "codex")).toBe(0);
    expect(read(path)).toContain(users);
    expect(read(path)).toContain("[mcp_servers.hindsight-code]");
    run(["uninstall", "codex"], c);
    expect(read(path)).toContain(users);
    expect(read(path)).not.toContain("hindsight-code");
  });

  it("CLI host (claude-code): never runs `claude mcp remove hindsight` on the user's server", () => {
    write(join(home, ".claude.json"), JSON.stringify({ mcpServers: { hindsight: USERS_SERVER } }));
    const c = ctx();
    install(c, "claude-code");
    run(["uninstall", "claude-code"], c);
    const calls = c.claudeMcp.mock.calls.map(([a]) => a.join(" "));
    expect(calls.filter((a) => a.startsWith("mcp remove"))).toEqual([]);
    expect(calls.some((a) => a.includes("add --scope user hindsight-code"))).toBe(true);
  });

  it("preflight host (factory-droid): --mcp-name is the way past a user-owned 'hindsight'", () => {
    const path = join(home, ".factory", "mcp.json");
    write(path, JSON.stringify({ mcpServers: { hindsight: USERS_SERVER } }));
    const logs: string[] = [];
    const blocked = { ...ctx(), log: (m: string) => logs.push(m) };
    expect(
      run(["install", "factory-droid", "--server", "self-hosted", "--api-url", "http://x"], blocked)
    ).toBe(1);
    expect(logs.join("\n")).toContain("--mcp-name");
    expect(install(ctx(), "factory-droid")).toBe(0);
    expect(JSON.parse(read(path)).mcpServers.hindsight).toEqual(USERS_SERVER);
  });

  it("rejects a name that is unsafe as a TOML key or tool prefix, before writing anything", () => {
    expect(run(["install", "cursor-cli", "--mcp-name", "my server"], ctx())).toBe(1);
    expect(existsSync(join(home, ".cursor"))).toBe(false);
  });
});

it("every harness either declares where its MCP servers live, or has none the installer writes", () => {
  // In-process hosts load our plugin and register no MCP server by name; Dcode's ships in
  // plugin.json, namespaced by the plugin; TraeCode's is per repo (core/traecode-mcp.ts), which
  // skips a foreign same-named entry itself.
  const NO_NAMED_REGISTRATION = ["opencode", "opencode2", "kilo", "pi", "dsh", "dcode", "traecode"];
  const without = INSTALLERS.filter((i) => !i.mcpRegistry).map((i) => i.name);
  expect(without.sort()).toEqual([...NO_NAMED_REGISTRATION].sort());
});

// Every host the installer registers a server in, enumerated so a new one cannot skip the rule.
describe.each(INSTALLERS.filter((i) => i.mcpRegistry).map((i) => i.name))("%s", (harness) => {
  const registry = (c: InstallCtx) => INSTALLERS.find((i) => i.name === harness)!.mcpRegistry!(c);
  const serversOf = (c: InstallCtx): Record<string, any> => {
    const { servers } = registry(c);
    return (servers ?? {}) as Record<string, any>;
  };
  /** Put the user's own "hindsight" server where this host keeps its servers. */
  const seedUsers = (c: InstallCtx) => {
    const { path } = registry(c);
    if (path.endsWith(".toml")) {
      write(path, `[mcp_servers.hindsight]\ncommand = "uvx"\nargs = ["hindsight-local-mcp"]\n`);
    } else if (harness === "zcode") {
      write(path, JSON.stringify({ mcp: { servers: { hindsight: USERS_SERVER } } }));
    } else {
      write(path, JSON.stringify({ mcpServers: { hindsight: USERS_SERVER } }));
    }
  };
  const users = (c: InstallCtx) => {
    const entry = serversOf(c).hindsight;
    return entry && { command: entry.command, args: entry.args };
  };

  it("registers under --mcp-name, never under the default", () => {
    const c = ctx();
    expect(install(c, harness)).toBe(0);
    const registered = JSON.stringify([serversOf(c), c.claudeMcp.mock.calls, c.qwenMcp.mock.calls]);
    expect(registered).toContain("hindsight-code");
    expect(serversOf(c)).not.toHaveProperty("hindsight");
  });

  it("refuses to take a name the user's own server holds, and leaves the file alone", () => {
    const c = ctx();
    seedUsers(c);
    const before = read(registry(c).path);
    const logs: string[] = [];
    const args = ["install", harness, "--server", "self-hosted", "--api-url", "http://x"];
    expect(run(args, { ...c, log: (m) => logs.push(m) })).toBe(1);
    expect(logs.join("\n")).toContain("--mcp-name");
    expect(read(registry(c).path)).toBe(before);
  });

  it("install --mcp-name then uninstall keeps the user's server", () => {
    const c = ctx();
    seedUsers(c);
    const theirs = users(c);
    expect(install(c, harness)).toBe(0);
    expect(run(["uninstall", harness], c)).toBe(0);
    expect(users(c)).toEqual(theirs);
    const removes = [...c.claudeMcp.mock.calls, ...c.qwenMcp.mock.calls]
      .map(([a]) => a.join(" "))
      .filter((a) => a.startsWith("mcp remove"));
    expect(removes).toEqual([]);
  });
});
