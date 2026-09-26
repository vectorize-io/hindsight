import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, beforeAll, afterAll, describe, expect, it } from "vitest";
import {
  enableWorkspaceMcpSetting,
  ensureTraecodeWorkspaceMcp,
  ensureWorkspaceMcpEnabled,
  markWorkspaceMcpEnabled,
  traecodeUserSettingsPath,
  traecodeWorkspaceEnabledKey,
  workspaceEnableHint,
  workspaceMcpGateState,
  workspaceMcpHint,
} from "./traecode-mcp";
import { HOOK_HARNESSES } from "../harness/hook-lifecycle";

/** Pin the userData-env vars to their defaults so path resolution lands inside the temp homes on
 *  every platform (a runner-set XDG_CONFIG_HOME would point the Trae-brand probe elsewhere). */
let savedEnv: Record<string, string | undefined>;
beforeAll(() => {
  savedEnv = { APPDATA: process.env.APPDATA, XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME };
  delete process.env.APPDATA;
  delete process.env.XDG_CONFIG_HOME;
});
afterAll(() => {
  for (const [k, v] of Object.entries(savedEnv)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
});

/** Mirrors traecodeUserDataDir's root choice (env pinned by the beforeAll above). */
const userDataRoot = (home: string) =>
  process.platform === "darwin"
    ? join(home, "Library", "Application Support")
    : process.platform === "win32"
      ? join(home, "AppData", "Roaming")
      : join(home, ".config");

/** The seed shells out to the system sqlite3 — absent on some runners, so the DB-backed tests
 *  gate on a probe instead of assuming. */
const hasSqlite3 = (() => {
  try {
    execFileSync("sqlite3", ["--version"], { stdio: ["ignore", "pipe", "pipe"] });
    return true;
  } catch {
    return false;
  }
})();

describe("ensureTraecodeWorkspaceMcp", () => {
  const dirs: string[] = [];
  const tmp = (p: string) => {
    const d = mkdtempSync(join(tmpdir(), p));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });

  /** A repo dir plus a fake dist carrying an (empty) mcp-server.js, and an isolated HOME so the
   *  hint's gate check never touches the developer's real Trae settings or the dev-tree state. */
  const repo = () => {
    const dir = tmp("traecode-mcp-repo-");
    const dist = tmp("traecode-mcp-dist-");
    const home = tmp("traecode-mcp-home-");
    writeFileSync(join(dist, "mcp-server.js"), "// stub");
    return { repo: dir, dist, home };
  };
  const mcpFile = (r: string) => join(r, ".trae", "mcp.json");
  const readJson = (r: string) => JSON.parse(readFileSync(mcpFile(r), "utf8"));

  it("writes the workspace registration when the file is absent", () => {
    const { repo: r, dist, home } = repo();
    ensureTraecodeWorkspaceMcp(r, { dist, home });
    const doc = readJson(r);
    expect(doc.mcpServers.hindsight).toEqual({
      command: "node",
      args: [join(dist, "mcp-server.js")],
      env: { HINDSIGHT_MCP_HARNESS: "traecode", HINDSIGHT_MCP_PROJECT_CWD: r },
    });
  });

  it("is idempotent: a second run rewrites nothing", () => {
    const { repo: r, dist, home } = repo();
    ensureTraecodeWorkspaceMcp(r, { dist, home });
    const before = readFileSync(mcpFile(r), "utf8");
    ensureTraecodeWorkspaceMcp(r, { dist, home });
    expect(readFileSync(mcpFile(r), "utf8")).toBe(before);
  });

  it("preserves sibling servers and merges env into an existing OUR entry", () => {
    const { repo: r, dist, home } = repo();
    mkdirSync(join(r, ".trae"), { recursive: true });
    writeFileSync(
      mcpFile(r),
      JSON.stringify({
        mcpServers: {
          hindsight: {
            command: "node",
            args: ["/old/hindsight-coding-agents/dist/mcp-server.js"],
            env: { HINDSIGHT_MCP_HARNESS: "traecode", MY_EXTRA: "keep" },
          },
          other: { command: "uvx", args: ["something"] },
        },
      })
    );
    ensureTraecodeWorkspaceMcp(r, { dist, home });
    const doc = readJson(r);
    expect(doc.mcpServers.other).toEqual({ command: "uvx", args: ["something"] });
    expect(doc.mcpServers.hindsight).toEqual({
      command: "node",
      args: [join(dist, "mcp-server.js")],
      env: {
        HINDSIGHT_MCP_HARNESS: "traecode",
        HINDSIGHT_MCP_PROJECT_CWD: r,
        MY_EXTRA: "keep",
      },
    });
  });

  it("never touches a foreign hindsight entry", () => {
    const { repo: r, dist, home } = repo();
    mkdirSync(join(r, ".trae"), { recursive: true });
    const foreign = { command: "python", args: ["-m", "my_hindsight"] };
    writeFileSync(mcpFile(r), JSON.stringify({ mcpServers: { hindsight: foreign } }));
    ensureTraecodeWorkspaceMcp(r, { dist, home });
    expect(readJson(r).mcpServers.hindsight).toEqual(foreign);
  });

  it("leaves an unparseable file alone", () => {
    const { repo: r, dist, home } = repo();
    mkdirSync(join(r, ".trae"), { recursive: true });
    writeFileSync(mcpFile(r), "{ not json");
    ensureTraecodeWorkspaceMcp(r, { dist, home });
    expect(readFileSync(mcpFile(r), "utf8")).toBe("{ not json");
  });

  describe("gitignore", () => {
    it("appends the ignore line to an existing .gitignore when creating the file", () => {
      const { repo: r, dist, home } = repo();
      writeFileSync(join(r, ".gitignore"), "node_modules\n");
      ensureTraecodeWorkspaceMcp(r, { dist, home });
      const gitignore = readFileSync(join(r, ".gitignore"), "utf8");
      expect(gitignore).toContain("node_modules\n");
      expect(gitignore).toContain(".trae/mcp.json");
    });

    it("creates no .gitignore when the repo has none", () => {
      const { repo: r, dist, home } = repo();
      ensureTraecodeWorkspaceMcp(r, { dist, home });
      expect(existsSync(join(r, ".gitignore"))).toBe(false);
    });

    it("does not append when .trae is already ignored", () => {
      const { repo: r, dist, home } = repo();
      writeFileSync(join(r, ".gitignore"), "node_modules\n.trae/\n");
      ensureTraecodeWorkspaceMcp(r, { dist, home });
      expect(readFileSync(join(r, ".gitignore"), "utf8")).toBe("node_modules\n.trae/\n");
    });
  });

  it("does nothing for home, root, or relative cwd", () => {
    const { repo: r, dist } = repo();
    ensureTraecodeWorkspaceMcp(homedir(), { dist });
    ensureTraecodeWorkspaceMcp("/", { dist });
    ensureTraecodeWorkspaceMcp("relative/path", { dist });
    expect(existsSync(join(homedir(), ".trae", "mcp.json"))).toBe(false);
    expect(existsSync(join(r, ".trae"))).toBe(false);
  });

  it("does nothing when the dist has no mcp-server.js (unbuilt tree)", () => {
    const { repo: r } = repo();
    const empty = tmp("traecode-mcp-empty-");
    ensureTraecodeWorkspaceMcp(r, { dist: empty });
    expect(existsSync(mcpFile(r))).toBe(false);
  });
});

/** The traecode SessionStart spec must carry the registration step; no other harness may. */
describe("traecode sessionStart spec wiring", () => {
  it("wires ensureMcpRegistration for traecode only", () => {
    const wired = Object.entries(HOOK_HARNESSES)
      .filter(([, spec]) => spec.sessionStart.ensureMcpRegistration !== undefined)
      .map(([name]) => name);
    expect(wired).toEqual(["traecode"]);
  });
});

// ── the workspace-MCP gate (trae.mcp.enableWorkspaceMcp) ──────────────────────────────────────

describe("workspace-MCP gate", () => {
  const dirs: string[] = [];
  const tmp = (p: string) => {
    const d = mkdtempSync(join(tmpdir(), p));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });

  /** A fake HOME with a Trae userData settings.json pre-written. */
  const homeWithSettings = (content?: string) => {
    const home = tmp("traecode-gate-home-");
    const settings = traecodeUserSettingsPath(home);
    if (content !== undefined) {
      mkdirSync(join(settings, ".."), { recursive: true });
      writeFileSync(settings, content);
    }
    return { home, settings };
  };

  describe("workspaceMcpGateState", () => {
    it("reads plain JSON settings", () => {
      expect(
        workspaceMcpGateState(homeWithSettings('{"trae.mcp.enableWorkspaceMcp":true}').settings)
      ).toBe("on");
      expect(workspaceMcpGateState(homeWithSettings("{}").settings)).toBe("off");
      expect(
        workspaceMcpGateState(homeWithSettings('{"trae.mcp.enableWorkspaceMcp":false}').settings)
      ).toBe("off");
    });

    it("falls back to a targeted regex for comment-bearing (JSONC) files", () => {
      expect(
        workspaceMcpGateState(
          homeWithSettings('{\n  // comment\n  "trae.mcp.enableWorkspaceMcp": true\n}').settings
        )
      ).toBe("on");
      expect(workspaceMcpGateState(homeWithSettings("{\n  // just a comment\n}").settings)).toBe(
        "unknown"
      );
    });

    it("reports unknown for an absent file", () => {
      expect(workspaceMcpGateState(join(tmp("traecode-gate-x-"), "nope.json"))).toBe("unknown");
    });
  });

  describe("enableWorkspaceMcpSetting", () => {
    it("merges into an existing plain-JSON settings file, preserving keys", () => {
      const { settings } = homeWithSettings('{"workbench.colorTheme":"Dark"}');
      expect(enableWorkspaceMcpSetting(settings)).toBe(true);
      const doc = JSON.parse(readFileSync(settings, "utf8"));
      expect(doc["trae.mcp.enableWorkspaceMcp"]).toBe(true);
      expect(doc["workbench.colorTheme"]).toBe("Dark");
    });

    it("creates the file when absent", () => {
      const { settings } = homeWithSettings();
      expect(enableWorkspaceMcpSetting(settings)).toBe(true);
      expect(JSON.parse(readFileSync(settings, "utf8"))["trae.mcp.enableWorkspaceMcp"]).toBe(true);
    });

    it("refuses to rewrite a comment-bearing file", () => {
      const raw = '{\n  // user comments\n  "a": 1\n}';
      const { settings } = homeWithSettings(raw);
      expect(enableWorkspaceMcpSetting(settings)).toBe(false);
      expect(readFileSync(settings, "utf8")).toBe(raw);
    });
  });

  describe("workspaceMcpHint", () => {
    const stateFile = (dir: string) => join(dir, "state.json");

    it("is silent when the gate reads on", () => {
      const { home } = homeWithSettings('{"trae.mcp.enableWorkspaceMcp":true}');
      expect(workspaceMcpHint({ home, stateFile: stateFile(tmp("s-")) })).toBeUndefined();
    });

    it("hints when the gate reads off, at most once a day", () => {
      const { home } = homeWithSettings("{}");
      const state = stateFile(tmp("s-"));
      expect(workspaceMcpHint({ home, stateFile: state, now: 1_000 })).toMatch(
        /enableWorkspaceMcp/
      );
      expect(workspaceMcpHint({ home, stateFile: state, now: 2_000 })).toBeUndefined();
      expect(workspaceMcpHint({ home, stateFile: state, now: 86_500_000 })).toMatch(
        /enableWorkspaceMcp/
      );
      expect(JSON.parse(readFileSync(state, "utf8")).lastHint).toBe(86_500_000);
    });

    it("stays silent on an unreadable settings file when the installer snapshot says enabled", () => {
      const { home } = homeWithSettings(); // no settings file: "unknown"
      const state = stateFile(tmp("s-"));
      markWorkspaceMcpEnabled({ stateFile: state });
      expect(workspaceMcpHint({ home, stateFile: state, now: 1_000 })).toBeUndefined();
    });

    it("hints on an unreadable settings file with no snapshot", () => {
      const { home } = homeWithSettings();
      expect(workspaceMcpHint({ home, stateFile: stateFile(tmp("s-")), now: 1_000 })).toMatch(
        /enableWorkspaceMcp/
      );
    });
  });
});

// ── the per-workspace enable switch (workspaceStorage state.vscdb) ────────────────────────────

describe("workspace enable switch seed", () => {
  const dirs: string[] = [];
  const tmp = (p: string) => {
    const d = mkdtempSync(join(tmpdir(), `traecode-enable-${p}`));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });

  /** A fake HOME whose Trae CN userData carries one workspaceStorage entry for `repo`, with an
   *  ItemTable pre-seeded via (key, value) pairs. Returns the DB path. */
  const homeWithWorkspace = (repo: string, rows: [string, string][] = []) => {
    const home = tmp("home-");
    const ws = join(userDataRoot(home), "Trae CN", "User", "workspaceStorage", "hash1");
    mkdirSync(ws, { recursive: true });
    writeFileSync(join(ws, "workspace.json"), JSON.stringify({ folder: pathToFileURL(repo).href }));
    const db = join(ws, "state.vscdb");
    execFileSync("sqlite3", [
      db,
      "CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB);",
      ...rows.map(
        ([k, v]) => `INSERT OR REPLACE INTO ItemTable (key, value) VALUES ('${k}', '${v}');`
      ),
    ]);
    return { home, db };
  };
  const query = (db: string, key: string) =>
    execFileSync("sqlite3", [db, `SELECT value FROM ItemTable WHERE key='${key}';`], {
      encoding: "utf8",
    }).trim();

  it.runIf(hasSqlite3)("seeds the switch in the matching workspace's DB, then reports on", () => {
    const repo = tmp("repo-");
    const { home, db } = homeWithWorkspace(repo, [["some.other.key", "1"]]);
    expect(ensureWorkspaceMcpEnabled(repo, { home })).toBe("seeded");
    expect(query(db, traecodeWorkspaceEnabledKey())).toBe("true");
    expect(query(db, "some.other.key")).toBe("1"); // the rest of the table is untouched
    expect(ensureWorkspaceMcpEnabled(repo, { home })).toBe("on"); // idempotent skip
  });

  it.runIf(hasSqlite3)("matches the repo by its file:// URL only", () => {
    const { home } = homeWithWorkspace(tmp("other-"));
    expect(ensureWorkspaceMcpEnabled(tmp("unrelated-"), { home })).toBe("failed");
  });

  it.runIf(hasSqlite3)("fails (and does not throw) when the DB is unreadable", () => {
    const repo = tmp("repo-");
    const { home, db } = homeWithWorkspace(repo);
    writeFileSync(db, "this is not a database"); // sqlite3 errors on open/query
    expect(ensureWorkspaceMcpEnabled(repo, { home })).toBe("failed");
  });

  it("fails without any workspaceStorage entry for the repo", () => {
    const repo = tmp("repo-");
    expect(ensureWorkspaceMcpEnabled(repo, { home: tmp("home-") })).toBe("failed");
  });
});

describe("workspaceEnableHint", () => {
  const dirs: string[] = [];
  const tmp = (p: string) => {
    const d = mkdtempSync(join(tmpdir(), `traecode-enablehint-${p}`));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });
  const stateFile = (dir: string) => join(dir, "state.json");

  it("hints at most once a day, on its own rate-limit field", () => {
    const state = stateFile(tmp("s-"));
    expect(workspaceEnableHint({ stateFile: state, now: 1_000 })).toMatch(/MCP panel/);
    expect(workspaceEnableHint({ stateFile: state, now: 2_000 })).toBeUndefined();
    expect(workspaceEnableHint({ stateFile: state, now: 86_500_000 })).toMatch(/MCP panel/);
    const saved = JSON.parse(readFileSync(state, "utf8"));
    expect(saved.lastEnableHint).toBe(86_500_000);
    expect(saved.lastHint).toBeUndefined(); // independent of the gate hint's field
  });
});

describe("ensureTraecodeWorkspaceMcp hint selection", () => {
  const dirs: string[] = [];
  const tmp = (p: string) => {
    const d = mkdtempSync(join(tmpdir(), `traecode-select-${p}`));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });
  const stateFile = (dir: string) => join(dir, "state.json");
  const homeWithSettings = (content: string) => {
    const home = tmp("home-");
    const settings = traecodeUserSettingsPath(home);
    mkdirSync(dirname(settings), { recursive: true });
    writeFileSync(settings, content);
    return home;
  };
  const dist = () => {
    const d = tmp("dist-");
    writeFileSync(join(d, "mcp-server.js"), "// stub");
    return d;
  };

  it("gate hint wins while the global gate is off", () => {
    const home = homeWithSettings("{}"); // gate off, seed will also fail (no storage)
    const hint = ensureTraecodeWorkspaceMcp(tmp("repo-"), {
      home,
      dist: dist(),
      stateFile: stateFile(tmp("s-")),
    });
    expect(hint).toMatch(/enableWorkspaceMcp/); // the FIRST blocker, not the panel switch
  });

  it("the enable hint appears once the gate is on but the switch could not be seeded", () => {
    const home = homeWithSettings('{"trae.mcp.enableWorkspaceMcp":true}');
    const hint = ensureTraecodeWorkspaceMcp(tmp("repo-"), {
      home,
      dist: dist(),
      stateFile: stateFile(tmp("s-")),
    });
    expect(hint).toMatch(/MCP panel/);
  });

  it.runIf(hasSqlite3)("silent when the gate is on and the seed succeeds", () => {
    const repo = tmp("repo-");
    const home = homeWithSettings('{"trae.mcp.enableWorkspaceMcp":true}');
    const ws = join(userDataRoot(home), "Trae CN", "User", "workspaceStorage", "hash1");
    mkdirSync(ws, { recursive: true });
    writeFileSync(join(ws, "workspace.json"), JSON.stringify({ folder: pathToFileURL(repo).href }));
    execFileSync("sqlite3", [
      join(ws, "state.vscdb"),
      "CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB);",
    ]);
    const hint = ensureTraecodeWorkspaceMcp(repo, {
      home,
      dist: dist(),
      stateFile: stateFile(tmp("s-")),
    });
    expect(hint).toBeUndefined();
  });
});
