import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  resolveClaudeBin,
  startCodebaseSurvey as startSurvey,
  SURVEY_AGENT,
  SURVEY_AGENT_CONFIG,
  SURVEY_PROMPT,
} from "./survey";
import { releaseLease, SURVEY_SPEC_ENV, type SurveySupervisorSpec } from "./survey-lease";

import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync, readdirSync, writeFileSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { pathToFileURL, fileURLToPath } from "node:url";
import { buildSync } from "esbuild";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveHostConfig } from "./host-client";
import { resolveConfig } from "./config";

vi.mock("./host-client", () => ({ resolveHostConfig: vi.fn() }));
let lockDir: string;
beforeEach(() => {
  lockDir = mkdtempSync(join(tmpdir(), "hindsight-survey-test-"));
  vi.mocked(resolveHostConfig).mockReturnValue({
    cfg: resolveConfig({ apiUrl: "https://api.example.test", apiToken: "test-token" }),
    bankId: "bank-1",
  });
});
afterEach(() => {
  vi.restoreAllMocks();
  rmSync(lockDir, { recursive: true, force: true });
});
function startCodebaseSurvey(repoDir: string, opts: Parameters<typeof startSurvey>[1] = {}) {
  return startSurvey(repoDir, {
    lease: { dir: lockDir },
    supervisorPath: "/x/survey-supervisor.js",
    ...opts,
  });
}

describe("resolveClaudeBin", () => {
  const ORIGINAL_ENV = process.env.HINDSIGHT_CLAUDE_BIN;

  afterEach(() => {
    if (ORIGINAL_ENV === undefined) delete process.env.HINDSIGHT_CLAUDE_BIN;
    else process.env.HINDSIGHT_CLAUDE_BIN = ORIGINAL_ENV;
  });

  it("an explicit argument wins over everything", async () => {
    process.env.HINDSIGHT_CLAUDE_BIN = "/env/claude";
    expect(resolveClaudeBin("/explicit/claude")).toBe("/explicit/claude");
  });

  it("HINDSIGHT_CLAUDE_BIN env var wins when no explicit arg is given", async () => {
    process.env.HINDSIGHT_CLAUDE_BIN = "/env/claude";
    expect(resolveClaudeBin()).toBe("/env/claude");
  });

  it("falls back to the bare 'claude' PATH lookup when nothing else resolves", async () => {
    delete process.env.HINDSIGHT_CLAUDE_BIN;
    const bin = resolveClaudeBin();
    expect(typeof bin).toBe("string");
    expect(bin.length).toBeGreaterThan(0);
  });
});

describe("startCodebaseSurvey", () => {
  function fakeSpawn() {
    return vi.fn().mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), {
        pid: process.pid,
        unref: vi.fn(),
        kill: vi.fn(),
      });
      vi.spyOn(child, "on");
      queueMicrotask(() => child.emit("spawn"));
      return child;
    });
  }
  const yes = () => true;
  /** The survey agent is launched through the lease supervisor: decode the agent's bin + argv
   *  from the supervisor's spec; the options are the real spawn's (the agent inherits them). */
  function launched(spawn: ReturnType<typeof fakeSpawn>, i = 0) {
    const [node, argv, options] = spawn.mock.calls[i];
    expect(node).toBe("node");
    // Only the script path on the command line: the payload rides in the environment (#4255).
    expect(argv).toEqual(["/x/survey-supervisor.js"]);
    const spec = JSON.parse(options.env[SURVEY_SPEC_ENV]) as SurveySupervisorSpec;
    return [spec.bin, spec.args, options, spec] as const;
  }

  // ── claude recipe (the default / self-contained inline-MCP one) ────────────────────────────────
  it("claude: spawns the resolved binary with the expected argv, sandbox, and options", async () => {
    const spawn = fakeSpawn();
    await startCodebaseSurvey("/repo", {
      model: "sonnet",
      mcpServerPath: "/x/mcp-server.js",
      claudeBin: "/bin/claude",
      spawn,
      exists: yes,
    });

    expect(spawn).toHaveBeenCalledTimes(1);
    const [bin, argv, options] = launched(spawn);
    expect(bin).toBe("/bin/claude");

    expect(argv).toContain("-p");
    expect(argv).toContain(SURVEY_PROMPT);
    expect(argv).toContain("--model");
    expect(argv).toContain("sonnet");
    expect(argv).toContain("--mcp-config");
    expect(argv).toContain("--strict-mcp-config");
    expect(argv).toContain("mcp__hindsight__hindsight_ingest_document");

    // Sandbox: no bypassPermissions (defeats --allowedTools), a --disallowedTools deny-list.
    expect(argv).not.toContain("--permission-mode");
    expect(argv).not.toContain("bypassPermissions");
    expect(argv).toContain("--disallowedTools");
    for (const t of ["Bash", "Write", "Edit", "NotebookEdit", "WebFetch", "WebSearch", "Task"]) {
      expect(argv).toContain(t);
    }
    expect(argv).toContain("--max-budget-usd");
    expect(argv).toContain("2");

    const mcpConfigJson = argv[argv.indexOf("--mcp-config") + 1];
    const parsed = JSON.parse(mcpConfigJson);
    expect(parsed.mcpServers.hindsight.args).toEqual(["/x/mcp-server.js"]);
    expect(parsed.mcpServers.hindsight.env.HINDSIGHT_MCP_PROJECT_CWD).toBe("/repo");
    // #3603: mcp-server.js requires the harness — an inline recipe that omits it used to be served
    // as claude-code by default, and now refuses to start at all.
    expect(parsed.mcpServers.hindsight.env.HINDSIGHT_MCP_HARNESS).toBe("claude-code");

    expect(options.cwd).toBe("/repo");
    expect(options.detached).toBe(true);
    expect(options.stdio).toBe("ignore");
    expect(options.windowsHide).toBe(true);
    expect(options.env.HINDSIGHT_DISABLE_HOOKS).toBe("1");

    const child = spawn.mock.results[0].value;
    expect(child.on).toHaveBeenCalledWith("error", expect.any(Function));
    expect(child.unref).toHaveBeenCalled();
  });

  it("claude: defaults model to 'haiku' and --max-budget-usd to 2", async () => {
    const spawn = fakeSpawn();
    await startCodebaseSurvey("/repo", { claudeBin: "/bin/claude", spawn, exists: yes });
    const argv = launched(spawn)[1];
    expect(argv[argv.indexOf("--model") + 1]).toBe("haiku");
    expect(argv[argv.indexOf("--max-budget-usd") + 1]).toBe("2");
  });

  // ── codex recipe (read-only sandbox + inline -c MCP) ───────────────────────────────────────────
  it("codex: spawns `codex exec --sandbox read-only` with inline MCP overrides + the prompt", async () => {
    const spawn = fakeSpawn();
    await startCodebaseSurvey("/repo", {
      harness: "codex",
      mcpServerPath: "/x/mcp-server.js",
      spawn,
      exists: (b) => b === "codex",
    });
    const [bin, argv, options] = launched(spawn);
    expect(bin).toBe("codex");
    expect(argv.slice(0, 3)).toEqual(["exec", "--sandbox", "read-only"]);
    expect(argv).toContain(SURVEY_PROMPT);
    expect(argv).toContain(`mcp_servers.hindsight.command="node"`);
    expect(argv).toContain(`mcp_servers.hindsight.args=["/x/mcp-server.js"]`);
    // Without this the survey's findings were stamped harness:claude-code and landed in Claude
    // Code's bank even though Codex ran the survey (#3603).
    expect(argv).toContain(`mcp_servers.hindsight.env.HINDSIGHT_MCP_HARNESS="codex"`);
    expect(argv).toContain(`mcp_servers.hindsight.env.HINDSIGHT_MCP_PROJECT_CWD="/repo"`);
    // No Claude-only flags leak into the codex recipe.
    expect(argv).not.toContain("--model");
    expect(argv).not.toContain("--disallowedTools");
    expect(options.env.HINDSIGHT_DISABLE_HOOKS).toBe("1");
  });

  // ── Antigravity recipe (plan read-only mode + global MCP config) ───────────────────────────────
  it("antigravity: spawns `agy -p` in plan mode", async () => {
    const spawn = fakeSpawn();
    await startCodebaseSurvey("/repo", {
      harness: "antigravity-cli",
      spawn,
      exists: (b) => b === "agy",
    });
    const [bin, argv, options] = launched(spawn);
    expect(bin).toBe("agy");
    expect(argv).toEqual(["-p", SURVEY_PROMPT, "--mode=plan"]);
    expect(options.env.HINDSIGHT_DISABLE_HOOKS).toBe("1");
  });

  // ── opencode recipe (our own read-only agent; tools from the loaded plugin) ────────────────────
  it("opencode: spawns `opencode run` under OUR survey agent, never the built-in plan agent", async () => {
    const spawn = fakeSpawn();
    await startCodebaseSurvey("/repo", {
      harness: "opencode",
      spawn,
      exists: (b) => b === "opencode",
    });
    const [bin, argv, options] = launched(spawn);
    expect(bin).toBe("opencode");
    expect(argv).toEqual(["run", "--agent", SURVEY_AGENT, SURVEY_PROMPT]);
    // `plan` appends a read-only system-reminder that talks models out of the ingest call the
    // survey exists to make (#3450) — the whole point is not to run under it.
    expect(argv).not.toContain("plan");
    expect(options.env.HINDSIGHT_DISABLE_HOOKS).toBe("1");
  });

  // The recipe above is only safe because the agent it names is read-only. opencode drops denied
  // tools from the model's tool list entirely, so this ruleset IS the sandbox.
  it("the survey agent denies everything except reading and the one ingest tool", async () => {
    expect(SURVEY_AGENT_CONFIG.permission["*"]).toBe("deny");
    expect(SURVEY_AGENT_CONFIG.permission.hindsight_ingest_document).toBe("allow");
    const allowed = Object.entries(SURVEY_AGENT_CONFIG.permission)
      .filter(([, v]) => v === "allow")
      .map(([k]) => k)
      .sort();
    expect(allowed).toEqual(["glob", "grep", "hindsight_ingest_document", "read"]);
    // No write, no bash, and no `task` — which would reach a subagent that CAN write.
    for (const escape of ["write", "edit", "bash", "task", "patch"])
      expect(SURVEY_AGENT_CONFIG.permission).not.toHaveProperty(escape, "allow");
  });

  // ── agent selection + fallback ─────────────────────────────────────────────────────────────────
  it("honors the HINDSIGHT_CODEX_BIN override for the codex binary", async () => {
    const spawn = fakeSpawn();
    process.env.HINDSIGHT_CODEX_BIN = "/opt/codex";
    try {
      await startCodebaseSurvey("/repo", {
        harness: "codex",
        spawn,
        exists: (b) => b === "/opt/codex",
      });
    } finally {
      delete process.env.HINDSIGHT_CODEX_BIN;
    }
    expect(launched(spawn)[0]).toBe("/opt/codex");
  });

  it("falls back to another available agent when the preferred harness's CLI is missing", async () => {
    const spawn = fakeSpawn();
    // Prefer Antigravity, but only codex is installed → survey runs under codex.
    await startCodebaseSurvey("/repo", {
      harness: "antigravity-cli",
      mcpServerPath: "/x/mcp-server.js",
      spawn,
      exists: (b) => b === "codex",
    });
    const [bin, argv] = launched(spawn);
    expect(bin).toBe("codex");
    expect(argv[0]).toBe("exec");
  });

  it("no capable agent found → no spawn (fail open; the git-log seed still ran)", async () => {
    const spawn = fakeSpawn();
    await startCodebaseSurvey("/repo", { harness: "antigravity-cli", spawn, exists: () => false });
    expect(spawn).not.toHaveBeenCalled();
  });

  // ── fail-safe ──────────────────────────────────────────────────────────────────────────────────
  it("fail-safe: a spawn that throws synchronously does not throw out of startCodebaseSurvey", async () => {
    const spawn = vi.fn().mockImplementation(() => {
      throw new Error("spawn EMFILE");
    });
    await expect(
      startCodebaseSurvey("/repo", { claudeBin: "/bin/claude", spawn, exists: yes })
    ).resolves.toBe(false);
  });

  it("fail-safe: an async 'error' event on the child does not crash the caller", async () => {
    const { EventEmitter } = await import("node:events");
    const child = new EventEmitter() as InstanceType<typeof EventEmitter> & { unref: () => void };
    child.unref = vi.fn();
    const spawn = vi.fn().mockReturnValue(child);
    const launched = startCodebaseSurvey("/repo", { claudeBin: "/bin/claude", spawn, exists: yes });
    expect(() => child.emit("error", new Error("ENOENT"))).not.toThrow();
    await expect(launched).resolves.toBe(false);
    await expect(startCodebaseSurvey("/repo", { spawn: fakeSpawn(), exists: yes })).resolves.toBe(
      true
    );
  });

  it("launches the agent under the detached lease supervisor, which owns the lease", async () => {
    const spawn = fakeSpawn();
    await expect(
      startCodebaseSurvey("/repo", { claudeBin: "/bin/claude", spawn, exists: yes })
    ).resolves.toBe(true);
    const [bin, , options, spec] = launched(spawn);
    expect(bin).toBe("/bin/claude");
    expect(options).toMatchObject({ cwd: "/repo", detached: true, stdio: "ignore" });
    expect(readdirSync(spec.lease.directory)).toEqual([spec.lease.owner]);
    expect(spawn.mock.results[0].value.unref).toHaveBeenCalled();
  });

  it("admits one concurrent survey and permits retry once the lease is released", async () => {
    const spawn = fakeSpawn();
    const results = await Promise.all(
      Array.from({ length: 6 }, () => startCodebaseSurvey("/repo", { spawn, exists: yes }))
    );
    expect(spawn).toHaveBeenCalledTimes(1);
    expect(results.filter(Boolean)).toHaveLength(1);
    releaseLease(launched(spawn)[3].lease); // what the supervisor does when the agent exits
    await expect(startCodebaseSurvey("/repo", { spawn, exists: yes })).resolves.toBe(true);
    expect(spawn).toHaveBeenCalledTimes(2);
  });

  it("releases the lease when the supervisor cannot be spawned", async () => {
    const child = Object.assign(new EventEmitter(), { unref: vi.fn() });
    const failing = vi.fn().mockReturnValue(child);
    const launch = startCodebaseSurvey("/repo", { spawn: failing, exists: yes });
    child.emit("error", Object.assign(new Error("spawn node ENOENT"), { code: "ENOENT" }));
    await expect(launch).resolves.toBe(false);
    await expect(startCodebaseSurvey("/repo", { spawn: fakeSpawn(), exists: yes })).resolves.toBe(
      true
    );
  });

  it("keys admission by the resolved API, credential and bank, not repository or asking harness", async () => {
    const spawn = fakeSpawn();
    await expect(startCodebaseSurvey("/repo-a", { spawn, exists: yes })).resolves.toBe(true);
    // A second directory / fallback harness can write the same destination.
    await expect(
      startCodebaseSurvey("/repo-b", { harness: "codex", spawn, exists: yes })
    ).resolves.toBe(false);
    for (const [apiUrl, apiToken, bankId] of [
      ["https://other.example.test", "test-token", "bank-1"],
      ["https://api.example.test", "other-token", "bank-1"],
      ["https://api.example.test", "test-token", "bank-2"],
    ]) {
      vi.mocked(resolveHostConfig).mockReturnValue({
        cfg: resolveConfig({ apiUrl, apiToken }),
        bankId,
      });
      await expect(startCodebaseSurvey("/repo-a", { spawn, exists: yes })).resolves.toBe(true);
    }
    expect(spawn).toHaveBeenCalledTimes(4);
  });

  it("a disabled plugin launches nothing", async () => {
    vi.mocked(resolveHostConfig).mockReturnValue({
      cfg: { ...resolveConfig({}), disabled: true },
      bankId: "",
    });
    const spawn = fakeSpawn();
    await expect(startCodebaseSurvey("/repo", { spawn, exists: yes })).resolves.toBe(false);
    expect(spawn).not.toHaveBeenCalled();
  });

  // Real processes end to end: bundled hook + bundled supervisor + a stand-in agent.
  it("single-flights independent hooks, survives their exit, and recovers a dead supervisor", async () => {
    const build = (entry: string, outfile: string) =>
      buildSync({
        entryPoints: [fileURLToPath(new URL(entry, import.meta.url))],
        bundle: true,
        platform: "node",
        format: "esm",
        outfile,
        banner: {
          js: 'import { createRequire } from "node:module"; const require = createRequire(import.meta.url);',
        },
      });
    const bundle = join(lockDir, "survey.mjs");
    const supervisor = join(lockDir, "survey-supervisor.mjs");
    build("./survey.ts", bundle);
    build("../survey-supervisor.ts", supervisor);
    const config = join(lockDir, "config.json");
    writeFileSync(
      config,
      JSON.stringify({ bankId: "bank-1", apiUrl: "https://api.example.test", apiToken: "t" })
    );
    // The stand-in agent records its pid and its supervisor's, then idles like a long survey.
    const agent = `const fs = require('node:fs'); if (process.env.${SURVEY_SPEC_ENV}) fs.writeFileSync(${JSON.stringify(lockDir)} + '/leaked-spec', ''); fs.writeFileSync(${JSON.stringify(lockDir)} + '/pid-' + process.pid + '-' + process.ppid, ''); setInterval(() => {}, 1000);`;
    const staleMs = 2_000;
    const hook = `
      import { startCodebaseSurvey } from ${JSON.stringify(pathToFileURL(bundle).href)};
      import { spawn } from 'node:child_process';
      const started = await startCodebaseSurvey(${JSON.stringify(lockDir)}, {
        exists: () => true,
        mcpServerPath: '/unused-mcp-server.js',
        supervisorPath: ${JSON.stringify(supervisor)},
        lease: { dir: ${JSON.stringify(join(lockDir, "locks"))}, staleMs: ${staleMs}, heartbeatMs: 200 },
        // Swap the real agent for the stand-in; the supervisor itself runs for real.
        spawn: (_node, argv, options) => {
          const spec = { ...JSON.parse(options.env.${SURVEY_SPEC_ENV}), bin: process.execPath, args: ['-e', ${JSON.stringify(agent)}] };
          return spawn(process.execPath, argv, { ...options, env: { ...options.env, ${SURVEY_SPEC_ENV}: JSON.stringify(spec) } });
        },
      });
      console.log(JSON.stringify(started));
    `;
    const runHook = async () => {
      const { stdout } = await promisify(execFile)(
        process.execPath,
        ["--input-type=module", "-e", hook],
        {
          env: {
            PATH: process.env.PATH,
            HOME: lockDir,
            HINDSIGHT_CONFIG: config,
            SystemRoot: process.env.SystemRoot,
          },
        }
      );
      return JSON.parse(stdout.trim()) as boolean;
    };
    const race = async () =>
      (await Promise.all(Array.from({ length: 6 }, runHook))).filter(Boolean).length;
    const agents = () =>
      readdirSync(lockDir)
        .filter((name) => name.startsWith("pid-"))
        .map((name) => {
          const [, pid, ppid] = name.split("-").map(Number);
          return { pid, ppid };
        });
    const alive = (pid: number) => {
      try {
        process.kill(pid, 0);
        return true;
      } catch {
        return false;
      }
    };
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    try {
      expect(await race()).toBe(1);
      await vi.waitFor(() => expect(agents()).toHaveLength(1));
      // The supervisor strips the spec before starting the agent.
      expect(readdirSync(lockDir)).not.toContain("leaked-spec");
      // Every hook has exited; past the stale window the heartbeat still holds the lease.
      await sleep(staleMs * 2);
      expect(await runHook()).toBe(false);

      // The agent finishes: the supervisor releases, and exactly one new survey is admitted.
      const first = agents()[0];
      process.kill(first.pid, "SIGTERM");
      await vi.waitFor(() => expect(alive(first.ppid)).toBe(false));
      expect(await race()).toBe(1);
      await vi.waitFor(() => expect(agents().filter((a) => alive(a.pid))).toHaveLength(1));

      // The supervisor dies without releasing: its lease goes stale, and one reclaimer wins.
      const second = agents().find((a) => alive(a.pid))!;
      process.kill(second.ppid, "SIGKILL");
      expect(await runHook()).toBe(false); // not stale yet
      await sleep(staleMs * 2);
      expect(await race()).toBe(1);
    } finally {
      for (const { pid, ppid } of agents()) {
        for (const p of [pid, ppid]) {
          try {
            process.kill(p, "SIGKILL");
          } catch {
            /* already exited */
          }
        }
      }
    }
  }, 30_000);
});
