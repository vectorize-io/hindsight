import { afterEach, describe, expect, it, vi } from "vitest";
import {
  resolveClaudeBin,
  startCodebaseSurvey,
  SURVEY_AGENT,
  SURVEY_AGENT_CONFIG,
  SURVEY_PROMPT,
} from "./survey";

describe("resolveClaudeBin", () => {
  const ORIGINAL_ENV = process.env.HINDSIGHT_CLAUDE_BIN;

  afterEach(() => {
    if (ORIGINAL_ENV === undefined) delete process.env.HINDSIGHT_CLAUDE_BIN;
    else process.env.HINDSIGHT_CLAUDE_BIN = ORIGINAL_ENV;
  });

  it("an explicit argument wins over everything", () => {
    process.env.HINDSIGHT_CLAUDE_BIN = "/env/claude";
    expect(resolveClaudeBin("/explicit/claude")).toBe("/explicit/claude");
  });

  it("HINDSIGHT_CLAUDE_BIN env var wins when no explicit arg is given", () => {
    process.env.HINDSIGHT_CLAUDE_BIN = "/env/claude";
    expect(resolveClaudeBin()).toBe("/env/claude");
  });

  it("falls back to the bare 'claude' PATH lookup when nothing else resolves", () => {
    delete process.env.HINDSIGHT_CLAUDE_BIN;
    const bin = resolveClaudeBin();
    expect(typeof bin).toBe("string");
    expect(bin.length).toBeGreaterThan(0);
  });
});

describe("startCodebaseSurvey", () => {
  function fakeSpawn() {
    return vi.fn().mockReturnValue({ on: vi.fn(), unref: vi.fn() });
  }
  const yes = () => true;

  // ── claude recipe (the default / self-contained inline-MCP one) ────────────────────────────────
  it("claude: spawns the resolved binary with the expected argv, sandbox, and options", () => {
    const spawn = fakeSpawn();
    startCodebaseSurvey("/repo", {
      model: "sonnet",
      mcpServerPath: "/x/mcp-server.js",
      claudeBin: "/bin/claude",
      spawn,
      exists: yes,
    });

    expect(spawn).toHaveBeenCalledTimes(1);
    const [bin, argv, options] = spawn.mock.calls[0];
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

  it("claude: defaults model to 'haiku' and --max-budget-usd to 2", () => {
    const spawn = fakeSpawn();
    startCodebaseSurvey("/repo", { claudeBin: "/bin/claude", spawn, exists: yes });
    const argv = spawn.mock.calls[0][1];
    expect(argv[argv.indexOf("--model") + 1]).toBe("haiku");
    expect(argv[argv.indexOf("--max-budget-usd") + 1]).toBe("2");
  });

  // ── codex recipe (read-only sandbox + inline -c MCP) ───────────────────────────────────────────
  it("codex: spawns `codex exec --sandbox read-only` with inline MCP overrides + the prompt", () => {
    const spawn = fakeSpawn();
    startCodebaseSurvey("/repo", {
      harness: "codex",
      mcpServerPath: "/x/mcp-server.js",
      spawn,
      exists: (b) => b === "codex",
    });
    const [bin, argv, options] = spawn.mock.calls[0];
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
  it("antigravity: spawns `agy -p` in plan mode", () => {
    const spawn = fakeSpawn();
    startCodebaseSurvey("/repo", {
      harness: "antigravity-cli",
      spawn,
      exists: (b) => b === "agy",
    });
    const [bin, argv, options] = spawn.mock.calls[0];
    expect(bin).toBe("agy");
    expect(argv).toEqual(["-p", SURVEY_PROMPT, "--mode=plan"]);
    expect(options.env.HINDSIGHT_DISABLE_HOOKS).toBe("1");
  });

  // ── opencode recipe (our own read-only agent; tools from the loaded plugin) ────────────────────
  it("opencode: spawns `opencode run` under OUR survey agent, never the built-in plan agent", () => {
    const spawn = fakeSpawn();
    startCodebaseSurvey("/repo", {
      harness: "opencode",
      spawn,
      exists: (b) => b === "opencode",
    });
    const [bin, argv, options] = spawn.mock.calls[0];
    expect(bin).toBe("opencode");
    expect(argv).toEqual(["run", "--agent", SURVEY_AGENT, SURVEY_PROMPT]);
    // `plan` appends a read-only system-reminder that talks models out of the ingest call the
    // survey exists to make (#3450) — the whole point is not to run under it.
    expect(argv).not.toContain("plan");
    expect(options.env.HINDSIGHT_DISABLE_HOOKS).toBe("1");
  });

  // The recipe above is only safe because the agent it names is read-only. opencode drops denied
  // tools from the model's tool list entirely, so this ruleset IS the sandbox.
  it("the survey agent denies everything except reading and the one ingest tool", () => {
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
  it("honors the HINDSIGHT_CODEX_BIN override for the codex binary", () => {
    const spawn = fakeSpawn();
    process.env.HINDSIGHT_CODEX_BIN = "/opt/codex";
    try {
      startCodebaseSurvey("/repo", { harness: "codex", spawn, exists: (b) => b === "/opt/codex" });
    } finally {
      delete process.env.HINDSIGHT_CODEX_BIN;
    }
    expect(spawn.mock.calls[0][0]).toBe("/opt/codex");
  });

  it("falls back to another available agent when the preferred harness's CLI is missing", () => {
    const spawn = fakeSpawn();
    // Prefer Antigravity, but only codex is installed → survey runs under codex.
    startCodebaseSurvey("/repo", {
      harness: "antigravity-cli",
      mcpServerPath: "/x/mcp-server.js",
      spawn,
      exists: (b) => b === "codex",
    });
    const [bin, argv] = spawn.mock.calls[0];
    expect(bin).toBe("codex");
    expect(argv[0]).toBe("exec");
  });

  it("no capable agent found → no spawn (fail open; the git-log seed still ran)", () => {
    const spawn = fakeSpawn();
    startCodebaseSurvey("/repo", { harness: "antigravity-cli", spawn, exists: () => false });
    expect(spawn).not.toHaveBeenCalled();
  });

  // ── fail-safe ──────────────────────────────────────────────────────────────────────────────────
  it("fail-safe: a spawn that throws synchronously does not throw out of startCodebaseSurvey", () => {
    const spawn = vi.fn().mockImplementation(() => {
      throw new Error("spawn EMFILE");
    });
    expect(() =>
      startCodebaseSurvey("/repo", { claudeBin: "/bin/claude", spawn, exists: yes })
    ).not.toThrow();
  });

  it("fail-safe: an async 'error' event on the child does not crash the caller", async () => {
    const { EventEmitter } = await import("node:events");
    const child = new EventEmitter() as InstanceType<typeof EventEmitter> & { unref: () => void };
    child.unref = vi.fn();
    const spawn = vi.fn().mockReturnValue(child);
    startCodebaseSurvey("/repo", { claudeBin: "/bin/claude", spawn, exists: yes });
    expect(() => child.emit("error", new Error("ENOENT"))).not.toThrow();
  });
});
