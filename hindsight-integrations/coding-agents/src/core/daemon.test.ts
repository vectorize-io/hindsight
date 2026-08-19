import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { resolveConfig } from "./config";
import { daemonEnv, detectLlm, ensureDaemon, startDaemonDetached } from "./daemon";

describe("connection modes", () => {
  it("daemon mode derives the URL from apiPort, ignoring apiUrl", () => {
    // Resolving here (rather than at each client construction) is what lets the eight existing
    // call sites stay untouched.
    const cfg = resolveConfig({ serverMode: "daemon", apiUrl: "https://cloud.example" });
    expect(cfg.apiUrl).toBe("http://127.0.0.1:9077");
  });

  it("uses a custom daemon port", () => {
    expect(resolveConfig({ serverMode: "daemon", apiPort: 9999 }).apiUrl).toBe(
      "http://127.0.0.1:9999"
    );
  });

  it("defaults to cloud, leaving today's behaviour unchanged", () => {
    const cfg = resolveConfig({});
    expect(cfg.serverMode).toBe("cloud");
    expect(cfg.apiUrl).toBe("https://api.hindsight.vectorize.io");
  });

  it("self-hosted keeps the configured apiUrl", () => {
    const cfg = resolveConfig({ serverMode: "self-hosted", apiUrl: "http://localhost:8888" });
    expect(cfg.apiUrl).toBe("http://localhost:8888");
  });

  // 9077, not hindsight-all's 8888: 8888 is the conventional port for a server the user runs, and
  // a daemon must never squat on it.
  it("defaults the daemon port to 9077", () => {
    expect(resolveConfig({ serverMode: "daemon" }).apiPort).toBe(9077);
  });
});

describe("detectLlm", () => {
  it("prefers an explicit provider, including one it knows nothing about", () => {
    const llm = detectLlm({
      HINDSIGHT_API_LLM_PROVIDER: "ollama",
      OPENAI_API_KEY: "sk-ignored",
    } as NodeJS.ProcessEnv);
    expect(llm?.provider).toBe("ollama");
  });

  it("falls back to a well-known key env", () => {
    const llm = detectLlm({ ANTHROPIC_API_KEY: "sk-ant" } as NodeJS.ProcessEnv);
    expect(llm?.provider).toBe("anthropic");
    expect(llm?.apiKey).toBe("sk-ant");
  });

  it("ignores a blank key", () => {
    const llm = detectLlm({ OPENAI_API_KEY: "   " } as NodeJS.ProcessEnv);
    expect(llm?.provider).not.toBe("openai");
  });
});

describe("daemonEnv", () => {
  const cfg = resolveConfig({ serverMode: "daemon", daemonIdleTimeout: 42 });

  it("passes the idle timeout through under the daemon's own env name", () => {
    expect(daemonEnv(cfg, {} as NodeJS.ProcessEnv).HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT).toBe("42");
  });

  // It used to send 300 whether or not anyone asked, quietly retiring a SHARED daemon out from
  // under an idle session — while every other Hindsight integration ships 0 (never exits).
  it("says nothing about the idle timeout when it is unset, leaving the daemon its own default", () => {
    const env = daemonEnv(resolveConfig({ serverMode: "daemon" }), {} as NodeJS.ProcessEnv);
    expect(env).not.toHaveProperty("HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT");
  });

  // Forwarding the whole HINDSIGHT_API_* namespace means a new server-side knob needs no change
  // here to reach the daemon.
  it("forwards arbitrary HINDSIGHT_API_* settings", () => {
    const env = daemonEnv(cfg, {
      HINDSIGHT_API_EMBEDDINGS_PROVIDER: "tei",
      UNRELATED: "no",
    } as NodeJS.ProcessEnv);
    expect(env.HINDSIGHT_API_EMBEDDINGS_PROVIDER).toBe("tei");
    expect(env.UNRELATED).toBeUndefined();
  });
});

describe("ensureDaemon", () => {
  // Cloud and self-hosted must not so much as probe a local port.
  it("touches nothing outside daemon mode", async () => {
    const cfg = resolveConfig({ apiUrl: "https://cloud.example" });
    const spawnFn = vi.fn(() => {
      throw new Error("must not spawn");
    });
    await ensureDaemon(cfg, "claude-code", {
      spawnFn: spawnFn as unknown as typeof import("node:child_process").spawn,
    });
    expect(spawnFn).not.toHaveBeenCalled();
  });

  // Side effect only: callers proceed either way, so a down daemon fails through the same client
  // error path as a down Cloud/self-hosted server rather than being skipped.
  it("resolves without reporting usability", async () => {
    const cfg = resolveConfig({ apiUrl: "https://cloud.example" });
    expect(await ensureDaemon(cfg, "claude-code", {})).toBeUndefined();
  });
});

/**
 * Daemon parity across harnesses (#3524).
 *
 * The ensure points were written for the fresh-process hook harnesses and lived in their hook
 * wrappers, so every harness added since — the persistent-plugin hosts, which call the shared
 * lifecycle directly — silently shipped without one. In daemon mode that means every request fails
 * with ECONNREFUSED and nothing ever starts a daemon; dsh shipped that way in 0.3.4.
 *
 * A per-harness unit test can't catch this: the harness that forgets is by definition the one whose
 * test nobody wrote. So this asserts the shape instead — a module that builds a client is a place a
 * request originates, and every one of them either ensures a daemon itself or delegates to
 * something that does. Adding a harness that does neither fails here, and an entrypoint that
 * genuinely must not start one has to say why in EXEMPT.
 */
describe("every harness entrypoint reaches a daemon", () => {
  const SRC = fileURLToPath(new URL("..", import.meta.url));

  /** Entrypoints that build a client but deliberately do NOT ensure a daemon, and why. */
  const EXEMPT: Record<string, string> = {
    "status.ts": "diagnostic — reports what is running; starting one would falsify the report",
    "deepen.ts": "spawned by the seed, which already ensured the daemon it inherits",
    "mcp-server.ts": "child of a hook harness whose SessionStart ensured the daemon first",
    "core/hook.ts":
      "the prompt path, deliberately: a cold start outlives the hook budget and stalls the turn",
  };

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  it("has no client-building module that neither ensures a daemon nor delegates to one", () => {
    const unreached = sourceFiles(SRC).filter((rel) => {
      if (rel in EXEMPT) return false;
      const src = readFileSync(join(SRC, rel), "utf8");
      if (!src.includes("new HindsightClient(")) return false;
      // RuntimeCore is the persistent-plugin hosts' shared lifecycle; it ensures at both points.
      return !src.includes("ensureDaemon") && !src.includes("RuntimeCore");
    });
    expect(unreached).toEqual([]);
  });

  // An exemption for a file that no longer builds a client is a stale claim about live code.
  it("keeps no exemption for a module that stopped building a client", () => {
    const stale = Object.keys(EXEMPT).filter(
      (rel) => !readFileSync(join(SRC, rel), "utf8").includes("new HindsightClient(")
    );
    expect(stale).toEqual([]);
  });
});

describe("startDaemonDetached", () => {
  it("spawns the starter detached, so it outlives the hook", () => {
    const child = { on: vi.fn(), unref: vi.fn() };
    const spawn = vi.fn(() => child);
    startDaemonDetached(
      resolveConfig({ serverMode: "daemon" }),
      "claude-code",
      spawn as unknown as typeof import("node:child_process").spawn
    );
    const [cmd, args, opts] = spawn.mock.calls[0] as unknown as [
      string,
      string[],
      { detached: boolean; stdio: string },
    ];
    expect(cmd).toBe("node");
    expect(args[0]).toMatch(/daemon-start\.js$/);
    expect(opts.detached).toBe(true);
    expect(opts.stdio).toBe("ignore");
    // An async spawn 'error' event would otherwise crash the hook process.
    expect(child.on).toHaveBeenCalledWith("error", expect.any(Function));
    expect(child.unref).toHaveBeenCalled();
  });
});
