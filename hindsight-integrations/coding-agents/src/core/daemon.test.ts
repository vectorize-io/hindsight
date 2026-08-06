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
