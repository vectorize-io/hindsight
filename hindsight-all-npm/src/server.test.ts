import { describe, it, expect, vi } from "vitest";
import { HindsightServer } from "./server.js";

const spawnMock = vi.hoisted(() => vi.fn());
vi.mock("child_process", () => ({ spawn: spawnMock }));

describe("HindsightServer construction", () => {
  it("defaults base URL to http://127.0.0.1:8888", () => {
    const server = new HindsightServer();
    expect(server.getBaseUrl()).toBe("http://127.0.0.1:8888");
    expect(server.getProfile()).toBe("default");
  });

  it("honours custom profile, port, and host", () => {
    const server = new HindsightServer({ profile: "app", port: 9077, host: "0.0.0.0" });
    expect(server.getProfile()).toBe("app");
    expect(server.getBaseUrl()).toBe("http://0.0.0.0:9077");
  });

  it("accepts open env pass-through without complaining about unknown keys", () => {
    const server = new HindsightServer({
      env: {
        HINDSIGHT_API_LLM_PROVIDER: "openai",
        HINDSIGHT_API_LLM_MODEL: "gpt-4o-mini",
        // A field that does not exist today — should still be accepted
        HINDSIGHT_FUTURE_FLAG: "enabled",
      },
    });
    expect(server).toBeInstanceOf(HindsightServer);
  });

  it("exposes checkHealth that returns false when no daemon is running", async () => {
    // Random high port that nothing is listening on.
    const server = new HindsightServer({ port: 1, readyTimeoutMs: 100 });
    const healthy = await server.checkHealth();
    expect(healthy).toBe(false);
  });

  it("hides daemon command windows on Windows", async () => {
    const child = {
      stdout: { on: vi.fn() },
      stderr: { on: vi.fn() },
      on: vi.fn((event: string, handler: (code?: number) => void) => {
        if (event === "exit") handler(0);
        return child;
      }),
    };
    spawnMock.mockReturnValueOnce(child);

    const server = new HindsightServer();
    await server.stop();

    expect(spawnMock).toHaveBeenCalledWith(
      "uvx",
      ["hindsight-embed@latest", "daemon", "--profile", "default", "stop"],
      expect.objectContaining({ stdio: "pipe", windowsHide: true })
    );
  });

  it("hides profile create and daemon start command windows on Windows", async () => {
    spawnMock.mockClear();
    const child = {
      stdout: { on: vi.fn() },
      stderr: { on: vi.fn() },
      on: vi.fn((event: string, handler: (code?: number) => void) => {
        if (event === "exit") handler(0);
        return child;
      }),
    };
    spawnMock.mockReturnValue(child);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));

    try {
      const server = new HindsightServer();
      await server.start();
    } finally {
      vi.unstubAllGlobals();
    }

    // start() spawns twice — `profile create` (configureProfile) and `daemon start`
    // (startDaemon), both through runCommand(). Both must pass windowsHide: true,
    // or a console-less parent (the detached daemon-start entrypoint) makes Windows
    // allocate a new visible console for the console-subsystem uvx.exe child.
    expect(spawnMock).toHaveBeenCalledTimes(2);
    const options = spawnMock.mock.calls.map((call) => call[2]);
    expect(options[0]).toEqual(expect.objectContaining({ stdio: "pipe", windowsHide: true }));
    expect(options[1]).toEqual(expect.objectContaining({ stdio: "pipe", windowsHide: true }));
    expect(spawnMock.mock.calls[0][1]).toEqual([
      "hindsight-embed@latest",
      "profile",
      "create",
      "default",
      "--merge",
      "--port",
      "8888",
    ]);
    expect(spawnMock.mock.calls[1][1]).toEqual([
      "hindsight-embed@latest",
      "daemon",
      "--profile",
      "default",
      "start",
    ]);
  });
});
