import { EventEmitter } from "node:events";
import { describe, expect, it, vi } from "vitest";
import { startBackgroundSeed, seedControl, DEFAULT_SEED_LIMIT } from "./seed";

describe("startBackgroundSeed", () => {
  function fakeSpawn() {
    return vi.fn().mockReturnValue({ on: vi.fn(), unref: vi.fn() });
  }

  it("spawns node against enginePath, detached, output kept (log file or ignore), with the default gitlog limit", () => {
    const spawn = fakeSpawn();
    startBackgroundSeed("/some/repo", { enginePath: "/dist/deepen.js", spawn });
    expect(spawn).toHaveBeenCalledWith(
      "node",
      ["/dist/deepen.js", "--repo", "/some/repo", "--gitlog-limit", String(DEFAULT_SEED_LIMIT)],
      { detached: true, stdio: expect.anything(), windowsHide: true }
    );
    expect(spawn.mock.results[0].value.unref).toHaveBeenCalled();
  });

  it("defaults enginePath to deepen.js next to the module", () => {
    const spawn = fakeSpawn();
    startBackgroundSeed("/some/repo", { spawn });
    expect(spawn).toHaveBeenCalledTimes(1);
    const args = spawn.mock.calls[0][1] as string[];
    expect(args[0].endsWith("deepen.js")).toBe(true);
    expect(args.slice(1)).toEqual([
      "--repo",
      "/some/repo",
      "--gitlog-limit",
      String(DEFAULT_SEED_LIMIT),
    ]);
  });

  it("opts.limit overrides the default limit", () => {
    const spawn = fakeSpawn();
    startBackgroundSeed("/some/repo", { enginePath: "/dist/deepen.js", spawn, limit: 50 });
    expect(spawn).toHaveBeenCalledWith(
      "node",
      ["/dist/deepen.js", "--repo", "/some/repo", "--gitlog-limit", "50"],
      { detached: true, stdio: expect.anything(), windowsHide: true }
    );
  });

  it("passes the calling harness to deepen so its config override is retained", () => {
    const spawn = fakeSpawn();
    startBackgroundSeed("/some/repo", {
      enginePath: "/dist/deepen.js",
      harness: "antigravity-cli",
      spawn,
    });
    expect(spawn).toHaveBeenCalledWith(
      "node",
      [
        "/dist/deepen.js",
        "--repo",
        "/some/repo",
        "--gitlog-limit",
        String(DEFAULT_SEED_LIMIT),
        "--harness",
        "antigravity-cli",
      ],
      { detached: true, stdio: expect.anything(), windowsHide: true }
    );
  });

  it("fail-safe: a spawn that throws does not throw out of startBackgroundSeed", () => {
    const spawn = vi.fn().mockImplementation(() => {
      throw new Error("spawn EMFILE");
    });
    expect(() =>
      startBackgroundSeed("/some/repo", { enginePath: "/dist/deepen.js", spawn })
    ).not.toThrow();
  });

  it("fail-safe: an async 'error' event on the child (ENOENT/EACCES/sandbox) does not crash the caller", () => {
    // Real spawn() failures like ENOENT arrive as an async 'error' event on the returned
    // ChildProcess, not as a synchronous throw — so the spy must be a real EventEmitter to
    // exercise that path (a plain object with unref() wouldn't emit anything).
    const child = new EventEmitter() as EventEmitter & { unref: () => void };
    child.unref = vi.fn();
    const spawn = vi.fn().mockReturnValue(child);
    expect(() =>
      startBackgroundSeed("/some/repo", { enginePath: "/dist/deepen.js", spawn })
    ).not.toThrow();
    // Emitting 'error' with no listener would normally throw (EventEmitter semantics) and crash
    // the process — proving a listener was attached means this does NOT throw.
    expect(() => child.emit("error", new Error("ENOENT"))).not.toThrow();
  });
});

describe("seedControl", () => {
  function fakeSpawn() {
    return vi.fn().mockReturnValue({ on: vi.fn(), unref: vi.fn() });
  }

  it("seed: spawns the background deepen engine and returns ok", () => {
    const spawn = fakeSpawn();
    const result = seedControl("seed", { repo: "/r", bankId: "b", spawn });
    expect(spawn).toHaveBeenCalled();
    expect(result.ok).toBe(true);
    expect(result.message).toContain("b");
  });

  it("decline (and any unknown command): not ok, usage message, no spawn", () => {
    // "decline" is no longer a supported command — the live bank is the only state, so there is
    // no declined flag to persist. It falls through to the usage error like any unknown command.
    const spawn = fakeSpawn();
    for (const command of ["decline", "bogus"]) {
      const result = seedControl(command, { repo: "/r", bankId: "b", spawn });
      expect(result.ok).toBe(false);
      expect(result.message).toBe("usage: hindsight-seed seed --repo <dir>");
    }
    expect(spawn).not.toHaveBeenCalled();
  });
});
