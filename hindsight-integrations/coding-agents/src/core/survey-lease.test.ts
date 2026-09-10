import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  acquireLease,
  heartbeatLease,
  LEASE_STALE_MS,
  releaseLease,
  superviseSurvey,
  type SurveyLease,
} from "./survey-lease";

let root: string;
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hindsight-lease-test-"));
});
afterEach(() => {
  vi.useRealTimers();
  rmSync(root, { recursive: true, force: true });
});

/** Backdate the lease's heartbeat, as if its holder died `ms` ago. */
function age(lease: SurveyLease, ms: number) {
  const then = new Date(Date.now() - ms);
  utimesSync(join(lease.directory, lease.owner), then, then);
}

describe("acquireLease", () => {
  it("admits exactly one holder per key, and leaves no staging behind", () => {
    const first = acquireLease(root, "k");
    expect(first).toBeDefined();
    expect(acquireLease(root, "k")).toBeUndefined();
    expect(acquireLease(root, "other")).toBeDefined();
    expect(readdirSync(root).sort()).toEqual(["survey-k.lock", "survey-other.lock"]);
    expect(readdirSync(first!.directory)).toEqual([first!.owner]);
  });

  it("is free again once released", () => {
    releaseLease(acquireLease(root, "k")!);
    expect(readdirSync(root)).toEqual([]);
    expect(acquireLease(root, "k")).toBeDefined();
  });

  it("keeps a fresh lease however old the process that took it is — liveness is the heartbeat", () => {
    const lease = acquireLease(root, "k")!;
    age(lease, LEASE_STALE_MS - 5_000);
    expect(acquireLease(root, "k")).toBeUndefined();
  });

  it("reclaims a lease whose heartbeat stopped, exactly once", () => {
    const dead = acquireLease(root, "k")!;
    age(dead, LEASE_STALE_MS + 5_000);
    const next = acquireLease(root, "k");
    expect(next).toBeDefined();
    expect(next!.owner).not.toBe(dead.owner);
    expect(acquireLease(root, "k")).toBeUndefined(); // the new holder is fresh
  });

  it("treats a heartbeat from the future (clock stepped back) as stale", () => {
    const lease = acquireLease(root, "k")!;
    age(lease, -(LEASE_STALE_MS + 60_000));
    expect(acquireLease(root, "k")).toBeDefined();
  });

  it("honors a custom stale window", () => {
    const lease = acquireLease(root, "k")!;
    age(lease, 3_000);
    expect(acquireLease(root, "k", 10_000)).toBeUndefined();
    expect(acquireLease(root, "k", 1_000)).toBeDefined();
  });

  it("reclaims an unowned directory left by an interrupted release", () => {
    mkdirSync(join(root, "survey-k.lock"));
    expect(acquireLease(root, "k")).toBeDefined();
  });

  it("never touches a lock directory it does not recognise", () => {
    const directory = join(root, "survey-k.lock");
    mkdirSync(join(directory, "a"), { recursive: true });
    mkdirSync(join(directory, "b"));
    expect(acquireLease(root, "k")).toBeUndefined();
    expect(readdirSync(directory).sort()).toEqual(["a", "b"]);
  });

  it("fails closed when the root cannot be created", () => {
    expect(acquireLease(join(root, "\0bad"), "k")).toBeUndefined();
  });
});

describe("releaseLease / heartbeatLease", () => {
  it("an old generation can neither release nor refresh its successor", () => {
    const old = acquireLease(root, "k")!;
    age(old, LEASE_STALE_MS + 5_000);
    const current = acquireLease(root, "k")!;
    releaseLease(old);
    expect(heartbeatLease(old)).toBe(false);
    expect(readdirSync(current.directory)).toEqual([current.owner]);
    expect(heartbeatLease(current)).toBe(true);
  });

  it("a heartbeat keeps an aged lease from being reclaimed", () => {
    const lease = acquireLease(root, "k")!;
    age(lease, LEASE_STALE_MS + 5_000);
    expect(heartbeatLease(lease)).toBe(true);
    expect(acquireLease(root, "k")).toBeUndefined();
  });
});

describe("superviseSurvey", () => {
  function fakeAgent() {
    const child = Object.assign(new EventEmitter(), { kill: vi.fn() });
    const spawn = vi.fn().mockReturnValue(child);
    return { child, spawn };
  }

  it("runs the agent and releases the lease when it exits", async () => {
    const lease = acquireLease(root, "k")!;
    const { child, spawn } = fakeAgent();
    const run = superviseSurvey({ lease, bin: "/bin/agent", args: ["-p", "x"] }, spawn);
    expect(spawn).toHaveBeenCalledWith("/bin/agent", ["-p", "x"], {
      stdio: "ignore",
      windowsHide: true,
    });
    child.emit("exit", 0);
    await run.done;
    expect(readdirSync(root)).toEqual([]);
  });

  it("heartbeats while the agent runs", async () => {
    vi.useFakeTimers();
    const lease = acquireLease(root, "k")!;
    const { child, spawn } = fakeAgent();
    const run = superviseSurvey({ lease, bin: "a", args: [], heartbeatMs: 1_000 }, spawn);
    age(lease, LEASE_STALE_MS + 5_000);
    vi.advanceTimersByTime(1_000);
    expect(acquireLease(root, "k")).toBeUndefined();
    child.emit("exit", 0);
    await run.done;
  });

  it("kills the agent, and keeps its hands off the lease, once the lease is taken over", async () => {
    vi.useFakeTimers();
    const lease = acquireLease(root, "k")!;
    const { child, spawn } = fakeAgent();
    const run = superviseSurvey({ lease, bin: "a", args: [], heartbeatMs: 1_000 }, spawn);
    age(lease, LEASE_STALE_MS + 5_000); // the machine slept past the stale window
    const successor = acquireLease(root, "k")!;
    vi.advanceTimersByTime(1_000);
    await run.done;
    expect(child.kill).toHaveBeenCalled();
    child.emit("exit", null);
    expect(readdirSync(successor.directory)).toEqual([successor.owner]);
  });

  it("does not start the agent if the lease was lost before it could", async () => {
    const lease = acquireLease(root, "k")!;
    releaseLease(lease);
    const { spawn } = fakeAgent();
    await superviseSurvey({ lease, bin: "a", args: [] }, spawn).done;
    expect(spawn).not.toHaveBeenCalled();
  });

  it("releases the lease when the agent cannot be spawned", async () => {
    const lease = acquireLease(root, "k")!;
    const { child, spawn } = fakeAgent();
    const run = superviseSurvey({ lease, bin: "missing", args: [] }, spawn);
    child.emit("error", new Error("spawn missing ENOENT"));
    await run.done;
    expect(readdirSync(root)).toEqual([]);

    const throwing = vi.fn(() => {
      throw new Error("EMFILE");
    });
    await superviseSurvey({ lease: acquireLease(root, "k")!, bin: "a", args: [] }, throwing).done;
    expect(readdirSync(root)).toEqual([]);
  });

  it("stop() kills the agent and releases", async () => {
    const lease = acquireLease(root, "k")!;
    const { child, spawn } = fakeAgent();
    const run = superviseSurvey({ lease, bin: "a", args: [] }, spawn);
    run.stop();
    await run.done;
    expect(child.kill).toHaveBeenCalled();
    expect(readdirSync(root)).toEqual([]);
  });
});
