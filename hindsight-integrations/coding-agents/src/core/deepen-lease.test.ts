import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { acquireDeepenLease, type ProcessIdentity } from "./deepen-lease";
import { acquireLease, heartbeatLease, releaseLease } from "./survey-lease";

let root: string;
let lock: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hindsight-deepen-lease-test-"));
  lock = join(root, "deepen-bank.lock");
});

afterEach(() => {
  vi.useRealTimers();
  rmSync(root, { recursive: true, force: true });
});

function legacy(pid: number, ts: number): void {
  writeFileSync(lock, JSON.stringify({ pid, ts }));
}

function quarantines(key = "bank"): string[] {
  return readdirSync(root)
    .filter((name) => name.startsWith(`deepen-${key}.legacy-`) && name.endsWith(".lock"))
    .map((name) => join(root, name));
}

const dead = (): ProcessIdentity => ({ alive: false });

function drainMarker(key = "bank"): string {
  return join(root, `deepen-${key}.lock`, "legacy-drain-complete");
}

describe("acquireDeepenLease", () => {
  it("backs off for the exact live legacy process generation however old its lock is", () => {
    const lockedAt = Date.now() - 75 * 60 * 1_000;
    legacy(41, lockedAt);

    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: () => ({ alive: true, startedAt: lockedAt - 100 }),
      listDeepenProcesses: () => [41],
    });

    expect(lease).toBeUndefined();
    expect(lstatSync(lock).isDirectory()).toBe(true);
    expect(quarantines()).toHaveLength(1);
    expect(JSON.parse(readFileSync(quarantines()[0], "utf8"))).toEqual({
      pid: 41,
      ts: lockedAt,
    });

    const afterExit = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
    });
    expect(afterExit).toBeDefined();
    expect(quarantines()).toHaveLength(0);
    afterExit!.release();
  });

  it("reclaims dead and PID-reused legacy generations", () => {
    legacy(42, 1_000);
    const deadLease = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
    });
    expect(deadLease).toBeDefined();
    expect(lstatSync(lock).isDirectory()).toBe(true);
    deadLease!.release();

    const reusedLock = join(root, "deepen-reused.lock");
    writeFileSync(reusedLock, JSON.stringify({ pid: 43, ts: 1_000 }));
    const reusedLease = acquireDeepenLease(root, "reused", {
      processIdentity: () => ({ alive: true, startedAt: 10_000 }),
      listDeepenProcesses: () => [],
    });
    expect(reusedLease).toBeDefined();
    reusedLease!.release();
  });

  it("leaves a permanent legacy fence that old cleanup cannot unlink", () => {
    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
    })!;

    expect(() => unlinkSync(lock)).toThrow();
    expect(lstatSync(lock).isDirectory()).toBe(true);
    lease.release();
    expect(lstatSync(lock).isDirectory()).toBe(true);
  });

  it("backs off without deleting a legacy generation that claims the handoff gap", () => {
    legacy(44, 1_000);
    const replacement = JSON.stringify({ pid: 45, ts: 2_000 });
    let replaced = false;

    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: (pid) => (pid === 45 ? { alive: true, startedAt: 1_500 } : { alive: false }),
      listDeepenProcesses: () => [],
      afterLegacyMoved: () => {
        if (replaced) return;
        replaced = true;
        writeFileSync(lock, replacement);
      },
    });

    expect(lease).toBeUndefined();
    expect(lstatSync(lock).isDirectory()).toBe(true);
    expect(quarantines()).toHaveLength(1);
    expect(readFileSync(quarantines()[0], "utf8")).toBe(replacement);
  });

  it("inspects the exact quarantine generation after a concurrent replacement", () => {
    legacy(48, 1_000);
    const replacement = JSON.stringify({ pid: 49, ts: 2_000 });
    let original: string | undefined;

    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: (pid) => (pid === 49 ? { alive: true, startedAt: 1_500 } : { alive: false }),
      listDeepenProcesses: () => [],
      beforeQuarantineInspect: (path) => {
        if (original) return;
        original = `${path}.original`;
        renameSync(path, original);
        writeFileSync(path, replacement);
      },
    });

    expect(lease).toBeUndefined();
    expect(original).toBeDefined();
    expect(readFileSync(original!, "utf8")).toBe(JSON.stringify({ pid: 48, ts: 1_000 }));
    expect(quarantines()).toHaveLength(1);
    expect(readFileSync(quarantines()[0], "utf8")).toBe(replacement);
  });

  it("drains hidden legacy contenders that overwrote the visible PID", () => {
    legacy(52, 2_000); // PID 51 also runs, but PID 52 overwrote its non-exclusive lock.
    let running = [51, 52];
    const options = {
      processIdentity: dead,
      listDeepenProcesses: () => running,
    };

    expect(acquireDeepenLease(root, "bank", options)).toBeUndefined();
    expect(lstatSync(lock).isDirectory()).toBe(true);
    running = [51];
    expect(acquireDeepenLease(root, "bank", options)).toBeUndefined();
    running = [];
    const lease = acquireDeepenLease(root, "bank", options);

    expect(lease).toBeDefined();
    lease!.release();
  });

  it("fails closed and preserves malformed or unreadable quarantines", () => {
    writeFileSync(lock, "{not-json");
    const malformed = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
    });
    expect(malformed).toBeUndefined();
    expect(readFileSync(quarantines()[0], "utf8")).toBe("{not-json");

    const unreadablePath = quarantines()[0];
    chmodSync(unreadablePath, 0o000);
    const unreadable = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
    });
    expect(unreadable).toBeUndefined();
    expect(lstatSync(unreadablePath).isFile()).toBe(true);
    chmodSync(unreadablePath, 0o600);
  });

  it("fails closed when legacy process enumeration is unavailable", () => {
    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => undefined,
    });

    expect(lease).toBeUndefined();
    expect(lstatSync(lock).isDirectory()).toBe(true);
  });

  it("publishes a per-fence drain marker and does not mistake v2 contenders for legacy", () => {
    const identities = new Map<number, ProcessIdentity>([
      [101, { alive: true, startedAt: 1_000 }],
      [102, { alive: true, startedAt: 2_000 }],
      [103, { alive: true, startedAt: 3_000 }],
    ]);
    const identityOf = (pid: number) => identities.get(pid) ?? dead();
    let second: ReturnType<typeof acquireDeepenLease>;
    const firstList = vi.fn(() => [102]);
    const first = acquireDeepenLease(root, "bank", {
      processId: 101,
      currentProcessIdentity: identities.get(101),
      processIdentity: identityOf,
      listDeepenProcesses: firstList,
      waitForProcessRegistration: () => {
        if (second) return;
        second = acquireDeepenLease(root, "other", {
          processId: 102,
          currentProcessIdentity: identities.get(102),
          processIdentity: identityOf,
          listDeepenProcesses: () => [101, 102],
          waitForProcessRegistration: () => {},
        });
      },
    });

    // PID 102 was visible just before reaching its registration write. Both different-bank v2
    // contenders complete their one-shot acquisition without either being classified as legacy.
    expect(first).toBeDefined();
    expect(second).toBeDefined();
    expect(firstList).toHaveBeenCalledTimes(2);
    expect(lstatSync(drainMarker()).isDirectory()).toBe(true);
    expect(lstatSync(drainMarker("other")).isDirectory()).toBe(true);

    const sameBankList = vi.fn(() => [101, 102, 103]);
    const sameBank = acquireDeepenLease(root, "bank", {
      processId: 103,
      currentProcessIdentity: identities.get(103),
      processIdentity: identityOf,
      listDeepenProcesses: sameBankList,
    });
    expect(sameBank).toBeUndefined();
    expect(sameBankList).not.toHaveBeenCalled();

    first!.release();
    second!.release();

    const shouldNotEnumerate = vi.fn(() => undefined);
    const afterMigration = acquireDeepenLease(root, "bank", {
      processId: 103,
      currentProcessIdentity: identities.get(103),
      processIdentity: identityOf,
      listDeepenProcesses: shouldNotEnumerate,
    });
    expect(afterMigration).toBeDefined();
    expect(shouldNotEnumerate).not.toHaveBeenCalled();
    afterMigration!.release();
  });

  it("treats a partial drain-marker directory as invalid and leaves it untouched", () => {
    const first = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
    })!;
    first.release();
    unlinkSync(join(drainMarker(), "fence-token"));
    const shouldNotEnumerate = vi.fn(() => []);

    const second = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: shouldNotEnumerate,
    });

    expect(second).toBeUndefined();
    expect(lstatSync(drainMarker()).isDirectory()).toBe(true);
    expect(readdirSync(drainMarker())).toHaveLength(0);
    expect(shouldNotEnumerate).not.toHaveBeenCalled();
  });

  it("blocks and leaves migration incomplete when quarantine discovery fails", () => {
    const denied = Object.assign(new Error("denied"), { code: "EACCES" });
    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
      quarantineIo: {
        readdir: () => {
          throw denied;
        },
      },
    });

    expect(lease).toBeUndefined();
    expect(existsSync(drainMarker())).toBe(false);
  });

  it("blocks and keeps the source discoverable when quarantine custody rename fails", () => {
    legacy(61, 1_000);
    const busy = Object.assign(new Error("busy"), { code: "EBUSY" });
    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
      quarantineIo: {
        rename: () => {
          throw busy;
        },
      },
    });

    expect(lease).toBeUndefined();
    expect(quarantines()).toHaveLength(1);
    expect(existsSync(drainMarker())).toBe(false);
  });

  it("blocks and keeps custody discoverable when hard links are unsupported", () => {
    legacy(62, 1_000);
    const unsupported = Object.assign(new Error("unsupported"), { code: "EPERM" });
    const lease = acquireDeepenLease(root, "bank", {
      processIdentity: () => ({ alive: true, startedAt: 500 }),
      listDeepenProcesses: () => [],
      quarantineIo: {
        link: () => {
          throw unsupported;
        },
      },
    });

    expect(lease).toBeUndefined();
    const retained = quarantines();
    expect(retained).toHaveLength(1);
    expect(retained[0]).toContain(".legacy-custody-");
    expect(readFileSync(retained[0], "utf8")).toBe(JSON.stringify({ pid: 62, ts: 1_000 }));
    expect(existsSync(drainMarker())).toBe(false);
  });

  it("never lets a completed marker bypass a later quarantine artifact", () => {
    const first = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => [],
    })!;
    first.release();
    expect(lstatSync(drainMarker()).isDirectory()).toBe(true);
    writeFileSync(join(root, "deepen-bank.legacy-unexpected"), "unknown");

    const second = acquireDeepenLease(root, "bank", {
      processIdentity: dead,
      listDeepenProcesses: () => {
        throw new Error("completed migration must not enumerate");
      },
    });

    expect(second).toBeUndefined();
    expect(readdirSync(root).some((name) => name.startsWith("deepen-bank.legacy-"))).toBe(true);
  });

  it("reports heartbeat loss and preserves the successor generation", () => {
    vi.useFakeTimers();
    const lost = vi.fn();
    const lease = acquireDeepenLease(root, "bank", {
      heartbeatMs: 100,
      staleMs: 1_000,
      processIdentity: dead,
      listDeepenProcesses: () => [],
      onLost: lost,
    })!;
    const leaseDirectory = join(root, "leases", "deepen-bank.lock");
    const owner = readdirSync(leaseDirectory)[0];
    const old = new Date(Date.now() - 2_000);
    utimesSync(join(leaseDirectory, owner), old, old);
    const successor = acquireLease(join(root, "leases"), "bank", 1_000, "deepen")!;

    vi.advanceTimersByTime(100);

    expect(lost).toHaveBeenCalledTimes(1);
    lease.release();
    expect(heartbeatLease(successor)).toBe(true);
    releaseLease(successor);
  });

  it("cannot release a successor owner-token lease", () => {
    const lease = acquireDeepenLease(root, "bank", {
      staleMs: 1_000,
      processIdentity: dead,
      listDeepenProcesses: () => [],
    })!;
    const leaseDirectory = join(root, "leases", "deepen-bank.lock");
    const owner = readdirSync(leaseDirectory)[0];
    const old = new Date(Date.now() - 2_000);
    utimesSync(join(leaseDirectory, owner), old, old);
    const successor = acquireLease(join(root, "leases"), "bank", 1_000, "deepen")!;

    lease.release();

    expect(heartbeatLease(successor)).toBe(true);
    releaseLease(successor);
  });
});
