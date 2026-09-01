import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type AutoUpdateOptions,
  CHECK_INTERVAL_MS,
  isNewer,
  maybeAutoUpdate,
  selfUpdatable,
  stateFile,
} from "./auto-update";

describe("isNewer", () => {
  it("compares releases numerically, not lexically", () => {
    expect(isNewer("0.4.3", "0.4.2")).toBe(true);
    expect(isNewer("0.10.0", "0.9.9")).toBe(true); // the classic string-compare trap
    expect(isNewer("1.0.0", "0.99.99")).toBe(true);
    expect(isNewer("0.4.2", "0.4.3")).toBe(false);
    expect(isNewer("0.4.2", "0.4.2")).toBe(false);
  });

  it("never pulls a machine from a release onto a prerelease of it", () => {
    expect(isNewer("1.2.0-rc.1", "1.2.0")).toBe(false);
    expect(isNewer("1.2.0", "1.2.0-rc.1")).toBe(true);
    expect(isNewer("1.3.0-rc.1", "1.2.0")).toBe(true); // a later version still wins
  });

  it("refuses to act on a version it cannot parse", () => {
    expect(isNewer("latest", "0.4.2")).toBe(false);
    expect(isNewer("0.4", "0.4.2")).toBe(false);
    expect(isNewer("0.4.x", "0.4.2")).toBe(false);
    expect(isNewer("0.4.3", "")).toBe(false);
  });
});

describe("maybeAutoUpdate", () => {
  const dirs: string[] = [];
  const tmp = () => {
    const d = mkdtempSync(join(tmpdir(), "hs-autoupdate-"));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
    delete process.env.HINDSIGHT_DISABLE_HOOKS;
  });

  /** A spawn stub recording (bin, args, options) — the three things this module's contract is
   *  about — without standing up a real ChildProcess. */
  const spawnMock = () =>
    vi.fn((_bin: string, _args: string[], _o: Record<string, unknown>) => ({
      on: vi.fn(),
      unref: vi.fn(),
    }));
  const asSpawn = (m: ReturnType<typeof spawnMock>): AutoUpdateOptions["spawn"] =>
    m as unknown as AutoUpdateOptions["spawn"];
  const asFetch = (m: unknown): AutoUpdateOptions["fetch"] => m as AutoUpdateOptions["fetch"];

  /** A staged runtime directory holding `version`, plus the seams the checker needs. */
  const staged = (version: string) => {
    const runtime = tmp();
    writeFileSync(join(runtime, "package.json"), JSON.stringify({ version }));
    const spawn = spawnMock();
    const fetchOk = (latest: string) =>
      vi.fn(async () => ({ ok: true, json: async () => ({ version: latest }) }));
    return { runtime, spawn, fetchOk };
  };

  const opts = (
    runtime: string,
    extra: {
      spawn?: ReturnType<typeof spawnMock>;
      fetch?: unknown;
      now?: number;
      lockFile?: string;
      selfUpdatable?: (dir: string) => boolean;
      binOnPath?: (bin: string) => boolean;
    }
  ): AutoUpdateOptions => ({
    pkgRoot: runtime,
    runtimeDir: runtime,
    // Per-test lock inside the throwaway runtime dir: the real one is machine-global, and tests
    // running in parallel would otherwise block each other for LOCK_STALE_MS.
    lockFile: extra.lockFile ?? join(runtime, "update.lock"),
    spawn: extra.spawn ? asSpawn(extra.spawn) : undefined,
    fetch: asFetch(extra.fetch),
    now: extra.now,
    // Default the ownership guards open so each test exercises the behaviour it is about; the
    // guards themselves have their own tests below.
    selfUpdatable: extra.selfUpdatable ?? (() => true),
    binOnPath: extra.binOnPath ?? (() => true),
  });

  it("spawns a detached stage-only update when the registry is ahead", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const started = await maybeAutoUpdate(
      { autoUpdate: true },
      opts(runtime, { spawn, fetch: fetchOk("0.4.3") })
    );

    expect(started).toBe("0.4.3");
    expect(spawn).toHaveBeenCalledTimes(1);
    const [bin, args, spawnOpts] = spawn.mock.calls[0];
    expect(bin).toBe("npx");
    // Pinned to the version we resolved, and `update` — never `install`, which would rewire hosts.
    expect(args).toEqual(["-y", "@vectorize-io/hindsight-coding-agents@0.4.3", "update"]);
    expect(args).not.toContain("install");
    expect(spawnOpts.detached).toBe(true);
  });

  it("does nothing when the staged version is already current", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.3");
    expect(
      await maybeAutoUpdate({ autoUpdate: true }, opts(runtime, { spawn, fetch: fetchOk("0.4.3") }))
    ).toBe("");
    expect(spawn).not.toHaveBeenCalled();
  });

  it("is off when autoUpdate is false — the flag reaches the registry call, not just the spawn", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const fetchImpl = fetchOk("0.4.3");
    expect(
      await maybeAutoUpdate({ autoUpdate: false }, opts(runtime, { spawn, fetch: fetchImpl }))
    ).toBe("");
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(spawn).not.toHaveBeenCalled();
  });

  it("never touches a checkout or an npx run — only the staged copy updates itself", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const checkout = tmp();
    writeFileSync(join(checkout, "package.json"), JSON.stringify({ version: "0.4.2" }));
    expect(
      await maybeAutoUpdate(
        { autoUpdate: true },
        { ...opts(runtime, { spawn, fetch: fetchOk("0.4.3") }), pkgRoot: checkout }
      )
    ).toBe("");
    expect(spawn).not.toHaveBeenCalled();
  });

  it("checks at most once per interval, and stamps the attempt even when it finds nothing", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const fetchImpl = fetchOk("0.4.2"); // up to date: nothing spawned, but the check still happened
    const t0 = 1_000_000_000;

    await maybeAutoUpdate(
      { autoUpdate: true },
      opts(runtime, { spawn, fetch: fetchImpl, now: t0 })
    );
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(JSON.parse(readFileSync(stateFile(runtime), "utf8")).lastCheck).toBe(t0);

    // A second session an hour later reads the stamp and does not call out again.
    await maybeAutoUpdate(
      { autoUpdate: true },
      opts(runtime, { spawn, fetch: fetchImpl, now: t0 + 3_600_000 })
    );
    expect(fetchImpl).toHaveBeenCalledTimes(1);

    // A day later it is due again.
    await maybeAutoUpdate(
      { autoUpdate: true },
      opts(runtime, { spawn, fetch: fetchImpl, now: t0 + CHECK_INTERVAL_MS })
    );
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("stamps a FAILED check too, so an offline machine asks once a day rather than every session", async () => {
    const { runtime, spawn } = staged("0.4.2");
    const offline = vi.fn(async () => {
      throw new Error("ENOTFOUND registry.npmjs.org");
    });
    const t0 = 1_000_000_000;

    expect(
      await maybeAutoUpdate({ autoUpdate: true }, opts(runtime, { spawn, fetch: offline, now: t0 }))
    ).toBe("");
    expect(JSON.parse(readFileSync(stateFile(runtime), "utf8")).lastCheck).toBe(t0);
    await maybeAutoUpdate(
      { autoUpdate: true },
      opts(runtime, { spawn, fetch: offline, now: t0 + 60_000 })
    );
    expect(offline).toHaveBeenCalledTimes(1);
    expect(spawn).not.toHaveBeenCalled();
  });

  // Several agents starting at once all read "due" before any has stamped it. Without a lock they
  // all spawn `update`, and two concurrent stageRuntime runs (rmSync dist, then cpSync) can leave a
  // half-written runtime with missing entry points.
  it("lets only ONE of several simultaneous session starts spawn an updater", async () => {
    const { runtime, fetchOk } = staged("0.4.2");
    const lock = join(tmp(), "auto-update.lock");
    const spawns = [spawnMock(), spawnMock(), spawnMock()];
    const now = 1_000_000_000;

    const started = await Promise.all(
      spawns.map((spawn) =>
        maybeAutoUpdate(
          { autoUpdate: true },
          opts(runtime, { spawn, fetch: fetchOk("0.4.3"), now, lockFile: lock })
        )
      )
    );

    expect(started.filter(Boolean)).toEqual(["0.4.3"]);
    expect(spawns.filter((s) => s.mock.calls.length).length).toBe(1);
  });

  it("frees the lock when the check finds nothing, so the next window is not blocked", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const lock = join(tmp(), "auto-update.lock");
    const t0 = 1_000_000_000;

    await maybeAutoUpdate(
      { autoUpdate: true },
      opts(runtime, { spawn, fetch: fetchOk("0.4.2"), now: t0, lockFile: lock })
    );
    expect(existsSync(lock)).toBe(false);

    // A day later the next check can claim it and act.
    expect(
      await maybeAutoUpdate(
        { autoUpdate: true },
        opts(runtime, {
          spawn,
          fetch: fetchOk("0.4.3"),
          now: t0 + CHECK_INTERVAL_MS,
          lockFile: lock,
        })
      )
    ).toBe("0.4.3");
  });

  it("treats a lock whose holder is gone as stale rather than waiting out the TTL", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const lock = join(tmp(), "auto-update.lock");
    // pid 2^22 is above every Linux/macOS pid_max — nothing can be running under it.
    writeFileSync(lock, JSON.stringify({ pid: 4_194_304, ts: Date.now() }));

    expect(
      await maybeAutoUpdate(
        { autoUpdate: true },
        opts(runtime, { spawn, fetch: fetchOk("0.4.3"), lockFile: lock })
      )
    ).toBe("0.4.3");
  });

  // A user who runs `npm i -g` (or vendors the package as a project dependency) manages its
  // version with npm. Re-staging behind their back would leave `npm ls -g` naming a version that
  // is no longer what runs, with no way to reconcile the two.
  it("leaves a runtime alone when it was not staged from an npx download", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const fetchImpl = fetchOk("0.4.3");
    expect(
      await maybeAutoUpdate(
        { autoUpdate: true },
        opts(runtime, { spawn, fetch: fetchImpl, selfUpdatable: () => false })
      )
    ).toBe("");
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(spawn).not.toHaveBeenCalled();
  });

  it("skips the check entirely when npx is not on PATH — there would be nothing to spawn", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const fetchImpl = fetchOk("0.4.3");
    expect(
      await maybeAutoUpdate(
        { autoUpdate: true },
        opts(runtime, { spawn, fetch: fetchImpl, binOnPath: (bin) => bin !== "npx" })
      )
    ).toBe("");
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(spawn).not.toHaveBeenCalled();
  });

  it("stamps both refusals, so neither repeats its reason on every session start", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    const t0 = 1_000_000_000;
    await maybeAutoUpdate(
      { autoUpdate: true },
      opts(runtime, { spawn, fetch: fetchOk("0.4.3"), now: t0, selfUpdatable: () => false })
    );
    expect(JSON.parse(readFileSync(stateFile(runtime), "utf8")).lastCheck).toBe(t0);
  });

  it("stays out of the survey's own headless session", async () => {
    const { runtime, spawn, fetchOk } = staged("0.4.2");
    process.env.HINDSIGHT_DISABLE_HOOKS = "1";
    expect(
      await maybeAutoUpdate({ autoUpdate: true }, opts(runtime, { spawn, fetch: fetchOk("0.4.3") }))
    ).toBe("");
    expect(spawn).not.toHaveBeenCalled();
  });

  it("never guesses when the staged version is unreadable", async () => {
    const runtime = tmp(); // no package.json at all
    const spawn = spawnMock();
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => ({ version: "9.9.9" }) }));
    expect(
      await maybeAutoUpdate({ autoUpdate: true }, opts(runtime, { spawn, fetch: fetchImpl }))
    ).toBe("");
    expect(spawn).not.toHaveBeenCalled();
  });

  it("survives a spawn that fails asynchronously", async () => {
    const { runtime, fetchOk } = staged("0.4.2");
    const handlers: Record<string, (e: Error) => void> = {};
    const spawn = vi.fn((_bin: string, _args: string[], _o: Record<string, unknown>) => ({
      on: (event: string, cb: (e: Error) => void) => {
        handlers[event] = cb;
      },
      unref: vi.fn(),
    }));
    await maybeAutoUpdate(
      { autoUpdate: true },
      {
        pkgRoot: runtime,
        runtimeDir: runtime,
        spawn: asSpawn(spawn as never),
        fetch: asFetch(fetchOk("0.4.3")),
      }
    );
    expect(() => handlers.error?.(new Error("spawn npx ENOENT"))).not.toThrow();
  });
});

/**
 * Session-start parity across harnesses — the same shape as the daemon guard in daemon.test.ts,
 * and for the same reason (#3524): session-start housekeeping keeps getting written for the
 * fresh-process hook harnesses and missed by the persistent-plugin hosts, which call the shared
 * lifecycle directly. A harness that never checks for updates is stuck on its installed version
 * forever with nothing to show for it, and the per-harness test that would catch it is by
 * definition the one nobody wrote.
 *
 * `buildSessionStartContext({` is the call, not the definition — every host that runs a session
 * start makes it, so it identifies the family without a hand-maintained list.
 */
describe("every session-start path checks for updates", () => {
  const SRC = fileURLToPath(new URL("..", import.meta.url));

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  it("has no session-start entry point that skips maybeAutoUpdate", () => {
    const sessionStarts = sourceFiles(SRC).filter((rel) =>
      readFileSync(join(SRC, rel), "utf8").includes("buildSessionStartContext({")
    );
    // If this list is empty the check has stopped checking anything — the call was renamed.
    expect(sessionStarts.length).toBeGreaterThan(0);
    const missing = sessionStarts.filter(
      (rel) => !readFileSync(join(SRC, rel), "utf8").includes("maybeAutoUpdate")
    );
    expect(missing).toEqual([]);
  });
});

/**
 * The origin marker is what separates "we downloaded this" from "somebody else's copy". It fails
 * closed on purpose: a wrong `true` overwrites a global install or a developer's built dist, while
 * a wrong `false` costs one manual `install`.
 */
describe("selfUpdatable", () => {
  const dirs: string[] = [];
  const tmp = () => {
    const d = mkdtempSync(join(tmpdir(), "hs-origin-"));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });

  const withOrigin = (source: unknown): string => {
    const d = tmp();
    writeFileSync(join(d, ".install-origin.json"), JSON.stringify({ source }));
    return d;
  };

  it("accepts a runtime staged from an npx cache", () => {
    expect(
      selfUpdatable(
        withOrigin("/Users/u/.npm/_npx/a1b2c3/node_modules/@vectorize-io/hindsight-coding-agents")
      )
    ).toBe(true);
    // Windows separators reach the same verdict — the marker records whatever path staged it.
    expect(
      selfUpdatable(withOrigin("C:\\Users\\u\\AppData\\npm-cache\\_npx\\a1\\node_modules\\pkg"))
    ).toBe(true);
  });

  it("refuses a global install, a project dependency and a checkout", () => {
    expect(
      selfUpdatable(withOrigin("/usr/local/lib/node_modules/@vectorize-io/hindsight-coding-agents"))
    ).toBe(false);
    expect(
      selfUpdatable(withOrigin("/home/u/proj/node_modules/@vectorize-io/hindsight-coding-agents"))
    ).toBe(false);
    expect(
      selfUpdatable(withOrigin("/home/u/dev/hindsight/hindsight-integrations/coding-agents"))
    ).toBe(false);
  });

  it("refuses a runtime with no marker, an unreadable one, or an empty source", () => {
    expect(selfUpdatable(tmp())).toBe(false);
    const broken = tmp();
    writeFileSync(join(broken, ".install-origin.json"), "not json");
    expect(selfUpdatable(broken)).toBe(false);
    expect(selfUpdatable(withOrigin(""))).toBe(false);
    expect(selfUpdatable(withOrigin(42))).toBe(false);
  });
});
