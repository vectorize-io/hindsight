import { describe, it, expect, vi, afterEach } from "vitest";
import { Logger, type LogLevel } from "./logger.js";

interface Captured {
  service: string;
  level: LogLevel;
  message: string;
  extra?: Record<string, unknown>;
}

function makeClient(): { log: ReturnType<typeof vi.fn>; calls: Captured[] } {
  const calls: Captured[] = [];
  const log = vi.fn((opts: { body: Captured }) => {
    calls.push(opts.body);
    return Promise.resolve();
  });
  return { log, calls };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Logger", () => {
  it("routes error/warn/info to OpenCode's app.log with the hindsight service", () => {
    const { log, calls } = makeClient();
    const logger = new Logger({ client: { app: { log } } });

    logger.info("init", { api: "http://localhost:8888" });
    logger.warn("watch out");
    logger.error("boom", new Error("network down"));

    expect(calls).toHaveLength(3);
    expect(calls.every((c) => c.service === "hindsight")).toBe(true);
    expect(calls[0]).toMatchObject({ level: "info", message: "init" });
    expect(calls[0].extra).toEqual({ api: "http://localhost:8888" });
    expect(calls[1]).toMatchObject({ level: "warn", message: "watch out" });
    expect(calls[2].level).toBe("error");
    expect(String(calls[2].extra?.error)).toContain("network down");
  });

  it("suppresses debug unless debug is enabled", () => {
    const off = makeClient();
    new Logger({ client: { app: { log: off.log } }, debug: false }).debug("verbose");
    expect(off.calls).toHaveLength(0);

    const on = makeClient();
    new Logger({ client: { app: { log: on.log } }, debug: true }).debug("verbose");
    expect(on.calls).toHaveLength(1);
    expect(on.calls[0]).toMatchObject({ level: "debug", message: "verbose" });
  });

  it("emits nothing when silent (used as the default in tests)", () => {
    const { log, calls } = makeClient();
    const logger = new Logger({ client: { app: { log } }, debug: true, silent: true });
    logger.info("x");
    logger.error("y");
    logger.debug("z");
    expect(calls).toHaveLength(0);
  });

  it("falls back to console.error when no OpenCode client is available", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    new Logger().error("no client here");
    expect(spy).toHaveBeenCalledTimes(1);
    expect(String(spy.mock.calls[0][0])).toContain("[Hindsight]");
    expect(String(spy.mock.calls[0][0])).toContain("no client here");
  });

  it("calls app.log with `this` bound to app (regression for detached-method crash)", () => {
    // Mirrors OpenCode's real client: app.log is a method that uses `this`.
    // A detached call (`const log = app.log; log(...)`) would throw on
    // `this._client` and silently log nothing.
    const calls: Captured[] = [];
    const app = {
      _client: { ok: true },
      log(opts: { body: Captured }) {
        // Throws if `this` is lost (i.e. called detached).
        if (!this || !this._client) throw new TypeError("this._client is undefined");
        calls.push(opts.body);
        return Promise.resolve();
      },
    };
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    new Logger({ client: { app } }).info("init", { api: "http://localhost:8888" });

    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ service: "hindsight", level: "info", message: "init" });
    // It went through app.log, not the console fallback.
    expect(spy).not.toHaveBeenCalled();
  });

  it("never throws when app.log throws", () => {
    const logger = new Logger({
      client: {
        app: {
          log: () => {
            throw new Error("log endpoint exploded");
          },
        },
      },
    });
    expect(() => logger.error("still fine")).not.toThrow();
  });

  it("swallows rejected app.log promises (no unhandled rejection)", async () => {
    const logger = new Logger({
      client: { app: { log: () => Promise.reject(new Error("server 500")) } },
    });
    expect(() => logger.info("fire and forget")).not.toThrow();
    // Give the rejected promise a tick to settle; it must be caught internally.
    await Promise.resolve();
  });
});
