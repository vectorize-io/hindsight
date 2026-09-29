import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { appendLogLine, describeError, LOG_MAX_BYTES } from "./log";

describe("appendLogLine", () => {
  it("creates an owner-only directory and rotates one generation past the cap", () => {
    const root = mkdtempSync(join(tmpdir(), "hs-log-"));
    try {
      const file = join(root, "logs", "diag.jsonl");
      appendLogLine(file, "first\n");
      expect(statSync(join(root, "logs")).mode & 0o777).toBe(0o700);

      writeFileSync(file, "x".repeat(LOG_MAX_BYTES));
      appendLogLine(file, "after\n");
      expect(readFileSync(file, "utf8")).toBe("after\n");
      expect(statSync(`${file}.1`).size).toBe(LOG_MAX_BYTES);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("describeError", () => {
  it("names the transport failure Node's fetch hides behind 'fetch failed'", () => {
    // The real shape of an unreachable apiUrl: undici reports the wrapper, and only the cause
    // says WHY. Logging the message alone reads as "fetch failed" and diagnoses nothing.
    const cause = Object.assign(new Error("connect ECONNREFUSED 127.0.0.1:8888"), {
      code: "ECONNREFUSED",
    });
    const error = Object.assign(new TypeError("fetch failed"), { cause });

    expect(describeError(error)).toBe(
      "fetch failed: connect ECONNREFUSED 127.0.0.1:8888 (ECONNREFUSED)"
    );
  });

  it("renders a plain error, a bare value and a nested chain", () => {
    expect(describeError(new Error("bank not found"))).toBe("bank not found");
    expect(describeError("something odd")).toBe("something odd");
    expect(
      describeError(new Error("outer", { cause: new Error("middle", { cause: "root" }) }))
    ).toBe("outer: middle: root");
  });

  it("does not repeat a cause that restates its wrapper", () => {
    expect(describeError(new Error("timeout", { cause: new Error("timeout") }))).toBe("timeout");
  });

  it("survives a self-referencing cause and caps the length", () => {
    const looping = new Error("loop") as Error & { cause?: unknown };
    looping.cause = looping;
    expect(describeError(looping)).toBe("loop");
    expect(describeError(new Error("x".repeat(500)))).toHaveLength(200);
  });
});
